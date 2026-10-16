"""Train and evaluate one run of one experimental condition.

Usage:
    python -m src.train --config configs/elliptic_baseline.yaml --seed 0

Writes a self-contained record of the run (config, seed, git commit,
metrics) to results/runs/{dataset}_{condition}_seed{seed}.json and the
best model weights to checkpoints/.

Proposal requirements implemented here, 2026-10-11 — phase graphs, class-weighted loss, resource
logging; log the prompt and cite per the ISEF generative-AI table):
  - Class-weighted loss: BCE with pos_weight = (#licit / #illicit) over the
    TRAINING labels only.
  - Message passing per phase uses only nodes from that split and earlier
    (data.node_split: 0 train, 1 val, 2 test; -1 = always present, e.g.
    IEEE-CIS-G metadata nodes). Training uses the train graph, validation /
    threshold selection the train+val graph, the test metrics the full graph.
  - Resource use (device, wall time, time per epoch, peak memory) is stored
    in the run record under "resources" for the cost/performance notebook.
"""

import argparse
import datetime
import json
import random
import resource
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score

from src.data import DATASETS
from src.evaluate import compute_metrics, select_threshold
from src.models import ConfigurableGCN
from src.models.propagation import build_operators

# torch prints a "sparse CSR is beta" warning on first use; CSR matmul is
# used deliberately (deterministic and supported on CPU and CUDA).
warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta")

RUNS_DIR = Path("results/runs")
CKPT_DIR = Path("checkpoints")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def git_dirty():
    """True if src/ or configs/ have uncommitted or untracked changes, i.e.
    git_commit() does not fully describe the code that produced the run."""
    try:
        return bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--", "src", "configs"], text=True
        ).strip())
    except Exception:
        return None


def phase_operators(data, which):
    """Build the operators for the train, val and test phase graphs: in phase
    k, keep only edges whose endpoints both have node_split <= k (nodes with
    node_split == -1 are always kept)."""
    split = data.node_split
    src, dst = data.edge_index
    # Featureless nodes form one side of a bipartite graph (IEEE-CIS-G);
    # needed for the implicit 2-hop operator. None for Elliptic.
    featureless_index = getattr(data, "featureless_index", None)
    side = featureless_index >= 0 if featureless_index is not None else None
    ops = {}
    for phase, k in (("train", 0), ("val", 1), ("test", 2)):
        allowed = split <= k  # -1 (always present) is <= every k
        keep = allowed[src] & allowed[dst]
        ops[phase] = build_operators(data.edge_index[:, keep], data.num_nodes, which, side)
    return ops


def peak_memory_mb(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2**20
    # ru_maxrss is bytes on macOS, kilobytes on Linux
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 2**10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR,
                        help="where the run JSON goes (use a sandbox outputs/ dir for smoke tests)")
    parser.add_argument("--epochs", type=int, help="override training.epochs (smoke tests only)")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t_start = time.perf_counter()
    data = DATASETS[cfg["dataset"]]().to(device)
    featureless_index = getattr(data, "featureless_index", None)
    num_featureless = int((featureless_index >= 0).sum()) if featureless_index is not None else 0
    model = ConfigurableGCN(in_dim=data.num_node_features, num_featureless=num_featureless,
                            **cfg["model"]).to(device)
    ops = phase_operators(data, model.required_operators())
    t_setup = time.perf_counter() - t_start

    def scores(phase):
        return torch.sigmoid(model(data.x, ops[phase], featureless_index))
    tr = cfg["training"]
    if args.epochs is not None:
        tr["epochs"] = args.epochs
    optimizer = torch.optim.Adam(
        model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"]
    )

    y = data.y.float()
    y_train = y[data.train_mask]
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    t_train = time.perf_counter()
    best_val, best_state, best_epoch, patience_left = -1.0, None, 0, tr["patience"]
    epochs_run = 0
    history = {"train_loss": [], "val_pr_auc": []}
    for epoch in range(tr["epochs"]):
        epochs_run = epoch + 1
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, ops["train"], featureless_index)
        loss = F.binary_cross_entropy_with_logits(
            logits[data.train_mask], y_train, pos_weight=pos_weight
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_scores = scores("val")
        val_pr_auc = average_precision_score(
            y[data.val_mask].cpu(), val_scores[data.val_mask].cpu()
        )
        history["train_loss"].append(round(float(loss), 6))
        history["val_pr_auc"].append(round(float(val_pr_auc), 6))
        if val_pr_auc > best_val:
            best_val, best_epoch, patience_left = val_pr_auc, epoch, tr["patience"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    train_seconds = time.perf_counter() - t_train

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_scores = scores("val").cpu().numpy()
        test_scores = scores("test").cpu().numpy()
    y = y.cpu().numpy()
    val_mask = data.val_mask.cpu().numpy()
    test_mask = data.test_mask.cpu().numpy()

    threshold = select_threshold(y[val_mask], val_scores[val_mask])
    metrics = compute_metrics(y[test_mask], test_scores[test_mask], threshold)
    val_metrics = compute_metrics(y[val_mask], val_scores[val_mask], threshold)

    run_name = f"{cfg['dataset']}_{cfg['condition']}_seed{args.seed}"
    CKPT_DIR.mkdir(exist_ok=True)
    torch.save(best_state, CKPT_DIR / f"{run_name}.pt")

    record = {
        "run": run_name,
        "dataset": cfg["dataset"],
        "condition": cfg["condition"],
        "seed": args.seed,
        "metrics": metrics,
        "best_epoch": best_epoch,
        "val_pr_auc": float(best_val),
        "val_metrics": val_metrics,
        "epochs_run": epochs_run,
        "history": history,
        "pos_weight": float(pos_weight),
        "num_params": sum(p.numel() for p in model.parameters()),
        "resources": {
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else device.type,
            "setup_seconds": round(t_setup, 2),
            "train_seconds": round(train_seconds, 2),
            "seconds_per_epoch": round(train_seconds / epochs_run, 4),
            "total_seconds": round(time.perf_counter() - t_start, 2),
            "peak_memory_mb": round(peak_memory_mb(device), 1),
            "torch": torch.__version__,
        },
        "config": cfg,
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    out = args.runs_dir / f"{run_name}.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"{run_name}: PR-AUC={metrics['pr_auc']:.4f} "
          f"F1={metrics['f1']:.4f} ROC-AUC={metrics['roc_auc']:.4f} | "
          f"{epochs_run} epochs, {record['resources']['total_seconds']}s, "
          f"peak {record['resources']['peak_memory_mb']} MB -> {out}")


if __name__ == "__main__":
    main()
