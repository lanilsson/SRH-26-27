"""Train and evaluate one run of one experimental condition.

Usage:
    python -m src.train --config configs/elliptic_baseline.yaml --seed 0

Writes a self-contained record of the run (config, seed, git commit,
metrics) to results/runs/{dataset}_{condition}_seed{seed}.json and the
best model weights to checkpoints/.
"""

import argparse
import datetime
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score

from src.data import DATASETS
from src.evaluate import compute_metrics, select_threshold
from src.models import ConfigurableGCN

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = DATASETS[cfg["dataset"]]().to(device)
    model = ConfigurableGCN(in_dim=data.num_node_features, **cfg["model"]).to(device)
    tr = cfg["training"]
    optimizer = torch.optim.Adam(
        model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"]
    )

    y = data.y.float()
    best_val, best_state, best_epoch, patience_left = -1.0, None, 0, tr["patience"]
    for epoch in range(tr["epochs"]):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(
            logits[data.train_mask], y[data.train_mask]
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(
                model(data.x, data.edge_index).squeeze(-1)
            )
        val_pr_auc = average_precision_score(
            y[data.val_mask].cpu(), scores[data.val_mask].cpu()
        )
        if val_pr_auc > best_val:
            best_val, best_epoch, patience_left = val_pr_auc, epoch, tr["patience"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(data.x, data.edge_index).squeeze(-1)).cpu().numpy()
    y = y.cpu().numpy()
    val_mask = data.val_mask.cpu().numpy()
    test_mask = data.test_mask.cpu().numpy()

    threshold = select_threshold(y[val_mask], scores[val_mask])
    metrics = compute_metrics(y[test_mask], scores[test_mask], threshold)

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
        "config": cfg,
        "git_commit": git_commit(),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNS_DIR / f"{run_name}.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"{run_name}: PR-AUC={metrics['pr_auc']:.4f} "
          f"F1={metrics['f1']:.4f} ROC-AUC={metrics['roc_auc']:.4f} -> {out}")


if __name__ == "__main__":
    main()
