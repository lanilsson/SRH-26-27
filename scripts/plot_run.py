"""Turn one run record into a self-contained HTML report with charts.

AI-generated: Claude Code (Opus 5.5), 2026-10-12. Figures made with this
script must be marked AI-generated and cited per the ISEF generative-AI table.

Usage: python scripts/plot_run.py path/to/run.json
Writes path/to/run.html next to the JSON (open it in a browser).

Charts: training loss and validation PR-AUC per epoch (from the record's
"history"), precision-recall and ROC curves on validation and test, and, for
Elliptic, fraud-class F1 per test time step. The curves need the model's
scores, so the script reloads the weights from checkpoints/{run}.pt and
recomputes them (no retraining). That checkpoint is overwritten whenever the
same dataset/condition/seed is trained again, so plot a run before rerunning it.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import DATASETS  # noqa: E402
from src.models import ConfigurableGCN  # noqa: E402
from src.train import CKPT_DIR, phase_operators  # noqa: E402

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta")
MAX_POINTS = 400  # points kept per curve


def thin(xs, ys):
    """Keep at most MAX_POINTS evenly spaced points of a curve, sorted by x."""
    order = np.lexsort((ys, xs))
    xs, ys = np.asarray(xs)[order], np.asarray(ys)[order]
    idx = np.unique(np.linspace(0, len(xs) - 1, min(len(xs), MAX_POINTS)).astype(int))
    return [[round(float(x), 4), round(float(y), 4)] for x, y in zip(xs[idx], ys[idx])]


def recompute_scores(record):
    cfg = record["config"]
    data = DATASETS[cfg["dataset"]]()
    featureless_index = getattr(data, "featureless_index", None)
    num_featureless = int((featureless_index >= 0).sum()) if featureless_index is not None else 0
    model = ConfigurableGCN(in_dim=data.num_node_features, num_featureless=num_featureless,
                            **cfg["model"])
    model.load_state_dict(torch.load(CKPT_DIR / f"{record['run']}.pt"))
    model.eval()
    ops = phase_operators(data, model.required_operators())
    with torch.no_grad():
        scores = {p: torch.sigmoid(model(data.x, ops[p], featureless_index)).numpy()
                  for p in ("val", "test")}
    return data, scores


def build_payload(record):
    data, scores = recompute_scores(record)
    y = data.y.numpy()
    threshold = record["metrics"]["threshold"]
    payload = {"record": {k: v for k, v in record.items() if k != "history"},
               "history": record.get("history"), "curves": {}, "check": {}}

    for split in ("val", "test"):
        mask = getattr(data, f"{split}_mask").numpy()
        yt, ys = y[mask], scores[split][mask]
        precision, recall, _ = precision_recall_curve(yt, ys)
        fpr, tpr, _ = roc_curve(yt, ys)
        payload["curves"][split] = {"pr": thin(recall, precision), "roc": thin(fpr, tpr),
                                    "fraud_rate": round(float(yt.mean()), 4)}
        # recomputed metrics must agree with the record, or the checkpoint is from another run
        payload["check"][split] = {"pr_auc": float(average_precision_score(yt, ys)),
                                   "roc_auc": float(roc_auc_score(yt, ys))}

    recorded = record["metrics"]["pr_auc"]
    if abs(payload["check"]["test"]["pr_auc"] - recorded) > 1e-4:
        raise SystemExit(f"checkpoints/{record['run']}.pt does not reproduce this record "
                         f"(test PR-AUC {payload['check']['test']['pr_auc']:.4f} vs {recorded:.4f}); "
                         "it was probably overwritten by a later run.")

    if hasattr(data, "time_step"):
        steps, mask = data.time_step.numpy(), data.test_mask.numpy()
        rows = []
        for s in sorted(set(steps[mask].tolist())):
            m = mask & (steps == s)
            yt, yp = y[m], (scores["test"][m] >= threshold).astype(int)
            rows.append({"step": int(s), "labelled": int(m.sum()), "fraud": int(yt.sum()),
                         "f1": round(float(f1_score(yt, yp, zero_division=0)), 4),
                         "precision": round(float(precision_score(yt, yp, zero_division=0)), 4),
                         "recall": round(float(recall_score(yt, yp, zero_division=0)), 4)})
        payload["by_step"] = rows
    return payload


def main():
    path = Path(sys.argv[1])
    record = json.loads(path.read_text())
    payload = build_payload(record)
    template = (Path(__file__).parent / "plot_run_template.html").read_text()
    out = path.with_suffix(".html")
    out.write_text(template.replace("__TITLE__", record["run"])
                   .replace("__PAYLOAD__", json.dumps(payload).replace("</", "<\\/")))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
