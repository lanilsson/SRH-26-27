# Effect of Heterophily-Aware GNN Architecture on Fraud Detection in Financial Transaction Graphs

Lucas Nilsson — SRH 2026–27

A controlled comparison of two heterophily-aware mechanisms from Zhu et al.
(2020) — ego–neighbor representation separation and explicit second-order
neighborhood aggregation — within a GCN-style architecture, evaluated on the
Elliptic Bitcoin graph and IEEE-CIS-G (H²GB). Full details in
[docs/proposal/Proposal-8-27.md](docs/proposal/Proposal-8-27.md).

## Layout

```
data/raw/         Original dataset CSVs (Git LFS for the large files; never modified)
data/processed/   Cached PyG graph objects (.pt) — gitignored, rebuilt from raw
src/data/         CSV -> graph builders for each dataset
src/models/       ConfigurableGCN: the four conditions via two boolean flags
src/train.py      Trains one (config, seed) run; writes a JSON record per run
src/evaluate.py   PR-AUC (primary), fraud-class F1, ROC-AUC
configs/          One YAML per condition (8 = 2 datasets x 4 architectures)
scripts/          run_all.sh (all 40 runs), aggregate.py (mean ± SEM tables)
results/runs/     One JSON per run (metrics + config + seed + git commit) — committed
results/tables/   Aggregated summary tables the Data Analysis section uses
results/figures/  Plots for the paper/poster
notebooks/        Exploratory analysis (imbalance, homophily measurement, ...)
checkpoints/      Trained weights — gitignored
docs/             Proposal and other written deliverables
```

## Experimental conditions

Each config sets two flags on the same base GCN; nothing else differs
between architectures:

| Condition      | `ego_separation` | `second_order` |
| -------------- | ---------------- | -------------- |
| baseline       | false            | false          |
| ego            | true             | false          |
| second_order   | false            | true           |
| combined       | true             | true           |

Hyperparameters are tuned on train+validation only, then the config files
are frozen (committed) before any test-set evaluation.

## Reproducing

```
git clone <repo> && cd <repo>
git lfs pull                        # large CSVs
pip install torch==2.7.1            # CUDA build matching the machine
pip install -r requirements.txt
bash scripts/run_all.sh             # 8 conditions x 5 seeds -> results/runs/
python scripts/aggregate.py         # -> results/tables/summary.csv
```

A single run: `python -m src.train --config configs/elliptic_baseline.yaml --seed 0`

`run_all.sh` skips runs whose JSON already exists, so it can resume after an
interruption. On the RunPod machine, commit `results/runs/` back before
shutting the pod down — checkpoints and processed caches are disposable.
