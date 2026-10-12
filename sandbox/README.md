# Sandbox — learn-and-test area

Scratch code for building the study one piece at a time. Nothing here is the
final system: `src/` is where the real code lands once a piece is understood
and works. Nothing here goes into the paper, so no citation needed.

Areas, in the order we build them:

| Folder                        | What gets figured out here                                  |
| ----------------------------- | ----------------------------------------------------------- |
| `01_data_loading/`            | Raw CSVs -> a PyG `Data` graph (nodes, edges, labels, masks) |
| `02_gnn_foundation/`          | The plain GCN: message passing, layers, forward pass         |
| `03_heterophily_mechanisms/`  | Ego–neighbor separation and second-order aggregation         |
| `04_training_loop/`           | Train / validate, early stopping, seeding, checkpoints       |
| `05_evaluation_metrics/`      | PR-AUC, fraud-class F1, ROC-AUC on imbalanced labels         |
| `06_testing/`                 | Sanity checks and unit tests for the pieces above            |
| `07_mini_experiments/`        | Small end-to-end runs before touching `scripts/run_all.sh`   |

Conventions:
- Each folder has its own `README.md` with what to learn and a checklist.
- Anything a script writes (plots, tiny `.pt` files, logs) goes in that
  folder's `outputs/`, which is gitignored.
- Use small subsets of the data here. Full runs happen from `src/`.
- Run scripts from the repo root with the project venv, e.g.
  `.venv/bin/python sandbox/02_gnn_foundation/some_script.py`
