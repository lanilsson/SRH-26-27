# Final study procedure (locked 2026-10-20)

> **AI-generated reference.** Compiled by Claude (Anthropic, Opus 5.5) on
> 2026-10-20 from the 10-11-26 proposal, the code in this repository and the
> decisions Lucas made (D1–D13 in [week-2026-10-12.md](week-2026-10-12.md),
> OI-1 in [open-issues.md](open-issues.md)). It records *what* is done; it
> contains no results or interpretation. Per the ISEF generative-AI table,
> log the prompt and write the notebook entry in your own words.


## 1. Design

|                     |                                                                                                                                                            |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Question            | How ego–neighbor separation and explicit second-order aggregation, individually and combined, affect fraud classification in a controlled GCN-style model |
| Factor 1            | Ego–neighbor representation separation: absent / present                                                                                                  |
| Factor 2            | Explicit second-order (2-hop) neighborhood aggregation: absent / present                                                                                   |
| Configurations      | A baseline · B ego only · C second-order only · D combined (2 × 2)                                                                                     |
| Datasets            | Elliptic Bitcoin; IEEE-CIS-G (H²GB). Analyzed separately                                                                                                  |
| Dependent variables | PR-AUC (primary), fraud-class F1, ROC-AUC                                                                                                                  |
| Replication         | 5 final seeds (0–4) per configuration × dataset → **40 final runs**                                                                               |
| Held constant       | Layers (2 aggregation layers), tuning search, optimizer, loss, max epochs, patience, splits, features, hardware, software versions                         |

The only difference between configurations is two flags in `configs/*.yaml`
(`ego_separation`, `second_order`).

---

## 2. Materials and environment

- **Data:** Elliptic Bitcoin dataset (Kaggle, `data/raw/elliptic/`);
  IEEE-CIS-G train CSVs (H²GB Kaggle release, `data/raw/ieee_cis_g/`). Raw
  files are never modified.
- **Software:** Python with the pinned `requirements.txt` (PyTorch, PyTorch
  Geometric, scikit-learn, NumPy, Pandas, SciPy, statsmodels). Versions are
  recorded with `scripts/check_env.py`. (TBB) pin `statsmodels`.
- **Hardware:** all tuning and final runs on **one NVIDIA RTX 4090 (24 GB)**
  on RunPod (D7). The laptop (M3, CPU) is used only for smoke tests.
- **Reproducibility record:** every run writes a JSON with config, seed, git
  commit, `git_dirty` flag, metrics, training history and resource use.

---

## 3. Procedure

### Step 1: Set up the environment  local /  RunPod

1. Create the virtual environment and install `requirements.txt`.
2. Run `scripts/check_env.py` on the RunPod 4090 pod and save the output.
3. Run `python -m pytest -q`; all tests must pass on the pod.

### Step 2: Prepare the Elliptic graph  (`src/data/elliptic.py`)

1. Load the raw CSVs with PyTorch Geometric's `EllipticBitcoinDataset`
   → 203,769 nodes, **165 features** (PyG drops the time-step column; D2).
2. Check that the classes file and features file list txIds in the same order.
3. Relabel *unknown* nodes as −1: they stay in the graph for message passing
   but are excluded from the loss and all metrics.
4. Split by time step: **1–29 train, 30–34 validation, 35–49 test**.
5. Make edges undirected (add the reverse of every edge).
6. Standardize every feature with the mean / std of training-split nodes only.
7. Cache to `data/processed/elliptic.pt`.

### Step 3: Prepare the IEEE-CIS-G graph  (`src/data/ieee_cis_g.py`)

Built by reproducing the H²GB construction code (not the H²GB library; D11).

1. Inner-join `train_transaction.csv` and `train_identity.csv` on
   TransactionID → 144,233 transactions (7.85% fraud), checked to be in
   TransactionDT order.
2. Create one metadata node per distinct value of 11 columns (card1–card6,
   ProductCD, addr1, addr2, P_emaildomain, R_emaildomain) → 9,647 nodes;
   NaN gives no node and no edge.
3. Connect each transaction to its metadata nodes, undirected
   → 2,873,472 directed edge entries.
4. Transaction features: categorical columns NaN → 0 then one-hot; numeric
   columns NaN → 0; TransactionAmt → log10(|x| + 1e-9).
5. Split by time: first **80% train, next 10% validation, last 10% test**
   (115,386 / 14,423 / 14,424).
6. Drop the 464 columns that are constant over training transactions
   (4,785 → **4,321 features**; D10).
7. Standardize with training-transaction mean / std only. This replaces
   H²GB's all-transaction z-score and row normalization (D9).
8. Metadata nodes have no features; each gets a learnable embedding.
9. Verify node, edge, per-type and split counts against H²GB
   (`tests/test_pipeline.py`).

### Step 4: Measure homophily (Elliptic  edge homophily / (TBB) adjusted)

1. Run `scripts/inspect_dataset.py elliptic` and `... ieee_cis_g`.
2. Report, for both graphs, **edge homophily** (Zhu et al., 2020) and
   **adjusted homophily** (Platonov et al., 2023; H²GB's H_adj), plus H²GB-style
   metapath versions for IEEE-CIS-G, comparing them with H²GB Table 2 (OI-1).
3. This step measures and reports; it is done before training and does not
   change the experiment.
4. (TBB) Lucas reads Platonov (2023) and Lim (2021) before this is final;
   (TBB) code for adjusted / metapath homophily.

### Step 5: Build the four configurations  (`src/models/gcn.py`, `src/models/propagation.py`)

Shared structure:

1. **Input layer:** Linear(features → hidden) + ReLU + dropout; metadata nodes
   use their embedding instead.
2. **Two aggregation layers**, each: build the channels, concatenate them,
   then Linear + ReLU + dropout.
   - A baseline: GCN neighbor average with self-loop, `Â_self·h`.
   - B ego separation: `[h, Â·h]`, ego kept separate from the self-loop-free
     neighbor average.
   - C second-order: `[Â_self·h, Â₂·h]`.
   - D combined: `[h, Â·h, Â₂·h]`.
   - `Â₂` = nodes at exactly distance 2, weighted by number of shared neighbors
     (path counts rather than H2GCN's 0/1; D8). On IEEE-CIS-G it is computed
     implicitly via the bipartite identity, which is exact.
   - All operators are symmetric-normalized (D^-1/2 M D^-1/2).
3. **Output layer:** Linear(hidden → 1), one fraud logit per node.
4. Parameter counts differ by configuration because of the extra channels
   (D3), to be stated in the methods.

### Step 6: Leakage controls  (`src/train.py`)

1. Message passing per phase uses only nodes from that split and earlier:
   training on the train graph, validation on train + val, test on the full
   graph (metadata nodes always present).
2. Every statistic (feature mean / std, class weight, F1 threshold, dropped
   columns) comes from training or validation data only.

### Step 7: Verify the code  (`tests/test_pipeline.py`)

`pytest` must pass: operator correctness, implicit = explicit 2-hop, GCN
normalization matches PyG, all four configurations run forward and backward,
embeddings, phase graphs hide later splits, same seed → same result, metrics
and validation threshold, Elliptic validity, IEEE-CIS-G matches H²GB.

### Step 8: Training procedure (per run)  (`src/train.py`)

1. Set the seed (Python, NumPy, PyTorch, CUDA).
2. Optimizer: Adam (weight decay 5e-4).
3. Loss: binary cross-entropy with `pos_weight` = #licit / #illicit over
   training labels.
4. Each epoch: one full-batch training step, then validation PR-AUC.
5. Early stopping: **patience 50** epochs on validation PR-AUC, **max 200
   epochs**; restore the best-validation weights. `epochs_run` records whether
   a run hit the 200 cap.

### Step 9: Hyperparameter tuning (TBB) script to build (D4)

1. Grid, identical for every configuration and dataset:
   learning rate {0.001, 0.01} × hidden dimension {32, 64} × dropout
   {0.3, 0.5} = **8 settings**.
2. **3 tuning seeds** per setting, different from final seeds 0–4.
3. Train and validation data only; the test set is not evaluated.
4. Selection: the setting with the **highest mean validation PR-AUC** over
   the 3 seeds, per configuration × dataset.
5. Total: 8 × 3 × 4 × 2 = **192 tuning runs**.

### Step 10: Validation gate (TBB) check to build (D12)

Before the final runs, the tuned baseline on each dataset must have, on every
tuning seed:

- validation PR-AUC clearly above the validation fraud rate (random-classifier
  level), and
- validation ROC-AUC > 0.5.

A failure means a pipeline bug is investigated before continuing. No
test-set data is used.

### Step 11: Freeze 

Write the selected settings into the 8 `configs/*.yaml` files and **commit
them before any test-set evaluation**. Final runs must have
`git_dirty: false`.

### Step 12: Final runs  (`scripts/run_all.sh`)

1. 4 configurations × 2 datasets × seeds 0–4 = **40 runs** on the 4090.
2. Per run: train as in Step 8 with the frozen settings; choose the F1
   threshold that maximizes fraud-class F1 on validation; apply it unchanged
   to test.
3. Test metrics (fraud = positive class, scikit-learn): PR-AUC (average
   precision), fraud-class F1, ROC-AUC.
4. Each run's JSON goes to `results/runs/`; finished runs are skipped if the
   script is restarted.

### Step 13: Data analysis  aggregation / (TBB) ANOVA (D5)

Per dataset, separately:

1. Mean ± SEM over the 5 seeds for each configuration and metric
   (`scripts/aggregate.py` → `results/tables/summary.csv`).
2. **Two-way ANOVA**: ego separation × second-order, with interaction.
   PR-AUC is the primary test; F1 and ROC-AUC use the same test as secondary
   outcomes. Significance at **p < 0.05**.
3. Assumption checks reported: Shapiro–Wilk on residuals, Levene for equal
   variance (low power at n = 5 per cell).

### Step 14: Baseline comparison with Weber et al. (2019)  (D12)

Report the baseline's Elliptic test F1 (mean ± SEM over the 5 final runs)
next to Weber et al.'s ≈ 0.63, listing the setup differences (train steps
1–29 vs 1–34, class-weighted loss, validation-selected threshold). This is
reported, not a pass/fail gate.

---

## 4. Departures from the source methods (to state in the paper)

| #   | Departure                                                   | Reason                                                                                 |
| --- | ----------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| D2  | 165 Elliptic features, not 166                              | PyG's loader drops the time step                                                       |
| D8  | 2-hop operator uses path counts, not 0/1                    | 0/1 version of IEEE-CIS-G needs ~10¹⁰ entries                                        |
| D9  | IEEE-CIS-G standardized on train only; no row normalization | Avoid val/test leakage; row normalization would undo standardization                   |
| D10 | 464 train-constant IEEE-CIS-G columns dropped               | Cannot be learned; decided from train data only                                        |
| D11 | H²GB construction reimplemented, not the library           | Library conflicts with the pinned environment; split lives in its loader; counts match |
| D3  | Parameter counts differ by configuration                    | Inherent to adding channels                                                            |

---

## 5. Open before the final runs

- [ ] Lucas: read Platonov (2023) and Lim (2021) (OI-1)
- [ ] Lucas: proposal edits listed in D13
- [ ] Code: tuning script (Step 9) and validation gate (Step 10)
- [ ] Code: adjusted + metapath homophily (Step 4)
- [ ] Code: two-way ANOVA in `aggregate.py` (Step 13); pin `statsmodels`
- [ ] Code: `run_all.sh` uses the venv's Python
- [ ] Re-run `check_env.py` and `pytest` on the RunPod 4090
