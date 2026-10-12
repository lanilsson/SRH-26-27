# 04 — Training loop

Goal: a clean train / validate loop that later becomes `src/train.py`.

Learn here:
- Cross-entropy on masked nodes only (transductive setting).
- Early stopping on validation PR-AUC with a patience counter.
- Seeding torch / numpy / random so runs are repeatable.
- Saving the best checkpoint and a JSON record of the run.

Checklist:
- [ ] Train the baseline on the tiny graph for a few epochs, print loss
- [ ] Add validation metric each epoch and early stopping
- [ ] Run the same seed twice and confirm identical results
- [ ] Write one run record to `outputs/`
