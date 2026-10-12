# 05 — Evaluation metrics

Goal: understand the metrics the study reports and why PR-AUC is primary.

Learn here:
- Why accuracy is useless when ~2% of nodes are fraud.
- PR-AUC vs ROC-AUC on imbalanced data (Saito & Rehmsmeier, 2015 in docs/).
- Fraud-class F1 and choosing a threshold.
- Computing all three with scikit-learn from raw logits.

Checklist:
- [ ] Build fake predictions and compute the three metrics by hand
- [ ] Show a case where ROC-AUC looks good but PR-AUC is bad
- [ ] Plot a PR curve into `outputs/`
