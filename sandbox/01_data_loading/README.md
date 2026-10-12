# 01 — Data loading

Goal: turn the raw CSVs in `data/raw/` into a PyTorch Geometric `Data` object
and understand every field on it.

Learn here:
- What `x`, `edge_index`, `y`, and the train/val/test masks are and their shapes.
- How Elliptic's classes file (1 = illicit, 2 = licit, unknown) maps to labels.
- How IEEE-CIS transactions become a graph (what is a node, what is an edge).
- How to load only a small slice so experiments run in seconds.

Checklist:
- [x] Load a few thousand Elliptic rows and build a tiny graph (`tiny_elliptic.py`)
- [x] Print node count, edge count, feature dim, class balance
- [x] Confirm edge_index is shape [2, num_edges] and 0-indexed
- [x] Look at the graph interactively (`visualize_graph.py` -> `outputs/graph_step1.html`)
- [ ] Same for a small IEEE-CIS-G slice
