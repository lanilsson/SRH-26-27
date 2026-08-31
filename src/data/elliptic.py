"""Elliptic Bitcoin dataset: CSVs in data/raw/elliptic/ -> PyG Data object.

Inputs:
  - elliptic_txs_features.csv: node id, time step, 166 features
  - elliptic_txs_edgelist.csv: directed payment-flow edges (txId1 -> txId2)
  - elliptic_txs_classes.csv:  labels ("1" = illicit, "2" = licit, "unknown")

Design decisions to implement (per the proposal):
  - Unlabeled nodes stay in the graph for message passing but are excluded
    from loss and evaluation (masks cover labeled nodes only).
  - Use the standard temporal split by time step (train on early steps,
    test on later ones) so results are comparable to Weber et al. (2019).
  - Cache the built graph to data/processed/elliptic.pt; rebuild only if
    the cache is missing.
"""

from pathlib import Path

RAW_DIR = Path("data/raw/elliptic")
CACHE = Path("data/processed/elliptic.pt")


def load_elliptic():
    """Return a torch_geometric.data.Data with x, edge_index, y,
    train_mask, val_mask, test_mask."""
    raise NotImplementedError("TODO: build the Elliptic graph (see module docstring)")
