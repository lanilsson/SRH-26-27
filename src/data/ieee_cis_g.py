"""IEEE-CIS-G dataset: CSVs in data/raw/ieee_cis_g/ -> PyG graph.

Inputs:
  - train_transaction.csv / train_identity.csv
  - test_transaction.csv  / test_identity.csv

Design decisions to implement (per the proposal):
  - Follow the H2GB construction (Lin et al., 2024): transaction nodes
    linked through shared metadata (card, address, email domain, device),
    using the benchmark's published train/val/test split so results are
    comparable to the benchmark.
  - Decide and document how the heterogeneous structure is presented to
    the GCN-style models (e.g. metadata entities as nodes vs. collapsing
    shared-entity links into transaction-transaction edges).
  - Cache the built graph to data/processed/ieee_cis_g.pt.
"""

from pathlib import Path

RAW_DIR = Path("data/raw/ieee_cis_g")
CACHE = Path("data/processed/ieee_cis_g.pt")


def load_ieee_cis_g():
    """Return a graph object with features, edges, labels, and
    train/val/test masks matching the H2GB split."""
    raise NotImplementedError("TODO: build the IEEE-CIS-G graph (see module docstring)")
