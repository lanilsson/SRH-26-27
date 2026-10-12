"""Elliptic Bitcoin dataset -> PyG Data object with the study's splits.

AI-generated: Claude Code (Opus 5.5), 2026-10-11. Log the prompt in the
project prompt log and cite per the ISEF generative-AI table.

Loaded with PyTorch Geometric's built-in EllipticBitcoinDataset (proposal,
Build Procedure step 1), reading the CSVs already in data/raw/elliptic/:
  - elliptic_txs_features.csv: col 0 = txId, col 1 = time step (1..49),
    cols 2..166 = 165 features. PyG drops the time step, so x has 165 columns
    (Weber et al. count it among their 166 features).
  - elliptic_txs_edgelist.csv: directed payment-flow edges (txId1 -> txId2)
  - elliptic_txs_classes.csv:  PyG maps licit -> 0, illicit -> 1, unknown -> 2

PyG's loader has train (steps 1-34) and test (35-49) masks only, directed
edges and globally pre-standardised features. On top of it, this module:
  - makes edges undirected (adds the reverse of every edge);
  - relabels unknown as -1. Unknown nodes stay in the graph for message
    passing but are in no mask, so they never enter the loss or the metrics;
  - splits by time step (Weber et al., 2019): 1-29 train, 30-34 val,
    35-49 test. Every node (labelled or not) gets a `node_split` of 0/1/2;
    train.py uses it to restrict message passing to "this split and
    earlier" in each phase. (Elliptic has no edges between time steps, so
    here the restriction does not change any prediction.);
  - re-standardises features with the mean / std of training-split nodes
    only (all nodes in steps 1-29, labelled or not).

Data fields returned:
  x [N, 165] float, edge_index [2, E] long (undirected), y [N] long,
  train_mask / val_mask / test_mask [N] bool (labelled nodes only),
  node_split [N] long (0 train, 1 val, 2 test), time_step [N] long.
"""

from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import EllipticBitcoinDataset
from torch_geometric.utils import to_undirected

from src.data.transforms import standardize_by_train

RAW_DIR = Path("data/raw/elliptic")
PYG_DIR = Path("data/processed/elliptic_pyg")
CACHE = Path("data/processed/elliptic.pt")

TRAIN_STEPS = (1, 29)
VAL_STEPS = (30, 34)
TEST_STEPS = (35, 49)


class _LocalElliptic(EllipticBitcoinDataset):
    """PyG's loader, pointed at the CSVs already in data/raw/elliptic/
    instead of downloading them again."""

    @property
    def raw_dir(self):
        return str(RAW_DIR)

    @property
    def processed_dir(self):
        return str(PYG_DIR)


def build_elliptic():
    pyg = _LocalElliptic(root=str(PYG_DIR))[0]

    # PyG assigns labels by row position, assuming the classes file lists
    # txIds in the same order as the features file. Check that it does.
    tx_ids = pd.read_csv(RAW_DIR / "elliptic_txs_features.csv", header=None, usecols=[0, 1])
    class_ids = pd.read_csv(RAW_DIR / "elliptic_txs_classes.csv", usecols=["txId"])
    assert (tx_ids[0].to_numpy() == class_ids["txId"].to_numpy()).all()
    time_step = torch.tensor(tx_ids[1].to_numpy(), dtype=torch.long)

    y = pyg.y.clone()
    y[y == 2] = -1

    node_split = torch.full_like(time_step, -1)
    for split_id, (lo, hi) in enumerate([TRAIN_STEPS, VAL_STEPS, TEST_STEPS]):
        node_split[(time_step >= lo) & (time_step <= hi)] = split_id
    assert (node_split >= 0).all(), "every Elliptic node must fall in a split"

    labelled = y >= 0
    return Data(
        x=standardize_by_train(pyg.x, node_split == 0),
        edge_index=to_undirected(pyg.edge_index, num_nodes=pyg.num_nodes),
        y=y,
        train_mask=labelled & (node_split == 0),
        val_mask=labelled & (node_split == 1),
        test_mask=labelled & (node_split == 2),
        node_split=node_split,
        time_step=time_step,
    )


def load_elliptic(rebuild=False):
    """Return the Elliptic graph, building and caching it on first use."""
    if CACHE.exists() and not rebuild:
        return torch.load(CACHE, weights_only=False)
    data = build_elliptic()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, CACHE)
    return data
