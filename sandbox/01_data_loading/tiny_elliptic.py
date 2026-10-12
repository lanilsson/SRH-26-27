"""Build a small Elliptic graph and inspect every field on it.

Run from the repo root:
    .venv/bin/python sandbox/01_data_loading/tiny_elliptic.py --max-step 3

Elliptic has 49 "time steps" (roughly two-week windows of Bitcoin activity).
Edges never cross time steps, so keeping only the first few steps gives a
self-contained small graph that is still real data.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

RAW = Path("data/raw/elliptic")
OUT = Path(__file__).parent / "outputs"


def build(max_step: int) -> Data:
    # ------------------------------------------------------------------ 1
    # Node features. The CSV has NO header row. Column 0 is the transaction
    # id, column 1 is the time step, and columns 2..166 are the 165 features
    # (94 "local" features about the tx itself + 71 aggregated neighbour
    # features that Elliptic precomputed).
    feats = pd.read_csv(RAW / "elliptic_txs_features.csv", header=None)
    feats = feats.rename(columns={0: "txId", 1: "step"})
    feats = feats[feats["step"] <= max_step].reset_index(drop=True)

    # Every node needs an integer index 0..N-1 for PyG. The row order of
    # `feats` *is* that index, so remember which txId sits at which row.
    tx_to_idx = {tx: i for i, tx in enumerate(feats["txId"])}

    x = torch.tensor(feats.iloc[:, 2:].to_numpy(), dtype=torch.float)

    # ------------------------------------------------------------------ 2
    # Edges. Directed payment flow txId1 -> txId2. Keep only edges whose
    # endpoints are both in our slice (they should all be, since edges
    # stay within a time step, but check anyway).
    edges = pd.read_csv(RAW / "elliptic_txs_edgelist.csv")
    keep = edges["txId1"].isin(tx_to_idx) & edges["txId2"].isin(tx_to_idx)
    edges = edges[keep]

    src = edges["txId1"].map(tx_to_idx).to_numpy()
    dst = edges["txId2"].map(tx_to_idx).to_numpy()
    # PyG wants shape [2, num_edges]: row 0 = source index, row 1 = target.
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)

    # ------------------------------------------------------------------ 3
    # Labels. "1" = illicit, "2" = licit, "unknown" = no label.
    # We want fraud to be the positive class (1) because PR-AUC is about the
    # positive class. Licit -> 0. Unknown -> -1 so it is easy to mask out.
    classes = pd.read_csv(RAW / "elliptic_txs_classes.csv")
    label_map = {"1": 1, "2": 0, "unknown": -1}
    classes["y"] = classes["class"].astype(str).map(label_map)
    # Align labels to node order.
    y_series = classes.set_index("txId").loc[feats["txId"], "y"]
    y = torch.tensor(y_series.to_numpy(), dtype=torch.long)

    # ------------------------------------------------------------------ 4
    # Bundle everything into one Data object. Extra attributes (like step
    # and labeled_mask) are allowed; PyG just carries them along.
    data = Data(x=x, edge_index=edge_index, y=y)
    data.step = torch.tensor(feats["step"].to_numpy(), dtype=torch.long)
    data.labeled_mask = y >= 0
    return data


def describe(data: Data) -> None:
    print(f"nodes        : {data.num_nodes}")
    print(f"edges        : {data.num_edges}")
    print(f"feature dim  : {data.num_node_features}")
    print(f"x shape      : {tuple(data.x.shape)}")
    print(f"edge_index   : {tuple(data.edge_index.shape)}  dtype={data.edge_index.dtype}")
    print(f"y shape      : {tuple(data.y.shape)}")
    print(f"time steps   : {sorted(data.step.unique().tolist())}")

    n_illicit = int((data.y == 1).sum())
    n_licit = int((data.y == 0).sum())
    n_unknown = int((data.y == -1).sum())
    n_labeled = n_illicit + n_licit
    print(f"illicit      : {n_illicit}")
    print(f"licit        : {n_licit}")
    print(f"unknown      : {n_unknown}")
    print(f"labeled frac : {n_labeled / data.num_nodes:.3f}")
    print(f"fraud rate   : {n_illicit / n_labeled:.3f}  (among labeled nodes)")

    # Sanity checks the study will rely on later.
    assert data.edge_index.min() >= 0
    assert data.edge_index.max() < data.num_nodes
    assert data.x.shape[0] == data.y.shape[0] == data.num_nodes
    data.validate(raise_on_error=True)  # PyG's own consistency check
    print("checks       : edge indices in range, shapes agree, PyG validate OK")

    # Are there isolated nodes? Degree tells us how connected the graph is.
    deg = torch.bincount(data.edge_index.flatten(), minlength=data.num_nodes)
    print(f"isolated     : {int((deg == 0).sum())} nodes with no edges")
    print(f"mean degree  : {deg.float().mean():.2f}   max degree: {int(deg.max())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-step", type=int, default=3, help="keep time steps 1..N")
    args = ap.parse_args()

    data = build(args.max_step)
    describe(data)

    OUT.mkdir(exist_ok=True)
    torch.save(data, OUT / f"tiny_elliptic_steps1-{args.max_step}.pt")
    print(f"saved        : outputs/tiny_elliptic_steps1-{args.max_step}.pt")
