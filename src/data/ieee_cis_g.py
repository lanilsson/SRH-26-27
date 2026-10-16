"""IEEE-CIS-G dataset: CSVs in data/raw/ieee_cis_g/ -> one PyG graph.



Reproduces the H2GB construction (Lin et al., 2025; code:
github.com/junhongmit/H2GB, H2GB/datasets/ieee_cis_dataset.py and
H2GB/loader/master_loader.py::preformat_IEEE_CIS), then flattens it to one
node type for the GCN-style models, as H2GB itself does for homogeneous
GNNs (HeteroData.to_homogeneous after the per-type encoder):

  Transactions  train_transaction INNER JOIN train_identity on TransactionID
                (H2GB also concatenates the unlabelled Kaggle test files, then
                drops every row without isFraud, so only these remain):
                144,233 transactions, 7.85% fraud, already in TransactionDT
                order (checked below).
  Metadata      11 node types: card1..card6, ProductCD, addr1, addr2,
                P_emaildomain, R_emaildomain. One node per distinct value of
                each column (numbered in order of first appearance); a NaN
                value gives no node and no edge. 9,647 metadata nodes.
  Edges         transaction <-> metadata value, undirected (H2GB applies
                ToUndirected): 1,436,736 pairs -> 2,873,472 directed entries.
  Features      transactions only. Categorical columns (M1-M9, DeviceType,
                DeviceInfo, id_12..id_38): NaN -> 0, then one-hot. Numeric
                (all other columns): NaN -> 0, log10(|TransactionAmt|+1e-9).
                x = [one-hot, numeric].
                Then, per the proposal (Build Procedure step 1), every column
                is standardised with the mean / std of TRAINING transactions
                only. This replaces H2GB's z-score over all transactions and
                its PyG NormalizeFeatures (row-sum scaling), which would use
                validation/test statistics or undo the standardisation.
                Columns constant over the training transactions are then
                dropped (464 of 4,785), before standardising.
                Dimension: 4,785 -> 4,321. (H2GB reports 4,823: its concat
                with the test files adds 38 columns named "id-01".."id-38",
                hyphen spelling in the Kaggle test file, that are all 0 for
                the kept rows.)
  Metadata      no features; each metadata node gets a learnable embedding
  nodes         (H2GB "Raw" encoder). Marked by `featureless_index`.
  Split         first 80% / next 10% / last 10% of transactions in time order
                (H2GB preformat_IEEE_CIS): 115,386 / 14,423 / 14,424.

Node order in the flat graph: transactions 0..T-1, then the metadata types
in the order above.

Data fields returned:
  x [N, 4321] float (rows of metadata nodes are 0), edge_index [2, E] long,
  y [N] long (-1 for metadata), train/val/test_mask [N] bool,
  node_split [N] long (0/1/2 for transactions, -1 = metadata, always
  present in message passing), featureless_index [N] long (-1 for
  transactions, else row in the embedding table), node_type [N] long
  (0 = transaction, 1..11 = METADATA_TYPES), transaction_dt [N] long.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.data.transforms import standardize_by_train

RAW_DIR = Path("data/raw/ieee_cis_g")
CACHE = Path("data/processed/ieee_cis_g.pt")

METADATA_TYPES = [
    "card1", "card2", "card3", "card4", "card5", "card6",
    "ProductCD", "addr1", "addr2", "P_emaildomain", "R_emaildomain",
]
CATEGORICAL_COLS = (
    [f"M{i}" for i in range(1, 10)] + ["DeviceType", "DeviceInfo"]
    + [f"id_{i}" for i in range(12, 39)]
)
EXCLUDE_COLS = ["TransactionID", "isFraud", "TransactionDT"]
SPLIT_FRACTIONS = (0.8, 0.1)  # train, val; test is the rest


def build_ieee_cis_g():
    transactions = pd.read_csv(RAW_DIR / "train_transaction.csv")
    identity = pd.read_csv(RAW_DIR / "train_identity.csv")
    df = pd.merge(transactions, identity, on="TransactionID").reset_index(drop=True)
    assert df["TransactionDT"].is_monotonic_increasing, "split assumes time order"
    num_tx = len(df)

    # --- transaction features (H2GB order: one-hot block, then numeric block)
    numeric_cols = [c for c in df.columns
                    if c not in METADATA_TYPES + EXCLUDE_COLS + CATEGORICAL_COLS]
    feats = df[numeric_cols + CATEGORICAL_COLS].fillna(0)
    feats["TransactionAmt"] = np.log10(np.abs(feats["TransactionAmt"]) + 1e-9)
    one_hot = torch.tensor(pd.get_dummies(feats[CATEGORICAL_COLS], columns=CATEGORICAL_COLS)
                           .to_numpy(dtype=np.float32))
    numeric = torch.tensor(feats[numeric_cols].to_numpy(dtype=np.float32))
    x_tx = torch.cat([one_hot, numeric], dim=-1)
    del feats, one_hot, numeric

    n_train = int(SPLIT_FRACTIONS[0] * num_tx)
    n_val = int(SPLIT_FRACTIONS[1] * num_tx)
    is_train = torch.zeros(num_tx, dtype=torch.bool)
    is_train[:n_train] = True
    # Drop columns that are constant over the training transactions (mostly
    # categories first seen later in time): the model cannot learn anything
    # from them, yet their untrained weights would still reach val/test
    # predictions. Decided from training rows only, so no leakage. (D10)
    varies_in_train = x_tx[is_train].std(0) > 0
    x_tx = standardize_by_train(x_tx[:, varies_in_train], is_train)

    # --- metadata nodes and edges
    src, dst, node_type = [], [], [np.zeros(num_tx, dtype=np.int64)]
    offset = num_tx
    for t, col in enumerate(METADATA_TYPES, start=1):
        codes, uniques = pd.factorize(df[col])  # first-appearance order, NaN -> -1
        has_value = codes >= 0
        src.append(np.flatnonzero(has_value))
        dst.append(codes[has_value] + offset)
        node_type.append(np.full(len(uniques), t, dtype=np.int64))
        offset += len(uniques)
    num_nodes = offset
    src, dst = np.concatenate(src), np.concatenate(dst)
    edge_index = torch.tensor(np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])]))

    # --- assemble the flat graph
    x = torch.zeros(num_nodes, x_tx.shape[1])
    x[:num_tx] = x_tx
    y = torch.full((num_nodes,), -1, dtype=torch.long)
    y[:num_tx] = torch.tensor(df["isFraud"].to_numpy(), dtype=torch.long)

    node_split = torch.full((num_nodes,), -1, dtype=torch.long)
    node_split[:n_train] = 0
    node_split[n_train:n_train + n_val] = 1
    node_split[n_train + n_val:num_tx] = 2

    featureless_index = torch.full((num_nodes,), -1, dtype=torch.long)
    featureless_index[num_tx:] = torch.arange(num_nodes - num_tx)
    transaction_dt = torch.full((num_nodes,), -1, dtype=torch.long)
    transaction_dt[:num_tx] = torch.tensor(df["TransactionDT"].to_numpy())

    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=node_split == 0,
        val_mask=node_split == 1,
        test_mask=node_split == 2,
        node_split=node_split,
        featureless_index=featureless_index,
        node_type=torch.tensor(np.concatenate(node_type)),
        transaction_dt=transaction_dt,
        num_nodes=num_nodes,
    )


def load_ieee_cis_g(rebuild=False):
    """Return the IEEE-CIS-G graph, building and caching it on first use."""
    if CACHE.exists() and not rebuild:
        return torch.load(CACHE, weights_only=False)
    data = build_ieee_cis_g()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, CACHE)
    return data
