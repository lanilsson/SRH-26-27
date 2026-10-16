"""Overfit a handful of labelled nodes on purpose to prove the pipeline trains.

Run from the repo root:
    .venv/bin/python sandbox/02_gnn_foundation/overfit_few_nodes.py
    .venv/bin/python sandbox/02_gnn_foundation/overfit_few_nodes.py --k 20 --epochs 300

If a model cannot drive the loss to ~0 on 10 illicit + 10 licit nodes,
something is broken (wrong loss, no gradient, bad shapes). If it can, the
forward pass, the loss, the backward pass and the optimiser all work. This
says nothing about how well it generalises — that is what 04/05 are for.
Writes outputs/overfit_loss.csv with the per-epoch curve.
"""

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F

from two_layer_forward import TwoLayerGCN, load_tiny

OUT = Path(__file__).parent / "outputs"


def pick_nodes(data, k: int, seed: int):
    """k illicit + k licit node indices, chosen at random from the labelled ones."""
    g = torch.Generator().manual_seed(seed)
    illicit = (data.y == 1).nonzero().flatten()
    licit = (data.y == 0).nonzero().flatten()
    illicit = illicit[torch.randperm(len(illicit), generator=g)[:k]]
    licit = licit[torch.randperm(len(licit), generator=g)[:k]]
    return torch.cat([illicit, licit])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10, help="nodes per class to memorise")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pyg", action="store_true", help="use GCNConv instead of HandGCNConv")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    data = load_tiny(undirected=True)
    train_idx = pick_nodes(data, args.k, args.seed)
    y_train = data.y[train_idx].float()  # BCE wants float targets 0.0 / 1.0
    print(f"graph        : {data.num_nodes} nodes, {data.num_edges} edges")
    print(f"memorising   : {len(train_idx)} nodes ({args.k} illicit + {args.k} licit)")

    # Dropout off (dropout=0): we WANT to overfit here.
    model = TwoLayerGCN(data.num_node_features, args.hidden, dropout=0.0, use_pyg=args.pyg)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    rows = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        # Full-batch: run the whole graph every step (neighbours are needed
        # for message passing) but only the chosen nodes contribute to loss.
        logits = model(data.x, data.edge_index)
        loss = F.binary_cross_entropy_with_logits(logits[train_idx], y_train)
        loss.backward()
        opt.step()

        with torch.no_grad():
            pred = (logits[train_idx] > 0).float()  # logit > 0  <=>  prob > 0.5
            acc = (pred == y_train).float().mean().item()
        rows.append((epoch, loss.item(), acc))
        if epoch == 1 or epoch % 20 == 0:
            print(f"epoch {epoch:4d}   loss {loss.item():.4f}   train acc {acc:.2f}")

    OUT.mkdir(exist_ok=True)
    with open(OUT / "overfit_loss.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "loss", "train_acc"])
        w.writerows(rows)
    print(f"saved        : outputs/overfit_loss.csv")

    final_loss, final_acc = rows[-1][1], rows[-1][2]
    assert final_acc == 1.0, f"did not memorise: acc {final_acc}"
    assert final_loss < rows[0][1] * 0.1, f"loss barely moved: {rows[0][1]:.3f} -> {final_loss:.3f}"
    print(f"checks       : loss {rows[0][1]:.3f} -> {final_loss:.4f}, all {len(train_idx)} nodes classified correctly")

    # The nodes it did NOT see. Expect this to be poor/uninformative — it is
    # only printed so the difference between memorising and learning is visible.
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
    held = data.labeled_mask.clone()
    held[train_idx] = False
    pred = (logits[held] > 0).long()
    yh = data.y[held]
    tp = int(((pred == 1) & (yh == 1)).sum())
    fp = int(((pred == 1) & (yh == 0)).sum())
    fn = int(((pred == 0) & (yh == 1)).sum())
    print(f"unseen nodes : {int(held.sum())} labelled, of which {int((yh == 1).sum())} illicit; "
          f"predicted illicit: {tp} correct, {fp} false alarms, {fn} missed")
