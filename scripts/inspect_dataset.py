"""Print a data-check report for one dataset: features, edges, labels,
splits, and the homophily ratio — the numbers to record in the notebook
before any training.

AI-generated: Claude Code (Opus 5.5), 2026-10-11. Log the prompt in the
project prompt log and cite per the ISEF generative-AI table.

Usage: python scripts/inspect_dataset.py elliptic
       python scripts/inspect_dataset.py ieee_cis_g

Homophily ratio (Zhu et al., 2020): among edges whose endpoints are BOTH
labelled, the fraction joining nodes of the same class. For a bipartite
graph (IEEE-CIS-G: transaction - metadata - transaction) direct edges never
join two labelled nodes, so the ratio is also reported over 2-hop pairs
(labelled nodes sharing a neighbour), counted once per shared neighbour.
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import DATASETS  # noqa: E402

SPLITS = ("train", "val", "test")


def edge_homophily(y, src, dst):
    both = (y[src] >= 0) & (y[dst] >= 0)
    if both.sum() == 0:
        return float("nan"), 0
    return (y[src][both] == y[dst][both]).float().mean().item(), int(both.sum())


def two_hop_homophily(y, edge_index, num_nodes):
    """Same-class fraction over labelled pairs (u, v), u != v, that share a
    neighbour w, without materialising A^2: for every w, count same-class
    and labelled pairs among its labelled neighbours."""
    src, dst = edge_index
    lab = y[src] >= 0
    w, lbl = dst[lab], y[src][lab]
    n_lab = torch.zeros(num_nodes).index_add_(0, w, torch.ones(len(w)))
    n_pos = torch.zeros(num_nodes).index_add_(0, w, lbl.float())
    n_neg = n_lab - n_pos
    pairs = n_lab * (n_lab - 1)
    same = n_pos * (n_pos - 1) + n_neg * (n_neg - 1)
    return (same.sum() / pairs.sum()).item(), int(pairs.sum().item())


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "elliptic"
    t = time.perf_counter()
    d = DATASETS[name]()
    print(f"== {name}  (loaded in {time.perf_counter() - t:.1f}s)")
    print(d)

    featured = torch.ones(d.num_nodes, dtype=torch.bool)
    if hasattr(d, "featureless_index"):
        featured = d.featureless_index < 0
        print(f"nodes: {d.num_nodes:,} ({int(featured.sum()):,} with features, "
              f"{int((~featured).sum()):,} featureless)")
    else:
        print(f"nodes: {d.num_nodes:,}")

    x = d.x[featured]
    std = x.std(0)
    print(f"features: dim {x.shape[1]}, NaN {int(torch.isnan(x).sum())}, "
          f"constant columns {int((std == 0).sum())}, "
          f"mean range [{x.mean(0).min():.3f}, {x.mean(0).max():.3f}], "
          f"std range [{std.min():.3f}, {std.max():.3f}]")

    src, dst = d.edge_index
    deg = torch.bincount(src, minlength=d.num_nodes)
    print(f"edges: {d.edge_index.size(1):,} directed entries "
          f"(symmetric: {_is_symmetric(d.edge_index, d.num_nodes)}), "
          f"self-loops {int((src == dst).sum())}")
    print(f"degree: mean {deg.float().mean():.2f}, median {deg.median().item()}, "
          f"max {deg.max().item():,}, isolated nodes {int((deg == 0).sum()):,}")

    print("labels (1 = fraud/illicit):")
    for s in SPLITS:
        m = getattr(d, f"{s}_mask")
        pos = int(d.y[m].sum())
        n = int(m.sum())
        print(f"  {s:5s}: {n:>8,} labelled, {pos:>6,} fraud ({pos / max(n, 1):.2%})")
    print(f"  unlabelled nodes in graph: {int((d.y < 0).sum()):,}")
    if hasattr(d, "node_split"):
        counts = {int(k): int(v) for k, v in zip(*torch.unique(d.node_split, return_counts=True))}
        print(f"  node_split counts (-1 = always present): {counts}")

    h, n = edge_homophily(d.y, src, dst)
    print(f"homophily ratio (1-hop edges, labelled-labelled): {h:.4f} over {n:,} edge entries")
    for c, label in ((1, "fraud"), (0, "licit")):
        from_c = (d.y[src] == c) & (d.y[dst] >= 0)
        if from_c.any():
            frac = (d.y[dst][from_c] == c).float().mean().item()
            print(f"  from {label} nodes: {frac:.4f} of labelled neighbours are also {label}")
    h2, n2 = two_hop_homophily(d.y, d.edge_index, d.num_nodes)
    print(f"homophily ratio (2-hop pairs via shared neighbour): {h2:.4f} over {n2:,} ordered pairs")


def _is_symmetric(edge_index, n):
    fwd = edge_index[0] * n + edge_index[1]
    rev = edge_index[1] * n + edge_index[0]
    return bool(torch.equal(torch.sort(fwd).values, torch.sort(rev).values))


if __name__ == "__main__":
    main()
