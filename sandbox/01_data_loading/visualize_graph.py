"""Export a viewable subgraph of the tiny Elliptic graph to an interactive HTML page.

Run from the repo root:
    .venv/bin/python sandbox/01_data_loading/visualize_graph.py --step 1 --max-nodes 3000
    open sandbox/01_data_loading/outputs/graph_step1.html

Add --3d for a rotatable three-dimensional version (graph3d_stepN.html).

A browser can comfortably lay out a few thousand nodes, and a single time
step of Elliptic is one connected blob of ~8k nodes. So we grow a subgraph
outward from the illicit nodes: seed with every fraud node, then add their
neighbours, then the neighbours' neighbours, until we hit --max-nodes. The
result is centred on fraud and every edge you see is a real edge between
two exported nodes. Edges to nodes we did not export are simply not drawn.
"""

import argparse
import json
from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx

HERE = Path(__file__).parent
OUT = HERE / "outputs"
TEMPLATE = HERE / "graph_template.html"
TEMPLATE_3D = HERE / "graph3d_template.html"


def grow_from_fraud(data: Data, step: int, max_nodes: int):
    # Restrict to one time step first.
    in_step = (data.step == step).nonzero().flatten()
    sub = data.subgraph(in_step)
    sub.orig_index = in_step  # remember position in the full tiny graph

    # Breadth-first growth. networkx gives us neighbours() for free; use the
    # undirected view so we expand along payments in both directions.
    g = to_networkx(sub, to_undirected=True)
    frontier = [int(i) for i in (sub.y == 1).nonzero().flatten()]
    chosen = set(frontier)
    hops = 0
    while frontier and len(chosen) < max_nodes:
        nxt = []
        for u in frontier:
            for v in g.neighbors(u):
                if v not in chosen:
                    chosen.add(v)
                    nxt.append(v)
                    if len(chosen) >= max_nodes:
                        break
            if len(chosen) >= max_nodes:
                break
        frontier = nxt
        hops += 1

    keep = torch.tensor(sorted(chosen), dtype=torch.long)
    out = sub.subgraph(keep)
    out.orig_index = sub.orig_index[keep]
    return out, sub.num_nodes, hops


def to_json(sub: Data, tx_ids):
    src, dst = sub.edge_index
    out_deg = torch.bincount(src, minlength=sub.num_nodes)
    in_deg = torch.bincount(dst, minlength=sub.num_nodes)
    label_name = {1: "illicit", 0: "licit", -1: "unknown"}
    nodes = [
        {
            "id": i,
            "idx": int(sub.orig_index[i]),
            "tx": int(tx_ids[i]),
            "cls": label_name[int(sub.y[i])],
            "step": int(sub.step[i]),
            "in": int(in_deg[i]),
            "out": int(out_deg[i]),
        }
        for i in range(sub.num_nodes)
    ]
    links = [{"s": int(s), "t": int(t)} for s, t in zip(src.tolist(), dst.tolist())]
    return {"nodes": nodes, "links": links}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default=str(OUT / "tiny_elliptic_steps1-3.pt"))
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--max-nodes", type=int, default=3000)
    ap.add_argument("--3d", dest="three_d", action="store_true", help="write the 3D viewer instead")
    args = ap.parse_args()

    data = torch.load(args.graph, weights_only=False)
    # tx ids were not saved on the Data object; recover them from the raw CSV
    # in the same row order tiny_elliptic.py used.
    import pandas as pd
    feats = pd.read_csv("data/raw/elliptic/elliptic_txs_features.csv", header=None, usecols=[0, 1])
    feats = feats[feats[1] <= int(data.step.max())].reset_index(drop=True)
    all_tx = feats[0].to_numpy()

    sub, n_step, hops = grow_from_fraud(data, args.step, args.max_nodes)
    payload = to_json(sub, all_tx[sub.orig_index.numpy()])

    print(f"time step {args.step}: {n_step} nodes total")
    print(f"exported {sub.num_nodes} nodes / {sub.num_edges} edges, "
          f"grown {hops} hops out from the illicit nodes")
    counts = {c: sum(1 for n in payload['nodes'] if n['cls'] == c) for c in ("illicit", "licit", "unknown")}
    print(f"classes: {counts}")

    template = TEMPLATE_3D if args.three_d else TEMPLATE
    html = template.read_text().replace("/*__GRAPH_JSON__*/", json.dumps(payload, separators=(",", ":")))
    target = OUT / f"graph{'3d' if args.three_d else ''}_step{args.step}.html"
    target.write_text(html)
    print(f"wrote {target.relative_to(Path.cwd())}")
