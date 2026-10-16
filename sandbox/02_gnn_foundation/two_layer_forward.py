"""A 2-layer GCN built from HandGCNConv, run forward on the tiny Elliptic graph.


Run from the repo root (after sandbox/01 has produced the .pt file):
    .venv/bin/python sandbox/02_gnn_foundation/two_layer_forward.py
    .venv/bin/python sandbox/02_gnn_foundation/two_layer_forward.py --pyg   # swap in GCNConv

Stacking two layers means each node's output depends on its 2-hop
neighbourhood: layer 1 mixes in direct neighbours, layer 2 mixes in the
neighbours' (already mixed) representations. The model ends in a single
logit per node — the binary "is this transaction illicit?" score that
src/models/gcn.py will also produce.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected

from gcn_layer import HandGCNConv

TINY = Path("sandbox/01_data_loading/outputs/tiny_elliptic_steps1-3.pt")


class TwoLayerGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, dropout=0.5, use_pyg=False):
        super().__init__()
        Layer = GCNConv if use_pyg else HandGCNConv
        self.conv1 = Layer(in_dim, hidden_dim)
        self.conv2 = Layer(hidden_dim, 1)  # one logit per node
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)            # [N, hidden]  1-hop information
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        logit = self.conv2(h, edge_index)        # [N, 1]       2-hop information
        return logit.squeeze(-1)                 # [N]


def load_tiny(undirected: bool):
    data = torch.load(TINY, weights_only=False)
    if undirected:
        # Elliptic edges are directed payment flow (tx1 -> tx2). A GCN only
        # passes messages along edge direction, so a node that only *sends*
        # payments would hear nothing but itself. Making the graph
        # undirected lets information move both ways, which is the usual
        # choice for GCN. Whether to keep it directed is a study decision.
        data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    return data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pyg", action="store_true", help="use GCNConv instead of HandGCNConv")
    ap.add_argument("--directed", action="store_true", help="keep edges directed")
    ap.add_argument("--hidden", type=int, default=64)
    args = ap.parse_args()

    torch.manual_seed(0)
    data = load_tiny(undirected=not args.directed)
    print(f"graph        : {data.num_nodes} nodes, {data.num_edges} edges "
          f"({'directed' if args.directed else 'undirected'}), {data.num_node_features} features")

    model = TwoLayerGCN(data.num_node_features, args.hidden, use_pyg=args.pyg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model        : {'GCNConv' if args.pyg else 'HandGCNConv'} x2, hidden={args.hidden}, "
          f"{n_params} parameters")
    print(model)

    model.eval()  # turn dropout off so the pass is deterministic
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = torch.sigmoid(logits)

    print(f"logits shape : {tuple(logits.shape)}   (one per node)")
    print(f"logits range : [{logits.min():.3f}, {logits.max():.3f}]  mean {logits.mean():.3f}")
    print(f"probs  range : [{probs.min():.3f}, {probs.max():.3f}]")
    print(f"first 5      : {[round(v, 3) for v in logits[:5].tolist()]}")
    assert logits.shape == (data.num_nodes,)
    assert torch.isfinite(logits).all()
    print("checks       : one finite logit per node")

    # Untrained model, so the numbers mean nothing yet — the point is that
    # the shapes line up and the pass runs. Compare an illicit vs. licit
    # node just to see they come out of the same pipeline.
    illicit = (data.y == 1).nonzero().flatten()[0].item()
    licit = (data.y == 0).nonzero().flatten()[0].item()
    print(f"example      : node {illicit} (illicit) logit {logits[illicit]:.3f}, "
          f"node {licit} (licit) logit {logits[licit]:.3f}  [untrained]")

    # Does the hand layer really see 2 hops? Perturb one node's features and
    # see how far the change travels in the output.
    with torch.no_grad():
        x2 = data.x.clone()
        x2[illicit] += 10.0
        changed = (model(x2, data.edge_index) - logits).abs() > 1e-6
    print(f"receptive    : bumping node {illicit}'s features changed "
          f"{int(changed.sum())} nodes' logits (its 2-hop neighbourhood incl. itself)")
