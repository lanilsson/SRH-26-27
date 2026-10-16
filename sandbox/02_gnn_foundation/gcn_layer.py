"""One GCN layer written by hand, checked against torch_geometric's GCNConv.

Run from the repo root:
    .venv/bin/python sandbox/02_gnn_foundation/gcn_layer.py

A GCN layer (Kipf & Welling, 2017) computes, for every node i,

    h_i = b + sum over j in N(i) ∪ {i} of  x_j W^T / sqrt(deg(j) * deg(i))

i.e. (1) add a self-loop so a node hears its own features, (2) take a
degree-normalised average of the neighbours' (linearly transformed)
features, (3) add a bias. That is the whole layer. Two equivalent ways to
write it are below:

  * gcn_layer_dense   — the textbook matrix form  D^-1/2 (A + I) D^-1/2 X W^T.
                        Easy to read, but the N x N adjacency matrix does
                        not fit in memory once N is tens of thousands.
  * gcn_layer_scatter — the "message passing" form: one row per edge, then
                        index_add_ to sum the messages at their target.
                        Memory is O(edges), so this is what scales.

Both are compared against GCNConv with the same weights on a toy graph and
must agree to floating-point precision.
"""

import math

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


# ---------------------------------------------------------------------------
# 1. Dense (matrix) form
# ---------------------------------------------------------------------------
def gcn_layer_dense(x, edge_index, weight, bias):
    """x: [N, in], edge_index: [2, E] (src -> dst), weight: [out, in], bias: [out]."""
    n = x.size(0)
    src, dst = edge_index

    # A_hat[i, j] = 1 if there is an edge j -> i  (message flows into row i),
    # plus the identity so every node is its own neighbour.
    a_hat = torch.zeros(n, n, dtype=x.dtype)
    a_hat[dst, src] = 1.0
    a_hat = a_hat + torch.eye(n, dtype=x.dtype)

    # deg(i) = number of messages arriving at i (including the self-loop).
    deg = a_hat.sum(dim=1)
    d_inv_sqrt = deg.pow(-0.5)
    a_norm = d_inv_sqrt[:, None] * a_hat * d_inv_sqrt[None, :]  # D^-1/2 A D^-1/2

    return a_norm @ (x @ weight.T) + bias


# ---------------------------------------------------------------------------
# 2. Scatter (message passing) form — same maths, O(E) memory
# ---------------------------------------------------------------------------
def gcn_layer_scatter(x, edge_index, weight, bias):
    n = x.size(0)
    loops = torch.arange(n, device=x.device)
    src = torch.cat([edge_index[0], loops])  # add the self-loops as edges
    dst = torch.cat([edge_index[1], loops])

    # deg[i] = how many edges (incl. the self-loop) point at node i.
    deg = torch.zeros(n, dtype=x.dtype, device=x.device)
    deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
    d_inv_sqrt = deg.pow(-0.5)
    norm = d_inv_sqrt[src] * d_inv_sqrt[dst]  # one scalar per edge

    h = x @ weight.T                       # transform first (cheaper: [N, out] not [E, in])
    messages = h[src] * norm[:, None]      # [E, out]: what each edge carries
    out = torch.zeros(n, h.size(1), dtype=x.dtype, device=x.device)
    out.index_add_(0, dst, messages)       # sum messages at their target node
    return out + bias


# ---------------------------------------------------------------------------
# 3. Wrap the scatter form as an nn.Module so it can be stacked and trained
# ---------------------------------------------------------------------------
class HandGCNConv(nn.Module):
    """Drop-in replacement for GCNConv(in_dim, out_dim) built from the code above."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        # Glorot / Xavier uniform, the same init GCNConv uses.
        bound = math.sqrt(6.0 / (in_dim + out_dim))
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, x, edge_index):
        return gcn_layer_scatter(x, edge_index, self.weight, self.bias)


# ---------------------------------------------------------------------------
# 4. Check all three agree on a toy graph
# ---------------------------------------------------------------------------
def toy_graph():
    #   0 -> 1 -> 2 -> 0   (a directed triangle)
    #   2 -> 3             (3 has one incoming edge)
    #   4                  (isolated: only its self-loop reaches it)
    edge_index = torch.tensor([[0, 1, 2, 2],
                               [1, 2, 0, 3]], dtype=torch.long)
    x = torch.randn(5, 4)
    return x, edge_index


if __name__ == "__main__":
    torch.manual_seed(0)
    x, edge_index = toy_graph()
    in_dim, out_dim = x.size(1), 3

    # Reference layer. Its parameters: conv.lin.weight [out, in], conv.bias [out].
    conv = GCNConv(in_dim, out_dim)
    ref = conv(x, edge_index)

    w, b = conv.lin.weight, conv.bias
    dense = gcn_layer_dense(x, edge_index, w, b)
    scatter = gcn_layer_scatter(x, edge_index, w, b)

    hand = HandGCNConv(in_dim, out_dim)
    with torch.no_grad():  # copy the same weights into the module
        hand.weight.copy_(w)
        hand.bias.copy_(b)
    module = hand(x, edge_index)

    print("toy graph      : 5 nodes, 4 edges, in_dim=4 -> out_dim=3")
    print(f"GCNConv output : {tuple(ref.shape)}")
    print(f"max |dense   - GCNConv| = {(dense - ref).abs().max():.2e}")
    print(f"max |scatter - GCNConv| = {(scatter - ref).abs().max():.2e}")
    print(f"max |module  - GCNConv| = {(module - ref).abs().max():.2e}")
    assert torch.allclose(dense, ref, atol=1e-5)
    assert torch.allclose(scatter, ref, atol=1e-5)
    assert torch.allclose(module, ref, atol=1e-5)
    print("checks         : all three hand-written versions match GCNConv")

    # Look at the normalisation for one node to see the maths concretely.
    # Node 0 receives from node 2 and from itself. deg(0)=2, deg(2)=2 (from 1
    # and its own loop), so both weights are 1/sqrt(2*2) = 0.5.
    print("node 0 output  = 0.5 * (x_0 W^T) + 0.5 * (x_2 W^T) + b")
    manual0 = 0.5 * (x[0] @ w.T) + 0.5 * (x[2] @ w.T) + b
    print(f"max |manual node0 - GCNConv node0| = {(manual0 - ref[0]).abs().max():.2e}")
    assert torch.allclose(manual0, ref[0], atol=1e-5)

    # The gradient must flow through the hand-written module too.
    hand.zero_grad()
    module.sum().backward()
    assert hand.weight.grad is not None and hand.weight.grad.abs().sum() > 0
    print("checks         : gradients flow through HandGCNConv")
