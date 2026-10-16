"""Sparse neighbourhood operators shared by all four conditions.



Each operator is a symmetric-normalised matrix D^-1/2 M D^-1/2 (D = row sums
of M), applied to node representations as `op @ h`:

  adj_self  M = A + I        standard GCN (Kipf & Welling, 2017) — the ego
                             node is mixed into its own neighbourhood average
  adj       M = A            1-hop neighbours only, no self-loop (used with
                             ego separation, Zhu et al., 2020)
  adj2      M = A^2 with the diagonal and every 1-hop entry set to 0: only
                             nodes at distance EXACTLY 2 (Zhu et al., 2020,
                             H2GCN design 2). Entry (i, j) is the number of
                             shared neighbours (2-hop paths). H2GCN binarises
                             these to 0/1; path counts are used instead so the
                             same definition can be computed on IEEE-CIS-G,
                             where A^2 is far too large to store.

adj2 is built in one of two equivalent ways:
  - explicitly (sparse A @ A) when the number of 2-hop paths is small enough
    (Elliptic: ~4M entries);
  - implicitly when the graph is bipartite (IEEE-CIS-G: transaction <->
    metadata). In a bipartite graph no 2-hop pair is also a 1-hop pair, so
    M h = A (A h) - deg * h and its row sums are A deg - deg; nothing of
    size A^2 is ever formed. A few metadata values (e.g. card4, ProductCD)
    are shared by ~144k transactions, which would make A^2 ~10^10 entries.

The operators are built from whatever edge_index is passed in, so train.py
can build a separate set per phase (train / val / test graph) to keep
message passing within "this split and earlier".
"""

import torch

OPERATORS = ("adj_self", "adj", "adj2")

# Above this many 2-hop paths (sum of squared degrees), adj2 is not
# materialised and the graph must be bipartite.
MAX_EXPLICIT_TWO_HOP_PATHS = 50_000_000


def _coo(edge_index, num_nodes, values=None):
    if values is None:
        values = torch.ones(edge_index.size(1), device=edge_index.device)
    return torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes)).coalesce()


def _inv_sqrt(deg):
    out = deg.pow(-0.5)
    out[torch.isinf(out)] = 0.0  # degree-0 rows stay 0
    return out


def _sym_normalise(m):
    """D^-1/2 M D^-1/2 for a coalesced COO matrix, returned as sparse CSR."""
    row, col = m.indices()
    s = _inv_sqrt(torch.sparse.sum(m, dim=1).to_dense())
    values = s[row] * m.values() * s[col]
    return torch.sparse_coo_tensor(m.indices(), values, m.shape).coalesce().to_sparse_csr()


def two_hop(edge_index, num_nodes):
    """A^2 with diagonal and 1-hop entries removed, as a coalesced COO matrix
    whose values are the number of shared neighbours."""
    a = _coo(edge_index, num_nodes)
    a2 = torch.sparse.mm(a, a).coalesce()
    row, col = a2.indices()
    keep = row != col
    # Drop pairs that are already 1-hop neighbours. Encode (row, col) as one
    # integer key so membership is a sorted search instead of a Python set.
    one_hop_keys = torch.sort(edge_index[0] * num_nodes + edge_index[1]).values
    keys = row * num_nodes + col
    pos = torch.searchsorted(one_hop_keys, keys).clamp(max=one_hop_keys.numel() - 1)
    keep &= one_hop_keys[pos] != keys
    return torch.sparse_coo_tensor(
        torch.stack([row[keep], col[keep]]), a2.values()[keep], a2.shape
    ).coalesce()


class BipartiteTwoHop:
    """Normalised adj2 on a bipartite graph without forming A^2:
    (D2^-1/2 M D2^-1/2) h = s * (A (A (s h)) - deg * (s h)), s = D2^-1/2."""

    def __init__(self, edge_index, num_nodes, side):
        src, dst = edge_index
        if (side[src] == side[dst]).any():
            raise ValueError("implicit adj2 needs a bipartite graph (every edge crosses sides)")
        self.a = _coo(edge_index, num_nodes).to_sparse_csr()
        self.deg = torch.bincount(src, minlength=num_nodes).float()
        two_hop_deg = self.a @ self.deg.unsqueeze(-1)
        self.s = _inv_sqrt(two_hop_deg.squeeze(-1) - self.deg).unsqueeze(-1)
        self.deg = self.deg.unsqueeze(-1)

    def __matmul__(self, h):
        h = self.s * h
        return self.s * (self.a @ (self.a @ h) - self.deg * h)


def build_operators(edge_index, num_nodes, which=OPERATORS, side=None):
    """Return {name: operator} for the requested operator names.

    edge_index must be undirected (both directions present) and free of
    self-loops. `side` ([N] bool) marks one side of a bipartite graph and is
    only needed when adj2 is too large to build explicitly.
    """
    ops = {}
    if "adj_self" in which:
        loops = torch.arange(num_nodes, device=edge_index.device).repeat(2, 1)
        ops["adj_self"] = _sym_normalise(_coo(torch.cat([edge_index, loops], 1), num_nodes))
    if "adj" in which:
        ops["adj"] = _sym_normalise(_coo(edge_index, num_nodes))
    if "adj2" in which:
        deg = torch.bincount(edge_index[0], minlength=num_nodes)
        if int((deg.double() ** 2).sum()) <= MAX_EXPLICIT_TWO_HOP_PATHS:
            ops["adj2"] = _sym_normalise(two_hop(edge_index, num_nodes))
        elif side is not None:
            ops["adj2"] = BipartiteTwoHop(edge_index, num_nodes, side)
        else:
            raise ValueError("adj2 too large to build explicitly and no bipartite `side` given")
    return ops
