"""Sanity checks for the study pipeline. Run from the repo root:

    .venv/bin/python -m pytest tests/ -q



Fast tests use a small random graph; tests marked `data` load the real
datasets from data/processed (built on first use) and take a few seconds.
"""

import networkx as nx
import numpy as np
import pytest
import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from src.evaluate import compute_metrics, select_threshold
from src.models import ConfigurableGCN
from src.models.propagation import BipartiteTwoHop, build_operators, two_hop
from src.train import phase_operators, set_seed

CONDITIONS = [(False, False), (True, False), (False, True), (True, True)]


def random_graph(n=60, m=120, seed=1):
    g = nx.gnm_random_graph(n, m, seed=seed)
    ei = torch.tensor(list(g.edges)).T
    return g, torch.cat([ei, ei.flip(0)], dim=1)


def test_two_hop_is_distance_two_weighted_by_shared_neighbours():
    g, ei = random_graph()
    m = two_hop(ei, g.number_of_nodes())
    got = {(u, v): c for (u, v), c in zip(m.indices().T.tolist(), m.values().tolist())}
    sp = dict(nx.all_pairs_shortest_path_length(g, cutoff=2))
    expected = {(u, v): len(list(nx.common_neighbors(g, u, v)))
                for u in sp for v, d in sp[u].items() if d == 2}
    assert got == expected


def test_implicit_bipartite_two_hop_matches_explicit():
    g = nx.bipartite.random_graph(40, 15, 0.2, seed=3)
    n = g.number_of_nodes()
    ei = torch.tensor(list(g.edges)).T
    ei = torch.cat([ei, ei.flip(0)], dim=1)
    side = torch.tensor([g.nodes[i]["bipartite"] == 1 for i in range(n)])
    explicit = build_operators(ei, n, ["adj2"])["adj2"]
    implicit = BipartiteTwoHop(ei, n, side)
    h = torch.randn(n, 8)
    assert torch.allclose(explicit @ h, implicit @ h, atol=1e-5)


def test_adj_self_matches_pyg_gcn_normalisation():
    g, ei = random_graph()
    n = g.number_of_nodes()
    ops = build_operators(ei, n)
    ei2, w = gcn_norm(ei, num_nodes=n, add_self_loops=True)
    h = torch.randn(n, 8)
    ref = torch.zeros_like(h).index_add_(0, ei2[1], w[:, None] * h[ei2[0]])
    assert torch.allclose(ops["adj_self"] @ h, ref, atol=1e-6)


@pytest.mark.parametrize("ego,second", CONDITIONS)
def test_each_condition_forward_and_backward(ego, second):
    g, ei = random_graph()
    n = g.number_of_nodes()
    model = ConfigurableGCN(16, 8, num_layers=2, ego_separation=ego, second_order=second)
    ops = build_operators(ei, n, model.required_operators())
    out = model(torch.randn(n, 16), ops)
    assert out.shape == (n,)
    out.sum().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_featureless_nodes_use_embeddings():
    g, ei = random_graph()
    n = g.number_of_nodes()
    featureless_index = torch.full((n,), -1)
    featureless_index[:10] = torch.arange(10)
    model = ConfigurableGCN(16, 8, num_featureless=10)
    out = model(torch.randn(n, 16), build_operators(ei, n), featureless_index)
    out.sum().backward()
    assert model.embedding.weight.grad.abs().sum() > 0


def test_phase_graphs_hide_later_splits():
    g, ei = random_graph()
    n = g.number_of_nodes()

    class D:  # minimal stand-in for a PyG Data object
        num_nodes = n
        edge_index = ei
        node_split = torch.tensor([i % 3 for i in range(n)])
        node_split[:5] = -1  # always-present nodes (like IEEE-CIS-G metadata)

    ops = phase_operators(D, ["adj"])
    split = D.node_split
    for phase, k in (("train", 0), ("val", 1), ("test", 2)):
        row, col = ops[phase]["adj"].to_sparse_coo().indices()
        assert (split[row] <= k).all() and (split[col] <= k).all()
    assert ops["test"]["adj"]._nnz() == ei.size(1)


def test_same_seed_same_result():
    g, ei = random_graph()
    n = g.number_of_nodes()
    x = torch.randn(n, 16)
    outs = []
    for _ in range(2):
        set_seed(0)
        model = ConfigurableGCN(16, 8, ego_separation=True, second_order=True)
        model.train()
        outs.append(model(x, build_operators(ei, n)))
    assert torch.equal(outs[0], outs[1])


def test_metrics_perfect_and_threshold_from_val():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    t = select_threshold(y, s)
    m = compute_metrics(y, s, t)
    assert m["pr_auc"] == m["f1"] == m["roc_auc"] == 1.0


@pytest.mark.data
def test_elliptic_graph_is_valid():
    from src.data.elliptic import load_elliptic

    d = load_elliptic()
    assert d.num_nodes == 203_769 and d.x.shape[1] == 165
    assert d.edge_index.min() >= 0 and d.edge_index.max() < d.num_nodes
    # undirected, no self-loops, and no edge crosses a time step
    src, dst = d.edge_index
    assert d.edge_index.size(1) == 2 * 234_355
    assert (src != dst).all()
    assert (d.time_step[src] == d.time_step[dst]).all()
    # masks: disjoint, labelled nodes only, and cover every labelled node
    masks = torch.stack([d.train_mask, d.val_mask, d.test_mask])
    assert (masks.sum(0) <= 1).all()
    assert (d.y[masks.any(0)] >= 0).all()
    assert masks.any(0).sum() == (d.y >= 0).sum() == 46_564
    assert d.y[d.y >= 0].sum() == 4_545  # illicit
    assert d.time_step[d.train_mask].max() == 29
    assert d.time_step[d.val_mask].min() == 30 and d.time_step[d.val_mask].max() == 34
    assert d.time_step[d.test_mask].min() == 35


@pytest.mark.data
def test_ieee_cis_g_matches_h2gb():
    """Counts from H2GB (Lin et al., 2025): Table 1 and Fig. 6."""
    from src.data.ieee_cis_g import METADATA_TYPES, load_ieee_cis_g

    d = load_ieee_cis_g()
    tx = d.node_type == 0
    assert d.num_nodes == 153_880 and int(tx.sum()) == 144_233
    assert d.edge_index.size(1) == 2_873_472
    src, dst = d.edge_index
    assert (tx[src] != tx[dst]).all()  # bipartite: transaction <-> metadata only
    fig6 = {"card1": 8_499, "card2": 482, "card3": 103, "card4": 4, "card5": 106,
            "card6": 3, "ProductCD": 4, "addr1": 255, "addr2": 72,
            "P_emaildomain": 59, "R_emaildomain": 60}
    for t, col in enumerate(METADATA_TYPES, start=1):
        assert int((d.node_type == t).sum()) == fig6[col], col
    # time-ordered 80/10/10 split over transactions; metadata always present
    assert [int(m.sum()) for m in (d.train_mask, d.val_mask, d.test_mask)] == [115_386, 14_423, 14_424]
    assert (d.node_split[~tx] == -1).all() and (d.featureless_index[~tx] >= 0).all()
    dt = d.transaction_dt
    assert dt[d.train_mask].max() <= dt[d.val_mask].min()
    assert dt[d.val_mask].max() <= dt[d.test_mask].min()
    assert int(d.y[tx].sum()) == 11_318  # 7.85% fraud
    # D10: no feature is constant over training transactions; standardised on train only
    train_x = d.x[d.train_mask]
    assert d.x.shape[1] == 4_321 and (train_x.std(0) > 0).all()
    assert train_x.mean(0).abs().max() < 1e-3
