# 02 — GNN foundation

Goal: understand and build the plain GCN that every condition in the study
starts from (the `baseline` row in the README table).

Learn here:
- What one GCN layer actually computes (normalize adjacency, aggregate
  neighbors, linear transform, nonlinearity).
- Why stacking layers = looking further out in the graph.
- How `hidden_dim`, `num_layers`, and `dropout` change the model.

Scripts (run from the repo root with `.venv/bin/python`; 01 must have
produced `tiny_elliptic_steps1-3.pt` first):

| Script                  | What it does                                                                 |
| ----------------------- | ---------------------------------------------------------------------------- |
| `gcn_layer.py`          | One GCN layer in dense-matrix form and in scatter (message-passing) form, wrapped as `HandGCNConv`; all three checked against `GCNConv` on a 5-node toy graph |
| `two_layer_forward.py`  | `TwoLayerGCN` (HandGCNConv or `--pyg` GCNConv) forward pass on the tiny graph; prints shapes and shows the 2-hop receptive field |
| `overfit_few_nodes.py`  | Trains the 2-layer model to memorise 10 illicit + 10 licit nodes; asserts loss -> ~0 and 100% train accuracy; writes `outputs/overfit_loss.csv` |

Checklist:
- [x] Write a single GCN layer by hand with plain tensor ops (`gcn_layer.py`)
- [x] Compare it against `torch_geometric.nn.GCNConv` on a toy graph (`gcn_layer.py`)
- [x] Build a 2-layer model and run a forward pass on the tiny graph from 01 (`two_layer_forward.py`)
- [x] Overfit a handful of nodes on purpose to prove the pipeline trains (`overfit_few_nodes.py`)

Things to try yourself:
- `--directed` on `two_layer_forward.py`: how many nodes does the receptive-
  field check touch when messages only flow along payment direction?
- `--hidden 8` / `--hidden 256` on the overfit script: does depth of the
  loss curve change? Does the CSV show it converging faster or slower?
- Replace `F.relu` in `TwoLayerGCN` with `torch.tanh` and rerun the overfit.

Note: the `.py` files here were AI-generated (Claude Code, 2026-09-24) and
are marked as such in their headers. Sandbox code is not part of the paper,
but the prompt should still go in the project prompt log per the ISEF
generative-AI table.
