"""Controlled GCN with two independently switchable mechanisms (Zhu et al., 2020).

The four experimental conditions are produced by two boolean flags, so the
architectures differ ONLY in these mechanisms — everything else (depth,
hidden size, dropout, training procedure) is shared:

  ego_separation=False, second_order=False  -> baseline (control)
  ego_separation=True,  second_order=False  -> ego-separation only
  ego_separation=False, second_order=True   -> second-order only
  ego_separation=True,  second_order=True   -> combined

Intended semantics:
  - Baseline: standard GCN layer; self-loops mix the ego representation
    into the neighborhood average at every layer.
  - ego_separation: aggregate neighbors WITHOUT self-loops, keep the ego
    embedding as a separate representation, and combine (concatenate)
    ego and neighbor representations before the layer output.
  - second_order: aggregate the 2-hop neighborhood as its own channel
    (from the squared adjacency, excluding 0/1-hop nodes) and combine it
    with the 1-hop channel, keeping the two distances distinguishable
    rather than reaching 2 hops only through stacked layers.
"""

import torch.nn as nn


class ConfigurableGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=2, dropout=0.5,
                 ego_separation=False, second_order=False):
        super().__init__()
        self.ego_separation = ego_separation
        self.second_order = second_order
        # TODO: build layers per the module docstring. Output is a single
        # logit per node for binary fraud classification.
        raise NotImplementedError("TODO: implement the four-condition GCN")

    def forward(self, x, edge_index):
        raise NotImplementedError
