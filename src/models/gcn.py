"""Controlled GCN with two independently switchable mechanisms (Zhu et al., 2020).

AI-generated: Claude Code (Opus 5.5), 2026-10-11. Log the prompt in the
project prompt log and cite per the ISEF generative-AI table.

The four experimental conditions are produced by two boolean flags, so the
architectures differ ONLY in these mechanisms — everything else (depth,
hidden size, dropout, encoder, classifier, training procedure) is shared:

  ego_separation=False, second_order=False  -> baseline (control)
  ego_separation=True,  second_order=False  -> ego-separation only
  ego_separation=False, second_order=True   -> second-order only
  ego_separation=True,  second_order=True   -> combined

Architecture (same for every condition):

  h0 = dropout(relu(encoder(x)))          per-node input encoder to hidden_dim;
                                          featureless nodes (IEEE-CIS-G metadata)
                                          get a learnable embedding instead
  for each of num_layers layers:
      parts = [A_self h]                  baseline: GCN aggregation, self-loop
                                          mixes ego into neighbour average
            | [h, A h]                    ego_separation: ego kept separate from
                                          the self-loop-free neighbour average
      parts += [A2 h]  if second_order    exact-2-hop neighbours as own channel
      h = dropout(relu(W [parts concatenated]))
  logit = classifier(h)                   one logit per node (fraud vs licit)

Concatenating before one linear map means W applies separate weights to the
ego, 1-hop and 2-hop channels, so they stay distinguishable (Zhu et al.'s
"combine" step). Operators come from src/models/propagation.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfigurableGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=2, dropout=0.5,
                 ego_separation=False, second_order=False, num_featureless=0):
        super().__init__()
        self.ego_separation = ego_separation
        self.second_order = second_order
        self.dropout = dropout

        self.encoder = nn.Linear(in_dim, hidden_dim)
        self.embedding = nn.Embedding(num_featureless, hidden_dim) if num_featureless else None

        num_parts = (2 if ego_separation else 1) + (1 if second_order else 0)
        self.layers = nn.ModuleList(
            nn.Linear(num_parts * hidden_dim, hidden_dim) for _ in range(num_layers)
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def required_operators(self):
        """Names of the operators (see propagation.py) this condition uses."""
        ops = ["adj" if self.ego_separation else "adj_self"]
        if self.second_order:
            ops.append("adj2")
        return ops

    def forward(self, x, ops, featureless_index=None):
        """x: [N, in_dim]; ops: dict from build_operators; featureless_index:
        [N] long, -1 for nodes with features, else the node's embedding row."""
        h = self.encoder(x)
        if self.embedding is not None:
            is_featureless = featureless_index >= 0
            h = h.clone()
            h[is_featureless] = self.embedding(featureless_index[is_featureless])
        h = F.dropout(F.relu(h), self.dropout, self.training)

        for layer in self.layers:
            if self.ego_separation:
                parts = [h, ops["adj"] @ h]
            else:
                parts = [ops["adj_self"] @ h]
            if self.second_order:
                parts.append(ops["adj2"] @ h)
            h = F.dropout(F.relu(layer(torch.cat(parts, dim=-1))), self.dropout, self.training)

        return self.classifier(h).squeeze(-1)
