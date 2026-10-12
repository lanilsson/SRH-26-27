# 03 — Heterophily-aware mechanisms

Goal: implement the two mechanisms from Zhu et al. (2020) as switchable flags
on top of the foundation from 02.

Learn here:
- Why plain GCN struggles when neighbors tend to have *different* labels
  (fraud nodes surrounded by legitimate ones).
- Ego–neighbor separation: keep the node's own features apart from the
  aggregated neighbor features instead of mixing them.
- Second-order aggregation: give the 2-hop neighborhood its own channel.
- How to measure homophily on a graph so you can see the problem in the data.

Checklist:
- [ ] Measure edge homophily on the tiny Elliptic graph
- [ ] Add `ego_separation` flag and check output shapes
- [ ] Add `second_order` flag and check output shapes
- [ ] Verify flags off reproduces the 02 baseline exactly
