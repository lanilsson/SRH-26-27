# Open issues to revisit

## OI-1 — Homophily: Elliptic measures as homophilous, not heterophilous

- **Found:** 2026-10-11, running `scripts/inspect_dataset.py` (checklist item D1)
- **Status:** deferred, does not block the code, pilot runs, or this week's tasks
- **Decide by:** before the 40 final runs on RunPod (settle together with D4
  tuning and D5 statistical test); at the latest, before writing the methods
  section

**What was found.** Build Procedure step 2 of the proposal (10-11-26) expects
the homophily check to "confirm that it is heterophilous". The edge homophily
ratio it names (Zhu et al., 2020) gives:

| Graph      | Measure                                                             | Value                                                                 |
| ---------- | ------------------------------------------------------------------- | --------------------------------------------------------------------- |
| Elliptic   | edge homophily, labelled–labelled edges (73,248 entries)           | **0.954**                                                       |
| Elliptic   | edges from fraud nodes: share of labelled neighbours that are fraud | 0.541                                                                 |
| Elliptic   | edges from licit nodes: share of labelled neighbours that are licit | 0.976                                                                 |
| Elliptic   | same-class share over 2-hop pairs                                   | 0.977                                                                 |
| IEEE-CIS-G | edge homophily, 1-hop                                               | undefined (every edge is transaction–metadata)                       |
| IEEE-CIS-G | same-class share over 2-hop pairs (via shared metadata)             | 0.859                                                                 |
| IEEE-CIS-G | H²GB's published values (Lin et al., Table 2)                      | H_edge 0.5917, H_adj 1.3151, H² Index 0.9846 (near 1 = heterophilic) |

**Context about the metric.** Edge homophily is inflated by class imbalance:
about 90% of Elliptic's labelled nodes are licit, so a randomly wired graph
would score about 0.82 (0.90² + 0.10²).

**To do when revisiting.**

1. Read about measures that correct for class imbalance, and verify each
   reference yourself before citing it:
   - class-insensitive homophily (Lim et al., 2021);
   - adjusted homophily (Platonov et al., 2023), which is H²GB's H_adj;
   - node homophily (Pei et al., 2020).
2. Ask Claude to add those measures to `scripts/inspect_dataset.py`, including
   H²GB-style metapath versions for IEEE-CIS-G. Matching H²GB's Table 2 values
   would also double-check the graph build.
3. Decide which measure(s) the study reports and what the result means for
   the hypothesis and framing.
4. Reword Build Procedure step 2: a measurement cannot be planned to "confirm"
   a result.

**Unaffected either way.** The 2×2 experiment, the loaders, the model and
the metrics do not change with this decision.
