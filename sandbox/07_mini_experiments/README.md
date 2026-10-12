# 07 — Mini experiments

Goal: run the full pipeline end to end on a small subset before scaling up
with `scripts/run_all.sh` on RunPod.

Learn here:
- Running all four conditions on one small dataset slice.
- Reading the run records and aggregating mean ± SEM across seeds.
- Spotting problems (no learning, NaNs, val metric never moving) cheaply.

Checklist:
- [ ] 4 conditions x 2 seeds on tiny Elliptic
- [ ] Aggregate into a small table in `outputs/`
- [ ] Note anything that needs fixing before moving code into `src/`
