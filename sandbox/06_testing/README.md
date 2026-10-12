# 06 — Testing

Goal: small automated checks so refactors into `src/` do not silently break
things.

Learn here:
- Writing `pytest` tests for shapes, masks, and metric edge cases.
- Testing that each config flag combination builds and runs a forward pass.
- Testing that the data builders produce valid graphs.

Checklist:
- [ ] Install pytest into the venv
- [ ] One test per builder: graph is valid
- [ ] One test per condition: forward pass works and shapes match
- [ ] One test: metrics on a perfect classifier give 1.0
