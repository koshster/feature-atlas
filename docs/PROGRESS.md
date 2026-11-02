# Progress Log

## Phase 0: Planning (2026-11-01)

### What was done

- Wrote `docs/PLAN.md`: architecture, model and layer choice, libraries, compute budget, extended ideas, timeline and risks.
- Wrote `CLAUDE.md` with project conventions.
- Checked library versions and free-tier terms against current sources instead of relying on memory.

No application code exists, so there are no results or tests yet.

### Findings that changed the design

1. **TransformerLens and SAELens conflict at the latest versions.** TL 4 removed `HookedTransformer`, and SAELens 6.54.12 requires `transformer-lens<4`. A dependency resolve confirmed that TL 3.9.0 works with SAELens and torch 2.15. Fix: pin `>=3.9,<4` and isolate TL in one adapter file.
2. **The reference SAE expects centred weights.** Its config sets `center_writing_weights: true`, so its activations differ from raw GPT-2's. Comparing on mismatched activations would produce a believable but wrong number.
3. **A full disk cache is impossible on free tiers.** 300M tokens × 3,072 bytes is about 922 GB. Fix: online activations for training, shards for tests, evaluation and analysis.
4. **Labelling all features locally isn't realistic.** At an estimated 20 s each, 24,576 features would take about 6 days. Fix: a stratified sample, reported as a sample.

### What you should understand before Phase 1

- Why an SAE's dictionary must have unit-norm columns.
- Why the residual stream suits a sparse linear model.
- Why tokens need shuffling before reaching the SAE.
- Why GPU training can't honestly be called bit-for-bit reproducible.

### Decisions confirmed

Recorded in section 11 of `docs/PLAN.md`: noreply commit email for this repo, no hard deadline, general ML engineering focus, TransformerLens 3.9 behind an adapter.

### Not yet verified

- The Mermaid diagram hasn't been rendered. Check it when GitHub first displays `docs/PLAN.md`.
- Throughput, session limits and notebook disk size are estimates or unofficial until Phase 1 measures them.
