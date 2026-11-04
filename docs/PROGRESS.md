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

## Phase 1: Foundations (2026-11-01)

### What was done

- Scaffolded the package with uv, ruff, mypy, pre-commit and GitHub Actions on Python 3.11 and 3.12.
- Built the pipeline: config, text streaming, TransformerLens adapter, atomic shards, shard and online sources, `atlas cache` and `atlas benchmark`.
- Added `notebooks/benchmark.ipynb` for the GPU measurement.

`hub.py` and `tracking.py` from the plan are deferred to Phases 2 and 3, where they first have a user.

### Measured results

59 fast tests pass in about 3 s, and 3 slow tests against real GPT-2 pass in about 1 minute.

| Measurement | Result |
| --- | --- |
| Throughput, M1 Pro MPS, batch 32 | 15,763 tokens/s |
| Throughput, M1 Pro CPU, batch 32 | 4,733 tokens/s |
| Throughput, Kaggle GPU, batch 32 | 24,317 tokens/s |
| Throughput, Kaggle GPU, batch 128 | 24,308 tokens/s |
| Free disk on Kaggle | 19.5 GB |
| Smoke cache | 2,000 contexts (256k tokens), 376 MB on disk, about 30 s |
| float16 relative error on real activations | 8.7e-5 |
| Processed model against Hugging Face | Log-probabilities within 1e-3 |

Throughput uses random tokens, so it measures the model alone. The smoke cache time includes startup and model loading. The Kaggle session had two Tesla T4s with 15 GB each, and the benchmark used one of them.

**What the Kaggle numbers change:**

- **Speed:** throughput is about half the plan's 50k tokens/s estimate. One 300M-token run needs about 3.4 hours of model time alone. The compute budget rises from about 20 to roughly 25 to 30 GPU hours, which spans one to two weekly quotas. The plan allowed for this.
- **Batch size:** quadrupling the batch didn't help, so the GPU is already saturated at batch 32. Half-precision inference is the obvious speedup, but it changes the activations, so it needs the same validation as float16 storage before Phase 3 uses it.
- **Disk:** 19.5 GB free caps a float16 cache at about 6M tokens once a safety margin is left. Evaluation and analysis caches will target 5M tokens.

### Failures and surprises

1. **`stop_at_layer` skips the hook we wanted.** `stop_at_layer=N` stops before block N, so `run_with_cache` captured nothing at `blocks.N.hook_resid_pre`. Fix: use the tensor it returns, with a test proving it equals the hook value.
2. **Position 0 is a large outlier.** Its mean norm is about 3,119 against about 102 elsewhere, and it holds 88% of the squared magnitude in the smoke cache. A normal word at position 0 behaves the same, so the cause is the position and not BOS. A BOS token mid-context does not, at about 156. Centring doesn't change it. Left alone, it could dominate the L2 loss and inflate explained variance.
3. **`UV_TORCH_BACKEND` doesn't affect `uv sync`.** My first CI draft relied on it. Fix: a Linux-only CPU index in `pyproject.toml`.
4. **Long command lines wrapped into repo metadata.** A wrapped terminal line put a newline in the repo description, and GitHub rejected it.

### What you should understand

- Why an atomic rename protects a cache, and why the manifest is written last.
- Why the shuffle buffer keeps half of each fill.
- Why `ShardSource` needs no saved RNG state to resume exactly.
- Why float16 is acceptable here, and how overflow is caught.

### Decision for Phase 2

Whether to exclude position 0 from SAE training and evaluation. Decided on 2026-11-03: exclude it from both, and evaluate the reference SAE the same way.

### Not yet verified

- `OnlineSource` is tested on the tiny model only, not at real scale.
- The Mermaid diagram in `docs/PLAN.md`.

## Phase 2: SAE from scratch (2026-11-04)

### What was done

- **Model:** a ReLU SAE with a unit-norm dictionary, decoder gradient projection, and inputs scaled to mean squared norm `d_model`.
- **Initialization:** input scale and a geometric-median `b_dec`, both set from data.
- **Dead features:** resampled toward badly reconstructed inputs, with Adam state reset for the changed weights.
- **Training:** the loop has learning-rate and λ warm-ups, gradient clipping, atomic checkpoints, config hash checks and exact resume.
- **Evaluation:** streaming explained variance, MSE, L0, dead and dense fractions, and loss recovered, spliced into GPT-2.
- **Commands:** `atlas train` and `atlas eval`. SAEs are saved as safetensors.
- **Position 0:** excluded from training and evaluation by default, as decided after Phase 1.

### Measured results

109 fast tests pass in about 4 s.

**Planted dictionary.** Synthetic data is built from 48 known unit directions, with about 2.9 active per token. A 64-feature SAE recovers every direction with cosine similarity of at least 0.989, against a mean of 0.42 for the untrained control. Across five other data and initialization seeds, the worst direction scored between 0.976 and 0.984. On the default seed, raising λ shows the expected tradeoff:

| λ | L0 | Explained variance |
| --- | --- | --- |
| 0.3 | 10.7 | 0.964 |
| 1.0 | 5.5 | 0.930 |
| 3.0 | 3.4 | 0.746 |

At λ=3, L0 approaches the true 2.9, but explained variance falls. That's L1 shrinkage: the penalty also pulls active features' values below their true size.

**Kill and resume.** A run killed at step 40, with resampling on both sides of the kill, resumes from its step-20 checkpoint to weights identical to an uninterrupted run.

**Laptop smoke run on real GPT-2.** 8× dictionary (6,144 features), 4M tokens, λ=5, about 6 min 40 s on M1 Pro MPS, roughly 10k tokens/s including the model. Held-out evaluation on 32.5k tokens:

| Metric | Value |
| --- | --- |
| Explained variance | 0.455 |
| L0 | 7.7 |
| CE clean / SAE / zero | 3.660 / 5.309 / 11.131 |
| Loss recovered | 0.779 |

These numbers only show the pipeline works end to end. λ=5 on so short a run is very sparse, and Phase 3 will tune it.

### Failures and surprises

1. **The first recovery test passed for the wrong reason.** The SAE's initialization and the synthetic data both used seed 0, so the SAE's random starting directions *were* the planted ones. The untrained control scored 1.0 and exposed it. Fix: separate seeds, and the control stays in the test.
2. **MPS has no float64.** Converting to float64 before moving to the CPU crashed the first real run. CI is CPU-only, so it couldn't catch this. Fix: convert on the CPU.
3. **I understated online resume loss.** A resumed `OnlineSource` loses up to the buffer *plus one model batch*, not just the buffer. A test caught it, and the docstring now says so.
4. **Evaluation's dead fraction depends on sample size.** It was 65% on 32.5k evaluation tokens, against 0.6% from training's 200-step window. With L0 about 8 across 6,144 features, many rare features simply never appear in a small sample. Phase 3 needs at least 1M evaluation tokens before reporting this number.
5. **A git step bundled three commits into one.** `git stash -u` also stashed staged files. Fix: `--keep-index`. Every Phase 2 commit was then tested on its own before committing.

### What you should understand

- Why unit-norm decoder rows stop the L1 penalty being dodged, and why the gradient is projected before Adam sees it.
- Why inputs are scaled, and why λ still can't be compared across implementations.
- Why `b_dec` starts at the geometric median and not the mean.
- Why resampling resets Adam's moments.
- Why the recovery test needs an untrained control.
- What L1 shrinkage is, and how it shows up in the λ table above.
- What loss recovered measures, and why position 0 is left untouched in both splices.

### Not yet verified

- Training throughput for a 32× SAE on a T4. The SAE's matrix multiplications may cost more than the model itself at that size, so Phase 3 measures this first.
- How the reference SAE scores under our evaluation, with position 0 excluded.
- Hub sync for checkpoints and W&B tracking. Both are deferred to Phase 3, where training first runs on Kaggle.
