# Phase 0 Research: RAVE Variational Bottleneck Parity

## Decision 1: Softplus variance parameterization (replace clamped log-var)

**Decision**: Variance branch outputs pre-activation `v`; use `σ² = softplus(v) + ε` (small ε for numerical stability, match acids-rave). Reparameterize as `z = μ + σ ⊙ ε` with `ε ~ N(0,I)` during worker training. KL uses standard Gaussian closed form against unit prior with learned `μ` and `log σ²`.

**Rationale**: FR-001/FR-003. Current OYB uses raw `logvar` with `clamp(-8, 8)` — diverges from reference and changes gradient flow.

**Alternatives considered**:
- Keep clamp: rejected (explicit spec parity)
- Exp(logvar) without softplus: rejected (reference uses softplus path)

## Decision 2: Grouped causal conv head (`groups=2`, default k=5)

**Decision**: Replace two full-width 1×1 convs with one **causal grouped Conv1d**:
- `in_channels = encoder_features`, `out_channels = 2 × (latent_size/2) = latent_size`
- `groups = 2`: group 0 → μ (first `latent_size/2` outputs), group 1 → variance pre-activation (second half)
- Each group sees `in_channels / 2` input channels
- `kernel_size` from node property (default **5**), causal left-padding `(k−1)` prepared off audio thread
- User may change `kernel_size`; group count is **not** configurable

**Rationale**: FR-004/FR-005; TODO and clarify session. Reference continuous layout uses k=5 grouped head on 1024-wide encoder features.

**Alternatives considered**:
- Two separate grouped convs: rejected (reference uses single grouped layer split)
- Configurable group count: rejected (clarify: fixed mean/variance pair)
- 1×1 with flexibility: rejected (legacy, breaking change)

## Decision 3: Live μ-only vs worker stochastic sampling

**Decision**:
- **Background worker** (`train_worker.py`): `module.train()` during stage 1; bottleneck forward samples `z = μ + σ⊙ε`
- **Live engine + Gold + checkpoints on audio path**: **always** `z = μ` (eval path); never inject ε on audio thread
- Hear-while-training checkpoint load uses same live rule

**Rationale**: FR-002; clarify session. Stable DAW monitoring; worker-only ELBO gradients.

**Alternatives considered**:
- Live mirrors worker during stage 1: rejected (noisy monitoring, clarify A)
- Snapshot-synced sampling: rejected (complexity, clarify C)

## Decision 4: Train/validation split for compactness PCA

**Decision**: At reconstruction job start, split selected corpus paths:
- **~98% train / ~2% validation**, `random.seed(42)`, validation list capped at **1000** segments (acids-rave `split_dataset` + `max_residual=1000`)
- Stage 1 optimization draws minibatches from **train split only**
- At **last step of stage 1** (before stage 2): set `module.eval()`, run full **validation pass** (all val segments), collect **μ only** (pre-sample), flatten to `[N, latent]`, run SVD PCA
- Stage 2 does **not** recompute PCA (encoder frozen)

**Rationale**: FR-007; clarify session. Fixes current OYB behavior (single training minibatch at stage-1 end ≈ Option C).

**Alternatives considered**:
- Full corpus PCA: rejected (training bias)
- Mid-stage-1 ad-hoc PCA for checkpoints: rejected (fidelity inactive until stage-1 end)

## Decision 5: Fidelity rank from singular-value cumulative sum

**Decision**: Keep existing OYB `applyFidelity` pattern but ensure PCA inputs are validation **μ** only. `cumulative[i] = sum(s[:i+1]) / sum(s)` from SVD singular values `s`. User fidelity percent maps to smallest `keep` where `cumulative[keep-1] >= fidelity/100`.

**Rationale**: FR-008. Current `compute_compactness` already uses SVD cumulative on singular values — wire correct data source.

**Alternatives considered**:
- Covariance eigenvalues: equivalent for PCA; SVD on centered matrix matches reference linear cumulative sum

## Decision 6: Weight storage and export layout

**Decision**:
- **Live C++**: two weight tensors — grouped conv weight `[latent, in/2, k]` per group packed as PyTorch grouped layout `[latent, in/groups, k]` with groups=2; optional separate storage as mean-group + var-group for clarity in tests
- **TorchScript export**: embed `latent_mean`, `latent_pca`, `cumulative_variance` buffers; `compactness.ready` bool in metadata and module
- **Unfreeze**: copy buffers from Gold metadata / sidecar into `GraphNode.compactnessReady`, `latentMean`, `latentPca`, `cumulativeVariance` on variational bottleneck node

**Rationale**: FR-009/FR-010; spec 005 Gold contract extended.

**Alternatives considered**:
- Recompute PCA on Unfreeze: rejected (must match trained artifact)

## Decision 7: Fidelity UI gating

**Decision**: Until `compactness.ready == true` (post stage-1 PCA success): fidelity slider disabled or ignored with status **"Compactness not ready"**; effective behavior = full latent width (keep = latent_size). Mid-stage-1 checkpoints export `compactness.ready: false`.

**Rationale**: FR-010; clarify session.

## Decision 8: Breaking change — no legacy bottleneck

**Decision**: Remove code paths for 1×1 dual conv bottleneck. Old `.pt` artifacts may fail load; graphs with missing `kernel_size` default to 5 on next prepare. No migration shim.

**Rationale**: FR-013; user confirmed no trained models in production.

**Alternatives considered**:
- Dual-path loader: rejected (maintenance, user waived compat)

## Reference anchors (acids-rave)

| Topic | Reference behavior |
|-------|-------------------|
| Split | `train, val = rave.dataset.split_dataset(dataset, 98)` seed 42, val cap 1000 |
| PCA timing | `validation_epoch_end`, phase 1 only |
| μ source | `validation_step` encoder mean before reparameterize |
| Head | Grouped conv k=5, groups=2 on encoder output |
| Variance | softplus on variance branch |

## NOTICE

No new third-party ML code import required. Retain existing MIT/Apache notices; acids-rave referenced for parity rules only (not vendored).
