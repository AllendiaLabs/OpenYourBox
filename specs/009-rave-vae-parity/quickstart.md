# Quickstart: RAVE Variational Bottleneck Parity

Validate reference-parity bottleneck behavior after implementation. Assumes spec **005** RAVE train shell is already working.

## Prerequisites

- Built OpenYourBox AU/VST3 + Python workers
- Copyright acknowledgment completed
- Mono test clip(s) in Training Library (≥ 10 files recommended so 2% val split is non-empty)

## 1. Layout defaults (kernel 5, grouped head)

1. Insert **Latest continuous RAVE layout** (mono).
2. Select the **Variational Bottleneck** node.
3. **Expect**: `Kernel Size = 5`, `Latent = 128`.
4. Trace encoder → bottleneck: upstream channel count **even** (layout default satisfies).

**Fail if**: kernel defaults to 3 or bottleneck uses 1×1-only weights in inspector.

## 2. Shape gate (odd channels)

1. Insert a **Linear** or conv reducing to **odd** channels before the bottleneck.
2. Attempt wire to bottleneck.

**Expect**: Red cable / tooltip — even channel count required.

## 3. Live mean-only (untrained)

1. Play audio through the layout (random weights).
2. Capture two consecutive encode outputs with identical input (debug hook or test harness).

**Expect**: Bit-identical latent outputs (no sampling noise).

## 4. Reconstruction train + validation PCA

1. Select ≥ 10 library clips; objective **Reconstruction**; arm layout; Run.
2. Watch Train panel through stage 1 → stage 2 transition.

**Expect at stage-1 end**:
- Train event reports `compactness.ready: true` (if enough val audio)
- Status **not** "Compactness not ready" after transition

**Expect during stage 1** (optional checkpoint load):
- Fidelity inactive / full width
- Live audio still deterministic (μ-only)

## 5. Fidelity sweep (post-train)

1. After success auto-load (Gold), sweep fidelity **100 → 50**.
2. Listen to forward reconstruction.

**Expect**: Audible coarsening at lower fidelity; monotonic effective rank (debug log if available).

## 6. Unfreeze buffer parity

1. Unfreeze Gold RAVE chain.
2. Inspect bottleneck node: `compactnessReady` true.
3. Repeat fidelity sweep on Blue bottleneck.

**Expect**: Same qualitative behavior as Gold at same fidelity setting.

## 7. Worker vs live sampling (automated)

Run Python/C++ tests:

```bash
# From repo root after build configures tests
ctest -R VariationalBottleneck --output-on-failure
python3 -m pytest Tests/test_train_worker.py -k "bottleneck or compactness or split" -v
```

**Expect**:
- Train forward ≠ eval forward for same module/weights
- Split 98/2 reproducible with seed 42
- Val paths excluded from stage-1 train sampler

## 8. Parity smoke (optional, manual)

Short train same clip in acids-rave (reference) and OYB with matched layout defaults.

Compare:
- Eval μ latent histograms (qualitative)
- Fidelity 50% vs 90% rank direction

**Pass**: Practitioner sign-off per SC-007 (no bit-exact weights required).

## Contracts

- Head geometry / sampling: [variational-bottleneck-contract.md](./contracts/variational-bottleneck-contract.md)
- Split / PCA / fidelity: [compactness-pca-contract.md](./contracts/compactness-pca-contract.md)
- Data entities: [data-model.md](./data-model.md)

## Common failures

| Symptom | Likely cause |
|---------|----------------|
| Noisy live audio during stage 1 | Live path sampling (violates FR-002) |
| Fidelity active mid-stage-1 | Checkpoint missing `ready: false` gate |
| PCA from train batch | Split/validation loop not wired |
| Odd channel crash | Missing shape gate |
| Old checkpoint load error | Expected breaking change — retrain |
