# Quickstart: TorchScript Checkpoint Loader Node

Validation guide for `015-torchscript-load-node`. Implementation details belong in `tasks.md`; this file lists runnable checks that prove the feature end-to-end.

## Prerequisites

- Build OpenYourBox VST (Debug or Release) with LibTorch linked as usual
- Host or standalone runner that loads the plugin and plays audio
- A known-good TorchScript export (pretrained RAVE `.ts` / `.pt` preferred), or a fixture `.pt` produced by the project’s freeze/train path that exposes `forward` (and ideally `encode`/`decode`)
- Optional: CTest target covering `LiveGraphEngine` tests

## Setup

```bash
# From repo root — adjust generator/config to your usual workflow
cmake --build build --target OpenYourBox

# After tests exist for this feature:
ctest --test-dir build -R LiveGraphEngine -V
```

## Scenario A — Place, load, hear (P1)

1. Open Factory; confirm **TorchScript Load** is listed.
2. Place Audio Input → TorchScript Load → Audio Output.
3. With empty path, play audio: confirm **dry passthrough**.
4. Browse to a valid checkpoint; wait until status ready (no audio engine stop).
5. Confirm processed (non-dry) output.
6. Select a different valid checkpoint; confirm swap without plugin restart.

**Expect**: SC-001, SC-002; FR-001–FR-004, FR-017; [external-checkpoint-load-contract.md](contracts/external-checkpoint-load-contract.md).

## Scenario B — Errors & shapes (P1)

1. Point at a missing path: recoverable error, **silence** (no prior model), host keeps running.
2. Point at an invalid file: error; if a prior model existed, it keeps running.
3. After a good load, attempt an illegal channel connection: refuse + explanation.
4. Edit channel overrides to break a cable: refuse/flag; Reset restores inferred.

**Expect**: SC-003, SC-004, SC-007; FR-005, FR-006, FR-011, FR-014; [pin-surface-and-channels-contract.md](contracts/pin-surface-and-channels-contract.md).

## Scenario C — Persist & restore (P2)

1. Load a valid path; set an intentional channel override.
2. Save session or named preset; reload.
3. Confirm path + override restored and model becomes ready without re-browsing (file still present).
4. Move/rename the file; reload: path string restored, recoverable missing-file error, no crash.

**Expect**: SC-005; FR-007; [persist-and-restore-contract.md](contracts/persist-and-restore-contract.md).

## Scenario D — RAVE-like encode/decode & fidelity (P3)

1. Load a checkpoint with `encode`/`decode` methods.
2. Confirm latent pins appear (same class as trained RAVE Gold).
3. If compactness attrs present, adjust fidelity; confirm audible/latent thinning behavior consistent with trained RAVE Gold.
4. If compactness absent, confirm fidelity hidden/disabled with explanation.
5. Confirm Unfreeze is unavailable/disabled for this node.

**Expect**: SC-006; FR-008, FR-012, FR-013, FR-016.

## Scenario E — Conditioning pin (when available)

1. Load a conditioned checkpoint (or project fixture that requires `forward(audio, cond)`).
2. Confirm Control pin appears; unconnected runs unconditioned; connecting Control applies conditioning.
3. Load a plain forward model; confirm no Control pin.

**Expect**: FR-015.

## Scenario F — Clear

1. From a ready node, Clear path.
2. Confirm return to empty: dry passthrough + choose-file prompt; optional pins removed.

**Expect**: FR-009, FR-017.
