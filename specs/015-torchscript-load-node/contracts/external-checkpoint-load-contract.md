# Contract: External Checkpoint Load & Runtime

## Purpose

Define load, prepare, swap, capability detection, and fail-safe audio behavior for `BlackBoxOrigin::externalLoad`.

## File choice

- UI: browse (`FileChooser`) and/or path text in property panel.
- Accepted extensions: at least `*.pt;*.pth;*.ts` (RAVE pretrained exports commonly use `.ts`).
- Directory selection → reject with recoverable message (error policy if no prior model).
- Clear control → unload to `empty` (dry passthrough).

## Prepare pipeline (message / worker thread — never audio thread)

1. Validate path (exists, regular file).
2. `TorchScriptBlackBoxFactory::load(path, inputChannelsHint, …)` with silence preservation **off** (third-party models may not preserve silence).
3. Detect encode/decode via `find_method("encode")` && `find_method("decode")`.
4. Detect conditioning via existing unconditioned → conditioned probe.
5. Publish factory into `publishedFrozenArtifacts[path]`.
6. `copyCompactnessFromArtifact` when attrs present; set `compactnessReady`.
7. Update node status `ready`; morph pins; prefill inferred channel fields / overrides defaults.
8. Atomic graph/runtime swap so audio picks up the new factory at buffer boundary.

## Capability → UI / DSP

| Capability | Detection | UI / pins | DSP |
|------------|-----------|-----------|-----|
| Forward | Always after successful load | Audio in/out | `forward` or conditioned forward |
| Encode/decode | Both methods present | Latent in/out pins | Same as trained RAVE Gold (encode→fidelity→decode; decode-from-latent when latent in connected) |
| Conditioning | Factory conditioned flag | Control pin only if true | Unconnected = unconditioned |
| Fidelity | Encode/decode + `compactnessReady` | Same fidelity control as trained RAVE; else hide/disable + explanation | `applyFidelity` on encode path |

## Fail-safe audio

| Condition | Output |
|-----------|--------|
| `empty` / cleared / never loaded | Dry passthrough main audio |
| `error` and no retained prior factory | Silence + error message |
| `loading` / failed reload with prior factory | Keep prior factory until success or clear |
| `ready` | Model output |

## Unfreeze

- Unfreeze modular restore: **disabled** for `externalLoad` with user-visible explanation.

## Errors (recoverable, no host stop)

- Missing / unreadable file
- `jit::load` or probe failure
- Incomplete shape when inference fails and overrides missing

## Acceptance

- FR-003, FR-004, FR-006, FR-008, FR-009, FR-010, FR-012–FR-017, SC-002, SC-003, SC-007.
