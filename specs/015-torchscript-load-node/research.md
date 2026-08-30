# Phase 0 Research: TorchScript Checkpoint Loader Node

## Decision 1: Reuse `NodeType::blackBox` + `BlackBoxOrigin::externalLoad`

**Decision**: Palette item **TorchScript Load** creates a `NodeType::blackBox` node with `blackBoxOrigin = externalLoad`, empty `artifactPath`, and Gold presentation when ready. Do not introduce a parallel live Blue TorchScript module.

**Rationale**: Constitution Dual-Engine already runs TorchScript only through the Frozen BlackBox path. `LiveGraphEngine` already implements forward / encode→decode / conditioning / fidelity once `resolveFrozenBlackBox` returns a factory. A new origin distinguishes Unfreeze policy and empty/error UX from `manualFreeze` / `trainAutoload`.

**Alternatives considered**:
- New `NodeType::torchScriptLoad` with a separate compile path: rejected for v1 (duplicates kernel wiring; higher churn).
- Place train-origin BlackBox and browse weights only: rejected (`setWeightsPath` today does not publish the registry; Unfreeze semantics wrong).

## Decision 2: Prepare via existing `TorchScriptBlackBoxFactory::load` + published registry

**Decision**: On successful file choice (and on restore), call a dedicated `prepareExternalArtifact` (same family as `prepareTrainedArtifact`): off-audio-thread `TorchScriptBlackBoxFactory::load`, then publish into `publishedFrozenArtifacts` keyed by absolute path. Audio resolves via existing `resolveFrozenBlackBox`.

**Rationale**: FR-003 / SC-002. Explore confirmed browse-weights alone sets path without factory publish — that gap must be closed for this feature (and benefits external load specifically).

**Alternatives considered**:
- Load synchronously on the message thread only for tiny files: rejected (large RAVE `.pt` would stall UI; still must not touch audio thread).
- Embed module bytes in the patch: rejected (out of scope; path reference only).

## Decision 3: Capability detection — encode/decode, conditioning, compactness

**Decision**:
- **Encode/decode**: `module.find_method("encode") && find_method("decode")` (already in factory load / kernel).
- **Conditioning**: Reuse factory probe — try unconditioned `forward(audio)`; on failure or hint, probe `forward(audio, cond)` with `detectConditioningDim`. Expose Control pin only when factory reports conditioned.
- **Compactness / fidelity**: Read attrs `compactness_ready`, `latent_mean`, `latent_pca`, `cumulative_variance` via existing `copyCompactnessFromArtifact`; show fidelity when encode/decode present and `compactnessReady`; else hide/disable with explanation.

**Rationale**: Matches clarifications (auto-detect encode/decode; Control only if advertised; fidelity when compactness path exists) and existing trained RAVE Gold behavior.

**Alternatives considered**:
- Require OpenYourBox freeze/train metadata JSON beside `.pt`: rejected (blocks third-party RAVE exports that only ship methods).
- Always show Control + latent: rejected (clarifications).

## Decision 4: Channel inference + editable overrides

**Decision**: On load, probe with a preferred input channel count (host channel count, else last override, else 1), taking output channels from probe `size(1)`. If probe fails, retry a small set of common widths `{1, 2, 4, …}` before failing to “incomplete shape / enter overrides.” Prefill editable override fields (`inputChannels`, `outputChannels`, optional `latentChannels`). Shape checking uses overrides when set; Reset restores last inferred values. Persist overrides with the node.

For latent width when encode/decode exists: probe `encode` output channel dim (or keep `defaultLatentSize` then refine); expose as override alongside audio I/O.

**Rationale**: FR-011; factory today takes **caller** input channels and infers output — overrides close the gap for odd exports without blocking RAVE.

**Alternatives considered**:
- Infer-only: rejected (clarification chose infer + overrides).
- User-only fields: rejected (breaks one-click pretrained load).

## Decision 5: Pin surface morph after load

**Decision**: Initial palette node: main audio in/out only (no Control, no latent), dry-passthrough runtime. After successful load:
- Add latent in/out pins when `hasEncodeDecode()` (labels/routing match trained RAVE Gold / `isLatentPin`).
- Add Control input when factory `acceptsConditioning` / conditioned flag.
- Remove those pins when clearing path or loading a file that lacks the capability (reconnect illegally → refuse / drop with shape feedback).

**Rationale**: FR-012 / FR-015 / FR-013; avoids dead pins on plain forward models.

**Alternatives considered**:
- Always allocate Control like `makeNode(blackBox)` today: rejected (clarification B).
- Separate encode-only / decode-only Factory elements: rejected (clarification chose auto-detect on one element).

## Decision 6: Empty / error audio policy

**Decision**:
- Empty path and never successfully loaded (including after clear): **dry passthrough** of main audio in → out + choose-file affordance.
- Load error / invalid file with **no** retained prior ready factory: **silence** from that node + recoverable error string.
- Prior ready model: keep running during failed or in-progress reload until successful atomic swap or explicit clear.

**Rationale**: Clarification Option C; FR-017 / SC-007.

**Alternatives considered**:
- Always silence: rejected (user chose C).
- Always dry passthrough: rejected (masks errors as “working” dry signal).

## Decision 7: Unfreeze disabled for external loads

**Decision**: Context menu Unfreeze (restore modular Blue subgraph) is unavailable or disabled with explanation when `blackBoxOrigin == externalLoad`. No `sourceSubgraph` is stored for these nodes.

**Rationale**: FR-008; constitution Unfreeze assumes an OpenYourBox-authored modular graph.

**Alternatives considered**:
- Attempt to reverse-engineer Blue nodes from TorchScript: rejected (out of scope, unreliable).

## Decision 8: Persist path + overrides; re-prepare on restore

**Decision**: Persist `artifactPath` / `weightsPath`, `blackBoxOrigin=externalLoad`, channel overrides, fidelity, compactness mirrors. On session/preset/`applyPatchSnapshot` restore, for each externalLoad (and ideally any Gold with `artifactPath`), if the file exists, call prepare again into the registry; if missing, restore path string, set error status, silence (no prior model).

**Rationale**: FR-007 / SC-005; explore confirmed registry is in-memory and restore currently does not re-`load`.

**Alternatives considered**:
- Rely on user re-browsing after every launch: rejected (fails SC-005).
- Copy `.pt` into project package in v1: deferred (assumption: path reference only).

## Decision 9: No Python worker changes for v1

**Decision**: Do not add freeze/train `build_module` types for TorchScript Load. External files are already TorchScript. Optional future: a converter worker for non-script checkpoints — out of scope.

**Rationale**: Spec targets runnable TorchScript artifacts (e.g. pretrained RAVE exports).

**Alternatives considered**:
- Auto-trace arbitrary `state_dict` in the backend: deferred (scope / reliability).
