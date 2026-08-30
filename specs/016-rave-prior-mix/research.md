# Phase 0 Research: RAVE Prior Mix & Insert Catalog

## Decision 1: Prior-mix math and processing order

**Decision**: For RAVE-capable Gold / external-load boxes with encode/decode:

1. If `priorMix < 1` (with a small epsilon for “full prior”), run `encode` to obtain base mean `μ_e` and spread `σ_e`.
2. Else skip encode; set `μ_e = 0`, `σ_e = 1` (broadcast to effective latent shape / time).
3. Linearly interpolate: `μ = (1 − α) μ_e + α · 0`, `σ = (1 − α) σ_e + α · 1` where `α = priorMix ∈ [0, 1]`.
4. Apply fidelity/compactness **before** this mix when compactness is active, on the encoder distribution (or on `μ_e` with existing `applyFidelity` semantics adapted so discarded dims behave as prior)—so mix/bias/scale see the effective latent space (FR-009).
5. `μ ← μ + bias`, `σ ← σ ⊙ scale` (disconnected → bias 0, scale 1).
6. Sample once: `z = μ + σ ⊙ ε` with preallocated `ε ~ N(0,1)`.
7. Publish `z` on latent out; `decode(z)`.

**Rationale**: Matches clarifications (lerp mean/spread toward 0/1, then bias/scale on every mode; one sample; encoder skip at full prior).

**Alternatives considered**:
- Sample posterior and prior then lerp `z`: rejected (clarification chose distribution lerp).
- Apply bias/scale only in prior mode: rejected (user wants both modes).
- Post-sample bias only (some VST UIs): rejected (spec is mean/spread).

## Decision 2: Live path must expose spread (upgrade from μ-only)

**Decision**: Today Gold/live encode is **μ-only** (`VariationalBottleneck::encodeMean`, TorchScript `encode` returning a single latent). This feature **requires** a live `(μ, σ)` path for RAVE-capable boxes:

- **OYB-trained / freeze RAVE exports**: Extend export and/or C++ bottleneck helpers so the Gold path can read softplus-std (`σ = softplus(scale) + ε`) alongside μ (same convention as `train_worker` / `VariationalBottleneck::softplusEpsilon`). Prefer a dedicated distribution encode helper used only by the prior-mix BlackBox path so unrelated μ-only consumers stay stable until migrated.
- **External TorchScript**: Prefer methods that yield `(μ, σ)` when present. If only a single latent tensor is available from `encode`, **fallback**: treat it as `μ_e` with `σ_e = 1` for mix math (document in quickstart). Prior path remains true 0/1; bias/scale still apply.

**Rationale**: Without spread, scale and intermediate mix are undefined. Fallback keeps third-party RAVE usable without blocking the feature.

**Alternatives considered**:
- Keep μ-only and reinterpret “scale” as post-sample gain: rejected (contradicts clarified mean/spread semantics).
- Require every external `.pt` to ship `(μ, σ)` or fail load: rejected (breaks many published RAVE checkpoints).

## Decision 3: `priorMix` as fidelity-class element property

**Decision**: Add `priorMix` as `NodeProperty` `PropertyKind::real` in `[0, 1]`, default `0` (full forward). Mirror on `GraphNode` (like `fidelityPercent`). Allow edits on frozen Gold (same exception pattern as `fidelity`). Feed `RuntimeControlState` / `CompiledElement` at compile/process. Persist via existing ValueTree node serialization. Not a host-only APVTS exception and not a required pin.

**Rationale**: Clarification — same pattern as activation gain / fidelity.

**Alternatives considered**: Host-only parameter or modulation pin: rejected in clarify session.

## Decision 4: Bias / scale pins replace latent input

**Decision**:
- Remove latent **input** pin from RAVE-capable surfaces (`applyExternalLoadSurface` + train-autoload Gold pin setup).
- Add input pins labeled for **bias** and **scale** (new helpers `isBiasPin` / `isScalePin`; distinct from `latentPinLabel`).
- Keep latent **output** pin.
- Compile indices on `CompiledElement` (bias/scale gather like Control/latent taps). Shape = effective latent channels (`flexibleTensorShape` / existing latent width rules).
- Disconnected: runtime constants 0 and 1 (no cable required).
- Remove decode-from-wired-latent branch that skipped encode whenever latent-in was connected; prior mix + bias/scale replace that drive model.
- Migration of old latent-in cables: **out of scope** (spec FR-011).

**Rationale**: Spec FR-004–007; RAVE-VST-style steering.

**Alternatives considered**:
- Keep latent-in alongside bias/scale: rejected (spec replaces).
- Remap old latent-in → bias: rejected (out of scope / wrong semantics).

## Decision 5: Latent out = effective sampled `z`

**Decision**: After mix→bias/scale→sample, write that tensor into `runtime.latentOutputs[i]` **before** `decode`, so existing latent-tap gather and the latent-out pin always expose decoder-driving values (including full prior).

**Rationale**: FR-008 / SC-005. Today latent out is post-fidelity μ or wired latent-in — both wrong after this feature.

**Alternatives considered**: Dual outs (pre/post sample): rejected (spec wants effective sampled only).

## Decision 6: Right-click catalog — shared Factory + User Library menu

**Decision**:
- Extract the hard-coded `paletteCategories` / `forEachPaletteItem` / place helpers from `NodeRenderer.cpp` into a shared **Factory palette** module used by **left Factory** and **Pin Add / Link Insert** context menus so both stay identical and current.
- Under Pin **Add** and Link **Insert**, add categorized Factory entries (existing `canInsertOnLink` / attach filters remain) plus a **User Library** submenu built from `UserBoxLibrary` folder/entry/snapshot-member tree (reuse 006/011 expand + `insertBox(..., nestedRootId)`).
- Scope is **right-click Pin/Link menus only** (FR-015). Empty-canvas node context need not gain Add unless already present.
- Library place onto pin/link: prefer place-at-cursor then best-effort wire when shapes allow; if auto-wire for groups is hard, place near pin/link and leave wiring to the user for v1 of library-on-pin (Factory attach/insert keep today’s `attachNodeToPin` / `insertNodeOnLink`). Document in contract.

**Rationale**: Context menus are stale because they flatten the same hand-maintained arrays; user library exists only in `UserBoxLibraryPanel`. Shared catalog fixes staleness; menu tree reuses proven insert APIs.

**Alternatives considered**:
- Auto-generate menu from every `NodeType`: rejected (would include non-palette types).
- Fix left Library only: rejected (clarify scoped to right-click).
- Duplicate user-library list in the menu without hierarchy: rejected (spec requires expandable hierarchy / nested subpart insert).

## Decision 7: Noise / RT safety for sampling

**Decision**: Allocate `ε` (and any mix work buffers) in prepare/compile sized for max block × latent × batch; fill with RNG off the hot path or with a pre-seeded buffer refreshed without heap ops on the audio thread. No `torch::randn` allocating on the audio thread.

**Rationale**: Constitution zero-allocation rule; fidelity path already uses `randn_like` in places — this feature must not regress that rule for the new sample step (replace or confine allocs to prepare).

**Alternatives considered**: Allocate per buffer: rejected (constitution).
