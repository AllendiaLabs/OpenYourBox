# Phase 0 Research: RAVE Architecture & Training

## Decision 1: Unify Train via `objective` on the existing worker — not a second product

**Decision**: Extend `TrainCoordinator` + `Backend/train_worker.py` and the existing Train panel. Add `train_options.objective`: `mapping` | `reconstruction`. Mapping keeps the Phase 3 steerable recipe unchanged. Reconstruction runs the two-stage RAVE recipe in the **same** process model (`train_steerable` operation, Run/Pause/Stop, progress events, hear-while-training checkpoints, success Gold auto-load). Persist last-used objective per plugin instance (ValueTree / processor state); default `mapping` if unset.

**Rationale**: Spec unification (FR-008, FR-008a). Constitution requires a single VST UI and one detached worker pattern. A second Train entry point would violate the product objective.

**Alternatives considered**:
- Separate `train_rave` operation and panel: rejected (two workflows).
- Infer objective from graph only: rejected (user chose explicit objective in clarify).
- New Python package wrapping acids-rave CLI: rejected (terminal/standalone app; constitution).

## Decision 2: Graph domains `audio` | `multiband` | `latent` plus integer rate factor

**Decision**: Extend `ShapeSignature` with `domain` values `audio`, `multiband`, `latent` and a `temporalRate` (samples of this domain per host sample, or equivalently hop product along the path). Connections MUST match domain **and** temporal rate **and** channel rules. Tooltips name the mismatch. Host Audio Input/Output remain `audio` at rate 1.

**Rationale**: FR-006. RAVE is not a same-rate effect: PQMF changes channel count (bands), encoder strides change temporal rate, bottleneck is a compact latent trajectory.

**Alternatives considered**:
- Overload `channels` only: rejected — 16-band stereo vs 16-ch audio would collide.
- Implicit reshape at every node: rejected — hides illegal graphs.

## Decision 3: New live node types; reuse TCN/Activation/Merge/Linear

**Decision**: Add palette types:

| Type | Role |
|------|------|
| `pqmfAnalysis` | Audio → multiband; default `nBand = 16` |
| `pqmfSynthesis` | Matching inverse; same `nBand` |
| `rateConv` | Causal Conv1d with integer `stride` and `direction` downsample \| upsample (transpose) |
| `variationalBottleneck` | Encoder features → latent sample; KL during reconstruction train; fidelity at inference |
| `noiseSynthesizer` | Learned filtered-noise addend (DDSP-style amp→IR) |

Original layout: strided `rateConv` encoder, bottleneck, decoder with waveform × loudness (Merge multiply + sigmoid) + optional noise. Latest continuous layout: residual dilated units expressed as existing **TCN** (or residual `rateConv` stacks with dilation) between down/up `rateConv` stages, amplitude modulation via channel-split × sigmoid.

**Rationale**: FR-001, FR-005. OpenYourBox philosophy is compose-from-elements, not a single opaque RAVE node (layouts are starting graphs).

**Alternatives considered**:
- One `RaveModel` mega-node: rejected — cannot inspect/edit like TCN.
- Port acids-rave Python modules into the live engine: rejected — live path is C++ LibTorch.

## Decision 4: Causal streaming with preallocated leftover/history buffers

**Decision**: Live `rateConv`, PQMF, and bottleneck use **causal** padding only (spec FR-016). Streaming uses the existing TCN pattern: preallocated history / leftover sample rings sized at graph prepare (GUI thread), **zero audio-thread allocations**. Downsample nodes emit when a full stride of new samples is available; upsample nodes expand in the prepared output buffer. Reconstruction **train** uses the same causal operators so weights match live.

**Rationale**: VST insert cannot use non-causal lookahead. Official RAVE `causal.gin` is the parity path.

**Alternatives considered**:
- Non-causal train + causal export: rejected (weight mismatch, spec).
- Block-aligned only (require host buffer multiple of 2048): rejected (DAW buffers vary).

## Decision 5: PQMF = Kaiser cosine-modulated bank matching acids-rave

**Decision**: Implement analysis/synthesis from the official RAVE PQMF (Kaiser prototype, cosine modulation, `n_band` default 16, high attenuation). Same coefficients in C++ live and Python train/freeze builders. Bypass analysis→synthesis MUST reconstruct with small error (near-perfect reconstruction of the filterbank).

**Rationale**: Paper Appendix B + `rave/pqmf.py`. Band count mismatch is a shape error (FR-002).

**Alternatives considered**:
- STFT multiband: rejected (not RAVE).
- Learned filterbank: out of scope.

## Decision 6: Reconstruction recipe = official continuous (v2) two-stage, one trainer

**Decision**: Worker-internal recipe when `objective = reconstruction`:

1. **Stage 1** (1,000,000 steps default): spectral distance (fullband + multiband if PQMF present), windows `{2048,1024,512,256,128}`, hops window/4; variational KL with beta warmup (v2: ~1e-6 → 5e-2 over ~20k steps). Encoder + decoder trained.
2. **Stage 2** (1,000,000 steps default): encoder **frozen** (`detach`); decoder + train-only discriminators (combined multi-period + multi-scale, hinge GAN, relative feature matching). Dual spectral terms continue.
3. Optimizers: generator Adam ~1e-3, discriminator Adam ~1e-4 (worker constants, not v1 UI).
4. Success auto-load **only** after both stages complete without Stop.
5. Original **layout** still uses this trainer (no legacy v1 trainer).

Discriminators exist only in the worker (FR-012).

**Rationale**: Spec FR-011; acids-rave 2.3 `v2.gin` + `model.py` warmup; clarify B for stage-2 length.

**Alternatives considered**:
- Mapping MR-STFT on autoencoders: allowed only if user picks mapping (explicit).
- acids-rave Lightning in-process: rejected (extra app/CLI).
- User-wired discriminator graph: rejected (out of scope).

## Decision 7: Gold RAVE = one TorchScript with `forward` / `encode` / `decode`

**Decision**: Reconstruction success (and freeze of a valid RAVE subgraph) exports a module exposing:

- `forward(audio) -> audio` (encode then decode, fidelity applied on latent)
- `encode(audio) -> latent`
- `decode(latent) -> audio`

Live Gold routes host audio to `forward` by default. Graph pins on the Gold node: audio in, audio out, latent out (encode), latent in (decode). Fidelity is a **runtime control** (0–100%) stored on the node, applied inside encode/forward/decode like FiLM `c` on Gold TCN — not a weight tensor. Compactness basis (`latent_pca`, `latent_mean`, `fidelity` cumulative variance) is baked into the artifact from stage-1 validation PCA (sklearn-equivalent in worker); if PCA cannot run, full latent width with UI status.

**Rationale**: Clarify C; FR-013, FR-014; nn~ RAVE method split.

**Alternatives considered**:
- Three Gold nodes: rejected (Unfreeze/auto-load complexity).
- Fidelity only after Unfreeze: rejected (user: always-on like XY→FiLM).

## Decision 8: Library tags, warn/filter, pair x+y for reconstruction

**Decision**: Extend Training Library entries:

- `kind`: `pair` | `clip` (system tag; also stored as tags `pair` / `unpaired`)
- User tags optional
- Import single file or Capture **Single** → `clip`
- Capture **Pair** / pair import → `pair`

Train panel objective drives Library:

- **mapping**: unpaired selected → **error**, Run blocked; filter/warn unpaired as ineligible
- **reconstruction**: selected pairs contribute **both x and y** as corpus clips; clips included as-is; channel count must match armed graph; mixed SR still blocked

**Rationale**: Clarify library rules; FR-009*.

**Alternatives considered**:
- Side picker for pairs: rejected (user: use x and y).
- Hide unpaired when mapping: warn+filter allowed; still error if selected.

## Decision 9: Capture kind Pair | Single in the same menu

**Decision**: Capture Samples adds `captureKind`: `pair` | `single`. Pair = existing master/slave Clean/Processed. Single = record this instance’s input into a `clip` library entry; no slave; default bypass still applies.

**Rationale**: FR-008b; unification.

**Alternatives considered**:
- Import-only unpaired: rejected (user chose B).
- Infer single when unpaired objective: rejected (capture kind is explicit).

## Decision 10: Causal RAVE delay vs constitution 5 ms / 7 ms

**Decision**: Feedforward TCN/effect graphs **keep** constitution live/frozen latency targets. RAVE layouts with PQMF + stride product 2048 **will** incur causal delay (filter + conv padding) that can exceed 5–7 ms. Product MUST: (1) remain causal (no lookahead), (2) **display delay** (samples/ms) on RAVE layouts and Gold, (3) use preallocated streaming state, (4) treat the 5/7 ms table as applying to non-rate-reducing graphs. This is a justified gate exception (see plan Complexity Tracking).

**Rationale**: Cannot satisfy paper RAVE architecture and sub-5 ms at 48 kHz simultaneously. Spec forbids non-causal.

**Alternatives considered**:
- Shrink to raspberry config by default: deferred (layouts are original/latest continuous).
- Non-causal Gold: rejected.

## Decision 11: Reconstruction validity gate before Run

**Decision**: When objective is reconstruction, Run requires an armed path: audio → (optional PQMF analysis) → encoder `rateConv`/TCN stack → `variationalBottleneck` → decoder stack → (optional matching PQMF synthesis) → audio, with matching `nBand` and channel width vs corpus. Missing bottleneck or broken decode path → refuse with reason. Mapping on a RAVE graph remains allowed (whole graph as x→y).

**Rationale**: FR-008, edge cases.

## Decision 12: NOTICE / license for RAVE method

**Decision**: Planning MUST add acids-rave / paper attribution to `NOTICE` when code is ported (PQMF, residual dilated units, recipe). Apache-2.0 plugin core unchanged; RAVE-derived snippets follow upstream license (MIT for acids-rave).

**Rationale**: Constitution NOTICE policy; spec assumption.

**Alternatives considered**:
- Ship acids-rave as a dependency CLI: rejected (constitution).
