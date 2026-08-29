# Phase 0 Research: DDSP Effects & Recurrent Layers

## Decision 1: Port Magenta DDSP semantics to LibTorch / PyTorch (no TensorFlow runtime)

**Decision**: Reimplement Magenta `ddsp/effects.py` effect behavior (parameter roles, IR synthesis, FFT convolution, LTV-FIR, ModDelay) in LibTorch for live and PyTorch for freeze/train. Do not embed TensorFlow or call Magenta at runtime.

**Rationale**: Constitution mandates LibTorch live + TorchScript freeze bridge. Magenta sources are Apache 2.0 and may be cited in NOTICE as a behavioral reference, matching acids-rave attribution practice.

**Alternatives considered**:
- Ship TensorFlow DDSP in-process: rejected (violates architectural mandate, dual runtime).
- Bind Python Magenta only in the worker: rejected (live Blue path must run in C++ without Python on the audio thread).

## Decision 2: Factory palette categories — Effects and Neural / Sequence

**Decision**: Replace the flat alphabetical Factory list with grouped sections: **Effects** (five DDSP types) and **Neural / Sequence** (LSTM, RNN), retaining existing types in current logical groups (or a default “Layers / Utilities” section). Payload remains `OPENYOURBOX_NODE_TYPE`; only presentation changes.

**Rationale**: Spec FR-001 / FR-010 and SC-004 require discoverability by category. No Effects/Neural grouping exists today (`NodeRenderer` `paletteItems`).

**Alternatives considered**:
- Flat list with renamed labels only: rejected (fails discovery success criterion).
- Separate windows/menus: rejected (unnecessary UI surface; constitution prefers single VST interface patterns).

## Decision 3: Live-safe `reverb_length` soft warning (InfoPanel / property affordance)

**Decision**: Allow any positive `reverb_length`. When length in milliseconds exceeds a live-safe threshold (default aligned with existing InfoPanel RF warning of **1000 ms**, or documented equivalent for IR samples at current sample rate), show a non-blocking orange warning. Do not clamp, refuse, or stop audio for length alone. Compile still fails only if required history would exceed `maximumHistorySamples` (hard safety net).

**Rationale**: FR-006a / SC-008. Closest precedent is `InfoPanel` RF &gt; 1000 ms warning.

**Alternatives considered**:
- Hard-clamp length: rejected (clarification).
- Refuse property edit: rejected (clarification).
- No UI feedback: rejected (SC-008).

## Decision 4: Reverb IR blend + empty external IR fallback

**Decision**: Reverb always owns an internal/trainable IR of `reverb_length`. Optional second audio/feature input accepts an external IR. When both are present, wet convolution uses a blendable IR: `ir = (1 − blend) * internal + blend * external` (normalized lengths via pad/crop policy documented in contracts). When external is connected but empty/zero-length, ignore external contribution, use internal, set recoverable `graphWarningMessage` (or property-level warning), continue audio.

**Rationale**: Clarifications for FR-006 / FR-006b.

**Alternatives considered**:
- External replaces internal when connected: rejected (user chose blend).
- Hard error / silence on empty IR: rejected (must not stop host).

## Decision 5: Magnitude grids — dimensions + init/randomize, no cell painter

**Decision**: FilteredNoiseReverb and FIRFilter expose `n_frames` (time steps) and `n_filter_banks` as editable ints plus `window_size`; magnitudes live as a trainable weight tensor `[n_frames, n_filter_banks]` (or channel-broadcast equivalent). UI provides init/randomize (existing weight randomization). No per-cell painter in v1. Illegal dimension changes that break connected shapes are refused with Shape Integrity messaging.

**Rationale**: Clarification Session 2026-08-29; FR-003 / FR-004.

**Alternatives considered**:
- Cell painter UI: deferred (out of v1).
- Magnitudes as non-trainable constants only: rejected (spec requires trainable + randomize).

## Decision 6: Custom single-layer RNN/LSTM with in-cell activation + gain

**Decision**: Implement custom single-layer unidirectional cells (optional bidirectional via forward+backward concat). Do **not** use stock `torch::nn::LSTM` / `nn.LSTM` when activation ≠ tanh or when gain ≠ 1, and prefer one code path that always uses the custom cell for parity. Activation choices and `negative_slope` / `gain` bounds match Activation/TCN (`ActivationType`, `gain` 0.1–10, LeakyReLU slope 0–1). Gain multiplies the pre-nonlinearity argument of the **primary** cell nonlinearity (RNN `nonlinearity`; LSTM candidate/`tanh` slot — gates remain sigmoid). Bias toggle maps to learnable bias on/off. `num_layers` is never exposed; depth = stacked nodes.

**Rationale**: FR-011 through FR-011e. Stock cells cannot replace in-cell nonlinearity or apply Activation/TCN gain.

**Alternatives considered**:
- Stock LSTM + post-activation: rejected (clarification: in-cell, not post-layer).
- Multi-layer `num_layers` property: rejected (clarification: stack elements).
- Explicit `input_size` property: rejected (infer from upstream channels).

## Decision 7: Hidden-state carry via `LiveGraphRuntime` slots; reset on prepare/randomize

**Decision**: Allocate per-element recurrent hidden (and LSTM cell) state in `LiveGraphRuntime` during `prepare`, sized for max batch/channels/hidden used by the compiled graph. Carry state across `process` calls. Call zero/reset when a new runtime is prepared (rebuild, reconnect, freeze/unfreeze swap) and when that element’s weights are re-init/randomized. Bidirectional maintains separate forward/backward states.

**Rationale**: FR-011a; mirrors causal `histories[]` reset model already used by Conv/TCN/PQMF.

**Alternatives considered**:
- Reset every buffer: rejected (clarification).
- Persist hidden across freeze swap: rejected (lifecycle reset required).

## Decision 8: Full sequence output; bidirectional doubles channels

**Decision**: For each input time step, emit a corresponding output time step. Unidirectional: `out_channels = hidden_size`. Bidirectional: concatenate forward and backward along channel dim → `out_channels = 2 * hidden_size`. Time length equals input time length. Shape inference updates output pin accordingly when `hidden_size` or `bidirectional` changes.

**Rationale**: FR-011b.

**Alternatives considered**:
- Last-timestep-only output: rejected (clarification).
- Bidirectional sum/mean instead of concat: rejected (spec: concat → 2× channels).

## Decision 9: Freeze/train registry — custom modules + Reverb multi-input

**Decision**: Add type string branches in `freeze_worker.build_module` and `train_worker.build_module` (and DAG/`build_rave_graph_module` or equivalent multi-input path for Reverb’s optional IR port). Export TorchScript-compatible `nn.Module` graphs. Trainable tensors: Reverb IR; FilteredNoise/FIR magnitudes; LSTM/RNN weights/biases. Scalar effect controls (`add_dry`, `decay`, dims, etc.) are module constructor / buffer / forward kwargs consistent with existing property→module patterns. Live-only gain on recurrent cells: follow Activation/TCN precedent (document whether freeze bakes gain; prefer baking gain into scripted forward for audible parity when frozen).

**Rationale**: FR-012 / FR-013; dual-engine constitution.

**Alternatives considered**:
- Live-only effects without freeze: rejected (explicit freeze eligibility).
- Always reject Reverb with IR input on freeze: rejected (legal dual-input graphs must freeze).

## Decision 10: Reuse NoiseSynthesizer FFT helpers where DDSP aligns; distinct Magenta IR recipes

**Decision**: Prefer existing `NoiseSynthesizer::fftConvolve` / amplitude→IR utilities for convolutional wet paths when math matches. Implement Magenta-specific ExpDecay (scaled gain × exp decay × noise), FilteredNoise IR synth, `frequency_filter` LTV-FIR, and ModDelay variable-length delay as dedicated helpers so acids-rave and Magenta semantics stay distinguishable in code and NOTICE.

**Rationale**: Avoid duplicating FFT infrastructure while preventing RAVE/DDSP behavior bleed.

**Alternatives considered**:
- Call Noise Synth element internally: rejected (wrong domain/API for DDSP params).
- Full independent FFT stack: unnecessary complexity.

## Decision 11: Naming — effect `gain` vs Activation/TCN `gain`

**Decision**: Keep Magenta property key `gain` on ExpDecayReverb and ModDelay (effect amplitude). Keep Activation/TCN/RNN/LSTM `gain` as pre-nonlinearity slope. Document in data model that keys share the string `"gain"` but semantics are type-scoped (same pattern as overloaded properties elsewhere). UI labels may disambiguate (“IR Gain” vs “Nonlinearity Gain”) without changing persistence keys unless an existing conflict forces a rename—prefer Magenta keys for effects and Activation keys for recurrent cells.

**Rationale**: Spec lists Magenta names; Activation/TCN already own `"gain"`. Type-scoped properties avoid migration.

**Alternatives considered**:
- Rename effect gain to `ir_gain`: diverges from Magenta FR-016 naming.
- Rename recurrent gain: breaks Activation/TCN parity (FR-011d2).

## Decision 12: Magenta DDSP NOTICE attribution

**Decision**: Add an Apache 2.0 Magenta/DDSP block to `NOTICE` stating behavioral reimplementation of `ddsp/effects.py` (no TensorFlow dependency), parallel to acids-rave attribution.

**Rationale**: Spec assumption + constitution licensing strategy.

**Alternatives considered**:
- No attribution: rejected (third-party notice practice).
- Vendor Magenta sources verbatim under TF: rejected (Decision 1).
