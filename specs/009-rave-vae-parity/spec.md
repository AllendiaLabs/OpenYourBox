# Feature Specification: RAVE Variational Bottleneck Parity

**Feature Branch**: `009-rave-vae-parity`

**Created**: 2026-08-27

**Status**: Clarified (ready for planning)

**Input**: User description: "oyb should reproduce acids-ircam rave code more faithfully, starting with: add softplus to vae and perform exactly the same parameterization/sampling; kernel size of variational convs should be 5 (but add as bottleneck parameter for flexibility) with grouped head instead of two full-width convs; PCA on μ in eval mode over a validation pass, use linear singular-value cumulative sum for r_f, and load compactness buffers onto the live bottleneck after Unfreeze."

## Clarifications

### Session 2026-08-27

- Q: When compactness PCA runs after representation learning, which audio should supply the mean latents (μ)? → A: **2% held-out validation subset** (matching acids-rave `split_dataset` at ~98% train / 2% val, fixed seed 42, validation size capped at 1000 segments). Validation clips are **never used in stage-1 training**. Compactness collects **μ only from validation batches in eval mode** at representation-stage end (phase 1), before reparameterized sampling — not from training audio, not from a random training minibatch, and not from a full-corpus pass.
- Q: Should the variational bottleneck’s grouped-head group count be fixed at 2 or exposed as a user-configurable property? → A: **Fixed at 2 groups** — group 1 is the **mean** branch, group 2 is the **variance** branch (reference pattern). Only **kernel size** is user-configurable alongside latent width and fidelity; group count is not a separate property.
- Q: While reconstruction training runs in the background, should the live Blue bottleneck on the audio path use mean-only or stochastic sampling during stage 1? → A: **Live audio path always mean-only (μ)**; **stochastic sampling only inside the background training worker**. Hear-while-training checkpoints follow the same live rule. Worker samples for ELBO/KL gradients; live path gives stable monitoring parity with reference eval encode.
- Q: Should fidelity/compactness controls stay inactive on hear-while-training checkpoints loaded before PCA completes? → A: **Yes — inactive until PCA completes** at representation-stage end. Mid-stage-1 checkpoints have no compactness buffers; fidelity defaults to full latent width with a clear **“compactness not ready”** status.
- Q: For RAVE models trained before this parity feature ships, should the plugin require retraining or preserve legacy behavior? → A: **No backward compatibility required** — pre-parity Gold artifacts and legacy bottleneck geometry may be broken on upgrade; users must retrain for reference-equivalent behavior. No migration path (no production models exist yet).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Train and Infer With Reference-Equivalent Latent Sampling (Priority: P1)

A user who trains RAVE reconstruction models in OpenYourBox expects the variational bottleneck to behave like the official acids-ircam RAVE implementation: the same variance parameterization, the same reparameterized sampling during training, and the same deterministic mean-only path during evaluation. Today, subtle differences in how variance is produced and when noise is injected can change learned weights, reconstruction quality, and how faithfully timbre transfer matches models trained in the reference toolchain.

**Why this priority**: Latent sampling is the mathematical core of the VAE; every downstream behavior (training convergence, encode/decode parity, compactness) depends on matching reference parameterization first.

**Independent Test**: Run reconstruction training on a small corpus with a RAVE layout whose bottleneck uses reference defaults; compare encode outputs at evaluation time (no stochastic sampling) against a reference run on the same graph topology and seed-controlled weights; confirm sampling is active only during training and mean-only during evaluation/live inference.

**Acceptance Scenarios**:

1. **Given** a variational bottleneck on a RAVE graph, **When** reconstruction training is active, **Then** latent values are drawn using the reference reparameterization (mean plus scaled noise from a positive variance derived via the reference softplus parameterization)
2. **Given** the same trained graph in evaluation or live inference, **When** audio passes through encode or forward, **Then** the bottleneck uses the mean path only (no injected sampling noise), matching reference eval behavior — **including while reconstruction training runs in the background**
3. **Given** reconstruction stage 1 is active in the worker, **When** audio plays through the live Blue bottleneck (or a hear-while-training checkpoint), **Then** the live path still uses **μ only**; stochastic sampling occurs **only inside the training worker**, not on the audio thread
4. **Given** a user inspects bottleneck behavior during training versus after train completes, **When** they compare stochastic versus deterministic paths, **Then** the worker uses sampling during stage-1 training forwards while the live path always uses mean-only
5. **Given** KL regularization is computed during representation learning, **When** the regularizer is applied, **Then** it uses the same closed-form expression implied by the reference variance parameterization

---

### User Story 2 - Configure Bottleneck Head Geometry Like Reference RAVE (Priority: P1)

Advanced users and layout authors need the variational head to match the reference architecture: a configurable temporal kernel (default matching reference RAVE at width 5), and a grouped projection head that splits input channels across parallel convolutions rather than two separate full-width projections. The kernel size must remain a user-facing bottleneck parameter so graphs can experiment without forking the element type.

**Why this priority**: Head geometry affects receptive field and parameter sharing; mismatches prevent weight-compatible graphs and change what the encoder can express before the latent.

**Independent Test**: Insert a latest-continuous RAVE layout; confirm the variational bottleneck exposes a kernel-size parameter defaulting to the reference value; wire a graph whose encoder output width matches the reference grouped-head split; confirm shape checking accepts legal configurations and refuses illegal channel/group combinations with a clear message.

**Acceptance Scenarios**:

1. **Given** a variational bottleneck element, **When** the user views its properties, **Then** they can set temporal kernel size (default matching reference RAVE at 5) alongside existing latent-width and fidelity controls
2. **Given** reference-default settings on a compatible encoder width, **When** the bottleneck is wired, **Then** the head uses **two fixed groups**: group 1 projects to the **mean (μ)** branch and group 2 projects to the **variance** branch (reference pattern: grouped conv with `groups=2`, each group seeing half the encoder channels)
3. **Given** a user changes kernel size within supported bounds, **When** the graph is prepared, **Then** live inference, training, and freeze/export all honor the chosen kernel without silent fallback to a different geometry
4. **Given** an illegal combination (e.g., channel count not divisible by required groups), **When** the user wires or arms the graph, **Then** shape checking refuses the configuration with an explanatory tooltip
5. **Given** insertable RAVE layouts, **When** the user inserts original or latest-continuous layouts, **Then** the embedded bottleneck uses reference-default kernel and grouped-head settings unless the user edits them afterward

---

### User Story 3 - Compactness From Mean Latents on a Validation Pass (Priority: P1)

After representation learning, RAVE identifies informative latent directions via principal-component analysis on **mean latents** collected in **evaluation mode** across a **validation pass** over held-out or reserved corpus audio—not from a single training minibatch or from sampled latents. The fidelity control (`r_f`) must select how many leading components to keep using the **linear cumulative sum of singular values** (equivalent to explained-variance ratio in the reference method). The resulting compactness basis must travel with the trained model to Gold auto-load **and** be restored on the live variational bottleneck after Unfreeze so fidelity behaves identically in both execution modes.

**Why this priority**: Compactness is a defining RAVE control; wrong statistics (sampled z, training mode, or partial data) produce misleading fidelity curves and break parity with published models and the reference toolchain.

**Independent Test**: Complete a short reconstruction train; confirm compactness analysis runs in eval mode over multiple validation segments; sweep fidelity from 100% down and verify monotonic coarsening; Unfreeze the Gold model and confirm the same fidelity setting and compactness metadata apply on the live bottleneck without retraining.

**Acceptance Scenarios**:

1. **Given** representation stage completes, **When** compactness analysis runs, **Then** it collects **mean latents (μ)** with the model in **evaluation mode** across the **held-out validation subset only** (~2% of corpus, capped at 1000 segments, excluded from stage-1 training), not from training clips, stochastic samples, or a single random training minibatch
2. **Given** collected mean latents, **When** the compactness basis is computed, **Then** informative rank is determined by **cumulative singular-value ratio** (linear sum of singular values), matching reference `r_f` semantics
3. **Given** compactness succeeds, **When** reconstruction training finishes and Gold auto-loads, **Then** the Gold artifact embeds mean vector, principal basis, and cumulative singular-value ratios used by the fidelity control
4. **Given** a Gold RAVE model with compactness ready, **When** the user Unfreezes, **Then** the live variational bottleneck receives the same compactness buffers and honors the current fidelity percent without requiring a new train
5. **Given** the user adjusts fidelity on Gold or on the live bottleneck after Unfreeze, **When** audio plays, **Then** dropped latent dimensions are filled with prior noise per the reference compactness method and kept dimensions follow the PCA basis
6. **Given** compactness cannot run (insufficient validation data), **When** training otherwise completes, **Then** the job still finishes, fidelity falls back to full latent width, and the UI reports compactness as unavailable (consistent with existing RAVE spec behavior)
7. **Given** a hear-while-training checkpoint loaded **before** representation stage completes, **When** the user adjusts fidelity, **Then** the control is **inactive** (full latent width, **“compactness not ready”** status) until PCA completes at stage-1 end

---

### User Story 4 - End-to-End Parity Smoke Test for Practitioners (Priority: P2)

A practitioner validating OpenYourBox against acids-ircam RAVE can assemble the reference layout, train briefly, and observe that bottleneck statistics, eval encode paths, and fidelity curves qualitatively track the reference implementation on the same audio—not identical weights without shared initialization, but the same rules for sampling, head geometry, and compactness.

**Why this priority**: Confirms the three parity pillars work together; lower priority because each pillar is independently testable.

**Independent Test**: Side-by-side short train on the same mono clip with matched layout hyperparameters; compare eval mean-latent distributions and fidelity sweep behavior; document any remaining intentional product differences.

**Acceptance Scenarios**:

1. **Given** matched layout defaults and corpus, **When** a short reconstruction train completes in OpenYourBox and in the reference toolchain, **Then** eval mean latents from both systems are statistically comparable (same shape rules, no spurious sampling noise in eval)
2. **Given** compactness buffers from both systems on the same validation audio, **When** fidelity is swept at 50% and 90%, **Then** the effective kept rank differs in the same direction (higher fidelity keeps more dimensions)
3. **Given** a trained OpenYourBox model, **When** the user freezes or uses Gold encode/decode, **Then** exported behavior preserves bottleneck parity settings embedded at train end

---

### Edge Cases

- What happens if kernel size is set to 1? Supported as a legal parameter value for experimentation; shape and causality rules still apply; default layouts use reference width 5.
- What happens if encoder channel count is not compatible with grouped-head splitting? Wiring or Train is refused with a clear reason; user adjusts encoder width or bottleneck settings.
- What happens if the held-out validation subset has fewer frames than latent width? Compactness falls back gracefully (full width, status message) without aborting an otherwise successful train.
- What happens if the corpus is too small to form a 2% validation split? Use all available clips for validation only when stage-1 training can still proceed on the remaining train portion; if the split is unusable, compactness falls back with a status message (consistent with spec 005).
- What happens when hear-while-training checkpoints are loaded during stage 1? Live path uses **mean-only** encode; **fidelity/compactness controls are inactive** (full latent width, **“compactness not ready”**) until PCA completes at representation-stage end.
- What happens if the user randomizes weights on an Unfrozen bottleneck that had compactness buffers? Compactness metadata clears or marks stale until a new representation stage recomputes it.
- What happens if fidelity is changed before compactness is ready? Control applies full latent width or last known basis per existing RAVE fidelity rules; UI indicates compactness not ready.
- What happens when training is Stopped before representation stage completes? No compactness analysis; no parity buffers exported; consistent with existing reconstruction train stop semantics.
- What happens on graphs with multiple variational bottlenecks? Out of scope unless explicitly armed; armed path follows existing RAVE single-bottleneck convention from spec 005.
- What happens to pre-parity RAVE Gold artifacts or graphs with legacy 1×1 bottleneck heads after this feature ships? **No migration** — legacy artifacts may fail to load or require graph rebuild; users retrain with reference-parity bottleneck geometry.

## Out of Scope

- Full numerical weight import from acids-ircam checkpoint files (parity of **rules**, not binary weight transfer)
- Backward compatibility with pre-parity RAVE Gold artifacts or legacy 1×1 variational bottleneck geometry (breaking change acceptable; no production models yet)
- Changes to PQMF, rate conv strides, discriminator recipe, or two-stage step counts (covered by spec 005 unless bottleneck parity exposes a bug there)
- Alternate bottleneck types (discrete, Wasserstein, spherical) from other RAVE variants
- Recomputing compactness on the live audio thread during performance
- Automatic hyperparameter search for kernel size or group count beyond user/properties and layout defaults

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The variational bottleneck MUST use the reference RAVE variance parameterization (positive variance via softplus on a learned pre-activation) and MUST NOT use ad-hoc clamping that diverges from reference behavior.
- **FR-002**: During reconstruction **training**, the **background worker only** MUST sample latents with the reference reparameterization (mean plus scaled noise). The **live audio path** (Blue bottleneck, hear-while-training checkpoints, **encode**, and **Gold forward/decode**) MUST **always** use the mean path only without injected sampling noise — including while stage 1 is running in the worker.
- **FR-003**: KL regularization during representation learning MUST be consistent with the reference parameterization chosen in FR-001.
- **FR-004**: The variational bottleneck MUST expose a user-configurable temporal **kernel size** property defaulting to the reference RAVE value (**5**). Live engine, training worker, and freeze/export MUST all respect the configured value.
- **FR-005**: The variational head MUST use a **grouped** projection with **exactly 2 groups**: group 1 → **mean (μ)** branch, group 2 → **variance** branch (reference `groups=2` semantics). Group count MUST NOT be a user property. Each group sees half the encoder input channels and contributes half the latent width. This replaces two independent full-width 1×1 convolutions over all input channels.
- **FR-006**: Shape checking MUST validate grouped-head legality (encoder channels divisible by 2, minimum kernel, causal padding) and refuse illegal graphs with an explanatory tooltip.
- **FR-007**: Before stage 2 begins, the training worker MUST split the reconstruction corpus into **~98% train / ~2% validation** (fixed seed **42**, validation capped at **1000** segments, matching acids-rave). Stage-1 training MUST use **train clips only**. At representation-stage end, compactness analysis MUST run in **evaluation mode**, collecting **encoder mean latents (μ)** from **validation batches only** (before reparameterized sampling), not from training clips, a single training minibatch, or sampled latents.
- **FR-008**: Compactness rank selection MUST use **cumulative singular-value ratio** (linear sum of singular values normalized to 1) to map user fidelity percent to kept component count, matching reference `r_f` semantics.
- **FR-009**: Compactness outputs (mean vector, principal basis, cumulative singular-value ratios) MUST be embedded in successful reconstruction Gold artifacts and MUST be copied onto the live variational bottleneck when the user Unfreezes, preserving the current fidelity setting.
- **FR-010**: Fidelity control behavior on Gold and live bottleneck after Unfreeze MUST match spec 005 intent: keep leading informative components, fill dropped dimensions with prior noise, fixed latent port width. Fidelity MUST remain **inactive** (full latent width, **“compactness not ready”** status) on hear-while-training checkpoints and live models until PCA completes at representation-stage end.
- **FR-011**: Insertable original and latest-continuous RAVE layouts MUST default the variational bottleneck to reference parity settings (kernel 5, grouped head on compatible encoder widths) unless the user edits them.
- **FR-012**: Training, freeze, and live inference MUST remain **causal** and MUST NOT introduce audio-thread allocations (constitution compliance).
- **FR-013**: The plugin MUST NOT preserve legacy pre-parity variational bottleneck geometry (1×1 full-width mean/logvar heads) or load pre-parity Gold RAVE artifacts; breaking change is acceptable. Reference-parity bottleneck geometry and compactness rules apply to all new trains and graph inserts after this feature ships.

### Key Entities

- **Variational Bottleneck**: Latent head with configurable kernel size, grouped input projections, mean and variance branches, training sampling versus eval mean-only behavior, fidelity percent, compactness readiness flag.
- **Compactness Basis**: Mean latent vector, principal component matrix, cumulative singular-value ratios derived from validation-pass mean latents in eval mode.
- **Validation Subset**: ~2% of reconstruction corpus clips (seed 42, max 1000 segments) held out from stage-1 training; sole source of mean latents for compactness PCA at representation-stage end.
- **Validation Pass**: Eval-mode encode of all validation-subset segments to collect μ (distinct from stochastic training minibatches).
- **Fidelity Control**: User-facing 0–100% compactness amount mapping to kept rank via cumulative singular-value thresholding.
- **RAVE Gold Artifact**: Trained frozen model carrying compactness buffers for encode/forward/decode fidelity application.
- **Live Bottleneck After Unfreeze**: Modular bottleneck restored with trained weights and compactness buffers copied from the Gold artifact.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In guided tests, eval encode outputs from the live bottleneck and Gold model show zero injected sampling noise (deterministic given input) in 100% of trials, while training steps show stochastic sampling in 100% of observed training forwards.
- **SC-002**: Default-inserted RAVE layouts expose bottleneck kernel size defaulting to 5; changing kernel size propagates to prepared live and training graphs without silent revert in 100% of property-edit tests.
- **SC-003**: Illegal grouped-head channel configurations are rejected at wire or arm time with a readable reason in 100% of intentionally invalid wiring attempts.
- **SC-004**: After a successful reconstruction train with sufficient validation audio, compactness buffers are present on Gold auto-load and on the live bottleneck immediately after Unfreeze in 100% of completion tests.
- **SC-005**: Sweeping fidelity from 100% to 50% on the same clip produces audibly coarser reconstruction within one control gesture and monotonic rank reduction (higher fidelity ≥ kept dimensions than lower fidelity) in guided listening tests.
- **SC-006**: When compactness analysis fails due to insufficient data, training still completes and the UI reports compactness unavailable without blocking Gold load, in 100% of low-data tests.
- **SC-007**: Short parity smoke runs on matched layouts show eval mean-latent statistics (mean/variance per dimension) within acceptable tolerance of reference RAVE on the same audio when architecture defaults are used (practitioner sign-off criterion, not automated bit-exact match).

## Assumptions

- Spec **005-rave-architecture-training** is implemented: RAVE elements, reconstruction train, Gold auto-load, fidelity control shell, and Unfreeze already exist; this feature **refines bottleneck parity** without replacing the unified Train workflow.
- **Reference implementation** means the official acids-ircam RAVE continuous variational bottleneck: softplus variance, k=5 grouped head on typical encoder widths (e.g., 1024→256 with two groups), eval mean-only encode, PCA on μ with singular-value cumulative fidelity.
- Default grouped-head layout: **2 groups fixed** (group 1 = mean, group 2 = variance); encoder feature channels MUST be divisible by 2. Only **kernel size** is user-configurable among head geometry properties (default 5).
- Compactness validation split matches acids-rave: **~98% train / ~2% val**, seed **42**, validation capped at **1000** segments (`split_dataset.max_residual` equivalent). Stage-1 trains on train split only; PCA runs once at representation-stage end on validation μ only.
- Parity targets **behavioral equivalence** of rules and defaults, not byte-identical weights across frameworks without shared initialization.
- **No backward compatibility** with pre-parity bottleneck implementations or Gold artifacts; breaking change is acceptable (no production-trained models yet).
- Constitution constraints hold: plugin-only UI, background worker for train/compactness, no audio-thread allocations, Blue live versus Gold frozen.
