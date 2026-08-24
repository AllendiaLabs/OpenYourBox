# Feature Specification: RAVE Architecture & Training

**Feature Branch**: `005-rave-architecture-training`

**Created**: 2026-08-25

- **Status**: Clarified (ready for planning)

**Input**: User description: "features to allow users to create rave architecture and train them in openyourbox"

## Clarifications

### Session 2026-08-25

- Q: When a reconstruction train finishes stage 1 (representation) and then runs the quality stage, at what point should the plugin treat the job as successful and auto-load the Gold model? → A: Run a default quality stage of **1,000,000** steps after representation, then auto-load. Stop during either stage must not auto-load.
- Q: After a successful reconstruction train, what should the Gold RAVE BlackBox expose to the user for live playing? → A: Gold supports **forward** (encode→decode) plus **encode** and **decode** (latent in/out available) so users can tap or drive latents without Unfreeze.
- Q: How should RAVE layouts and reconstruction training handle channel count relative to the plugin’s usual stereo audio path? → A: User picks **mono or stereo** when inserting a layout; Train requires corpus channel count to match the graph.
- Q: During a long reconstruction train, should the plugin let the user hear intermediate models before the job finishes? → A: Periodic **hear-while-training checkpoints**; user may **optionally** load a checkpoint into the live path while training continues. Final success auto-load only after both stages complete. Stop still does not count as success auto-load.
- Q: Where should the fidelity / compactness control live after a RAVE model is trained and auto-loaded as Gold? → A: Control **always applies** to the active RAVE model (Gold after auto-load and live bottleneck after Unfreeze), same pattern as Knob/XY conditioning into a frozen Gold TCN via FiLM.
- Q: Should RAVE training be a separate workflow from steerable training, or unified? → A: **Unification is the objective** — match the steerable training workflow as much as possible; not two distinct product workflows.
- Q: In a unified Train flow, how should the plugin decide whether to run the steerable mapping recipe or the RAVE reconstruction recipe? → A: User picks **objective** in the **same** Train panel (still one UI); mapping vs reconstruction is an explicit choice, not a separate Train product.
- Q: How should unpaired (single-instance) capture fit into the existing Capture Samples flow so it stays one workflow? → A: Same Capture Samples menu; user chooses capture kind **Pair** (two instances, x/y) or **Single** (this instance, unpaired library audio).
- Q: When Train objective is reconstruction, how should the shared Library decide which selected entries are eligible for Run? → A: Library entries have **tags**; UI **warns and filters when necessary**. Reconstruction uses **both x and y** of selected pairs as training audio. Mapping **errors** if unpaired entries are selected.
- Q: When the user opens the unified Train panel, what should the default Train objective be? → A: Remember the **last-used** objective **per plugin instance**.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Assemble a RAVE Graph From Live Elements (Priority: P1)

A sound designer wants to build a Realtime Audio Variational AutoEncoder (RAVE) inside the plugin, not in a separate tool. They add graph elements that together form an encoder–latent–decoder path: split the incoming sound into frequency bands, squeeze it into a compact latent representation, then expand it back to audio. They can hear the untrained (random-weight) graph immediately, inspect shapes while wiring, and freeze a selection later if they wish.

They reuse existing building blocks where those already exist (temporal convolution stacks, activations, merge, linear projections). They gain new blocks that RAVE requires and that the current palette cannot express: multiband split/join, rate-changing (down/up) convolution, a variational latent bottleneck, and an optional filtered-noise synthesizer for the decoder.

**Why this priority**: Without a live graph that can express RAVE, training has nothing legal to optimize, and users cannot explore RAVE-shaped networks in the same modular way they already explore TCN effects.

**Independent Test**: From an empty processing graph, assemble an encoder–decoder with multiband split/join, a latent bottleneck, and matching up/down rate changes; play audio through it; confirm illegal band/latent/audio connections are refused; confirm existing TCN effect graphs still assemble as before.

**Acceptance Scenarios**:

1. **Given** the graph palette, **When** the user looks for RAVE-related elements, **Then** they can add multiband analysis, multiband synthesis, rate-changing convolution (downsampling and upsampling), a variational latent bottleneck, and an optional noise synthesizer without leaving the plugin
2. **Given** those elements plus existing temporal, activation, merge, and projection elements, **When** the user wires an encoder–latent–decoder path, **Then** the graph is accepted when band counts, latent width, and audio/latent domains match
3. **Given** an audio-domain output connected to a latent-domain input (or mismatched band counts), **When** the user attempts the connection, **Then** the cable is refused and a tooltip explains the mismatch
4. **Given** a legally wired RAVE-shaped graph with random weights, **When** audio plays, **Then** live processing continues without dropouts attributable to the new element types
5. **Given** an existing effect-style graph (no autoencoder), **When** the user ignores the new RAVE elements, **Then** prior palette behavior and training-as-mapping remain available
6. **Given** a RAVE-shaped live graph, **When** the user Freezes a selection, **Then** Freeze still produces a Gold BlackBox using the same freeze workflow as other graphs

---

### User Story 2 - Use the Same Library and Capture Shell for Reconstruction Audio (Priority: P1)

RAVE learns by reconstructing a body of sound, not by cloning an effect from clean/processed pairs. The user stays in the **same Training Library and Capture Samples workflow** as steerable training. Capture Samples offers **Pair** or **Single** kind. **Single** (or file import) adds unpaired recordings alongside existing pairs. Library entries carry **tags** (at least distinguishing pair vs unpaired/clip). The Library **warns and filters when necessary** based on Train objective. Before Train, the user multi-selects entries as today. For **reconstruction**, selected **pairs contribute both x and y** as training audio. For **mapping**, selecting unpaired entries is an **error**. Copyright acknowledgment remains the same gate.

**Why this priority**: Unification requires extending the existing library/capture shell, not a second corpus product.

**Independent Test**: Import single files and/or Single-capture into the existing Library; confirm tags; set objective reconstruction and select pairs (x and y both used) and/or unpaired clips; set objective mapping with unpaired selected and confirm Run errors; confirm copyright gate unchanged.

**Acceptance Scenarios**:

1. **Given** the existing Training Library, **When** the user imports one or more single audio files (no paired wet file required), **Then** each file appears in the **same** library UI as an unpaired/clip entry with appropriate **tags**
2. **Given** the Capture Samples menu, **When** the user chooses capture kind **Single**, **Then** recording from this instance alone adds unpaired audio to the same library (no peer required) with appropriate tags
3. **Given** the Capture Samples menu, **When** the user chooses capture kind **Pair**, **Then** the existing dual-instance Clean/Processed pairing flow is used to add an x/y pair with appropriate tags
4. **Given** Train objective is **reconstruction** and one or more **pairs** are selected, **When** Run starts, **Then** **both x and y** of each selected pair are used as reconstruction training audio
5. **Given** Train objective is **reconstruction** and unpaired/clip entries are selected, **When** Run starts, **Then** those clips are included in the reconstruction corpus
6. **Given** Train objective is **mapping** and any **unpaired** entry is selected, **When** the user attempts Run, **Then** Train shows an **error** and does not start
7. **Given** the Library with mixed pair and unpaired entries, **When** the user changes Train objective, **Then** the UI **warns and filters when necessary** so ineligible combinations are visible before Run
8. **Given** selected library audio, **When** sample rates differ across the selection, **Then** Train is blocked with a clear message (same mixed-rate rule as today)
9. **Given** no copyright acknowledgment, **When** the user views Train, **Then** Train stays disabled until the existing copyright modal is acknowledged (same gate for any objective)
10. **Given** selected reconstruction audio shorter than a usable train window after context, **When** the user presses Run, **Then** Train is refused with a clear reason
11. **Given** the Library and Capture panels, **When** the user switches between pairs and unpaired clips, **Then** they do not open a separate RAVE-only capture or library application or modal workflow

---

### User Story 3 - Train From the Same Panel With an Explicit Objective (Priority: P1)

With an armed graph, selected library audio, and copyright acknowledgment, the user starts Train from the **same** master Train panel used for steerable discovery. They set **objective** to **mapping** or **reconstruction** in that panel. Mapping keeps the existing steerable recipe. Reconstruction uses the two-stage RAVE recipe (representation then quality) but the same Run / Pause / Stop, live loss, hear-while-training checkpoints, non-blocking audio, arm/disarm snapshot, and success Gold auto-load shell. Stage readout appears as progress detail inside that panel when objective is reconstruction—not a second Train UI.

**Why this priority**: Unification means one Train UX; recipes differ under the hood by objective, not by product mode.

**Independent Test**: Open the existing Train panel; set objective reconstruction; arm a RAVE-shaped graph; select library audio; Run; confirm same controls as mapping; confirm stage 1→2 status; Pause/Resume/Stop; optional checkpoint load. Full success proceeds to Story 4.

**Acceptance Scenarios**:

1. **Given** copyright acknowledgment, an armed graph, and selected library audio, **When** the user opens Train, **Then** they see one panel with Run/Pause/Stop, loss/progress, and an **objective** control (mapping | reconstruction) defaulting to that instance’s **last-used** objective (or mapping if none yet)—not a separate RAVE Train entry point
2. **Given** the user sets objective to reconstruction then closes and reopens Train on the same instance, **When** they view the objective control, **Then** it shows **reconstruction** (last-used remembered per instance)
3. **Given** objective is reconstruction, a RAVE-shaped armed graph, and valid selected audio, **When** the user presses Run, **Then** training starts in the background and the plugin UI stays responsive
4. **Given** training is running, **When** the DAW continues playback, **Then** audio uses the previously loaded model (unless the user optionally loads a checkpoint) with no training-induced dropouts on the audio path
5. **Given** stage 1 (representation), **When** the user views the Train panel, **Then** they see that representation learning is active and live loss/step update within the same panel chrome as mapping Train
6. **Given** stage 1 has reached its configured duration, **When** stage 2 begins, **Then** the encoder is held fixed and the decoder is trained with an adversarial quality objective plus continued spectral (and feature-matching) terms; the same panel shows the quality stage
7. **Given** stage 2 reaches its default duration of **1,000,000** steps without Stop, **When** the job completes, **Then** the run is treated as successful and Gold auto-load proceeds (Story 4)
8. **Given** training is running, **When** the user Pause/Resume/Stop, **Then** behavior matches mapping Train controls; Stop during either stage never counts as success auto-load
9. **Given** training is running, **When** a hear-while-training checkpoint is available, **Then** the user may optionally load that checkpoint into the live path without ending the job (same hear-while-training pattern as mapping Train)
10. **Given** objective is mapping and an effect-style armed graph, **When** the user Runs Train, **Then** the steerable mapping recipe is unchanged
11. **Given** objective is reconstruction and the armed graph is not a valid RAVE autoencoder, **When** the user presses Run, **Then** Run is refused with a reason (missing bottleneck, missing decoder, domain mismatch)
12. **Given** objective is reconstruction and the job proceeds without interrupt, **When** both stages complete, **Then** it used the prescribed two-stage durations (each default **1,000,000** steps), dual spectral distances, variational regularization with warmup, and an adversarial quality stage off the audio thread
13. **Given** objective is mapping on a RAVE autoencoder with selected pairs, **When** the user Runs Train, **Then** mapping remains allowed and the UI makes the chosen objective explicit so recipes are not confused

---

### User Story 4 - Auto-Load a Trained RAVE as Gold and Perform Timbre Transfer (Priority: P1)

When reconstruction training succeeds, the armed encoder–decoder chain becomes a Gold BlackBox. That Gold node exposes **forward** (audio in → reconstructed audio out), plus **encode** (audio in → latent out) and **decode** (latent in → audio out) so the user can tap or drive the latent path without Unfreeze. Knob/XY sources that were not part of the armed chain remain Blue. Unfreeze restores the modular graph **with trained weights**. The user can play material the model never saw and hear it rendered in the learned timbre.

**Why this priority**: Live reconstruction after train is the user-visible payoff; without auto-load, RAVE training is incomplete relative to existing mapping Train.

**Independent Test**: Complete a successful reconstruction train; confirm armed RAVE chain becomes Gold; pass new audio through and hear reconstructed output; Unfreeze and confirm weights remain.

**Acceptance Scenarios**:

1. **Given** reconstruction train succeeds, **When** the result is ready, **Then** the armed RAVE processing chain is replaced by a Gold BlackBox without a separate Freeze action
2. **Given** the Gold RAVE BlackBox is loaded, **When** audio plays through **forward**, **Then** the plugin outputs the model’s reconstruction of the input (timbre transfer when the input is out of domain)
3. **Given** the Gold RAVE BlackBox, **When** the user uses **encode**, **Then** audio is mapped to a latent-domain output without requiring Unfreeze
4. **Given** the Gold RAVE BlackBox, **When** the user uses **decode**, **Then** a latent-domain input is mapped to audio without requiring Unfreeze
5. **Given** control sources existed outside the armed chain, **When** auto-load completes, **Then** they remain Blue and unabsorbed
6. **Given** a Gold RAVE BlackBox, **When** the user Unfreezes, **Then** the modular encoder–latent–decoder graph returns with trained weights until randomize or retrain
7. **Given** auto-load fails after an artifact exists, **When** the user is notified, **Then** the prior model stays active and a retry-load is offered without re-running the full train when possible

---

### User Story 5 - Control Latent Compactness and Inspect the Latent Path (Priority: P2)

After representation learning, RAVE can drop uninformative latent dimensions while keeping reconstruction quality (the paper’s **fidelity** trade-off). The user sets a compactness/fidelity amount that **always applies** to the active RAVE model—on the Gold BlackBox after auto-load and on the live bottleneck after Unfreeze—the same way Knob/XY conditioning continues to steer a frozen Gold TCN. Advanced users can also treat the latent stream as a first-class graph domain: insert allowed processing between encoder and decoder, or tap latent channels for analysis views.

**Why this priority**: Compactness is a defining RAVE control, but a user can already train and hear reconstruction without it; latent tapping is the creative “high-level manipulation” workflow.

**Independent Test**: After a reconstruction train (or on a graph that exposes a bottleneck), sweep fidelity from high to low and confirm fewer effective latent dimensions and a coarser reconstruction; confirm analysis views still plot all latent channels on shared axes.

**Acceptance Scenarios**:

1. **Given** a Gold RAVE BlackBox or a live variational bottleneck after Unfreeze, **When** the user sets fidelity/compactness, **Then** the control always applies to the active model (decoder/forward driven by the reduced informative subset, uninformative dimensions replaced as specified by the compactness method) without restarting the DAW and without requiring Unfreeze to change fidelity on Gold
2. **Given** fidelity near maximum, **When** reconstruction is compared subjectively to a mid/low setting, **Then** higher fidelity is at least as faithful to the input as lower fidelity on the same clip
3. **Given** a latent-domain cable, **When** the user opens analysis at that point, **Then** all latent channels appear on the same plots (constitution: any feature dimension, not stereo-only)
4. **Given** the user inserts a legal processing element on the latent path, **When** audio plays, **Then** encode → process z → decode remains live and shape-checked
5. **Given** a Gold RAVE BlackBox with fidelity set, **When** the user Unfreezes, **Then** the restored bottleneck retains the same fidelity setting (control continuity, analogous to free-c after freeze)

---

### User Story 6 - Insert Original and Latest RAVE Layouts as Starting Graphs (Priority: P2)

Users who do not want to wire every block by hand can insert a **layout** that matches either the **original** RAVE architecture (paper / first official implementation) or the **latest continuous** architecture (current official improved model). When inserting, the user chooses **mono (1 channel)** or **stereo (2 channels)**; the populated graph’s audio I/O width matches that choice. Layouts are starting graphs: users may then edit, arm, freeze, or train them. Reconstruction Train requires selected corpus entries to match the armed graph’s channel count. This does not lock the product to a single version; it documents both lineages in the UI.

**Why this priority**: Speeds first success and makes “original vs latest” a user choice rather than hidden knowledge; still testable without a full multi-day train.

**Independent Test**: Insert each layout into an empty graph; confirm element types and wiring match the published original vs latest continuous structures described in Assumptions; play audio; confirm either layout can be selected for reconstruction Train (recipe still the latest continuous training procedure unless the user is only auditioning random weights).

**Acceptance Scenarios**:

1. **Given** an empty or selected insertion point, **When** the user chooses Insert original RAVE layout, **Then** they pick **mono or stereo** and the graph contains multiband split, a simple strided encoder, a variational bottleneck, and a decoder with waveform, loudness envelope, and noise branches as in the original method at that channel count
2. **Given** the same insertion affordance, **When** the user chooses Insert latest continuous RAVE layout, **Then** they pick **mono or stereo** and the graph contains multiband split, residual dilated encoder/decoder with amplitude modulation, and a variational bottleneck as in the current official continuous model at that channel count
3. **Given** either layout, **When** the user edits nodes afterward, **Then** the graph remains a normal live graph (delete, rewire, freeze, arm)
4. **Given** the latest-continuous layout armed for reconstruction Train, **When** Run is enabled, **Then** the two-stage recipe in Story 3 applies and selected corpus channel count must match the graph
5. **Given** the original layout armed for reconstruction Train, **When** Run is enabled, **Then** reconstruction training still runs in two stages (representation then quality) against that graph; the product does not require a separate legacy trainer; corpus channels must match the graph
6. **Given** a mono RAVE graph on a stereo host, **When** the user has not otherwise wired channel adaptation, **Then** shape checking refuses illegal channel mismatches (no silent default downmix/upmix around the model unless the user adds explicit adaptation elements)

---

### Edge Cases

- What happens if reconstruction Train is started on a mapping-style graph with no bottleneck? Run is refused; mapping Train remains available for that graph.
- What happens if mapping Train is started on a RAVE autoencoder with x/y pairs? Mapping Train remains allowed (learn a transform of x toward y through the whole graph) but is not the RAVE reconstruction recipe; the UI must make the objective explicit so users do not confuse the two.
- What happens if only one of encoder or decoder is armed? Reconstruction Train is refused until the bottleneck path required for autoencoding is armed.
- What happens if the user stops during stage 2? No auto-load; prior model stays; partial stage-2 weights are not silently swapped.
- What happens if compactness analysis cannot run (too little validation audio)? Train can still finish; fidelity control falls back to using the full latent width with a clear status.
- What happens if captured reconstruction audio is silence-only? Train may run but the UI warns that the corpus looks empty/silent before Run when detectable.
- What happens if band count of analysis and synthesis differ? Connection or Train is refused with a mismatch message.
- What happens if live graph uses a non-causal (lookahead) rate-changing convolution? Live audio path refuses or forces causal mode so the plugin stays real-time; training of RAVE graphs uses the same causal constraint so weights match live.
- What happens if the user randomizes a trained RAVE Gold after Unfreeze? Trained weights are replaced; fidelity metadata tied to those weights is cleared or recomputed only after a new representation stage.
- What happens if selected reconstruction corpus channel count does not match the armed RAVE graph? Train is blocked with a clear message (same class of gate as mixed sample-rate).
- What happens if mapping Train has unpaired library entries selected? Run is refused with an **error** (unpaired not valid for mapping).
- What happens if reconstruction Train has pairs selected? **Both x and y** of each selected pair are used as reconstruction audio (no side-picker required).
- What happens if the user optionally loads a hear-while-training checkpoint then Stops? The job ends without success auto-load; the live path may retain the last optionally loaded checkpoint or revert per explicit UI choice, but Stop is never treated as successful completion of both stages.
- What happens to dual-instance Capture Samples? Remains the **Pair** kind inside the **same** Capture shell; **Single** kind records unpaired audio without a slave.

## Out of Scope

- Discrete (codebook) RAVE, Wasserstein or spherical bottlenecks, and other alternate regularizers beyond variational reconstruction
- Latest experimental extras: Snake activations, Adaptive Instance Normalization style transfer, mel-spectrogram hybrid encoders, recurrent latent layers, and third-party discriminator variants not required by the default continuous recipe
- Separate latent **prior** models (unconditional latent generation)
- Non-causal / offline-only RAVE (lookahead models that cannot run as a real-time plugin insert)
- Export or hosting workflows aimed at other environments (dedicated RAVE-only editors, Max/MSP-style externals as a product surface)
- Cloud training, marketplace, and remote corpora
- Changing the steerable mapping recipe itself, dual-instance Capture pairing for mapping, or the copyright modal—except to extend the **same** shells for unpaired audio and a reconstruction objective
- A second Train / Library / Capture product surface for RAVE
- User-authored discriminator graphs in the live palette (the quality stage is training machinery, not a live sound element)
- Guaranteeing paper-quality models in a short default step count; reconstruction trains are long-running compared with mapping trains, and the user may Stop early

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The plugin MUST provide live graph elements sufficient to assemble both the **original** RAVE architecture and the **latest continuous** RAVE architecture: multiband analysis, multiband synthesis, rate-changing convolution (downsample and upsample), variational latent bottleneck, optional noise synthesizer, and reuse of existing temporal residual/activation/merge/projection elements.
- **FR-002**: Multiband analysis MUST split audio into a configurable number of sub-bands (default **16**); multiband synthesis MUST invert a matching bank so that a bypassed analysis→synthesis path reconstructs the input within a small audible error on a legal graph.
- **FR-003**: Rate-changing convolutions MUST support integer downsampling and upsampling factors used by RAVE layouts (including the original stride sequence **4, 4, 4, 2** and the same sequence on the latest continuous layout), with live **causal** operation (no future samples).
- **FR-004**: A variational bottleneck MUST map encoder features to a latent distribution, produce a latent trajectory for the decoder, and during reconstruction training apply a variational regularizer toward a standard prior; after the representation stage the encoder MUST be frozen for the quality stage.
- **FR-005**: The optional noise synthesizer MUST add a learned filtered-noise component to the decoder waveform path; original-layout graphs MUST be able to combine waveform, loudness envelope, and noise as in the original method; latest-continuous graphs MUST be able to use amplitude modulation of the waveform (and optional noise) as in the current official continuous model.
- **FR-006**: Shape checking MUST distinguish **audio**, **multiband**, and **latent** domains (or equivalent explicit dimension rules) so illegal mix of domains is refused with a tooltip.
- **FR-007**: Existing mapping Train (clean→processed, steerable spectral recipe, ~2500 steps) MUST remain available for non-autoencoder armed graphs and MUST NOT be replaced by the RAVE recipe.
- **FR-008**: There MUST be a **single** Train panel (the steerable Train shell). Users MUST choose Train **objective** inside that panel: **mapping** or **reconstruction**. The panel MUST restore each plugin instance’s **last-used** objective (default **mapping** if none stored yet). There MUST NOT be a separate RAVE-only Train entry point, panel, or wizard. Reconstruction MUST require a valid armed autoencoder path and selected reconstruction-capable library audio.
- **FR-008a**: Library, Capture Samples, copyright acknowledgment, arm/disarm, Run/Pause/Stop, live loss, hear-while-training checkpoints, success Gold auto-load, and Unfreeze MUST be the **same workflows** as Phase 3 steerable training, extended only as needed for unpaired audio and reconstruction recipe details.
- **FR-008b**: Capture Samples MUST offer a capture **kind** in the same menu: **Pair** (two-instance Clean/Processed x/y, existing behavior) or **Single** (record this instance only into the library as unpaired audio). Single MUST NOT require a slave peer.
- **FR-009**: The Training Library MUST be a **single** list that holds **pair** and **unpaired/clip** entries, each with **tags** (at least system tags distinguishing pair vs unpaired; user tags MAY be supported). The Library/Train UI MUST **warn and filter when necessary** based on the current Train objective.
- **FR-009a**: When objective is **reconstruction**, selected **pairs** MUST contribute **both x and y** as reconstruction training audio; selected unpaired/clip entries MUST be included as corpus clips.
- **FR-009b**: When objective is **mapping**, selecting any **unpaired** entry MUST produce an **error** and MUST block Run until the selection is valid (≥1 pair and no unpaired entries).
- **FR-009c**: Unpaired entries MUST be creatable via file import and via Capture kind **Single** in the same Library/Capture UI.
- **FR-010**: Reconstruction Train MUST reuse the existing copyright acknowledgment gate; it MUST NOT block the audio thread; Run/Pause/Stop and live loss MUST remain; audio MUST keep the previously loaded model until the user optionally loads a hear-while-training checkpoint or until successful auto-load after both stages complete.
- **FR-010a**: Reconstruction Train MUST export periodic hear-while-training checkpoints; the user MUST be able to optionally load a checkpoint into the live path while the job continues. Optional mid-run loads MUST NOT end the job. Final success auto-load MUST occur only after both default stages complete without Stop. Stop MUST NOT be treated as successful completion.
- **FR-011**: Reconstruction Train MUST follow a **two-stage** procedure: (1) representation — spectral reconstruction distance plus variational regularizer with a warmup of regularizer strength; (2) quality — encoder frozen, decoder trained with hinge-style adversarial loss, feature matching, and continued spectral distance. Default stage-1 duration MUST be **1,000,000** optimization steps and default stage-2 duration MUST be **1,000,000** optimization steps unless the user Stops earlier. Successful completion of both default stages MUST trigger Gold auto-load; Stop in either stage MUST NOT auto-load. Dual spectral distance MUST score **full-band** audio and **multiband** signals when a multiband path exists. Spectral scales MUST match the official continuous defaults (**windows 2048, 1024, 512, 256, 128**).
- **FR-012**: Reconstruction Train MUST keep discriminator/quality-stage auxiliary networks off the real-time audio path; they exist only inside the background training job.
- **FR-013**: On successful reconstruction Train, the master MUST auto-load the armed RAVE chain as a Gold BlackBox that exposes **forward** (audio→audio), **encode** (audio→latent), and **decode** (latent→audio) without requiring Unfreeze; Unfreeze MUST restore the modular graph with trained weights until randomize or retrain.
- **FR-014**: After representation learning (and on loaded trained models), users MUST be able to set a **fidelity / compactness** control (0–100%) that **always applies** to the active RAVE model: on the Gold BlackBox after auto-load (affecting forward/decode) and on the live bottleneck after Unfreeze, without requiring Unfreeze to change fidelity on Gold. Behavior MUST follow the same live-control-through-Gold pattern as Knob/XY conditioning into a frozen Gold TCN. The control MUST reduce the latent to the informative subset identified by post-stage compactness analysis, replacing dropped dimensions with prior noise as in the RAVE compactness method. Unfreeze MUST preserve the current fidelity setting on the restored bottleneck.
- **FR-015**: The plugin MUST offer insertable **layouts** labeled as original RAVE and latest continuous RAVE; at insert time the user MUST choose **mono (1)** or **stereo (2)** channel width for the layout’s audio I/O; the populated graph MUST match that choice.
- **FR-015a**: Reconstruction Train MUST require every selected corpus entry’s channel count to match the armed RAVE graph’s audio channel width; mismatches MUST block Run with a clear message. Mixed sample-rate selections remain blocked as for mapping Train.
- **FR-016**: Live RAVE graphs and reconstruction training MUST use **causal** temporal operators so trained weights are valid for real-time plugin inserts.
- **FR-017**: UI MUST remain usable under reconstruction-train load (constitution: interface stays fluid while the worker is busy); freeze of RAVE subgraphs MUST obey existing freeze/unfreeze policy.
- **FR-018**: Reconstruction Train MUST report stage, step, and loss at least as often as mapping Train; if compactness analysis succeeds, the UI MAY show estimated informative latent width at high-fidelity checkpoints.

### Key Entities

- **Plugin Instance Pairing**: Master/slave link used when Capture kind is **Pair**; not required for **Single** capture.
- **RAVE Graph**: Armed encoder–latent–decoder processing chain, optionally wrapped in matching multiband analysis/synthesis, assembled from live elements.
- **Multiband Bank**: Paired analysis/synthesis filter bank with a band count and invertibility requirement for bypass.
- **Variational Bottleneck**: Latent distribution, sampled latent trajectory, regularizer weight, freeze-after-representation flag, compactness basis (informative directions and mean).
- **Reconstruction Corpus Entry**: Unpaired audio clip in the Training Library (import or **Single** capture), with duration, sample rate, channels, source, and **tags**; shown in the same library as pairs.
- **Library Entry Tags**: At least system tags for pair vs unpaired/clip; used with objective-based warnings and filters; optional user tags.
- **Train Objective**: Mapping (pair x→y) versus Reconstruction (audio→audio through the autoencoder); chosen in the single shared Train panel; last-used value remembered per plugin instance (mapping if none yet).
- **Unified Train Shell**: Shared Phase 3 Train / Library / Capture / arm / Gold / Unfreeze workflow; recipes and eligible library selection differ by objective only; objective last-used per instance.
- **Reconstruction Job**: Snapshot of armed RAVE graph, selected corpus, two-stage state, losses (spectral, regularizer, adversarial, feature matching), compactness result, hear-while-training checkpoint paths.
- **Fidelity Control**: User-facing compactness amount (0–100%) that always applies to the active RAVE model (Gold or live bottleneck); preserved across Unfreeze like free-c conditioning on Gold TCN.
- **RAVE Layout**: Named starting graph for original versus latest continuous architectures; insert includes an explicit mono or stereo channel-width choice.
- **Steerable Gold BlackBox (RAVE)**: Trained frozen RAVE node replacing the armed chain; exposes forward, encode, and decode; Weights property shows the trained artifact path; fidelity control may apply on latent used by forward/decode.
- **Quality-Stage Discriminator**: Train-only adversary used in stage 2; not a live palette element.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In a guided test, a new user can insert a latest-continuous RAVE layout (or build the equivalent from the palette) and pass audio through it in under 10 minutes without leaving the plugin.
- **SC-002**: Illegal audio/band/latent connections are rejected on the first attempt in guided wiring tests, with a readable reason, in 100% of mismatched cases tried.
- **SC-003**: A user can import unpaired files (or Single-capture) in the **same** Library/Capture shell, see tags, select pairs and/or unpaired clips, set Train objective to reconstruction (pairs contribute x and y), and enable Run after copyright acknowledgment in under 5 minutes; mapping with unpaired selected shows an error and does not start.
- **SC-011**: Changing Train objective updates Library warnings/filters so ineligible selections are visible before Run in guided tests; reopening Train on the same instance restores the last-used objective.
- **SC-004**: During reconstruction Train, continuous playback shows no training-induced audio dropouts; the audible model remains the pre-train model until an optional checkpoint load or success auto-load, in at least 95% of interrupt/load tests.
- **SC-005**: Pause, Resume, and Stop during reconstruction Train behave like mapping Train; Stop never replaces the active model in at least 95% of stop tests.
- **SC-006**: After a successful reconstruction train, the armed chain becomes Gold with forward, encode, and decode available; new input through forward is audibly reconstructed (including out-of-domain material as timbre transfer) without a separate freeze step; encode/decode work without Unfreeze in guided tests.
- **SC-007**: Users can switch fidelity from high to low on a Gold RAVE BlackBox (without Unfreeze) in one control gesture and hear a coarser reconstruction within one second; after Unfreeze the same setting remains active on the bottleneck in guided tests.
- **SC-008**: Mapping Train on a non-RAVE graph still completes its existing short-run recipe; reconstruction Train is not required to finish in that same short duration.
- **SC-009**: Original and latest-continuous insertable layouts are distinguishable in the graph (different encoder/decoder structure as specified) and both play audio on the first insert in guided tests.
- **SC-010**: Unfreeze after RAVE auto-load restores an editable graph that still produces the trained reconstruction until the user randomizes or retrains.

## Assumptions

- Phase 3 mapping Train, Training Library, copyright modal, arm/disarm, Gold auto-load, Freeze/Unfreeze, and hear-while-training are the baseline; RAVE extends that **unified** shell rather than introducing a parallel training product.
- Unification means one Train panel with an explicit **objective** (mapping | reconstruction), **last-used remembered per plugin instance** (mapping if none yet), one Library with tags/warn/filter, one Capture Samples shell (Pair | Single), same arm/auto-load/Unfreeze patterns; under-the-hood recipes and data shapes may differ by objective.
- **Latest continuous RAVE** means the current official improved continuous model (residual dilated encoder/decoder, amplitude modulation, combined multi-period plus multi-scale quality stage, variational bottleneck, dual spectral distance, causal option). That training procedure is the default reconstruction recipe.
- **Original RAVE** means the paper / first official architecture (simple strided encoder, decoder with waveform × loudness + noise, 16-band split, 128-wide latent, strides 4-4-4-2). Users can build or insert it; they do not get a second incompatible trainer.
- Discrete, Wasserstein, spherical, Snake, Adaptive Instance Normalization, mel-hybrid, and latent-prior variants are deferred.
- Live plugin use requires **causal** models; non-causal RAVE is out of scope even though the official tool allows it for offline work.
- Reconstruction trains are **long**: default **1,000,000** representation steps then **1,000,000** quality steps before success auto-load. Users may Stop early for smoke tests; quality comparable to published demos is not promised for short runs. Periodic hear-while-training checkpoints allow optional mid-run audition without ending the job.
- Discriminators and other quality-stage networks are not user-wired live nodes.
- Default multiband count is 16; default latent width is 128; users may change these within shape-legal limits. Insertable layouts ask for **mono or stereo** at insert time; reconstruction Train requires corpus channels to match the graph (no silent downmix/upmix of the train set).
- Compactness/fidelity uses the RAVE post-representation analysis of informative latent directions (singular-value / principal-component family described in the paper). The fidelity control always applies to the active model (Gold or Unfrozen bottleneck), analogous to Knob/XY → FiLM on a frozen Gold TCN.
- Reconstruction capture does not require master/slave pairing when capture kind is **Single**; **Pair** kind retains existing master/slave pairing for x/y.
- Existing pair library entries are not deleted or migrated; unpaired clips are an additional entry kind with **tags** in the same library. Reconstruction uses both sides of selected pairs; mapping rejects unpaired selections with an error.
- Constitution constraints hold: plugin-only UI, background worker, no audio-thread allocations, Blue live vs Gold frozen, copyright before first Train (already stored acknowledgment counts).
- Auto-load of a RAVE Gold BlackBox exposes **forward**, **encode**, and **decode** without Unfreeze; Unfreeze still restores the modular graph with trained weights.
