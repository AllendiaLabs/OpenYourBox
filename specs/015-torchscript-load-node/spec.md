# Feature Specification: TorchScript Checkpoint Loader Node

**Feature Branch**: `015-torchscript-load-node`

**Created**: 2026-08-30

**Status**: Draft

**Input**: User description: "implement a new factory node that loads torchscript checkpoint and runs it, to load eg pretrained rave models or else"

## Clarifications

### Session 2026-08-30

- Q: For checkpoints that expose multiple entry points (for example RAVE-style encode, decode, and full forward), what should this Factory element offer in v1? → A: Auto-detect: forward always; if the checkpoint has encode/decode, expose latent encode/decode pins like trained RAVE Gold boxes
- Q: How should the element learn the input and output channel (or feature) counts used for shape checking after a checkpoint loads? → A: Auto-infer on load and pre-fill editable override fields; overrides win for shape checking until cleared/reset
- Q: Should a loaded checkpoint expose a Control (conditioning) input pin like some existing Gold black-box and TCN elements, or only the audio / latent ports required by forward and encode/decode? → A: Expose Control only when the checkpoint advertises conditioning; otherwise omit it
- Q: When a loaded checkpoint exposes encode/decode (RAVE-style), should this element also offer the same live fidelity control that trained RAVE Gold boxes use, or omit fidelity for externally loaded models in v1? → A: Same fidelity control as trained RAVE Gold boxes when encode/decode is present (disabled/hidden if the checkpoint lacks the needed compactness path)
- Q: While the element has no successfully loaded model (empty path, load error, or still loading with nothing previous), what should its audio output do? → A: Silence when errored; dry passthrough only when path is empty and never loaded

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Load and Hear an External Checkpoint (Priority: P1)

A sound designer opens the Factory palette, places a **TorchScript Load** (or equivalently named) element on the canvas, chooses a pretrained checkpoint file from disk (for example a published RAVE model or any other exported TorchScript artifact), connects it between Audio Input and Audio Output, and hears the model process live audio without leaving the plugin or running an external tool.

**Why this priority**: External pretrained models are unusable in-graph today without rebuilding them as modular Blue nodes or going through freeze/train. Loading and running a checkpoint is the minimum valuable product.

**Independent Test**: Place Audio In → TorchScript Load → Audio Out, select a known-good checkpoint, play audio, confirm processed output and uninterrupted transport.

**Acceptance Scenarios**:

1. **Given** the graph editor is open, **When** the user opens the Factory palette, **Then** a TorchScript Load element is available to place
2. **Given** a TorchScript Load element on the canvas with no file selected, **When** the user chooses a valid checkpoint via the element properties, **Then** the element loads that artifact and is ready to process audio
3. **Given** a successfully loaded checkpoint connected in a legal audio chain, **When** audio plays, **Then** the element’s output reflects the model’s processing and the host continues without dropouts caused by the load path after load completes
4. **Given** a checkpoint is already loaded, **When** the user selects a different valid checkpoint, **Then** the element swaps to the new model without requiring a plugin restart

---

### User Story 2 - Clear Failure and Shape Feedback (Priority: P1)

A user points the element at a missing, corrupt, or incompatible file, or connects it with channel counts that do not match the model. The editor refuses illegal connections or shows a recoverable error. On load error (or while replacing a failed load with nothing previous ready), that node outputs silence; when the path is still empty and never successfully loaded, it dry-passthroughs the main audio input. The host keeps running and the user understands what to fix.

**Why this priority**: External files fail often; opaque crashes or silent wrong shapes would make the feature unusable and unsafe for live use.

**Independent Test**: Attempt load of a missing path and an incompatible shape connection; confirm non-crashing feedback and continued host audio.

**Acceptance Scenarios**:

1. **Given** the user selects a path that does not exist or cannot be read, **When** load is attempted, **Then** the element shows a clear recoverable error, outputs silence from that node (unless a previous model remains active), does not crash the plugin, and does not interrupt the host’s audio engine
2. **Given** a file that is not a valid runnable checkpoint for this element, **When** load is attempted, **Then** the element rejects it with an explanatory message and keeps the previous successfully loaded model if one exists; if none exists, that node outputs silence
3. **Given** a TorchScript Load element with an empty path that has never successfully loaded, **When** audio plays through a legal chain, **Then** the main audio input is dry-passthrough to the output with a visible prompt to choose a file
4. **Given** a successfully loaded checkpoint, **When** the user views properties, **Then** inferred input/output (and latent, when applicable) channel fields are shown and editable as overrides
5. **Given** the user changes an override to a value that mismatches a connected cable, **When** the change is applied, **Then** the editor refuses or flags the illegal shape with explanatory feedback
6. **Given** inference fails for an otherwise loadable file, **When** the user has not yet entered valid overrides, **Then** the element does not treat connections as legal until overrides are provided
7. **Given** a loaded model with known input/output channel expectations (inferred or overridden), **When** the user tries to connect an incompatible upstream or downstream shape, **Then** the connection is refused and the mismatch is explained (same class of feedback as other graph shape violations)
8. **Given** load is in progress on a background path, **When** audio is playing, **Then** the live graph keeps using the previous safe model if one exists; if none exists and the path was empty/never loaded, dry passthrough continues until the new model swaps in atomically; if the in-progress load follows an error with no prior model, silence continues until ready

---

### User Story 3 - Persist Path Across Session and Preset (Priority: P2)

A user saves the project or a named preset with a TorchScript Load element that points at a checkpoint. On reload, the element restores the same path and reloads the model (or reports a clear missing-file error if the file moved), so workflows with pretrained RAVE or other models survive relaunch.

**Why this priority**: Without persistence, every session requires re-picking files; with it, external models become first-class graph citizens alongside freeze/train Gold boxes.

**Independent Test**: Save graph with a loaded path, reload session/preset, confirm path and successful reload (or missing-file warning if the file was removed).

**Acceptance Scenarios**:

1. **Given** a TorchScript Load element with a valid loaded path and optional channel overrides, **When** the user saves the graph or a preset and reloads it, **Then** the same path and overrides are restored and the model loads again if the file is still available
2. **Given** a saved path whose file was moved or deleted, **When** the session or preset loads, **Then** the element restores the path string, shows a recoverable missing-file error, and does not crash
3. **Given** multiple TorchScript Load elements in one graph, **When** the graph is saved and reloaded, **Then** each element restores its own path and overrides independently

---

### User Story 4 - Treat External Load as Opaque Frozen Processing (Priority: P3)

A user treats the loaded checkpoint like other optimized Gold processing: it participates in the graph as a frozen black-box (not modular Blue layers), cannot be “unfrozen” into editable Blue nodes (there is no source modular subgraph), and can sit alongside live Blue nodes and freeze-origin Gold boxes in the same graph.

**Why this priority**: Aligns with the product’s Live vs Frozen mental model and avoids promising reconstruction of third-party modular graphs that were never authored in OpenYourBox.

**Independent Test**: Place a TorchScript Load node next to Blue layers; confirm Gold-style presentation, no Unfreeze-to-modular action, and legal mix with other node types.

**Acceptance Scenarios**:

1. **Given** a successfully loaded TorchScript Load element, **When** the user views it on the canvas, **Then** it is presented as frozen/opaque processing (same visual class as other Gold black-box elements)
2. **Given** a TorchScript Load element whose checkpoint came from outside the editor, **When** the user looks for Unfreeze into modular Blue nodes, **Then** that action is unavailable or clearly disabled with an explanation
3. **Given** Blue nodes and a TorchScript Load element in one graph, **When** audio runs, **Then** both execute in the shared live graph without forcing the user to freeze the whole graph
4. **Given** a checkpoint that advertises encode/decode (e.g. pretrained RAVE), **When** it loads successfully, **Then** the element exposes the same latent encode/decode pin surface as trained RAVE Gold boxes
5. **Given** a checkpoint with only forward (no encode/decode), **When** it loads successfully, **Then** the element exposes forward audio ports only and does not show latent encode/decode pins
6. **Given** a checkpoint that advertises conditioning, **When** it loads successfully, **Then** a Control input pin is available; when conditioning is not advertised, no Control pin appears
7. **Given** a checkpoint with encode/decode and a usable compactness path, **When** the user adjusts fidelity, **Then** latent thinning behaves like trained RAVE Gold boxes; if compactness is unavailable, fidelity is hidden or disabled with an explanation

---

### Edge Cases

- Checkpoint path is empty and never successfully loaded: dry passthrough of the main audio input with a prompt to choose a file
- Load error with no previous ready model: silence from that node with a recoverable error message
- User clears a previously loaded path: returns to empty/never-loaded behavior (dry passthrough) with a prompt to choose a file
- User picks a directory instead of a file: rejected with a clear message; treated as error (silence if no previous model)
- Very large checkpoint: load may take noticeable time; UI shows loading state; audio must not stall on the real-time path
- Same file path re-selected after an external overwrite: reload uses the new file contents
- Model expects multi-channel or latent-width I/O that does not match stereo audio: shape gate refuses illegal cables; user adjusts overrides or uses compatible models
- Checkpoint advertises conditioning but Control is left unconnected: element runs unconditioned; connecting Control applies conditioning
- Checkpoint does not advertise conditioning: no Control pin is shown
- Host sample rate / buffer size changes while loaded: element continues to run if the checkpoint is sample-rate agnostic; if the model is rate-specific and mismatch is detectable, show a recoverable warning (do not crash)
- Duplicate elements pointing at the same path: each runs independently; unloading one must not break the other

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a Factory palette element dedicated to loading and running an external TorchScript checkpoint on the live graph
- **FR-002**: Users MUST be able to choose a checkpoint file from the local filesystem via the element’s properties (browse and/or path entry)
- **FR-003**: System MUST load a valid selected checkpoint off the real-time audio path and swap it into the graph only when ready, without allocating or compiling on the audio thread
- **FR-004**: Once loaded, the element MUST process connected audio through the checkpoint’s forward path on each buffer when no encode/decode latent path is in use; when a Control pin is present and connected, forward MUST apply that conditioning per the checkpoint’s advertised convention
- **FR-005**: System MUST refuse illegal port connections when the loaded model’s input/output channel (or feature) counts do not match the connected cable shapes, with the same class of shape-mismatch feedback used elsewhere in the editor
- **FR-006**: System MUST surface recoverable, user-visible errors for missing files, unreadable files, and files that are not valid runnable checkpoints for this element
- **FR-007**: System MUST persist each element’s selected checkpoint path and any channel overrides with the graph and with named presets, and MUST attempt reload on session/preset restore
- **FR-008**: A TorchScript Load element MUST appear and behave as opaque frozen (Gold) processing: not weight-randomizable like Blue layers, and not unfreezeable into a modular Blue subgraph when no OpenYourBox source graph exists for that file
- **FR-009**: Users MUST be able to clear or change the selected path; clearing unloads the model and returns the element to the empty never-loaded state (dry passthrough + choose-file prompt) without crashing
- **FR-010**: The element MUST support pretrained RAVE (and similar) TorchScript exports as primary examples, and MUST not be limited to RAVE-only filenames or a single vendor layout when the file is a valid runnable checkpoint matching the element’s supported calling convention
- **FR-011**: On successful load, the system MUST auto-infer input and output channel (or feature) counts—including latent widths when encode/decode pins are exposed—and MUST pre-fill editable override fields with those values. While an override is set, shape checking MUST use the override; a clear/reset action MUST restore the last inferred values. If inference fails, the element MUST enter an error or incomplete-shape state that prompts the user to enter overrides before connections are treated as legal
- **FR-012**: The element MUST always support forward (audio in → audio out). When a loaded checkpoint advertises encode/decode entry points, the element MUST automatically expose the same latent encode/decode pin surface used by trained RAVE Gold boxes; when encode/decode is absent, those latent pins MUST NOT appear
- **FR-013**: Encode/decode pin layout and routing behavior for checkpoints that advertise those entry points MUST match the established trained-RAVE Gold-box conventions (including decode-from-latent when a latent input is connected) so users can reuse the same latent workflows
- **FR-014**: Changing an override MUST re-validate existing cables; illegal connections MUST be refused or flagged with the same class of shape-mismatch feedback used elsewhere
- **FR-015**: The element MUST expose a Control (conditioning) input pin only when the loaded checkpoint advertises conditioning support; when conditioning is not advertised, that pin MUST NOT appear. An unconnected Control pin on a conditioning-capable checkpoint MUST run as unconditioned forward (same class of behavior as other optional Control inputs in the editor)
- **FR-016**: When encode/decode is present, the element MUST offer the same live fidelity control as trained RAVE Gold boxes. If the checkpoint lacks the compactness path required for fidelity, that control MUST be hidden or disabled with a clear explanation; forward-only checkpoints MUST NOT show fidelity
- **FR-017**: While the path is empty and the element has never successfully loaded a model, the element MUST dry-passthrough the main audio input and show a visible prompt to choose a file. On load error (or any failed/unready state with no previous ready model retained), the element MUST output silence from that node while showing a recoverable error. When a previous model is still active during a failed or in-progress reload, that previous model MUST keep running until a successful atomic swap or an explicit clear

### Key Entities

- **TorchScript Load Element**: Factory graph node that references one checkpoint path, holds load status (empty / loading / ready / error), exposes audio ports plus optional latent and Control ports based on what the checkpoint advertises, and stores inferred plus optional override channel counts used for shape checking
- **Checkpoint Reference**: Local filesystem path to a runnable TorchScript artifact; persisted with graph/preset; may become stale if the file moves
- **Load Status**: User-visible state describing whether the element can process audio and, on failure, a short reason
- **Channel Override**: User-editable input/output (and latent, when applicable) counts that default to inferred values and take precedence for shape checking until cleared/reset

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A user can place the element, select a known-good pretrained checkpoint, and hear processed output in under 2 minutes without leaving the plugin
- **SC-002**: After a successful load, switching the host transport and playing audio continues without audio-thread stalls attributable to checkpoint I/O (load work completes off the real-time path; swaps are atomic)
- **SC-003**: 100% of missing-file and invalid-file attempts in manual test cases produce a recoverable on-element or property-panel error and zero plugin crashes
- **SC-004**: Illegal shape connections involving a loaded element are refused with explanatory feedback in 100% of mismatch cases exercised in acceptance tests
- **SC-005**: Saving and reloading a session or preset that contains one or more TorchScript Load elements restores each path and channel overrides; when files remain available, models become ready again without re-picking paths
- **SC-006**: At least one publicly documented pretrained RAVE (or equivalent) TorchScript export can be demonstrated end-to-end: load → connect → hear in the live graph
- **SC-007**: In acceptance tests, an empty never-loaded element dry-passthroughs; a load-error element with no prior model outputs silence; both cases keep the host audio engine running without crashes

## Assumptions

- This element is distinct from “Freeze Selection” / train-autoloaded Gold boxes: those originate from in-editor modular graphs; this element originates from an external file the user supplies
- Empty never-loaded path: dry passthrough of main audio with a choose-file prompt. Load error with no retained prior model: silence from that node with a recoverable error. Prior ready model retained during failed/in-progress reload until successful swap or clear
- Checkpoint files remain on disk at the stored path; v1 does not require copying the artifact into the project package (path reference only), though a future “embed in project” option is out of scope
- Channel counts are inferred on load and shown as editable overrides; overrides are persisted with the element so intentional fixes survive reload
- The primary calling convention in v1 is forward (audio tensor in → audio tensor out); encode/decode latent pins appear automatically when the checkpoint advertises those entry points, matching trained RAVE Gold boxes
- Live fidelity is available for encode/decode loads that expose the compactness path expected by existing RAVE Gold boxes; otherwise fidelity is hidden or disabled
- Users are responsible for using checkpoints they have rights to run; no marketplace download browser is required for this feature
- Sample-rate–specific models are supported best-effort: if mismatch cannot be detected, the model still runs; if it can, warn rather than hard-fail unless processing would be unsafe
- Weight randomization (RONN-style) does not apply to this element
- Training this opaque external checkpoint in-place is out of scope; users who need trainable modular RAVE use existing architecture elements and train/freeze flows
- Marketplace download, cloud fetch, and in-place fine-tuning of external checkpoints are out of scope for this feature
