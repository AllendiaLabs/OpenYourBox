# Feature Specification: DDSP Effects & Recurrent Layers

**Feature Branch**: `014-ddsp-effects-rnn`

**Created**: 2026-08-29

**Status**: Draft

**Input**: User description: "Add DDSP effects from magenta/ddsp/effects.py as graph editor elements (Reverb, ExpDecayReverb, FilteredNoiseReverb, FIRFilter, ModDelay) and add LSTM and RNN."

## Clarifications

### Session 2026-08-29

- Q: Should LSTM and RNN keep their hidden state from one audio buffer to the next while the graph is running live, or reset hidden state at the start of every buffer? → A: Carry hidden state across buffers; reset on structural/lifecycle events (rebuild, reconnect, freeze swap, re-init)
- Q: How should users edit the magnitude grids (time × filter banks) on FilteredNoiseReverb and FIRFilter in v1? → A: Dimension controls (time steps × banks) + init/randomize; magnitudes trainable; no cell-by-cell painter in v1
- Q: When a Reverb has both an internal impulse response and an external IR input connected, which IR should the convolution use? → A: Mix or blend internal and external IRs with a wet-style control
- Q: What should happen when the user sets a `reverb_length` that is too long for safe live (Blue) processing? → A: Allow any length; show a non-blocking performance warning when above a live-safe threshold
- Q: If the external IR input on Reverb is connected but empty or zero-length, what should the element do? → A: Fall back to internal IR; show a recoverable warning; do not stop audio
- Q: How many stacked layers should a single LSTM or RNN element contain? → A: Exactly 1 layer; users deepen recurrence by grouping and stacking multiple LSTM/RNN elements
- Q: For each time step in the incoming buffer, should an LSTM or RNN output a value for every time step (full sequence), or only the final time step’s hidden state? → A: Full sequence out (same time length as input; channels = hidden size)
- Q: Should LSTM and RNN support bidirectional processing in v1 (forward and backward passes), or stay unidirectional only? → A: Unidirectional by default, optional bidirectional toggle
- Q: Should the LSTM/RNN input feature size be inferred automatically from the connected upstream channel count, or set as an explicit editable parameter? → A: Infer input size from upstream connection (no separate input-size field)
- Q: For the vanilla RNN element, which nonlinearity should v1 expose? → A: Same activation choices as Activation and TCN (ReLU, Sigmoid, Tanh, LeakyReLU, PReLU) on both RNN and LSTM; default Tanh; LeakyReLU uses the same negative-slope rules as Activation/TCN
- Q: Should LSTM and RNN expose PyTorch’s `bias` flag (learnable bias terms on/off), or always keep bias enabled? → A: Expose `bias` toggle; default on (`True`)
- Q: Where should the shared activation (ReLU / Sigmoid / Tanh / LeakyReLU / PReLU) apply on LSTM and RNN? → A: In-cell: replace the cell’s primary nonlinearity (RNN nonlinearity; custom LSTM cell) with the chosen function
- Q: Should LSTM and RNN also expose the same pre-nonlinearity gain control that Activation and TCN use, or omit gain on recurrent cells in v1? → A: Same gain control as Activation/TCN on both LSTM and RNN

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Place and Hear Differentiable Effects (Priority: P1)

A sound designer opens the element menu and finds a new **Effects** category containing Reverb, ExpDecayReverb, FilteredNoiseReverb, FIRFilter, and ModDelay. They drag an ExpDecayReverb between Audio Input and Audio Output, adjust gain, decay, reverb length, and dry/wet mix from the element properties, and immediately hear the wet signal change while the plugin continues processing audio without interruption.

**Why this priority**: Differentiable effects are the core creative value of this feature and deliver usable sound design without training or freeze.

**Independent Test**: Place Audio In → ExpDecayReverb → Audio Out, change decay and add_dry, confirm audible reverb and live parameter response.

**Acceptance Scenarios**:

1. **Given** the graph editor is open, **When** the user opens the element menu, **Then** Reverb, ExpDecayReverb, FilteredNoiseReverb, FIRFilter, and ModDelay appear as available elements
2. **Given** an ExpDecayReverb is on the canvas between audio in and out, **When** the user increases decay, **Then** the reverberant tail becomes longer in the audible output
3. **Given** any effect element with an `add_dry` (or equivalent dry-mix) control, **When** the user disables adding dry, **Then** only the processed (wet) path is heard
4. **Given** an effect element is live (not frozen), **When** the user edits a numeric parameter, **Then** the change applies without stopping audio or requiring a restart

---

### User Story 2 - Shape Spaces with Frequency-Domain Filters (Priority: P2)

A user inserts an FIRFilter (or FilteredNoiseReverb) to sculpt timbre over time. They set magnitude-grid dimensions (time steps × filter banks) and window size from the property UI, initialize or randomize the magnitude values, hear the spectral shaping update, and can refine magnitudes further through training. They can combine these nodes with existing neural layers in the same graph.

**Why this priority**: LTV-FIR and filtered-noise reverbs enable spectral design that simple gain/decay reverbs cannot, but they depend on the basic effects palette existing first.

**Independent Test**: Place FIRFilter in a minimal audio chain, change grid dimensions or window size and randomize magnitudes, verify audible spectral change and that incompatible connections are refused with clear feedback.

**Acceptance Scenarios**:

1. **Given** an FIRFilter on the canvas, **When** the user views its properties, **Then** magnitude-grid dimensions (time steps and filter banks) and window size are editable, and init/randomize controls for magnitudes are available
2. **Given** a FilteredNoiseReverb on the canvas, **When** the user views its properties, **Then** magnitude-grid dimensions, window size, reverb length, and add_dry are editable, and init/randomize controls for magnitudes are available
3. **Given** magnitude dimensions that would break the current signal shape, **When** the user attempts an illegal connection or illegal size, **Then** the editor refuses the change and explains the constraint
4. **Given** FilteredNoiseReverb or FIRFilter in v1, **When** the user looks for a cell-by-cell magnitude painter, **Then** no such painter is required; magnitudes are changed via dimensions, init/randomize, and training

---

### User Story 3 - Modulated Delay Color (Priority: P2)

A user adds ModDelay to create chorus, flanger, or vibrato-like motion. They set center delay, modulation depth, gain, phase, and dry mix, and hear time-varying delay coloration suitable for creative processing or as a trainable effect block in a larger architecture.

**Why this priority**: ModDelay completes the DDSP effects set from the source library and is independently useful for modulation FX.

**Independent Test**: Audio In → ModDelay → Audio Out; sweep depth_ms and phase; confirm modulation character changes.

**Acceptance Scenarios**:

1. **Given** a ModDelay element, **When** the user views properties, **Then** center_ms, depth_ms, gain, phase, and add_dry are available
2. **Given** depth_ms is set above zero, **When** audio plays, **Then** the output exhibits audible delay modulation relative to depth zero
3. **Given** add_dry is enabled, **When** audio plays, **Then** dry and wet are mixed according to the element’s dry-add behavior

---

### User Story 4 - Convolutional Reverb with Optional IR (Priority: P2)

A user places a Reverb element that convolves dry audio with an impulse response. They can use an internal / trainable IR controlled by reverb length and add_dry, supply an external IR via an optional input, and when both are available blend internal and external IRs with a dedicated mix control.

**Why this priority**: Full convolutional reverb is the most general DDSP reverb; parameterized variants (ExpDecay, FilteredNoise) cover common cases without requiring an IR input.

**Independent Test**: Place Reverb with default internal IR and hear convolution; connect an optional IR source and sweep the internal/external blend; confirm the wet character moves between internal and external IRs.

**Acceptance Scenarios**:

1. **Given** a Reverb element with no external IR connected, **When** audio plays, **Then** convolution uses the element’s internal IR of the configured reverb length
2. **Given** a Reverb element with an external IR connected, **When** the user adjusts the internal/external IR blend control, **Then** the wet character interpolates between the internal IR and the supplied external IR according to that control
3. **Given** a Reverb element with an external IR connected and blend set fully to external, **When** audio plays, **Then** convolution follows the external IR
4. **Given** reverb_length is reduced, **When** audio plays, **Then** the effective internal IR / tail length shortens
5. **Given** an external IR input is connected but empty or zero-length, **When** audio plays, **Then** Reverb uses the internal IR, shows a recoverable warning, and audio continues

---

### User Story 5 - Recurrent Layers in the Graph (Priority: P3)

A model builder adds LSTM and RNN elements from the neural / sequence category of the element menu. Each element is a single recurrent layer; they deepen the stack by placing multiple LSTM/RNN nodes in series (or grouping them). They configure hidden size and related recurrent parameters, connect them like other layers, and use them in live graphs and in freeze/train workflows alongside existing linear and convolutional elements.

**Why this priority**: Recurrent layers expand architecture expressiveness but are secondary to shipping the DDSP effects palette called out as the primary TODO block.

**Independent Test**: Build Audio In → (projection if needed) → LSTM or RNN → Audio Out (or a valid feature path), edit hidden size, confirm the graph runs and freeze/train still accept the element type; stack two single-layer recurrent elements to deepen the network.

**Acceptance Scenarios**:

1. **Given** the element menu is open, **When** the user browses neural / sequence elements, **Then** LSTM and RNN are listed
2. **Given** an LSTM or RNN on the canvas, **When** the user views properties, **Then** hidden size, a bidirectional toggle (default off / unidirectional), a bias toggle (default on), an activation choice matching Activation/TCN (ReLU, Sigmoid, Tanh, LeakyReLU, PReLU; default Tanh), and the same pre-nonlinearity gain control as Activation/TCN are editable, and there is no multi-layer / num_layers control
3. **Given** an LSTM or RNN with LeakyReLU selected, **When** the user views properties, **Then** the same negative-slope control and validation rules as Activation/TCN apply
4. **Given** a graph containing LSTM or RNN with legal shapes, **When** the user freezes the selection, **Then** freeze completes and the Gold BlackBox replaces the selection without stopping the plugin UI
5. **Given** incompatible channel / feature sizes at an LSTM or RNN port, **When** the user tries to connect, **Then** the connection is refused with a shape mismatch explanation
6. **Given** an LSTM or RNN processing consecutive live audio buffers, **When** no rebuild, reconnect, freeze/unfreeze swap, or re-init occurs, **Then** hidden state carries from the previous buffer into the next
7. **Given** an LSTM or RNN whose graph is rebuilt, ports are reconnected, the subgraph is frozen or unfrozen, or the user re-inits/randomizes the element, **When** the next audio buffer runs, **Then** hidden state starts from a fresh reset
8. **Given** two LSTM or RNN elements connected in series (or grouped), **When** shapes match, **Then** the user can deepen recurrence without any single element exposing a layer-count greater than one
9. **Given** an LSTM or RNN with a legal input sequence and bidirectional off, **When** audio or features are processed, **Then** the output has the same time length as the input and channel count equal to the element’s hidden size
10. **Given** an LSTM or RNN with bidirectional on, **When** audio or features are processed, **Then** the output has the same time length as the input and channel count equal to twice the hidden size (forward and backward concatenated)
12. **Given** an LSTM or RNN with a non-default activation selected, **When** the element processes a sequence, **Then** that activation is applied as the cell’s primary in-cell nonlinearity (not merely after the full layer output)

---

### Edge Cases

- Extremely long `reverb_length` relative to buffer size: processing remains live and stable; the UI shows a non-blocking performance warning when length exceeds a live-safe threshold rather than clamping, rejecting the value, or crashing the host.
- Empty or zero-length external IR on Reverb: fall back to the internal IR (external blend contribution unused), show a recoverable UI warning, and continue audio without stopping the host.
- Magnitude grids with one time step or one filter bank: elements still run with valid defaults.
- ModDelay with `depth_ms` exceeding `center_ms` such that delay would go negative: values are clamped or rejected with a clear property validation message.
- LSTM/RNN with sequence length of one buffer: elements process correctly while carrying hidden state into the next buffer (no offline full-file context required).
- Freezing a subgraph that mixes effects and recurrent layers: freeze succeeds when shapes are legal; on failure the user sees a progress/error message and the live graph remains unchanged.
- Randomize Weights on elements that own trainable tensors (e.g., IR, magnitudes, recurrent weights): only that element’s trainable parameters randomize; non-weighted controls (e.g., add_dry flags) are unchanged unless they are defined as part of that element’s randomizable parameter set by existing project rules.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The element menu MUST expose five DDSP-style effect elements: Reverb, ExpDecayReverb, FilteredNoiseReverb, FIRFilter, and ModDelay
- **FR-002**: ExpDecayReverb MUST expose editable parameters for gain, decay, reverb length, and whether dry signal is added
- **FR-003**: FilteredNoiseReverb MUST expose editable magnitude-grid dimensions (time steps × filter banks), window size, reverb length, and whether dry signal is added; magnitude values MUST be initializable/randomizable and trainable, without requiring a cell-by-cell magnitude painter in v1
- **FR-004**: FIRFilter MUST expose editable magnitude-grid dimensions (time steps × filter banks) and window size; magnitude values MUST be initializable/randomizable and trainable, without requiring a cell-by-cell magnitude painter in v1
- **FR-005**: ModDelay MUST expose editable parameters for center delay (ms), depth (ms), gain, phase, and whether dry signal is added
- **FR-006**: Reverb MUST perform convolutional (FIR) reverberation of the dry input against an impulse response, expose reverb length and dry-add controls, support an optional external IR input alongside an internal/trainable IR, and expose a blend control that mixes internal and external IRs when the external input is connected
- **FR-006a**: When `reverb_length` (or equivalent IR length on related reverb elements) exceeds a live-safe performance threshold, the system MUST allow the value and continue live processing while showing a non-blocking performance warning; it MUST NOT hard-clamp or refuse the value solely for length
- **FR-006b**: When an external IR input is connected but empty or zero-length, Reverb MUST fall back to the internal IR, show a recoverable warning, and continue processing audio without stopping the host
- **FR-007**: All five effect elements MUST be placeable, movable, connectable, and deletable using the same graph interactions as existing processing elements
- **FR-008**: Effect and recurrent element parameters MUST be editable from the same property / parameter UI patterns used by other graph elements, with changes reflecting in live processing when the element is not frozen
- **FR-009**: Shape and connection validation MUST apply to these new elements the same way as existing elements (illegal cables refused; mismatch explained)
- **FR-010**: The element menu MUST expose LSTM and RNN as graph elements suitable for sequence / feature processing in the same graph as other neural layers
- **FR-011**: Each LSTM and RNN element MUST be exactly one recurrent layer (no multi-layer / num_layers parameter); users MUST deepen stacks by connecting or grouping multiple single-layer LSTM/RNN elements. Each element MUST expose at least hidden size, a bidirectional toggle (default unidirectional / off), a bias toggle (default on), and an activation/nonlinearity choice as editable parameters, with sensible defaults so a newly placed element runs without further configuration when connected legally
- **FR-011a**: While an LSTM or RNN remains in a stable live graph, its hidden state MUST carry across consecutive audio buffers; hidden state MUST reset on graph rebuild, port reconnect, freeze or unfreeze swap, or explicit re-init / randomize of that element
- **FR-011b**: LSTM and RNN MUST emit a full sequence aligned with the input time axis (same number of time steps as the input); when unidirectional, output channel count MUST equal hidden size; when bidirectional, output channel count MUST equal twice hidden size (forward and backward concatenated)
- **FR-011c**: LSTM and RNN input feature size MUST be inferred from the connected upstream output channel count (no separate editable input-size parameter); illegal or missing connections MUST be refused with the same shape-mismatch feedback as other elements
- **FR-011d**: Both LSTM and RNN MUST offer the same activation choices as Activation and TCN elements (ReLU, Sigmoid, Tanh, LeakyReLU, PReLU), defaulting to Tanh; the chosen function MUST replace the cell’s primary nonlinearity in-cell (not only as a post-layer activation)—for RNN this is the cell nonlinearity; for LSTM this is the custom cell’s corresponding primary nonlinearity—while when LeakyReLU is selected, the same negative-slope parameter and out-of-range refusal rules as Activation/TCN MUST apply
- **FR-011d2**: Both LSTM and RNN MUST expose the same pre-nonlinearity gain control as Activation and TCN, with the same defaults and semantics, applied to the in-cell nonlinearity
- **FR-011e**: Both LSTM and RNN MUST expose a bias toggle corresponding to learnable bias terms, defaulting to on
- **FR-012**: Graphs containing any of these new elements MUST remain eligible for manual freeze and unfreeze when shapes are legal
- **FR-013**: Graphs containing any of these new elements MUST remain eligible for training workflows that already accept modular graph architectures
- **FR-014**: New elements that own trainable parameters MUST support the project’s existing weight randomization and seed behavior where applicable
- **FR-015**: Element identity, parameters, and connections for these types MUST persist across project / plugin state save and reload
- **FR-016**: Behavior of each DDSP-style effect MUST match the corresponding Magenta DDSP effect semantics at a user-audible / parameter level (same controls and roles), without requiring users to leave the VST
- **FR-017**: A compressor effect and other TODO items not listed in the input (positional encoding, dropout, diffusion, etc.) are out of scope for this feature

### Key Entities

- **Effect Element**: A graph node that applies a differentiable audio effect (reverb family, FIR filter, or modulated delay), with typed audio/feature ports and editable effect parameters
- **Impulse Response (IR)**: Time-domain filter kernel used by convolutional Reverb; may be internal/trainable, supplied externally, or blended when both are present via an IR mix control
- **Magnitude Grid**: Time × filter-bank table controlling spectral envelopes for FilteredNoiseReverb and FIRFilter; in v1 users edit dimensions and init/randomize (and train) values rather than painting individual cells
- **Recurrent Element**: A single-layer LSTM or RNN graph node whose hidden state persists across consecutive live audio buffers until a structural or lifecycle reset (rebuild, reconnect, freeze/unfreeze swap, re-init); deeper stacks are built by chaining or grouping multiple such elements
- **Element Menu Category**: Grouping in the palette (e.g., Effects vs Neural/Sequence) so users can discover the new types quickly

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can add any of the five effect elements from the menu and hear a distinct, intended effect character within 30 seconds of placing a minimal Audio In → effect → Audio Out graph
- **SC-002**: For each effect element, 100% of the scalar / dimension parameters listed in the feature input are visible and editable in the property UI without leaving the VST; for magnitude grids, dimension controls plus init/randomize satisfy editability (no cell painter required)
- **SC-003**: 100% of legal freeze attempts on graphs that include at least one new effect or recurrent element complete without requiring the user to restart the host or plugin
- **SC-004**: In a guided first-use test, at least 9 out of 10 users can locate LSTM and RNN in the element menu and place one into a graph without documentation beyond on-screen labels
- **SC-005**: Changing a single exposed parameter on a live effect updates the audible output within one user-perceptible interaction (no mode switch, export, or offline bounce required)
- **SC-006**: Save then reload of a project containing all seven new element types restores every element’s type, connections, and parameter values with no manual repair
- **SC-007**: Shape-mismatch attempts involving the new elements are refused in 100% of tested illegal connections, each with an explanatory message or tooltip
- **SC-008**: When a user sets reverb length above the live-safe threshold, processing continues and a non-blocking performance warning is visible within one interaction (no clamp, no forced freeze)

## Assumptions

- Target users are the same OpenYourBox graph users already building live and trainable architectures in the VST (no separate app or CLI).
- DDSP effect behavior is defined by Magenta DDSP `effects.py` semantics (parameter names/roles and audible behavior), adapted to this product’s existing graph, live, freeze, and train conventions.
- LSTM and RNN follow the same tensor / channel conventions as existing neural elements (sequence along the time dimension of the graph’s signal), not a separate offline sequence editor; live hidden state carries across buffers and resets on structural/lifecycle events (see FR-011a). Each LSTM/RNN element is exactly one layer; multi-layer depth is achieved by stacking/grouping elements. Output is a full sequence; channels equal hidden size when unidirectional, or twice hidden size when bidirectional (see FR-011b). Bidirectional defaults to off. Input feature size is inferred from the upstream connection (see FR-011c). Both LSTM and RNN use the same activation menu as Activation/TCN (ReLU, Sigmoid, Tanh, LeakyReLU, PReLU), default Tanh, applied as the cell’s primary in-cell nonlinearity with the same pre-nonlinearity gain control as Activation/TCN (see FR-011d / FR-011d2). Bias defaults to on and is user-toggleable (see FR-011e).
- Default parameter values are chosen so a newly dropped element produces a safe, audible, non-silent effect or a pass-compatible neural transform when connected with matching shapes.
- Optional external IR for Reverb is modeled as an optional secondary input; when disconnected, the internal/trainable IR of `reverb_length` is used; when connected, a blend control mixes internal and external IRs.
- Magnitude grids may be edited via dimension controls plus init/randomize and training; a cell-by-cell magnitude painter or full waveform IR editor beyond existing property patterns is not required for v1.
- Custom cell types beyond LSTM and vanilla RNN, and a DDSP compressor, are deferred unless already covered by other features.
- Licensing attribution for Magenta/DDSP-derived behavior will follow the project’s existing third-party notice practices during implementation planning (not a user-facing workflow requirement).
- Live-performance safety for long FIRs uses a non-blocking warning above a live-safe threshold; values are not hard-clamped or rejected for length alone (exact threshold is an implementation detail as long as SC-001, SC-008, and host stability hold).
