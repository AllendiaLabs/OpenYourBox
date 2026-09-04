# Feature Specification: Generalize Training Graph

**Feature Branch**: `019-generalize-training-graph`

**Created**: 2026-09-04

**Status**: Clarified (ready for planning)

**Input**: User description: "now that we have a good mvp, I would like to improve the code and make it more general. it should now be of extremely good quality, ready for production and easily implementing future features — refactor inference and training to be architecture-agnostic (no explicit RAVE/TCN/steerable references; run/train whatever is on the graph); add a data loader node with controllable renamable outputs and example-count matching utilities; training fails without a data loader; only train nodes connected to or downstream of a data loader; data loader may connect to pins that already have a connection (e.g. alongside audio in to mimic inference); data loader cables use a distinct color and show no RMS (N/A); allow grouping a single box; remove reconstruction vs mapping train mode (objective follows the data); expose general training parameters so users can reproduce prior recipes themselves; save/load training configurations (hyperparameters); remove TCN (replaceable by stacked Conv1D) and Linear (Conv1D with stride=kernel=dilation=1)."

## Clarifications

### Session 2026-09-04

- Q: Without a reconstruction vs mapping mode, how should the product decide what the network is supervised against during training? → A: **Add loss nodes**; the user connects them to the desired output pins of their architecture. **No regression**: users must still be able to do everything they could under prior specs (mapping, reconstruction-style training, and equivalent creative outcomes) via the new general graph + data loader + loss + training-configuration model.
- Q: How should existing projects that used the old mapping or reconstruction Train objective move to the new data-loader + loss-node model without losing capability? → A: **No need to care about old projects** — legacy session/project migration for the old objective model is out of scope; no automatic compatibility wiring required.
- Q: If the canvas has more than one Data Loader, what should happen when the user presses Run? → A: **Allow multiple Data Loaders**; the user designates exactly one as **active** for the Run; others are ignored.
- Q: How should training audio be assigned to each Data Loader output? → A: **Per-output bindings** — each data-loader output owns its own ordered example list (chosen from the Training Library / capture / import).
- Q: Where should saved training configurations (hyperparameters) live so users can reuse them? → A: **Both** — a user-level library of named training configs (reusable across projects/sessions), and projects may also store/load a config snapshot used with that project.

### Session 2026-09-04 (continued)

- Q: When several loss nodes are wired into the graph, how should training combine them? → A: **B and D** — multiple losses with per-loss weights (weighted sum), **and** ordered loss **stages/schedules** (e.g. run one loss set for N steps, then another). Scope MUST include everything needed to reproduce RAVE/reconstruction-style multi-term and multi-stage training without a built-in architecture mode.
- Q: During live playback (not training), should Data Loader and Loss nodes affect the audible signal path? → A: **Training-only** — live playback ignores Data Loader and Loss; the audible path uses live cables only (e.g. Audio In).
- Q: Should the equal example-count rule apply to every Data Loader output, or only to outputs that are actually connected—and when is it enforced? → A: Enforce **only when starting training (Run)**. At that check, **only connected outputs** must have equal example counts; unconnected outputs are ignored for the gate.
- Q: Should this feature ship ready-made example graphs and/or training configurations that show how to reproduce prior mapping-style and reconstruction-style (RAVE-like) recipes? → A: **Ship both** example graph templates and example training configs (clearly labeled as examples, not modes).
- Q: In a processing chain, where may Data Loader connect? → A: Data Loader **only replaces external/inference data**. If A→B→C and A already has a data-loader feed on its external/inference input, **B and C cannot also take a data-loader on the pins fed by A**—those pins carry internal graph data, not external/inference replacement.
- Q: If a mid-chain node has a separate external pin (e.g. conditioning) while its main audio comes from upstream, may a Data Loader connect there? → A: **Yes, per external source input**—any input that should participate in training (and thus train that node and its downstream) MUST be fed by the Data Loader. That includes **Knob / XY Trackpad (and similar) source nodes**, not a special “conditioning pin” on removed TCN-style blocks. Utilities MUST support supplying e.g. a **scalar (or constant) value copied across all examples** when live conditioning would otherwise be zeros. If a required training feed is missing, **Start training MUST fail**. On the **Train tab**, elements that will be trained appear normal; elements not on the data-loader path appear **slightly transparent** (refined later: armed on-path normal; passthrough/off-path transparent).

### Session 2026-09-04 (more)

- Q: With Data Loader connectivity defining what can train, does the product still need a separate “arm for training” action, or is being on the data-loader path enough? → A: **Keep arm for training**. Data-loader path defines what participates in the training forward graph; **arm** means the element’s parameters are updated (backprop). An element on the path that is **not armed** is **passthrough only**—used during training but not updated.
- Q: When a new processing element is added to the graph, should it start armed for training or disarmed? → A: New elements default to **armed**.
- Q: If every element on the data-loader path is disarmed (all passthrough, nothing armed to update), what should Start training do? → A: **Refuse Start training** — require at least one armed on-path element with trainable parameters.
- Q: When a frozen Gold (BlackBox) element sits on the data-loader path, can it be armed for training, or must it stay passthrough-only? → A: Gold on the path is always **passthrough-only** — cannot arm; weights stay fixed.
- Q: Where should the user designate which Data Loader is active when more than one is on the canvas? → A: **Only in the Train panel** (picker of loaders on the canvas)—because only one can be active for a training session.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Train Whatever Is On The Graph (Priority: P1)

A sound designer builds any legal processing graph from general building blocks. They do not pick a named architecture recipe (such as a built-in “RAVE” or “steerable” mode). Training and live inference both operate on the graph as wired: whatever trainable elements sit on the trainable subgraph are optimized; whatever is connected for live play runs at inference. Product language and training controls describe the graph and training settings, not a fixed architecture brand.

**Why this priority**: Architecture-agnostic run/train is the core production shift of this feature; every later story depends on it.

**Independent Test**: Assemble a non-branded encoder–decoder or effect-style graph; start training with a valid data loader and config; confirm training updates only the trainable subgraph; confirm live audio still follows the live graph wiring without requiring an architecture mode selector.

**Acceptance Scenarios**:

1. **Given** a legally wired graph built only from general palette elements, **When** the user opens Train, **Then** they see no architecture-specific mode or recipe selector that names prior product architectures
2. **Given** that graph plus a valid data loader wiring and training configuration, **When** the user Runs training, **Then** the system trains the trainable subgraph defined by data-loader connectivity (see Story 3), not a hard-coded architecture path
3. **Given** the same graph at inference time, **When** audio plays, **Then** live processing follows the live graph connections without architecture-specific special cases in the user-facing train/inference workflow
4. **Given** product surfaces (Train panel, messages, presets, docs strings visible to users), **When** the user reads them, **Then** they do not require choosing or acknowledging named prior architectures to train a generic graph

---

### User Story 2 - Feed Training With A Data Loader Node (Priority: P1)

The designer adds a **Data Loader** node that supplies training examples into the graph. The node exposes a controllable number of outputs; each output can be renamed so the designer can label roles such as “input”, “target”, or custom names. **Each output owns its own ordered example list** (bound from the Training Library, capture, or import)—not a single global multi-select that all outputs share. Equal example counts across outputs are validated **only when starting training (Run)**, and only for **outputs that are connected** into the graph; unconnected outputs are ignored for that gate. When connected outputs differ in count, the designer uses provided utilities (for example, repeating/copying one example) so those connected outputs match before Run succeeds.

**Why this priority**: Without a data loader, generalized training has no architecture-neutral data path; training must refuse to start without one.

**Independent Test**: Insert a data loader; configure two renamed outputs with equal example counts; connect them into the graph; Run train; then mismatch counts and confirm Run is blocked until a repeat/copy utility restores equal counts.

**Acceptance Scenarios**:

1. **Given** the palette, **When** the user adds a Data Loader, **Then** a data-loader node appears with a controllable output count (at least one output) and each output can be renamed
2. **Given** a data loader with multiple **connected** outputs, **When** the user starts training (Run), **Then** the system requires equal example counts across those connected outputs before training can start
3. **Given** unequal example counts between connected outputs, **When** the user applies a provided utility to copy/repeat examples on a shorter output and Starts training again, **Then** example counts become equal and the mismatch blocking condition clears
4. **Given** a need to feed a Knob/Trackpad-style source for every example with a fixed value, **When** the user applies a Data Loader constant/scalar-copy utility on an output, **Then** that value is supplied for all examples and can satisfy equal-count with other connected outputs
5. **Given** two data-loader outputs, **When** the user binds different example lists to each, **Then** each output retains its independent ordered list (global Library multi-select alone does not replace per-output bindings)
6. **Given** an unconnected data-loader output with a different example count than connected outputs, **When** the user Starts training, **Then** that unconnected output is ignored for the equal-count gate and does not by itself block Run
7. **Given** mismatched connected-output counts while the user is still editing (not Starting training), **When** they continue editing, **Then** the product does not hard-block editing solely for equal-count; the equal-count refusal happens at Start training
8. **Given** no data loader on the graph (or no usable data-loader configuration), **When** the user attempts Run, **Then** training fails/refuses with a clear message that a data loader is required
9. **Given** a data loader whose outputs feed the graph, **When** training runs, **Then** batches presented to the trainable subgraph come from those data-loader outputs as connected
10. **Given** two or more Data Loaders on the canvas, **When** the user selects one as active in the **Train panel** and Runs, **Then** only the active loader feeds training and non-active loaders are ignored
11. **Given** two or more Data Loaders with no active designation in the Train panel, **When** the user attempts Run, **Then** training is refused with a clear message to choose the active loader
12. **Given** exactly one Data Loader on the canvas, **When** the user Runs without an explicit designation, **Then** that sole loader is treated as active

---

### User Story 3 - Train Only The Data-Loader Subgraph (Priority: P1)

Only nodes that are connected to a data loader or lie downstream of a data-loader feed are included in the trainable set. Nodes that never receive data-loader signal (and are not downstream of one) keep their current weights for training purposes and are not optimized. Live inference wiring remains usable in parallel: a data-loader cable may attach to a pin that already has a **live external/inference** connection (for example, alongside Audio In on the same pin) so training can mimic the inference stream without tearing down live play wiring. **Data Loader only replaces external/inference feeds**—not pins already driven by an upstream processing node. In a chain A→B→C, if A’s external input is data-loader-fed, B and C must not also receive data-loader connections on the pins fed by A; training data reaches them through A.

**Any external source input** that must participate for a subgraph to train—including **Knob**, **XY Trackpad**, and similar source nodes—MUST be connected to a Data Loader output (or a Data Loader utility that supplies a constant/scalar copied per example). Missing required feeds cause Start training to fail. **Arm for training** remains: elements on the data-loader path that are armed have parameters updated; elements on the path that are **not armed** are **passthrough only** (used in the forward pass, not updated by backprop). On the **Train tab**, elements whose parameters will be updated (armed + on path) appear at normal opacity; elements that are passthrough-only or off the data-loader path appear **slightly transparent** (passthrough may use a distinct subtle treatment from fully excluded if helpful, but both are visually de-emphasized relative to armed trainable elements).

**Why this priority**: Correct trainable-subgraph selection prevents accidental training of unrelated branches and enables train-while-graph-still-plays patterns.

**Independent Test**: Build a graph with Audio In → A → B → C and a Knob into A; connect data loader to Audio In’s pin on A and to the Knob source; refuse data loader on B’s audio-from-A pin; arm A and C but not B; Run and confirm B is used as passthrough (weights unchanged) while A and C update; open Train tab and confirm armed-on-path elements look normal while passthrough/off-path look slightly transparent; remove knob data-loader feed and confirm Start training fails unless a scalar/copy utility supplies values.

**Acceptance Scenarios**:

1. **Given** a graph where some nodes are reachable from a data loader and others are not, **When** training Runs, **Then** only nodes that are both on the data-loader path and **armed** have parameters optimized; off-path nodes are unchanged
2. **Given** a node on the data-loader path that is **not armed**, **When** training Runs, **Then** that node participates as **passthrough only** (forward use, no parameter updates)
3. **Given** a pin that already has a live **external/inference** connection (e.g. from Audio In), **When** the user connects a data-loader output to that same pin, **Then** the connection is allowed and both the live and training feeds remain conceptually associated with that pin for their respective runtimes
4. **Given** a chain A→B→C where A’s external input already has a data-loader feed, **When** the user tries to attach a data loader to B’s or C’s input pin that is fed by A, **Then** the connection is refused with a clear reason (data loader only replaces external/inference data; internal graph pins cannot take a data loader)
5. **Given** a Knob or XY Trackpad (or similar) source that feeds the trainable path, **When** the user wants that path trained, **Then** they connect a Data Loader output to that source (or use a Data Loader utility to supply a scalar/constant copied for every example)—not a dedicated conditioning pin on a removed TCN-style block
6. **Given** a required external source on the trainable path has no data-loader feed and no constant/scalar utility binding, **When** the user Starts training, **Then** Run fails with a clear message identifying the missing feed
7. **Given** the Train tab is open, **When** the user views the graph, **Then** armed elements on the data-loader path appear at normal opacity and passthrough-only or off-path elements appear slightly transparent
8. **Given** nodes with no path from any data loader, **When** training completes, **Then** those nodes’ trainable parameters are unchanged by the run
9. **Given** a data-loader cable on the canvas, **When** the user inspects it, **Then** it uses a distinct visual color from ordinary signal cables and does not show RMS (displayed as N/A / not applicable)
10. **Given** Data Loader and/or Loss nodes present on the canvas, **When** the user plays live audio without training, **Then** the audible path follows live cables only and ignores Data Loader and Loss (training-only nodes)
11. **Given** the user adds a new processing element, **When** it appears on the canvas, **Then** it defaults to **armed** for training
12. **Given** every on-path element is disarmed (passthrough only), **When** the user Starts training, **Then** Run is refused with a clear message that nothing is armed to train
13. **Given** a Gold (BlackBox) element on the data-loader path, **When** the user views arm controls, **Then** the Gold element cannot be armed and remains passthrough-only (weights unchanged by training)

---

### User Story 4 - Supervise With Loss Nodes And Save Training Configs (Priority: P1)

The designer adds **loss nodes** and connects them to the architecture output pins they want supervised (and to the corresponding target/reference feeds from the data loader as required by each loss). Multiple losses may be active with **per-loss weights** (weighted sum). Training configurations may also define an ordered **loss stage schedule** (different loss sets and/or weights over successive step ranges)—enough to reproduce RAVE/reconstruction-style multi-term and multi-stage training without a built-in architecture mode. Same-data vs different-data intent follows from what the data loader supplies on input vs target wires. Training also exposes a general hyperparameter surface; the user can save and load named training configurations in a **user training-config library** and as **project config snapshots**. **No regression**: every creative training outcome previously available under prior specs (including mapping-style and reconstruction-style results and equivalent recipe behaviors) remains achievable by combining graph wiring, data loader materials, loss nodes, stage schedules, and training configurations.

**Why this priority**: Explicit loss wiring is how objectives become architecture-agnostic; save/load configs and no-regression are required for production continuity.

**Independent Test**: Wire a loss from a network output pin to a data-loader target feed; save/load a training config; Run once with identical input/target materials and once with distinct paired materials without any mode toggle; confirm both succeed and that a prior mapping-style and reconstruction-style scenario can each be reproduced end-to-end.

**Acceptance Scenarios**:

1. **Given** the palette, **When** the user adds a loss node, **Then** they can connect it to the desired architecture output pin(s) and to the supervision/target feed(s) needed for that loss
2. **Given** at least one valid loss wiring plus data loader and training configuration, **When** the user Runs training, **Then** optimization follows the connected loss path(s)—not a reconstruction/mapping mode selector
3. **Given** multiple validly wired loss nodes each with a weight, **When** training Runs in a single stage, **Then** the optimized objective is the weighted sum of those losses
4. **Given** a training configuration with an ordered loss stage schedule (e.g. stage 1 losses for N steps, then stage 2 losses), **When** training Runs, **Then** the active loss set/weights follow the schedule across stages without requiring a built-in reconstruction mode
5. **Given** the need to reproduce prior reconstruction-style (RAVE-like) multi-term/multi-stage training, **When** the user configures loss types, weights, and stages plus data loader materials, **Then** they can express that recipe fully in the general model (capability-class parity)
6. **Given** the Train panel, **When** the user inspects training controls, **Then** they can edit a general hyperparameter set sufficient to express previously built-in recipe differences (steps, learning-related settings, loss weights/stages, schedule/regularization options that the product exposes)—without a reconstruction/mapping mode control
7. **Given** configured training settings, **When** the user saves a training configuration under a name to the **user training-config library**, **Then** that configuration is stored and can be listed and loaded in other projects/sessions
8. **Given** configured training settings, **When** the user saves or keeps a **project config snapshot**, **Then** that project can later restore those train settings with the project
9. **Given** a saved training configuration (from user library or project snapshot), **When** the user loads it, **Then** the Train panel hyperparameters and related train settings match the saved values
10. **Given** a data loader feeding the same examples into both the network input path and the loss target path, **When** the user Runs with a suitable config and loss wiring, **Then** training proceeds as a same-data (reconstruction-style) objective without selecting a reconstruction mode
11. **Given** a data loader feeding distinct paired materials into input versus loss target paths, **When** the user Runs with a suitable config and loss wiring, **Then** training proceeds as a different-data (mapping-style) objective without selecting a mapping mode
12. **Given** the Train panel, **When** the user looks for reconstruction vs mapping mode, **Then** no such mode selector exists
13. **Given** a scenario that was achievable under prior specs (mapping-style effect clone or reconstruction-style autoencoder train), **When** the user builds it **anew** with data loader + loss nodes + stage schedule + training configuration, **Then** they can obtain an equivalent trainable workflow and creative outcome class (no capability-class regression; old project migration not required)

---

### User Story 5 - Group A Single Box (Priority: P2)

A designer selects exactly one allowed box and creates a group. The resulting group contains that single member plus the usual group interface hubs, enabling later nesting, copies, library save, and collapse without requiring a second member first.

**Why this priority**: Small UX unlock for hierarchy and library workflows; independent of training but part of the same production-quality pass.

**Independent Test**: Select one Conv1D (or other allowed box), create a group, confirm the group exists with one member and hubs; save/reload; ungroup preserves the member.

**Acceptance Scenarios**:

1. **Given** exactly one allowed non–Audio-I/O box selected, **When** the user creates a group, **Then** a named group is created containing that single member and the mandatory group hubs
2. **Given** a one-member group, **When** the project is saved and reloaded, **Then** the single-member group structure is preserved
3. **Given** a one-member group, **When** the user ungroups, **Then** the member returns to the parent scope with connections preserved

---

### User Story 6 - Simplify Palette By Removing Redundant Blocks (Priority: P2)

Temporal-convolution-stack and dedicated linear-projection blocks are removed from the palette because they are expressible with Conv1D (stacked Conv1D for former TCN depth; Conv1D with stride = kernel = dilation = 1 for linear). Designers rebuild those behaviors from Conv1D and activations. **Legacy projects are out of scope** for this feature: no migration or compatibility path is required for old graphs that still contain removed block types or the former Train objective model.

**Why this priority**: Palette simplification reduces architecture-specific surface area and duplicates; secondary to data-loader training but required for the stated production cleanup.

**Independent Test**: Confirm TCN and Linear are absent from the palette; recreate a former TCN-depth stack with multiple Conv1D; confirm new graphs never offer those types.

**Acceptance Scenarios**:

1. **Given** the element palette, **When** the user browses available blocks, **Then** dedicated TCN and Linear element types are not offered for new insertion
2. **Given** the need for a former TCN-style depth, **When** the user stacks multiple Conv1D (and activations as needed), **Then** they can express equivalent temporal depth without a TCN block
3. **Given** the need for a former Linear projection, **When** the user configures Conv1D with stride = kernel = dilation = 1 (and matching channel settings), **Then** they can express the same projection role without a Linear block
4. **Given** this feature’s scope, **When** planning or testing load of pre-change projects, **Then** no automated migration of TCN/Linear nodes or old Train objectives is required (legacy project care is explicitly out of scope)

---

### Edge Cases

- What happens when multiple data loaders exist on one graph? Allowed. The user designates exactly one **active** Data Loader for the Run **in the Train panel**; non-active loaders are ignored for training feeds and data-loader path discovery. If no active designation exists when more than one loader is present, Run is refused with a clear message to choose the active loader. With exactly one loader, it is treated as active by default.
- What happens when a data-loader output is left unconnected? It is ignored for the equal-count gate at Start training. Training may still Run if connected outputs supply a valid trainable path and satisfy equal-count among themselves.
- What happens when example counts match but durations/sample rates differ across outputs? Mixed sample rates remain blocked (existing library rule); duration mismatch within a matched example index is handled by the training windowing rules already used for corpus slices, with a clear refusal if a window cannot be formed.
- What happens when the user connects a data loader only to frozen Gold boxes? Trainable set is empty → Run refused with a clear reason.
- What happens when no loss node is connected (or loss wiring is incomplete)? Run refused with a clear reason that supervision/loss wiring is required.
- What happens when a loss is connected to an output pin outside the data-loader-reachable trainable subgraph? Run refused with a clear diagnostic—training MUST NOT silently optimize an empty/disconnected path.
- What happens when multiple losses are wired with weights but no stage schedule? Training uses a single stage: the weighted sum of all validly wired losses for the full run.
- What happens when a stage schedule references a loss that is missing or unwired? Run refused with a clear message identifying the broken stage/loss reference.
- What happens when a Knob/Trackpad (or similar) source feeds the trainable path but has no data-loader binding? Start training fails unless the user binds a data-loader output or a constant/scalar-copied-across-examples utility.
- What happens when the Train tab is open? Armed on-path elements render at normal opacity; passthrough-only (on path, not armed) and off-path elements render slightly transparent.
- What happens when a node is on the data-loader path but disarmed? It runs as passthrough during training (forward use, no parameter updates).
- What happens when every on-path element is disarmed? Start training is refused with a clear message that nothing is armed to train.
- What happens when a Gold BlackBox is on the data-loader path? It is always passthrough-only (cannot arm); training does not update its weights.
- What happens when live Audio In and data loader both feed the same **external** pin? Inference continues to use the live source only; training uses the data-loader feed for that pin’s training-time input; Data Loader and Loss never contribute to the audible live path; RMS metering on the data-loader cable remains N/A.
- What happens when the user deletes the data loader while training is armed or running? Arming becomes invalid / Run Stopped with a clear message; training cannot continue without a data loader.
- What happens when a saved training configuration references hyperparameters no longer present after a product update? Load applies known fields, ignores unknown fields, and warns about missing fields that fall back to defaults.
- What happens when grouping is attempted on a single Audio Input or Audio Output? Still refused (Audio I/O exclusion unchanged).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST run inference and training against the authored graph in an architecture-agnostic way: no user-facing train/inference workflow may require selecting a named prior architecture (RAVE, steerable, TCN recipe, or equivalent).
- **FR-002**: System MUST provide a Data Loader graph node with a user-controllable number of outputs and user-renamable output labels.
- **FR-003**: When the user starts training (Run), the system MUST require equal example counts across the **connected** outputs of the active Data Loader, and MUST provide utilities to (a) copy/repeat examples so users can equalize counts and (b) supply **constant/scalar (and similar) values copied across all examples** for feeds such as Knob/Trackpad conditioning. Unconnected outputs MUST NOT participate in that equal-count gate. Equal-count MUST NOT hard-block graph editing outside Start training. Each data-loader output MUST own its own ordered example list (bindings from Training Library / capture / import / constant utilities); a shared global Library multi-select MUST NOT be the sole assignment model.
- **FR-004**: System MUST refuse to start training when no usable Data Loader is present, with a clear user-visible reason.
- **FR-005**: System MUST include in the training forward graph nodes on the active data-loader path. System MUST update parameters only for nodes that are **both** on that path **and armed for training**. Nodes on the path that are not armed MUST run as **passthrough only** (used during training, not updated by backprop). Nodes off the path MUST NOT have their parameters updated by the training run.
- **FR-006**: System MUST allow a data-loader connection to external/inference **source inputs**—including Audio In pins and **Knob / XY Trackpad (and similar) source nodes**—including when a live cable already exists on that pin. System MUST NOT allow a data-loader connection on a pin that is already driven by an upstream processing node (internal graph feed). In a chain A→B→C, if A is data-loader-fed on its external input, B and C MUST NOT accept data-loader connections on pins fed by A. For a path to be trained, every external source input required by that path MUST have a data-loader feed (audio list and/or constant/scalar utility); otherwise Start training MUST fail with a clear message.
- **FR-007**: Data-loader cables MUST use a distinct color from ordinary signal cables and MUST NOT display RMS (show N/A / not applicable).
- **FR-008**: System MUST NOT expose a reconstruction-versus-mapping training mode; objective differences MUST follow from data-loader materials, loss-node wiring to architecture output pins, and the user’s general training configuration.
- **FR-009**: System MUST provide loss nodes that users connect to the desired architecture output pins (and to required target/reference feeds) to define what is supervised during training. Multiple validly wired losses MUST be combinable via **per-loss weights** (weighted sum). Training configurations MUST support ordered **loss stage schedules** (changing active losses and/or weights over step ranges). Together with the loss catalog and hyperparameters, this MUST be sufficient to reproduce RAVE/reconstruction-style multi-term and multi-stage training without a built-in architecture mode.
- **FR-010**: System MUST refuse to start training when no usable loss wiring is present, with a clear user-visible reason.
- **FR-011**: System MUST expose a general training parameter surface (hyperparameters and related train settings, including loss weights and stage schedules) sufficient for users to configure runs that reproduce prior built-in recipe behaviors without those recipes being hard-coded product modes.
- **FR-012**: Users MUST be able to save and load named training configurations that capture those hyperparameters and related train settings (including loss weights and stage schedules), in **both** forms: (1) a **user training-config library** reusable across projects/sessions, and (2) a **project config snapshot** stored with the project.
- **FR-013**: Users MUST be able to create a group from a selection of exactly one allowed box (Audio I/O exclusion unchanged).
- **FR-014**: System MUST remove dedicated TCN and Linear element types from the palette for new graphs; equivalent behavior MUST remain achievable with Conv1D configurations (stacked Conv1D; and Conv1D with stride = kernel = dilation = 1).
- **FR-015**: Legacy project migration (old Train objective enum, TCN/Linear instances, pre-change session files) is **out of scope** — the system is NOT required to auto-migrate or preserve loadability of pre-change projects for this feature.
- **FR-016**: System MUST keep the VST as the sole training UI and MUST keep training off the real-time audio thread (constitution: non-blocking live audio during train).
- **FR-017**: The canvas MAY contain multiple Data Loader nodes. For a train Run, the user MUST designate exactly one Data Loader as **active** via a **Train panel picker** (not a per-node Active toggle)—because only one loader is active for a training session. Only the active loader supplies training feeds and defines the data-loader path. Non-active loaders MUST be ignored for that Run. If more than one loader exists and none is designated active in the Train panel, Run MUST be refused with a clear message. If exactly one Data Loader exists, it MUST be treated as active by default.
- **FR-018**: **No regression of capability class**: System MUST preserve the ability for users building **new** graphs to achieve every training and creative outcome class previously available under prior shipped specs (including mapping-style and reconstruction-style workflows and equivalent recipe behaviors) via the generalized graph, data loader, loss nodes (weighted and staged), and training configurations—even though architecture-named modes are removed. This does **not** require migrating or loading old project files.
- **FR-019**: Data Loader and Loss nodes MUST be **training-only**: during live playback they MUST NOT affect the audible signal path; live audio MUST follow ordinary live cables only (e.g. Audio In). Training-time execution MAY use data-loader feeds on pins that also have live connections.
- **FR-020**: System MUST ship a small set of **example graph templates** and **example training configurations** that demonstrate how to reproduce prior mapping-style and reconstruction-style (RAVE-like) recipes. These MUST be clearly presented as examples/templates, not as architecture modes or required Train selectors.
- **FR-021**: While the **Train tab** is active, elements that are armed and on the data-loader path (parameters will update) MUST be drawn at normal opacity; passthrough-only (on path, not armed) and off-path elements MUST be drawn **slightly transparent**.
- **FR-022**: Users MUST be able to arm and disarm elements for training so an element can remain on the training path as passthrough without parameter updates. Newly added processing elements MUST default to **armed**. Start training MUST be refused when no armed on-path element with trainable parameters exists, with a clear message. Frozen **Gold** (BlackBox) elements on the data-loader path MUST always be **passthrough-only** (cannot be armed; weights stay fixed).
- **FR-023**: Attempting to arm a Gold element MUST be refused or unavailable in the UI, with the element remaining passthrough on the training path.

### Key Entities

- **Data Loader**: Graph source used only for training feeds; owns N renamable outputs, per-output example assignments, and equal-count constraints/utilities. When multiple exist, exactly one is selected as **active** for a training session via the **Train panel** (sole loader defaults to active). Ignored during live playback.
- **Data-Loader Cable**: A training-feed connection with distinct visualization and no RMS metering; may coexist on a pin with a live cable; ignored for audible live path.
- **Loss Node**: Graph element that defines supervision; user connects it to chosen architecture output pin(s) and required target/reference feeds; exposes a weight; training may combine multiple losses as a weighted sum and may change active losses/weights across ordered stages. Ignored during live playback.
- **Loss Stage Schedule**: Ordered list of training stages in a training configuration; each stage specifies duration (e.g. step count) and which losses/weights are active—used to express multi-stage recipes such as reconstruction representation then quality.
- **Trainable Subgraph**: Nodes that are on the active data-loader path **and armed**—their parameters are updated during training.
- **Passthrough Element**: A node on the data-loader path that is **not armed**, or a Gold BlackBox on the path; used in the training forward pass without parameter updates.
- **Training Configuration**: Named, reusable bundle of training hyperparameters and related train settings (not architecture identity); may live in the user training-config library and/or as a project config snapshot.
- **Training Material Binding**: Per data-loader output: an ordered list of library/capture/import examples owned by that output; example count is that list’s cardinality for equal-count checks.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In usability checks with at least 5 representative graphs (effect-style and autoencoder-style), testers can start a valid training run without choosing any architecture-named mode, with 100% of successful runs using a data loader.
- **SC-002**: 100% of attempted Runs without a usable data loader are refused with an actionable message (no silent no-op train).
- **SC-003**: In graphs that mix a data-loader-fed branch and an unfed branch, and that mix armed vs passthrough-only nodes on the fed path, post-train inspection shows parameter change only on armed on-path nodes in 100% of sampled runs (passthrough and off-path unchanged).
- **SC-004**: Users can create a one-box group in under 10 seconds from selection to named group visible on the canvas.
- **SC-005**: Users can save a training configuration to the user library and reload it in another project/session, and can also restore a project config snapshot, with all exposed hyperparameter fields restored identically in at least 95% of save/load trials (allowing only documented forward-compatible ignored unknown fields after product updates).
- **SC-006**: New-graph palette insertion attempts for TCN and Linear fail at a 100% rate (types unavailable); equivalent Conv1D constructions remain insertable.
- **SC-007**: Data-loader cables are visually distinguishable from ordinary cables in under 1 second of inspection in guided review, and RMS readouts on those cables are always N/A.
- **SC-008**: Testers can reproduce both same-data and different-data training intents by wiring loss nodes plus changing data-loader material assignments and hyperparameters—without a reconstruction/mapping switch—in guided scenarios with 100% task completion.
- **SC-009**: In a guided regression suite covering prior mapping-style and reconstruction-style **capability classes** (expressed on new graphs), 100% of covered outcome classes remain achievable under the new model (data loader + weighted/staged loss nodes + training configuration; no architecture mode required). Legacy project-file load/migration is not part of this criterion.
- **SC-010**: 100% of attempted Runs without usable loss wiring are refused with an actionable message.
- **SC-011**: Testers can configure a multi-loss weighted sum and a multi-stage loss schedule sufficient to express a reconstruction-style (RAVE-like) training recipe, and complete a short staged run with stage transitions visible in progress, in guided scenarios with 100% task completion.
- **SC-012**: With Data Loader and Loss present alongside live Audio In wiring, live playback produces audible output driven only by the live path in 100% of guided checks (train-only nodes do not alter live sound).
- **SC-013**: Guided review confirms that attempting to attach a data loader to an internal (upstream-fed) pin is refused 100% of the time, while attaching to external/inference sources (Audio In, Knob, XY Trackpad, and similar) remains allowed.
- **SC-014**: Shipped example graph templates and example training configs for mapping-style and reconstruction-style recipes are present and loadable; testers can open each example without selecting an architecture mode, in 100% of guided checks.
- **SC-015**: On the Train tab, armed on-path elements are visually distinguishable from passthrough-only and off-path elements via slight transparency on the latter in 100% of guided checks.
- **SC-016**: Start training fails with an actionable message in 100% of guided cases where a required Knob/Trackpad (or similar) source on the trainable path lacks both a data-loader feed and a constant/scalar utility binding.
- **SC-017**: In a guided passthrough scenario (on-path node disarmed), the node is used during training and its parameters remain unchanged in 100% of checks.
- **SC-018**: 100% of Start training attempts with no armed on-path trainable elements are refused with an actionable message.
- **SC-019**: In guided checks, Gold on-path elements remain un-updatable (passthrough-only) in 100% of train runs; arming Gold is unavailable or refused.

## Assumptions

- “Architecture-agnostic” means removing built-in architecture *modes* and user-facing recipe branding from train/inference workflows; educational docs or example presets that *demonstrate* how to wire a RAVE-like or steerable-like graph remain allowed if clearly presented as user-authored examples, not mandatory product modes.
- Equal example count means equal number of training examples (entries/slices) per **connected** data-loader output at **Start training**, not necessarily identical audio duration of every file before windowing. Example lists are **per output**, chosen from the shared Training Library / capture / import pool. Unconnected outputs are ignored for the gate; editing is not hard-blocked for count mismatch before Run.
- Copy/repeat utilities are user-initiated equalization tools on data-loader outputs (e.g. repeat a single example to match another output’s count), not automatic silent duplication without user action—except that the product may suggest the utility when counts mismatch.
- Exactly one **active** Data Loader per training Run is the production rule; selection is made in the **Train panel** (not a per-node toggle). Multiple loaders may exist on the canvas; only the designated active one feeds training (sole loader is active by default; multiple with no designation → refuse Run).
- General training parameters include the knobs previously buried inside mapping/reconstruction recipes (step counts, learning-rate-related controls, loss weights/terms the product already conceptually supports, schedule/warmup/regularization options exposed in Train)—presented as one configuration surface. Exact field list is finalized in planning; the requirement is completeness relative to formerly built-in recipes, not inventing unrelated research knobs.
- Removing reconstruction/mapping mode does not remove the ability to train autoencoders or input→output maps; it removes the explicit objective enum. Users express those intents with data loader materials + loss nodes connected to architecture outputs.
- **No regression of capability class** is a hard product constraint for **new** authoring: prior outcome classes from shipped specs remain achievable; only the interaction model changes (modes → graph wiring + configs). Caring for old project files / auto-migration is explicitly **out of scope**.
- Loss nodes are first-class palette elements with per-loss weights; training configurations include optional **loss stage schedules**. Exact loss type catalog (spectral, adversarial, feature-matching, variational regularization, etc.) MUST cover what prior reconstruction/RAVE and mapping recipes needed—field/type list finalized in planning, with capability-class parity as the acceptance bar.
- Group-of-one does not change Audio I/O exclusion or group hub rules from the groups feature.
- Legacy TCN/Linear and old Train-objective project migration are **not required** for this feature.
- Cloud and local training destinations both consume the same generalized train package (graph + data loader bindings + training configuration); destination choice is unchanged in spirit from existing Train destination controls.
- Training configurations persist in both a **user training-config library** (cross-project) and optional **project config snapshots**.
- Copyright acknowledgment and library sample-rate consistency rules remain in force.
- RMS-on-cables remains for ordinary live/signal cables; data-loader cables are exempt (N/A).
- Data Loader and Loss are training-only graph elements; they do not participate in the audible live signal path.
- Data Loader replaces **external/inference** feeds only (Audio In, Knob, XY Trackpad, and similar sources—including sharing a pin with a live cable). Pins driven by upstream processing nodes cannot take a data loader; downstream nodes in a chain receive training data through the upstream feed. Every external source required for a trainable path must be data-loader-fed (list and/or constant/scalar-copied utility) or Start training fails.
- Example graph templates and example training configs for prior recipe classes are in scope for this feature and must not reintroduce architecture modes.
- Train-tab visualization uses normal opacity for armed on-path elements and slight transparency for passthrough-only and off-path elements. Arm-for-training selects backprop updates; disarmed on-path elements are passthrough. New processing elements default to armed. Start training requires at least one armed on-path trainable element. Frozen Gold elements are always passthrough-only and cannot be armed.
