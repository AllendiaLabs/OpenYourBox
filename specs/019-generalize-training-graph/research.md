# Research: Generalize Training Graph

**Feature**: `019-generalize-training-graph`  
**Date**: 2026-09-04  
**Status**: Complete — all Technical Context unknowns resolved

## Decision 1: Replace objective modes with graph + config

**Decision**: Remove `TrainObjective { mapping, reconstruction }` from the Train panel and from start gates. Training intent is expressed only by (1) Data Loader material bindings, (2) Loss node wiring + weights, and (3) Training Configuration (including optional loss stage schedules). Worker no longer branches on `objective`.

**Rationale**: Spec FR-001/FR-008 require architecture-agnostic train without reconstruction/mapping selectors; capability parity is via wiring + configs (FR-018), not legacy modes.

**Alternatives considered**:
- Keep objective enum as a “preset that wires losses” — rejected; reintroduces branded modes.
- Infer objective from graph heuristics (bottleneck present → reconstruction) — rejected; silent architecture coupling.

## Decision 2: Generalized train IPC operation

**Decision**: Introduce a new start envelope operation name `train_graph` (schema_version bumped for cloud). Keep Run/Pause/Stop/progress/event line protocol. Payload includes: `graph_fragment` (full relevant topology including Data Loader + Loss nodes), `active_data_loader_id`, `armed_element_ids`, `data_loader_bindings` (per-output ordered material paths / constants), `loss_schedule` (stages with loss refs + weights + step counts), and general `train_options` (optimizer, device, segment, LR schedules, adversarial helper knobs that are not loss-node-local). Retire `objective` and nested `reconstruction` as required fields; map former reconstruction knobs into general options / loss-node properties / stage entries.

**Rationale**: Local ChildProcess and CloudService already share `train_request()`; one schema avoids dual maintenance. Renaming away from `train_steerable` removes architecture branding from the wire.

**Alternatives considered**:
- Keep `operation: "train_steerable"` and ignore `objective` — rejected; branding remains on the wire and in logs/MLflow tags.
- Separate IPC ops per loss family — rejected; over-fragmented.

## Decision 3: Data Loader node and bindings

**Decision**: New palette element `dataLoader` (training-only). User-controllable output count (≥1), per-output rename, per-output ordered binding list (library pair/clip refs, import paths, or constant/scalar utility values). Equal example-count enforced only at Start for **connected** outputs of the **active** loader. Utilities: copy/repeat examples; constant/scalar copied across examples (for Knob/XY). Multiple loaders allowed; Train panel designates exactly one active (sole loader defaults active).

**Rationale**: Matches FR-002/003/017 and clarifications (per-output bindings, active picker in Train panel only).

**Alternatives considered**:
- Global Library multi-select only — rejected by clarification.
- Per-node “Active” toggle on the Data Loader — rejected; Train panel picker only.
- Auto-equalize counts silently — rejected; user-initiated utilities (product may suggest).

## Decision 4: External-only data-loader connections

**Decision**: Data Loader cables may attach to **external/inference source inputs** (Audio In destination pins, Knob, XY Trackpad, and similar sources), including when a live cable already exists on that pin. Refuse connection when the destination pin is already driven by an upstream **processing** node. Downstream nodes receive training data through the upstream feed. Missing required external feeds on the trainable path → refuse Start.

**Rationale**: FR-006 / SC-013; prevents ambiguous double-feeds on internal edges.

**Alternatives considered**:
- Allow data loader anywhere and override live — rejected; breaks live/train duality and chain semantics.
- Special “conditioning pin” on removed TCN — rejected; sources are first-class (Knob/XY).

## Decision 5: Trainable subgraph = path ∩ arm

**Decision**: Discover the training forward graph from the active Data Loader’s connected outputs (BFS/DFS downstream). Update parameters only for nodes that are on that path **and** `armedForTraining`. On-path disarmed nodes are passthrough (forward, no grad). Off-path nodes unchanged. New weighted processing elements default armed. Gold (`frozenGold`) always passthrough-only (cannot arm). Refuse Start if no armed on-path trainable element. Train-tab opacity: armed-on-path normal; passthrough/off-path slightly transparent.

**Rationale**: Clarifications keep arm as backprop selector; data-loader path replaces Audio-In-reachable snapshot as the participation rule.

**Alternatives considered**:
- Drop arm and train everything on path — rejected by clarification.
- Keep current Audio-In reachable snapshot — rejected; does not match Data Loader model.

## Decision 6: Loss catalog and stage schedules

**Decision**: First-class **Loss** palette elements. Initial catalog sufficient for prior recipes:

| Loss type | Role (parity) |
|-----------|----------------|
| `mr_stft` | Mapping multiresolution STFT (auraloss sizes) |
| `spectral_distance` | RAVE v1 AudioDistanceV1 (fullband + multiband when PQMF present) |
| `kl` | Variational KL / bottleneck regularization |
| `adversarial` | Train-only multi-scale discriminator hinge (stage 2) |
| `feature_matching` | Discriminator feature matching (paired with adversarial) |

Each loss node exposes a **weight** and typed properties (FFT sizes, spectral windows, etc.). Training Configuration holds an ordered **loss stage schedule**: each stage has `steps` and a list of `{ loss_node_id, weight_override? }`. If no schedule is set, a single stage runs the weighted sum of all validly wired losses for `total_steps`. Adversarial discriminators remain **worker-only** (never in VST live engine).

**Rationale**: FR-009/011/018; SC-011; reuse existing `train_worker` spectral/GAN helpers without hard-coding stages in an objective enum.

**Alternatives considered**:
- One mega “Loss” node with checkboxes for all terms — rejected; less graph-explicit, harder to stage.
- Stages as separate graph timeline nodes — deferred; config-side schedule is enough and matches “training configuration” entity.

## Decision 7: Live path ignores Data Loader and Loss

**Decision**: Live Modular Engine and audible path use ordinary live cables only. Data Loader and Loss nodes are excluded from live compilation / processBlock signal flow. Pins may hold both a live cable and a data-loader cable; live uses live; train uses data-loader.

**Rationale**: FR-019 / SC-012; Absolute Law (non-blocking live audio).

**Alternatives considered**:
- Mute live when Data Loader present — rejected; train-while-graph-still-plays is required.

## Decision 8: Data-loader cable visuals

**Decision**: Distinct cable color (dedicated constant, not idle/RMS live colors). No RMS fill; tooltip/readout shows N/A. Ordinary cables keep RMS metering.

**Rationale**: FR-007 / SC-007.

**Alternatives considered**:
- Dashed line only without color change — weaker distinguishability; color + N/A preferred.

## Decision 9: Training configuration persistence

**Decision**: Two stores:
1. **User TrainingConfigLibrary** under plugin user data (named entries: hyperparameters, stage schedule, related train settings — not graph topology or library audio).
2. **Project config snapshot** on the patch/session (same field set) restored with the project.

Load applies known fields, ignores unknown, warns on missing → defaults (edge case).

**Rationale**: Clarification “Both”; FR-012 / SC-005. Mirrors patterns from UserBoxLibrary / presets without coupling audio corpus to configs.

**Alternatives considered**:
- Only project snapshot — rejected; cross-project reuse required.
- Store full graph inside config — rejected; examples ship separately as graph templates.

## Decision 10: Remove TCN and Linear from palette

**Decision**: Remove `tcn` and `linear` from FactoryPalette / new insertion. Document equivalents: stacked Conv1D (+ activations) for TCN depth; Conv1D with stride = kernel = dilation = 1 for Linear. No migration for old projects. Live/worker may delete or leave dead handlers; new graphs must not offer types (FR-014/015). FiLM/conditioning for steerable-like graphs uses Knob/XY + Merge into Conv1D chains (existing control sources), not TCN’s dedicated control pin.

**Rationale**: Spec P2 cleanup; Linear already maps to 1×1 Conv in worker.

**Alternatives considered**:
- Keep types hidden but loadable — unnecessary if legacy migration is out of scope; prefer hard remove from palette for clarity.

## Decision 11: Group of one

**Decision**: Lower minimum selection size for `createGroup` from 2 to 1. Audio I/O exclusion and hub rules unchanged (006).

**Rationale**: FR-013 / SC-004.

**Alternatives considered**: None material.

## Decision 12: Example templates and configs (no modes)

**Decision**: Ship clearly labeled **example graph templates** and **example training configs** for (a) mapping-style effect clone and (b) reconstruction-style (RAVE-like) multi-stage train. Presented as examples/templates in Library/Presets/docs surfaces — **not** Train mode selectors.

**Rationale**: FR-020 / SC-014; Assumptions allow educational examples without reintroducing modes.

**Alternatives considered**:
- Only docs markdown — rejected; must be loadable artifacts.
- Built-in Train “recipes” combo — rejected; modes by another name.

## Decision 13: Cloud package alignment

**Decision**: Cloud job package and `train_runner.materialize_train_request` consume the same `train_graph` schema. Destination Local | Allendia UX unchanged in spirit; entitlement gates unchanged (017/018).

**Rationale**: Constitution Phase 4 + FR-016; Assumptions on shared package.

**Alternatives considered**: Cloud-only objective shim — rejected; dual paths would diverge.

## Decision 14: Gold naming and MLflow tags

**Decision**: Auto-loaded Gold labels and MLflow tags use neutral language (`Trained Graph` / tags like `train`, `graph`) instead of “Trained Steerable” / “Trained RAVE” / `steerable` / `rave` as required product modes. Example templates may mention RAVE-like in **example** names only.

**Rationale**: FR-001 / Story 1 product language.

**Alternatives considered**: Keep RAVE/Steerable Gold names based on graph sniffing — rejected; architecture branding.

## Decision 15: Snapshot / absorb set

**Decision**: Train snapshot / success absorb includes the training forward subgraph (data-loader path processing nodes needed for the module), not only armed nodes — so passthrough on-path helpers remain inside the compiled module / Gold, matching today’s “armed + reachable helpers” spirit but with data-loader reachability. Control sources (Knob/XY) stay outside Gold as today.

**Rationale**: Passthrough-on-path must exist in the exported module; off-path branches stay out.

**Alternatives considered**: Absorb only armed nodes — would break passthrough topology inside Gold.
