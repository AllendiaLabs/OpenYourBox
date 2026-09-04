# Quickstart: Generalize Training Graph

**Feature**: `019-generalize-training-graph`  
**Purpose**: Runnable validation that the generalized Data Loader + Loss + config model works end-to-end without architecture modes.  
**Not**: full implementation guide (see `tasks.md` after `/speckit-tasks`).

## Prerequisites

- Build OpenYourBox (AU/VST3) with embedded Backend workers.
- Training Library with at least one pair (mapping-style) and enough clips/pairs for reconstruction-style.
- Copyright acknowledgment available for first Train.
- Optional: CloudService only if validating Allendia destination.
- C++ graph/train tests (CMake `BUILD_TESTING`): add `Tests/GeneralizedTrainGraphTests.cpp` as `OpenYourBoxGeneralizedTrainTests` / `add_test(NAME OpenYourBoxGeneralizedTrain …)` next to `OpenYourBoxGroupTests` in root `CMakeLists.txt`. Covers equal-count, active-loader, data-loader connect gates, path∩arm, group-of-one, and palette exclusions (`tcn` / `linear`).
- Python: existing `add_test(NAME OpenYourBoxTrainWorker COMMAND python Tests/test_train_worker.py)` — extend for `train_graph`, weighted losses, stages, and recipe-parity smoke.

## Setup

1. Launch the plug-in in a DAW (master instance).
2. Open tabs: Library, Capture (optional), Train, graph canvas.
3. Confirm Train panel has **no** Mapping/Reconstruction objective combo.
4. Confirm palette has **Data Loader** and **Loss**, and does **not** list TCN or Linear.

## Scenario A — Mapping-style (different data) without modes

1. Build a small Conv1D (+ activation) effect graph with Audio In → … → Audio Out; add Knob if conditioning is used.
2. Insert **Data Loader** with outputs e.g. `input`, `target` (and `cond` if needed).
3. Bind distinct ordered materials per output; equalize counts with utilities if needed.
4. Connect Data Loader `input` to the external pin (empty or alongside Audio In); connect `target` into an `mr_stft` **Loss target**; wire **prediction** from the graph output (live only).
5. Arm at least one on-path trainable element (default armed).
6. In Train panel, select active loader (auto if sole), load mapping-style **example config** or set short `total_steps` / stage weights, Destination Local.
7. Run → expect training progress; live playback still follows Audio In only.
8. On success, Gold auto-loads with neutral naming (not “Trained Steerable” as a required brand).

**Fail checks**: Remove Data Loader → Start refused. Disarm all → refused. Mismatched connected counts → refused until utility. Connect Data Loader to an internal A→B pin → refused. Data Loader → Loss prediction or live → Loss target → refused.

## Scenario B — Reconstruction-style (same data, staged losses)

1. Load **example reconstruction-style graph template** (or build encoder/decoder-style from general blocks + bottleneck as needed).
2. Data Loader feeds input=target materials (same examples on both outputs) plus any Knob scalars via constant utility.
3. Wire `spectral_distance` + `kl` losses; add `adversarial` + `feature_matching` for stage 2.
4. Load reconstruction-style **example training config** (two-stage schedule); optionally expand **Freeze** on stage 2 and check encoder boxes.
5. Run short staged job (reduced steps for smoke) → progress shows stage transitions; encoder frozen in stage 2 when listed.
6. Confirm no reconstruction mode selector was used.

## Scenario C — Train-tab opacity & passthrough

1. On a data-loader path with nodes A, B, C: disarm B; leave A,C armed.
2. Open Train tab → A,C normal opacity; B and off-path slightly transparent.
3. Run → A,C params change; B unchanged; Gold on path cannot arm.

## Scenario D — Group of one

1. Select a single Conv1D → Create Group → group with one member + hubs.
2. Save/reload project → structure preserved; Ungroup restores member.
3. Pre-wire Audio In + Data Loader on the same member pin → Create Group → **one** input hub receives both external cables.

## Scenario E — Config library + project snapshot

1. Edit HP/stages → Save to user training-config library under a name.
2. New project/session → Load that config → fields match.
3. Save project snapshot → reload project → restore snapshot → fields match.

## Scenario F — Cloud (optional)

1. Same package as Local; Destination Allendia with valid entitlement.
2. Job completes → artifact load path unchanged in spirit; schema is `train_graph`.

## Expected outcomes checklist

| Check | Expected |
|-------|----------|
| Architecture mode in Train | Absent |
| TCN/Linear insert | Unavailable |
| Run without Data Loader / loss / arm | Refused with clear message |
| Data-loader cable | Distinct color; RMS N/A |
| Live + Data Loader present | Audible = live only |
| Empty pin + Data Loader | Allowed; processing-driven pin refused |
| Loss pins | DL→target only; live→prediction only |
| Loss weight | Stage schedule only (box has no weight) |
| Stage freeze | Collapsed Freeze tree; `freeze_element_ids` per stage |
| Shared Audio In + DL then group | One input hub for that pin |
| Example templates/configs | Loadable; labeled examples |
| Capability class | Mapping-style and reconstruction-style achievable on **new** graphs |

## References

- Spec: `specs/019-generalize-training-graph/spec.md`
- Data model: `data-model.md`
- Contracts: `contracts/`
