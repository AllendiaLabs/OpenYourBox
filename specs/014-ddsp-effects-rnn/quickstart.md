# Quickstart: DDSP Effects & Recurrent Layers

Validation guide for `014-ddsp-effects-rnn`. Implementation details belong in `tasks.md`; this file lists runnable checks that prove the feature end-to-end.

## Prerequisites

- Build OpenYourBox VST (Debug or Release) with LibTorch linked as usual
- Python env with Backend requirements (`Backend/requirements.txt`)
- Host or standalone runner that loads the plugin and plays audio
- Optional: pytest for worker tests

## Setup

```bash
# From repo root — adjust generator/config to your usual workflow
cmake --build build --target OpenYourBox   # or project-standard target
python -m pytest Tests/test_freeze_worker.py Tests/test_train_worker.py -q
```

C++ live coverage (after tests are added for this feature):

```bash
# Project-standard CTest / Catch target that includes LiveGraphEngineTests
ctest --test-dir build -R LiveGraphEngine -V
```

## Scenario A — Effects palette & ExpDecayReverb (P1)

1. Open the graph editor Factory / element menu.
2. Confirm **Effects** lists Reverb, ExpDecayReverb, FilteredNoiseReverb, FIRFilter, ModDelay.
3. Graph: Audio Input → ExpDecayReverb → Audio Output.
4. Play audio; increase `decay`; confirm longer tail.
5. Disable `add_dry`; confirm wet-only.
6. Edit a scalar while playing; confirm no audio stop / restart.

**Expect**: SC-001 / SC-005; FR-001 / FR-002 / FR-007 / FR-008.

## Scenario B — FIRFilter / FilteredNoiseReverb grids (P2)

1. Place FIRFilter (and separately FilteredNoiseReverb) in a minimal chain.
2. Edit `n_frames`, `n_filter_banks`, `window_size`; use init/randomize.
3. Confirm no cell-by-cell painter is required.
4. Attempt an illegal dimension/connection; confirm refuse + explanation.

**Expect**: FR-003 / FR-004 / FR-009; see [ddsp-effects-contract.md](contracts/ddsp-effects-contract.md).

## Scenario C — ModDelay (P2)

1. Audio In → ModDelay → Audio Out.
2. Set `depth_ms = 0`, note sound; raise `depth_ms` and sweep `phase`.
3. Toggle `add_dry`.

**Expect**: Audible modulation change; FR-005.

## Scenario D — Reverb IR blend & empty IR (P2)

1. Place Reverb; play with internal IR only.
2. Connect a non-empty IR source; sweep `ir_blend` fully internal → fully external.
3. Connect an empty/zero-length IR; confirm fallback to internal, recoverable warning, audio continues.
4. Set `reverb_length` above the live-safe threshold; confirm soft warning, value retained, audio continues.

**Expect**: FR-006 / FR-006a / FR-006b / SC-008; [data-model.md](data-model.md).

## Scenario E — LSTM / RNN (P3)

1. Factory → **Neural / Sequence** → place LSTM (and RNN).
2. Confirm properties: `hidden_size`, `bidirectional` (default off), `bias` (default on), activation menu + gain (+ negative slope for LeakyReLU); no `num_layers` / `input_size`.
3. Legal feature path: process a buffer sequence; unidirectional out channels = `hidden_size`; enable bidirectional → `2 * hidden_size`; time length unchanged.
4. Stack two LSTMs; confirm deeper recurrence without multi-layer control.
5. Live: consecutive buffers carry state; rebuild or randomize → reset (automated test preferred).
6. Freeze a legal selection including LSTM/RNN; confirm Gold swap without UI freeze/stop.
7. Illegal channel cable → red refuse + tooltip.

**Expect**: FR-010–FR-011e / FR-012; [recurrent-layers-contract.md](contracts/recurrent-layers-contract.md).

## Scenario F — Persist & workers

1. Save a project containing all seven new types; reload; confirm types, cables, properties.
2. Freeze worker: minimal fragment per type scripts successfully (`test_freeze_worker`).
3. Train worker: weighted types expose parameters; LSTM/RNN forward shapes match contract (`test_train_worker`).

**Expect**: FR-013–FR-015 / SC-006; [freeze-train-type-registry-contract.md](contracts/freeze-train-type-registry-contract.md).

## Scenario G — Attribution

1. Confirm `NOTICE` includes Magenta/DDSP Apache 2.0 behavioral-reimplementation note.

## Pass criteria summary

| ID | Check |
|----|-------|
| SC-001 | Distinct effect character within ~30 s of minimal In→effect→Out |
| SC-002 | All listed scalar/dimension params editable in Properties |
| SC-003 | Legal freeze with new types succeeds without host restart |
| SC-004 | LSTM/RNN findable under Neural / Sequence labels |
| SC-005 | Live param change audible in one interaction |
| SC-006 | Save/reload restores all seven types |
| SC-007 | Illegal connections refused with explanation |
| SC-008 | Long reverb length warns, does not clamp |
