# Quickstart: RAVE Architecture & Training

Validation for unified Train + RAVE graphs. Assumes Phase 3 (Library, Capture Pair, mapping Train, Gold auto-load) already works.

## Prerequisites

- OpenYourBox AU/VST3 in a DAW
- Phase 3 Python worker env (PyTorch + auraloss)
- Reconstruction smoke: shorten `stage1_steps` / `stage2_steps` in a test build if needed (product defaults are 1e6+1e6)
- Optional: mono wav clips; stereo host for shape-mismatch check

## Scenario 1 — Latest continuous layout plays (no train)

1. Insert **Latest continuous** layout, pick **mono** or **stereo** to match the insert’s intended width.
2. Confirm PQMF analysis/synthesis, rate conv / TCN stacks, bottleneck, amplitude path.
3. Play audio if channel width matches host; if mono graph on stereo host without adapters, connection/shape is refused.

**Expect**: Illegal domain cables red; delay readout present; existing TCN-only graphs still work.

## Scenario 2 — Original layout distinguishable

1. Insert **Original** RAVE layout (same channel choice).
2. Confirm strided encoder (no residual-dilated v2 stacks) and waveform × loudness + noise branches.

**Expect**: Visibly different from latest continuous; still plays when shapes legal.

## Scenario 3 — Library tags + Single capture + import clip

1. Library: import a **single** wav → entry tagged `unpaired` / kind `clip`; preview.
2. Capture Samples: kind **Single** → Record → Stop → clip appended with Capture source.
3. Capture kind **Pair** still adds x/y pairs as today.

**Expect**: One library list; tags visible; no second library product.

## Scenario 4 — Objective gates

1. Select only unpaired clips; objective **mapping** → Run **errors**.
2. Set objective **reconstruction**; select a **pair** → both x and y used; optionally add clips.
3. Change objective: warnings/filters update; last-used objective restores after close/reopen on this instance.

**Expect**: FR-009*; SC-011.

## Scenario 5 — Reconstruction refuse invalid graph

1. Objective reconstruction; armed TCN effect graph (no bottleneck).
2. Run.

**Expect**: Refused with missing-bottleneck (or equivalent) reason; mapping still available.

## Scenario 6 — Non-blocking reconstruction + Stop

1. Valid RAVE graph armed; valid corpus; copyright ack.
2. Run reconstruction; playback continues on prior model; panel shows stage + loss.
3. Optional: load a hear-while-training checkpoint (audio may change); Stop.

**Expect**: Stop is not success auto-load; UI stays usable.

## Scenario 7 — Success Gold forward / encode / decode

1. Complete both stages (or test-shortened success path).
2. Armed chain → Gold with forward; encode/decode ports; fidelity on Gold.
3. Sweep fidelity without Unfreeze; Unfreeze keeps weights and fidelity.

**Expect**: Timbre transfer on out-of-domain input via forward; SC-006/007/010.

## Scenario 8 — Mapping unchanged

1. Objective mapping; pair only; TCN steerable graph; short train.

**Expect**: Phase 3 recipe (~2500 steps, MR-STFT 32/128/512/2048) unchanged.

## Scenario 9 — Mixed SR / channel mismatch

1. Reconstruction selection with mixed sample rates, or clip channels ≠ graph width.

**Expect**: Blocked with clear message.

## References

- `spec.md`
- `research.md`, `data-model.md`, `plan.md`
- `contracts/rave-graph-ui-contract.md`
- `contracts/unified-train-ipc.md`
- `contracts/library-capture-extension.md`
- `contracts/rave-gold-runtime-contract.md`
