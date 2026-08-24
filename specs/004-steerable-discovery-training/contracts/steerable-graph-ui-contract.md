# Contract: Steerable Graph & Training UI

## Purpose

UI/graph editor behaviors for FiLM/residual/PReLU, arm/disarm, Weights property, copyright modal, Capture Samples menus, and ml_forge-style Train panel — all inside the plug-in window.

## Anchors

- `OpenYourBox/Source/graph/GraphTypes.h`, `NodeGraph.*`, `NodeRenderer.*`
- `OpenYourBox/Source/ui/` (Capture menu, Train panel, copyright modal)
- `OpenYourBox/Source/PluginEditor.*`
- Phase 2.2: Knob Input, XY Trackpad, Merge (`specs/003-signal-analysis-controls/contracts/graph-control-ui-contract.md`)

## TCN / Activation Extensions

| Control | Element | Behavior |
|---------|---------|----------|
| FiLM input pin | TCN | Conditioning input; connect from Knob, XY, or Merge |
| Dilation growth | TCN | Integer **Dilation growth** slider/stepper (RONN naming), default **2**, typical range 1–16; live readout of dilations `1→G→G²→…` and RF (samples/ms); optional presets that set G to 2 / 8 / 10 |
| Residual checkbox | TCN | Enables residual path when checked (on for steerable-nafx parity) |
| PReLU | Activation, TCN activation property | Selectable alongside existing activations |
| Gain | Activation, TCN | Unchanged Phase 2.2 slope control |

Shape violations: illegal FiLM/audio connections refuse with red cable + tooltip (existing shape integrity).

## Arm / Disarm for Training

- Shown only on elements with **trainable parameters**.
- Default: **armed**.
- Disarmed → excluded from train snapshot and from auto-load Gold absorption.
- Control sources (Knob, XY, Audio In/Out, etc.): **no** arm control; never absorbed.

## Weights Property

- Random weights → display `seed N`.
- File/train-backed → display path.
- Browse → native file chooser → load compatible weights; reject incompatible with error; keep prior weights.
- Gold BlackBox after train → path to trained artifact.
- After Unfreeze → restored nodes keep trained weights/path until randomize or retrain/reload.

## Copyright Modal

- Blocking before first Train when no local acknowledgment.
- Checkbox certification text (original / royalty-free).
- On confirm: persist local log; enable Train when other gates pass.
- Train button grayed until ack + ≥1 pair + ≥1 armed trainable element.

## Capture Samples & Training Library UI

See **`training-library-ui-contract.md`** for the full library browser (v1 + long-term).

**Master (summary)**
- Dedicated Library panel: list+detail, multi-select, import, rename, delete, x/y preview; Capture **adds** pairs.
- Discover/pair peer, assign Clean/Processed, bypass toggle (default on), Record start/stop as ingest into library.
- Train panel shows `N pairs selected · ~T min` with link to Library.

**Slave**
- Reduced menu: pair status, role, bypass, record sync indicator as applicable.
- No library ownership / full Train workflow.

## Train Panel (master, ml_forge-style)

| Control | Behavior |
|---------|----------|
| Run | Start job if gates pass |
| Pause | Suspend optimization |
| Stop | End job; no auto-load |
| Loss / step | Live readout from worker progress |
| Info | RF-aware crops; `Train window ≈ N s`; selected pair count (no raw segment-length editor in v1) |

During job: UI remains usable; audio uses prior model.

## Auto-Load & Unfreeze

- On train `success`: replace armed trainable chain with Gold BlackBox; keep control sources Blue and wired for **c**.
- Unfreeze: Phase 2 behavior + **retain trained weights** on restored Blue nodes until randomize/retrain.

## Non-Goals

- Embedding ml_forge / Dear PyGui application
- Cloud training / marketplace (Phase 4)
- User-editable train hyperparameters in v1

## Implementation Anchors

- Graph fields (FiLM pin, residual, PReLU, `dilationGrowth`, arm, Weights): `OpenYourBox/Source/graph/GraphTypes.h`, `OpenYourBox/Source/graph/NodeGraph.cpp`
- Arm / Weights / growth UI: `OpenYourBox/Source/graph/NodeRenderer.cpp`
- Live FiLM / residual / PReLU / Gⁿ: `OpenYourBox/Source/dsp/TCNModel.cpp`, `OpenYourBox/Source/dsp/LiveGraphEngine.cpp`
- Train panel + copyright modal: `OpenYourBox/Source/ui/TrainPanel.cpp`, `OpenYourBox/Source/ui/CopyrightModal.cpp`
- Weight browse/load: `OpenYourBox/Source/dsp/WeightLoader.cpp`
