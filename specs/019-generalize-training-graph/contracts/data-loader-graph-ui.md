# Contract: Data Loader Graph UI

## Purpose

Define Data Loader element behavior, connection legality, cable visuals, binding UX, and equal-count utilities inside the VST graph editor.

## Palette & element

- Insert type: **Data Loader** (`dataLoader`).
- Category: training / data (alongside or near Library-related tools—not live audio I/O).
- Default: ≥1 output pin; user can add/remove outputs (minimum 1).
- Each output has a renamable label (inline or property panel).
- Element is **training-only**: excluded from live audible compilation.

## Per-output bindings

- Each output owns an ordered example list (Training Library entries, capture, import) **or** a constant/scalar utility binding.
- Global Library multi-select MUST NOT be the sole assignment model for training feeds.
- UI MUST allow inspecting/editing bindings per output without requiring Start.

## Equal-count

- **Do not** hard-block graph editing when counts differ.
- At **Start training**, for the **active** Data Loader only: every **connected** output MUST have equal example counts.
- Unconnected outputs are ignored for the gate.
- Provide utilities:
  - Copy/repeat examples on an output to match another’s count.
  - Constant/scalar copied across examples (for Knob/XY-style feeds).
- Product MAY suggest the utility when Start fails due to mismatch.

## Connection rules

| Destination | Data Loader connect |
|-------------|---------------------|
| Empty input pin (no live cable yet) | ALLOW |
| Pin with live **Audio In** or **group-input** hub feed | ALLOW (coexist with live cable) |
| Pin already driven by upstream **processing** node | REFUSE + tooltip reason |
| **Loss prediction** pin | REFUSE (Data Loader → **target** only; see `loss-nodes-and-stages.md`) |
| **Loss target** pin | ALLOW |
| Audio Output / another Data Loader | REFUSE |

In chain A→B→C, if A’s external input is data-loader-fed, B/C MUST NOT accept data-loader on pins fed by A.

### Group hub dedupe

When a member pin has both a live Audio In cable and a Data Loader cable and the member is grouped, `groupBoundaryPorts` / hub creation MUST emit **one** group input hub for that `memberPinId` (both external cables retarget onto that hub; one interior hub→member cable). MUST NOT allocate one hub lane per crossing link.

## Cable visuals

- Distinct color from ordinary signal cables (dedicated theme constant).
- No RMS fill animation; any meter readout shows **N/A**.
- Ordinary live cables keep existing RMS behavior.

## Active loader

- Designation UI lives in the **Train panel** only (see `train-panel-generalized-ux.md`).
- Canvas MUST NOT require a per-node Active toggle for compliance with FR-017.

## Live playback

- Data Loader never contributes to audible output.
- Live path follows live cables only.

## Implementation anchors

- `OpenYourBox/Source/graph/FactoryPalette.h`
- `OpenYourBox/Source/graph/GraphTypes.h` / `NodeGraph.cpp` (`connect`, `groupBoundaryPorts`)
- `OpenYourBox/Source/graph/NodeRenderer.cpp` (cable color / RMS skip)
- `OpenYourBox/Source/dsp/LiveGraphEngine.cpp` (exclude from live)
- `OpenYourBox/Source/ui/TrainingLibraryPanel.*` (binding pickers as needed)
- `Tests/GeneralizedTrainGraphTests.cpp`
