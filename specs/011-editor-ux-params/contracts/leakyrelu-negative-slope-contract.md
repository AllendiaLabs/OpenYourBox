# Contract: LeakyReLU Negative Slope

## Purpose

Expose LeakyReLU negative slope as an editable element parameter (FR-001).

## Property

| Key | Kind | Default | Range | On invalid |
|-----|------|---------|-------|------------|
| `negative_slope` | real | `0.01` | `[0, 1]` | Refuse edit; keep prior; message (no clamp) |

- Shown when activation choice is LeakyReLU (index 3); hidden or disabled for other activations.
- Persists on the node; included in freeze/train architecture snapshots like `gain`.

## Runtime wiring

| Path | Behavior |
|------|----------|
| Live (`LiveGraphEngine::applyActivation`) | Use property value instead of hardcoded `0.01` |
| TCN (`TCNModel`) | Same when activation is leaky ReLU |
| `Backend/train_worker.py` `_activation` | Construct `LeakyReLU(negative_slope=…)` from element |
| `Backend/freeze_worker.py` | Same |

Discriminator/RAVE helper code that hardcodes `0.2` for non-element paths MAY remain; element graphs MUST be property-driven.

## Implementation anchors

- `NodeGraph.cpp` activation factory properties (alongside `gain`)
- `LiveGraphEngine.cpp`, `TCNModel.cpp`
- `Backend/train_worker.py`, `Backend/freeze_worker.py`
- `NodeRenderer` property visibility

## Non-goals

- Changing PReLU learned slope into this field
- Global default preference overriding per-element values
