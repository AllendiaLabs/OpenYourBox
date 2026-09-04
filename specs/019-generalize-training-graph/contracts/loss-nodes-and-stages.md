# Contract: Loss Nodes and Stage Schedules

## Purpose

Define Loss palette elements, wiring, weights, and Training Configuration stage schedules so users can reproduce prior mapping and reconstruction (RAVE-like) recipes without architecture modes.

## Loss node

- Palette element **Loss** with selectable `loss_type`:
  - `mr_stft` — multiresolution STFT (default sizes matching prior mapping recipe)
  - `spectral_distance` — RAVE v1 spectral / AudioDistanceV1 behavior
  - `kl` — variational / bottleneck KL regularization
  - `adversarial` — train-only multi-scale discriminator loss
  - `feature_matching` — feature matching (typically paired with adversarial)
- Pins: type-dependent **prediction** and **target/reference** (KL may attach to bottleneck stats only).
- Property **weight** (float ≥ 0).
- Training-only: ignored on live audible path.

## Wiring rules

- User connects loss prediction to architecture output (or intermediate) pins they want supervised.
- Target feeds typically come from Data Loader outputs (same-data vs different-data is material choice, not a mode).
- Start MUST refuse if no usable loss wiring exists.
- Start MUST refuse if a loss prediction is outside the active data-loader-reachable path.

## Combining losses

### Single stage (default)

Weighted sum of all validly wired losses using each node’s `weight` for `total_steps`.

### Multi-stage schedule

Training Configuration stores ordered stages:

```json
{
  "stages": [
    {
      "name": "representation",
      "steps": 1000000,
      "losses": [
        { "loss_node_id": 21, "weight": 1.0 },
        { "loss_node_id": 22, "weight": 0.1 }
      ]
    },
    {
      "name": "quality",
      "steps": 1000000,
      "losses": [
        { "loss_node_id": 21, "weight": 1.0 },
        { "loss_node_id": 23, "weight": 1.0 },
        { "loss_node_id": 24, "weight": 10.0 }
      ]
    }
  ]
}
```

- Stage `weight` overrides node default when present.
- Missing/unwired `loss_node_id` in a stage → refuse Start with clear message.
- Progress UI SHOULD show current stage name/index.

## Train-only adversarial machinery

- Discriminators exist only inside the worker for stages that include `adversarial` / `feature_matching`.
- Never loaded into the VST live engine or Gold artifact as separate user nodes.
- Related HP (`discriminator_lr`, `update_discriminator_every`, augmentations) live on Training Configuration general options.

## Capability parity (acceptance bar)

Users MUST be able to express:

1. **Mapping-style**: Data Loader input≠target materials + `mr_stft` (or equivalent) + mapping-like HP defaults via example config.
2. **Reconstruction-style**: same-data materials + spectral + KL (+ later adversarial/FM) with two-stage schedule matching prior defaults via example config.

## Implementation anchors

- `OpenYourBox/Source/graph/*` — loss element
- `OpenYourBox/Source/ui/TrainPanel.cpp` — stage editor
- `Backend/train_worker.py` — evaluate schedule; reuse spectral/GAN helpers
- `Tests/test_train_worker.py`
