# Contract: Train Panel Generalized UX

## Purpose

Describe Train tab / panel behavior after removing architecture modes: active Data Loader picker, general HP + stages, Start gates, Train-tab graph opacity, examples, and destination controls.

## Removed controls

- Objective combo (**Mapping** / **Reconstruction**) — MUST NOT appear.
- Any Train control that requires choosing RAVE / steerable / TCN as a mode.

## Active Data Loader

- When ≥1 Data Loader exists on the canvas, Train panel shows a **picker** listing loaders (by label/id).
- Exactly one active for a Run.
- Exactly one loader on canvas → treated active (picker may show selection or auto).
- Multiple loaders and none selected → Start refused with message to choose active loader.
- Non-active loaders ignored for path discovery and feeds.

## Hyperparameter surface

- Single general editor covering settings formerly split across mapping vs reconstruction UIs (steps or stage steps, LRs, segment, batch, device, checkpoint/export, RF crop, adversarial helpers, etc.).
- Loss **stage schedule** editor (optional): ordered stages with steps and, for each included Loss, a **weight** (edited in the stage row—not on the Loss box).
- Per-stage **Freeze** tree (collapsed by default): armable elements only; disarmed force-frozen gray; group check cascades; mixed parent state when children differ.
- Save / Load to **user training-config library**; keep/restore **project snapshot**.
- Config JSON: `loss_stage_schedule[].losses[{ loss_node_id, weight }]` and optional `freeze_element_ids` (accept legacy `loss_node_ids` as weight 1.0).

## Arm & Train-tab visuals

- Arm checkbox remains on eligible Blue weighted elements (Parameters / node UI).
- New processing elements default armed.
- Gold: arm control unavailable/refused; always passthrough.
- While **Train tab** is active: armed on-path = normal opacity; passthrough-only and off-path = slightly transparent.

## Start / Stop

- Run uses Destination Local | Allendia (unchanged spirit).
- Preflight gates (clear messages): copyright; active loader; equal-count; required external feeds; loss wiring; schedule refs; ≥1 armed on-path trainable; cloud entitlement when applicable.
- Stop ≠ success auto-load.
- Progress shows loss and stage when multi-stage.

## Examples

- Entry points to load **example graph templates** and **example training configs** MUST be labeled as examples/templates (e.g. “Example: mapping-style”, “Example: reconstruction-style”) — not as modes that replace the general panel.

## Group of one / palette

- Not Train-panel-specific, but same production pass: group menu allows one allowed box; palette omits TCN and Linear.
- Shared Audio In + Data Loader on one member pin → one group input hub (see `data-loader-graph-ui.md`).

## Implementation anchors

- `OpenYourBox/Source/ui/TrainPanel.h/.cpp`
- `OpenYourBox/Source/PluginEditor.cpp`
- `OpenYourBox/Source/graph/NodeRenderer.cpp` (opacity, group menu)
- `OpenYourBox/Source/graph/FactoryPalette.h`
