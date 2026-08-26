# Contract: Edit History (Undo/Redo)

## Purpose

Session-scoped **undo/redo** for live plugin patch edits, including treating **preset load** as a single history step and restoring **current-preset / dirty** association with that step.

## Scope of undoable changes

| Included | Excluded |
|----------|----------|
| Add/remove/move boxes, links, groups | Canvas pan, zoom, selection |
| Box layout position changes | In-flight train/freeze jobs (mid-progress) |
| Parameter / knob / XY value changes | Catalog-only mutations (rename/delete preset file) |
| Weight randomization / similar one-shot sonic actions | Host project load (host owns that restore) |
| Completed freeze/train patch results once applied | History across plugin instance destruction |
| Preset **load** into the instance | |

## Transactions

1. **Discrete action** — one `HistoryStep` (before/after `PatchSnapshot` + CurrentPreset)
2. **Gesture** — `beginGesture` … `endGesture` coalesces into one step (knob/XY/slider/node drag)
3. **One-shot sonic** — e.g. randomize = one step restoring prior weights/sonic state
4. **Preset load** — one step; label may include preset name; stores prior/next CurrentPreset
5. **Suppress** — while applying a snapshot for undo/redo/host restore/preset apply, do not push new steps

## Stack rules

- Max depth **≥ 50** (default 50); drop oldest when exceeded
- Undo applies step `before` + `beforeCurrentPreset`; Redo applies `after` + `afterCurrentPreset`
- New push after undo **clears redo**
- `canUndo` / `canRedo` drive UI enablement (buttons + shortcuts)
- Empty undo/redo = no-op with disabled affordances (FR-017)

## Input

| Path | Binding |
|------|---------|
| UI buttons | Undo / Redo in editor chrome |
| Shortcuts | Platform undo/redo (Cmd/Ctrl+Z; Shift+Cmd/Ctrl+Z or Ctrl+Y) |

Both paths MUST call the same `EditHistory` API.

## Apply path

- Must use shared `PatchSnapshot` apply (GUI thread + atomic runtime publish)
- Must not stop host transport
- Target: user-visible update within 1 s for typical patches (SC-005)
- View-only changes must not create steps or mark dirty (FR-007b)

## Failure modes

| Case | Behavior |
|------|----------|
| Apply fails during undo | Leave state consistent; prefer keeping current live patch; no crash |
| Gesture cancelled | No history push |
| Load preset failed | No history push |

## Non-Goals

- Persisted undo across sessions
- Selective per-node undo UI (optional labels on steps are enough)
- Collaborative/multi-instance shared history
- Undoing pan/zoom/selection

## Implementation anchors

- NEW: `OpenYourBox/Source/state/EditHistory.h/.cpp` (JUCE `UndoManager` façade)
- Wire from `PluginEditor` (shortcuts + buttons) and mutation sites in graph/UI
- Preset load orchestration pushes one step around successful apply
- Randomize / completed train-absorb / freeze completion push one step after patch lands
