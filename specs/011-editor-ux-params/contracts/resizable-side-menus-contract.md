# Contract: Resizable Side Menus

## Purpose

User-resizable left and right chrome panels with clamps so the graph canvas remains usable (FR-002).

## Panels

| Panel | Default (current) | Notes |
|-------|-------------------|--------|
| Left element palette (Library + Project structure) | 200 px | Inside graph area |
| Right Info / tabs column | ~332 px | `PluginEditor` graph area inset |

## Interaction

- Drag splitter / edge handle; width updates live.
- Clamp each menu to `[minMenu, maxMenu]`.
- Enforce `minCanvasWidth` so canvas never collapses.
- After release, width stays stable (no per-frame snap-back).

## Persistence

- Keep for the session at minimum.
- If plugin already has UI layout prefs, persist left/right widths there; else session-only is acceptable (spec assumption).

## Implementation anchors

- `OpenYourBox/Source/graph/NodeRenderer.cpp` — left `BeginChild("ElementPalette", ImVec2(200,…))`
- `OpenYourBox/Source/PluginEditor.cpp` — right column / graph area `-340` / Info `332`

## Non-goals

- Resizing floating popups/modals
- Vertical split of Library vs Project structure beyond ImGui tree collapse
- Per-monitor DPI profile UI (use ImGui/JUCE defaults)
