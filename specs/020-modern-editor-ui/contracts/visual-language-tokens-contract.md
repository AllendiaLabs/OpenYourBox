# Contract: Visual language tokens

## Purpose

Single source of named dark-mode colours, type, and icons for the OpenYourBox editor (spec FR-001, FR-007, FR-008, FR-010, FR-016).

## Apply

- `ImGuiHost::newOpenGLContextCreated` (and DPI/scale changes) MUST load Inter Regular + SemiBold, merge the icon font, then call `VisualLanguage::applyStyle()`.
- MUST NOT leave `ImGui::StyleColorsDark()` as the effective theme.
- imgui-node-editor `Style` MUST be updated from the same tokens (background, grid, node, pin, select).
- `chromeColourForType` MUST return token colours (Frozen → `frozen`; Live families → family tokens). `NodeState` enumerator names stay.

## Accent vs canvas roles

- `accent` is for chrome focus, active tab edge, and primary actions only.
- `accent` MUST NOT be used as Live or Frozen box fill.
- Illegal links stay `danger` (unambiguously red).

## Type

- Body text, numbers, pins, status: Regular.
- Titles, tabs, primary buttons: SemiBold.
- Toolkit default font MUST NOT remain in the atlas as the default face.

## Icons

- Actions use the bundled outline icon font.
- MUST NOT link or embed VS Code Codicons.

## Non-goals

- User-selectable themes or light mode
- Renaming `liveBlue` / `frozenGold` persistence keys
