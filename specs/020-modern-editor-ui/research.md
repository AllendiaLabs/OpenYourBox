# Research: Modern Editor UI

## Decision 1: Layout stays graph-first (no code-editor workbench)

**Decision**: Keep today’s information architecture: Factory/User Library as an overlay on the graph canvas, centre graph as the dominant surface, right inspector tabs (Info, Parameters, Library, Capture, Train, Presets), top session/status chrome. Do **not** add an activity-bar icon rail, bottom terminal panel, or IDE status bar.

**Rationale**: Spec FR-005 and the superseded “full VS Code reconstruction” clarification: VS Code is inspiration for materials only. A plugin window is small; a rail or extra docks steal the graph. The current overlay library already keeps the canvas large. Parameters-on-select (013) stays on the right inspector.

**Alternatives considered**:
- VS Code activity bar + sidebars + panel: rejected (spec SC-002/SC-008/SC-010; wastes plugin width).
- Permanent left dock for Library: optional later if overlay proves cramped; not required; would shrink the graph.
- Moving analysis to a bottom panel: rejected; analysis stays with Info / selected box.

## Decision 2: Typeface is Inter Regular + SemiBold

**Decision**: Bundle [Inter](https://github.com/rsms/inter) (SIL Open Font License 1.1), weights **400 Regular** and **600 SemiBold** only. Load both into the ImGui atlas from `BinaryData` in `ImGuiHost`. Use Regular for body/trees/fields/numbers/pins; SemiBold for titles, tabs, and primary buttons. Ship `OFL.txt` next to the fonts and a NOTICE block.

**Rationale**: Spec FR-010 — one contemporary UI sans, two-weight ramp, bundled so every DAW looks the same. Inter is a proven dense UI face, OFL-bundlable with Apache-2.0 software, and is not the reverted Bounded / Computer Says No pair.

**Alternatives considered**:
- System UI fonts (SF/Segoe): rejected (spec: looks different per OS/host).
- IBM Plex Sans / Source Sans 3: acceptable OFL/SIL alternatives if Inter files are awkward to subset; Inter remains the default.
- Bounded + Computer Says No: rejected (spec out of scope).

## Decision 3: Icons are Phosphor Regular (MIT), not Codicons

**Decision**: Bundle a **Phosphor Regular** outline TTF (MIT) as a merged ImGui icon font (`ImFontConfig.MergeMode`). One style on actions (freeze, folder, warning, etc.). Do not ship VS Code Codicons. NOTICE + license text.

**Rationale**: Spec FR-016 — generic open UI icons, TTF-friendly for ImGui, visually distinct from Codicons. Phosphor ships a font; Lucide is primarily SVG and would need a one-off TTF build.

**Alternatives considered**:
- Lucide TTF subset (ISC): fine if a maintained TTF exists; extra build step.
- Font Awesome: heavier, more “web app.”
- Codicons: rejected (spec).
- Emoji / original pictograms: rejected (spec).

## Decision 4: OpenYourBox-owned dark token set (not a VS Code clone)

**Decision**: Introduce named tokens in `VisualLanguage` (see `data-model.md`). Values are a **dark instrument family**: canvas slightly lifted over a darker page, teal **accent** for chrome focus, **Live** cooler blue, **Frozen** warmer gold, family roles remapped to sit in the same family. Illegal-cable **red** stays a safety colour, not a Tokyo/VS Code token. Do not copy VS Code Dark Modern hexes as identity; they may inform value (dark vs darker) only.

**Rationale**: Spec FR-001/FR-002/FR-007/FR-008. Constitution Live=Blue / Frozen=Gold **roles** stay; RGB may move. Accent must not equal Live fill (arm’s-length test).

**Alternatives considered**:
- Keep current `liveBlueColour{100,180,255}` and only recolour ImGui: fails “one family” and still reads as default toolkit blue.
- Use VS Code `#0078D4` as accent and as Live: collision (rejected in clarify).
- Pale plaza / TokyoS: reverted; out of scope.

## Decision 5: Custom ImGui draw-list widgets, not StyleColorsDark + stock controls

**Decision**: Stop calling `ImGui::StyleColorsDark()` as the visible theme. `VisualLanguage::applyStyle()` sets rounding, padding, separators, and maps `ImGuiCol_*` to tokens (never saturated default-toolkit blue fills). **InstrumentWidgets** wrap or replace tabs, tree rows, buttons, inputs, checkboxes, Dry/Wet slider, Knob, and XY with `ImDrawList` chrome (inset wells, thin tracks, circular knobs, crafted hover). imgui-node-editor `Style` colours/rounding follow the same tokens. Default `ImGui::Button` / `BeginTabBar` MUST NOT remain the unstyled visible default.

**Rationale**: Spec FR-004/FR-009 — recolouring stock widgets fails. Instrument look includes trees and tabs. Layout stays ImGui immediate-mode (60 FPS, existing host).

**Alternatives considered**:
- JUCE LookAndFeel for the editor: rejected (constitution/architecture is ImGui + node editor).
- Third-party ImGui theme pack: still stock widgets; plus alien identity.

## Decision 6: Standalone host + window-only stills

**Decision**: Use the existing `FORMATS … Standalone` target. Capture **window-only** stills into `.ignore/visual-refs/` (already gitignored via `.ignore/`). On macOS: `screencapture -x -l"$WID"` (or equivalent window-id capture). Include a **before** still before the first restyle. Every visual slice adds at least one new still; include typical plugin size (~900×600 class) and a large window. Do not commit stills.

**Rationale**: Spec FR-012/FR-013/SC-007/SC-010. Constitution “no standalone **product**” is satisfied by treating Standalone as QA-only (plan Complexity Tracking).

**Alternatives considered**:
- Headless framebuffer dump: more setup, less “what the user sees.”
- Full-desktop capture: invalid per spec.

## Decision 7: Keep `NodeState` enumerators; remap RGB only

**Decision**: Do not rename `NodeState::liveBlue` / `frozenGold` or persistence strings `frozen_gold`. Change `liveBlueColour`, `frozenGoldColour`, and sibling family colours in `GraphTypes.h` (or route them through `VisualLanguage` tokens). Lock icon stays.

**Rationale**: Avoid a graph-wide rename; spec allows hue change, not role change.

**Alternatives considered**: Rename enums to `live` / `frozen`: large churn, no user value.

## Decision 8: Tests vs visual QA

**Decision**: CTest covers token tables (unique roles, Live ≠ Frozen ≠ accent ≠ danger, both font weights registered). Beauty and “not a toolkit / not VS Code” are quickstart + stills, not pixel-diff CI (private stills are gitignored).

**Rationale**: Spec SC-001–SC-010 are reviewer judgements. CI cannot store the reference board.
