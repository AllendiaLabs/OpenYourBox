# Data Model: Modern Editor UI

Visual restyle only. No new graph persistence. Tokens are compile-time constants (or an equivalent immutable table) applied once when the ImGui context is created and when DPI scale changes.

## Entity: Colour token

Named dark-mode colour used by chrome and canvas. RGBA, opaque unless a documented overlay (e.g. selection 20%).

| Token | Role | Distinct from |
|-------|------|----------------|
| `surface.page` | Full-window / host clear | light/day pages |
| `surface.canvas` | Graph well (slightly lifted, still dark) | `surface.page`, `surface.panel` |
| `surface.panel` | Inspector and overlay library | canvas |
| `surface.raised` | Buttons, fields, wells | panel |
| `text.primary` | Labels, trees, pins, numbers | |
| `text.muted` | Secondary / disabled | primary |
| `accent` | Focus ring, active tab edge, primary action | **Live**, Frozen, danger |
| `live` | Learned Live box fill (cool) | accent, Frozen |
| `frozen` | Frozen / Black Box fill (warm) + lock | Live, accent |
| `family.audioIo` | Audio / group I/O | other families |
| `family.conditioning` | Knob Input, XY | other families |
| `family.helper` | Helpers / DSP utilities | other families |
| `family.trainOnly` | Data Loader, Loss | other families |
| `danger` | Illegal cable, blocking errors | accent, Live, Frozen |
| `warning` | Graph warning / muted-input caution | danger |
| `border` | 1 px separators | |

**Validation**:
- All surfaces stay dark (relative luminance well below a light page).
- Pairwise arm’s-length distinct: Live, Frozen, accent, danger.
- Family roles remain unique vs each other and vs Live/Frozen.
- `chromeColourForType` reads these tokens (Frozen state → `frozen`; else family).

**State**: Immutable for v1 (no user theme switch; light mode out of scope).

## Entity: Type ramp

| Style | Face | Weight | Use |
|-------|------|--------|-----|
| `type.body` | Inter | Regular 400 | trees, fields, numbers, pins, status |
| `type.strong` | Inter | SemiBold 600 | titles, tabs, primary buttons |

**Validation**: Exactly two weights. Toolkit default font must not appear after `ImGuiHost` font load.

## Entity: Icon glyph

Codepoint in the merged icon font (Phosphor Regular). One outline style.

**Validation**: No Codicon font files. No emoji as the rail/action default.

## Entity: Instrument control

A visible widget class that MUST use instrument chrome:

- Tab / section switch
- Tree row (Factory / User Library / Project structure)
- Button (primary / secondary / danger)
- Text/numeric field
- Checkbox
- Dry/Wet (and similar sliders)
- Knob Input
- XY Trackpad
- Modal card (copyright, error)
- Graph box (name + pins) and cables

**State**: visual only; hit-testing and callbacks stay as today.

## Entity: Windowed-host still

Gitignored file under `.ignore/visual-refs/`.

| Field | Rule |
|-------|------|
| Capture | Editor window only |
| Host | JUCE Standalone (not a DAW) |
| Sizes | Typical plugin (~900×600 class) and large window |
| Required set | before; idle; mixed Live/Frozen; each inspector tab; analysis; one modal |

**Relationships**: Stills evidence token + instrument-control application; they are not build inputs.

## Mapping from current code

| Today | After |
|-------|--------|
| `ImGui::StyleColorsDark()` | `VisualLanguage::applyStyle()` |
| `liveBlueColour` etc. in `GraphTypes.h` | Token-backed RGB (enum names unchanged) |
| Stock `BeginTabBar` / `Button` / `Slider` | `InstrumentWidgets` |
| Default ImGui font | Inter + merged icons from BinaryData |
| Top `TextDisabled` session line | Same data, instrument chrome |
| Library overlay in `NodeRenderer` | Same placement, instrument tree rows |
