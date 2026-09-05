# Contract: Instrument controls

## Purpose

Every visible control uses one custom instrument language (spec FR-009, SC-009). Layout stays ImGui; widgets are not stock toolkit defaults and not code-editor settings clones.

## Control classes

| Control | Instrument treatment |
|---------|----------------------|
| Tabs | Quiet active indicator (edge or underline), dim inactive, SemiBold label; not raised toolkit tabs |
| Tree rows | Full-row hover/selection well; no default bright-blue `Header` fill as the look |
| Buttons | Crafted fill; primary uses `accent`; danger uses `danger` |
| Fields | Inset well on `surface.raised`; token border |
| Checkbox | Crafted mark; token colours |
| Dry/Wet (and sliders) | Thin track, round thumb, accent fill — not the default thick bar |
| Knob Input | Circular knob + value arc in the modern family |
| XY Trackpad | Dark rounded well + visible handle |
| Graph box | Slim name + pins; family/Live/Frozen fill from tokens; Frozen lock remains |
| Cables | Family/Live colours; illegal = `danger` |
| Modals | Card on `surface.panel`; same accept/dismiss behaviour |

## Hit testing

Hit targets MUST remain at least as large as today (spec edge case). Visual restyle must not steal pin vs box drag.

## Non-goals

- Changing parameter semantics, freeze, train, or library behaviour
- A separate performance-view editor
