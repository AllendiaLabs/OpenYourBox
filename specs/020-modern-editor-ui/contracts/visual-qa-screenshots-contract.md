# Contract: Visual QA screenshots

## Purpose

Window-only stills from the existing Standalone host (spec FR-012, FR-013, FR-015, SC-007, SC-010).

## Host

- Build and launch **OpenYourBox Standalone** (JUCE format already in `CMakeLists.txt`).
- Do not treat Standalone as a customer product or a second Train UI.

## Capture rules

- Editor **window only** (no desktop, menu bar of other apps, Dock).
- macOS: window-id capture (e.g. `screencapture -x -l"$WID"`). Full-desktop grabs are invalid.
- Save under `.ignore/visual-refs/` (gitignored). Abstract or dated names. **Do not commit.**

## Required stills

| Id | Surface | Size |
|----|---------|------|
| `before` | Current editor idle (capture **before** restyle) | plugin + large |
| `idle` | Restyled idle graph + library overlay + inspector | plugin + large |
| `mixed-live-frozen` | Learned Live + Frozen on canvas | plugin |
| `tab-info` … `tab-presets` | Each inspector tab | plugin |
| `analysis` | Analysis well on Info | plugin |
| `modal` | Copyright or error card | plugin |

## Cadence

Every implementation slice that changes pixels MUST add at least one new still of the affected surface. Feature review requires the full set above.

## Reference board

Same folder MAY hold public inspiration stills (VS Code dark materials, ImHex, Tracy). Those stills MUST NOT be used as a layout to copy. They MUST NOT be committed.
