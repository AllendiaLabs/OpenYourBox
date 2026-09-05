# Contract: Editor chrome layout

## Purpose

Graph-first plugin layout (spec FR-005, FR-006, SC-010). VS Code workbench parts are **not** required.

## Required regions

| Region | Content | Rule |
|--------|---------|------|
| Top chrome | Session line (RF / params), freeze/train status, error/warning | Restyle; keep data; do not clone an IDE status bar |
| Graph (centre) | Node canvas | MUST keep the majority of a typical plugin window |
| Library overlay | Factory + User Library + Project structure | Stay on/over the graph (current NodeRenderer overlay); instrument tree chrome |
| Right inspector | Info, Parameters, Library (training), Capture, Train, Presets | Same tabs; Parameters-on-select unchanged (013) |
| Modals | Copyright, errors | Instrument cards; same blocking behaviour |

## Allowed (not required)

Icon rail, extra left dock, bottom panel, or IDE-like status bar **only if** they improve graph-instrument use at plugin size without making the canvas a thin strip. Default implementation: do not add them.

## Must not

- Activity bar, command palette, editor file tabs, minimap
- Relocating analysis into a fake terminal
- Hiding Train, Capture, Presets, or Parameters
- A second product window besides the VST (Standalone is QA-only)

## Splitters

Existing right-inspector splitter and minimum widths remain in force unless a later change is needed for instrument chrome (must not drop below today’s usable minima).
