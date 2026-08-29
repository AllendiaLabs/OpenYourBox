# Contract: Element Palette Categories

## Purpose

Define Factory palette grouping so Effects and Neural / Sequence types are discoverable without changing insert payloads or persistence.

## Surface

- **UI**: Factory / element list in `NodeRenderer` (or equivalent palette host)
- **Payload**: unchanged `"OPENYOURBOX_NODE_TYPE"` + `NodeType` / type string
- **Non-goals**: Separate windows, search redesign, drag semantics changes

## Categories (v1)

| Category ID | Label | Types (insertable) |
|-------------|-------|--------------------|
| `effects` | Effects | Reverb, ExpDecayReverb, FilteredNoiseReverb, FIRFilter, ModDelay |
| `neural_sequence` | Neural / Sequence | LSTM, RNN |
| `default` / existing | Existing section(s) | Current palette types (Linear, Conv1D, Activation, TCN, …) |

## Invariants

1. Every insertable `NodeType` appears in exactly one category section.
2. Section headers are visible without scrolling past an unlabeled mega-list when the palette is at default size (SC-004).
3. Drag/drop and click-insert behavior match pre-feature Factory interactions.
4. Audio I/O, BlackBox, and non-insertable types remain excluded from the palette as today.

## Acceptance Checks

- Opening the element menu shows an Effects section containing all five DDSP types.
- Neural / Sequence lists LSTM and RNN.
- Inserting from a category creates the same `GraphNode` as `NodeGraph::addNode` for that type.
