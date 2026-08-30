# Contract: Factory Palette — TorchScript Load

## Purpose

Define how users discover and place the external checkpoint loader in the Factory palette.

## Palette

| Item | Value |
|------|--------|
| Label | `TorchScript Load` |
| Category | Prefer **Neural** / **Models** (or existing “Layers / Utilities” if no Models group); must not bury under Effects |
| Payload | Existing `OPENYOURBOX_NODE_TYPE` drag/double-click insert |
| Created type | `NodeType::blackBox` |
| Initial origin | `BlackBoxOrigin::externalLoad` |
| Initial path | empty |
| Initial pins | Main audio input + main audio output only |
| Initial status | `empty` (dry passthrough + choose-file prompt in properties) |

## Non-goals

- Marketplace or remote download browser
- Auto-adding Control or latent pins before a successful load
- Weight randomization entry points for this element

## Acceptance

- Palette lists **TorchScript Load** and placing it creates one canvas node with empty path (FR-001).
- Node is visually Gold / opaque frozen class when shown (FR-008), including empty state styling consistent with “not Blue”.
