# Contract: Group Copies (N Blocks)

## Purpose

Defines the per-group **copies** parameter N that materializes independent serial block instances when I/O shapes allow (deep-learning-style repeated blocks).

## Parameter

| Field | Rules |
|-------|--------|
| `copies` (N) | Integer ≥ 1; default **1** |
| Independence | Each copy has its **own** parameters and weights (not shared) |
| Topology (v1) | **Serial chain only**: copy i outputs → copy i+1 matching inputs |
| External attach | Graph outside the stack connects to **first** copy inputs and **last** copy outputs |
| Nested groups | Each group has its own N |

## Materialization

- N = 1: single block (the group’s members as today).
- N → K (K &gt; current, legal): append new trailing copies by **cloning the last** copy’s structure and parameters/weights; re-assert serial links between copies.
- N → J (J &lt; current): remove trailing copies safely; keep copies 0 .. J-1; rewire external outs to new last copy.
- Illegal N (shapes cannot chain): **refuse or clamp** with clear message; **no** orphan/partial nodes.

## UI

- Expose N on the group (inline property on expanded group and/or collapsed group inspector), using ImGui controls consistent with other node properties.
- Optional: show copy index badges on materialized instances when expanded.

## Persistence & library

- Persist `copies` and all materialized instances in `GraphDocument`.
- Box-library save/insert of a group includes N and all copies (insert still forces groups collapsed per FR-012a).

## Interaction with other features

| Feature | Rule |
|---------|------|
| Collapse | Presentation only; all copies remain in the DSP graph |
| Freeze | Selection expansion includes freezable members of **all** copies; per-member freeze |
| Ungroup | Define clearly in implementation: ungroup with N&gt;1 either flattens all copy members to parent or requires N=1 first — **prefer**: allow ungroup that lifts all copy members preserving serial links |

## Non-Goals (v1)

- Shared-weight repeats
- Parallel fan-out / merge of copies
- Automatic residual bypass around the whole stack

## Implementation anchors

- `OpenYourBox/Source/graph/GraphTypes.h` (`copies` on `GraphGroup`)
- `OpenYourBox/Source/graph/NodeGraph.cpp` (setCopies / materialize / chain validation)
- `OpenYourBox/Source/graph/NodeRenderer.cpp` (N property UI)
- `specs/006-element-groups-library/data-model.md`
