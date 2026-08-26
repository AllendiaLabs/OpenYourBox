# Contract: Freeze Per Member (with Groups)

## Purpose

Defines Freeze behavior when the selection includes groups and/or multiple elements, aligning with FR-001b: freeze freezable members **individually**; never replace the whole selection/group with one Gold BlackBox.

## Scope

- UI action: existing **Freeze Selection** (and equivalents) in graph context menus.
- Out of scope: Train **absorb** path that still creates a single `blackBox` from an armed chain (`absorbArmedChain`) — leave as-is unless a later feature unifies naming.

## Selection expansion

1. Start from `ed` selected nodes (and selected group nodes if groups are selectable as nodes).
2. If a **group** is selected (collapsed or expanded), include all **descendant leaf nodes** in that group’s membership tree, including members of **all materialized copies** when `copies` N &gt; 1.
3. Partition candidates into:
   - **Freezable**: live nodes eligible under current freeze rules (weighted/process types accepted by today’s freeze; exclude fixed Audio I/O, and any type freeze already rejects).
   - **Skip**: already Gold, BlackBox-only as appropriate, Knob/XY if not freezable, etc.
4. Run freeze **once per freezable node** (or per minimal legal freeze unit that still yields **one Gold outcome per member**, not one Gold for the whole selection). Prefer true per-node compile/artifact when feasible; if the existing pipeline requires a trivial single-node chain, use that.

## Outcomes

| Outcome | Behavior |
|---------|----------|
| Success per member | That node becomes `frozenGold` with its artifact metadata; stays in the same group |
| Partial success | Some members Gold, skips listed; group remains |
| Failure on one member | Error for that member; others may still succeed; no wholesale rollback required unless easier to implement transactionally |
| Empty freezable set | Informative message; no graph change |

## Explicit non-behaviors

- MUST NOT delete the group and replace members with one BlackBox on Freeze.
- MUST NOT use Train absorb semantics for this menu action.
- Collapse state MUST NOT block freeze (collapsed group selection still expands to members for freeze).

## Relation to current code

Today `NodeGraph::freezeSelection` + `partitionFreezeChains` freezes **connected chains**, keeping nodes and sharing `artifactPath`. This contract **tightens** product intent to **per freezable member** when groups/multi-select are involved. Implementation may:

- Special-case: if selection expands to N freezable nodes, enqueue N freeze jobs; or
- Partition so each freezable node is its own chain of size 1 when invoked from group-aware freeze.

Document the chosen approach in code comments and keep UI copy accurate (“Freezing N elements…”).

## Implementation anchors

- `OpenYourBox/Source/graph/NodeGraph.cpp` (`freezeSelection`, `partitionFreezeChains`, `createFreezeRequest`)
- `OpenYourBox/Source/PluginEditor.cpp` (`handleFreeze`, `applyCompletedFreeze`)
- `OpenYourBox/Source/freeze/FreezeCoordinator.*`
