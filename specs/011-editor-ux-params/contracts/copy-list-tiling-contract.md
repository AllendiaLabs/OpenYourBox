# Contract: Copy-List Tiling

## Purpose

Define authored vs expanded multi-value parameters under nested group copies (FR-008…FR-010a).

## Copy-count vector

- Walk from **outermost** ancestor group with copies to the **innermost** group containing the node: `C = [c0, c1, …, ck-1]`.
- `P = Π ci` (= existing `effectiveCopyCount`).
- **Dividing set** `D(C) = {1} ∪ { Π_{j=i}^{k-1} c_j | i = 0..k-1 }`  
  Example `C=[O,M,N]` → `D = {1, N, M×N, O×M×N}`.

## Commit rules

| User list length L | Behavior |
|--------------------|----------|
| L ∈ D(C) | Accept; store authored length L; expand to P by tiling outer axes |
| L ∉ D(C) | Refuse; keep previous authored values; clear message listing allowed lengths |

## Expansion (tiling)

- Authored values repeat to fill outer dimensions so slot index `s` in `[0,P)` maps to authored index `s % L` when L is an inner suffix product (equivalently: block-repeat along outer axes). Contract tests MUST include `O=2,M=2,N=2` with L=2 and L=4.

## UI

- Editable field: authored CSV (length L).
- Read-only preview: expanded P values (not editable).
- When P=1: single-value UI as today (no preview required).

## Ancestor copy-count change

| Authored L vs new D | Behavior |
|---------------------|----------|
| L ∈ D′ | Keep authored; re-tile to new P; update preview |
| L ∉ D′ | Set invalid; keep authored text/values; runtime unresolved; message until user edits |

Invalid params MUST NOT silently rewrite L; sibling properties unchanged.

## Persistence

- Serialize authored vectors only.
- On load: compute D from restored groups; valid → expand; invalid → flag.

## Implementation anchors

- `OpenYourBox/Source/graph/GraphTypes.h` — `parsePropertyCopyList`, `ensurePropertyCopyCount`
- `OpenYourBox/Source/graph/NodeGraph.cpp` — `setPropertyCopyValues*`, `setGroupCopies`, `effectiveCopyCount`
- `OpenYourBox/Source/graph/NodeRenderer.cpp` — property row UI

## Non-goals

- Arbitrary divisors of P outside D(C)
- Editing the expanded preview directly
- Shared weights across copies (unchanged from 006)
