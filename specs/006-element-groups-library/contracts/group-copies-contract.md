# Contract: Group Copies (N Blocks)

## Purpose

Defines the per-group requested **copies** parameter N, the derived effective runtime count, serial legality, persistence, and diagnostics for independent repeated blocks.

## Parameter

| Field | Rules |
|-------|--------|
| Requested `copies` (N) | User-authored integer ≥ 1; default **1**; persisted even while inactive |
| Effective runtime copies | Requested N when legal; otherwise **1** |
| Independence | Each copy has its **own** parameters and weights (not shared) |
| Topology (v1) | **Serial chain only**: copy i outputs → copy i+1 matching inputs |
| External attach | Graph outside the stack connects to **first** copy inputs and **last** copy outputs |
| Nested groups | Each group has its own N |

## Explicit interface and legality

- Each group has one mandatory, non-removable **Group Input** hub and one mandatory, non-removable **Group Output** hub.
- Each hub declares a variable number of ordered lanes (minimum one). These lanes are the group interface; dangling member pins do not implicitly add ports.
- Requested N &gt; 1 is active only when all of the following hold:
  1. Group Input and Group Output declare the same nonzero lane count.
  2. A directed graph path exists from Group Input through the group to Group Output.
  3. Every Group Output lane shape is compatible with the corresponding Group Input lane shape.
- Optional conditioning/control inputs do not by themselves prevent a legal signal through-path.

## Requested and effective behavior

- Requested N = 1: effective runtime copies is 1.
- Requested N &gt; 1, legal: effective runtime copies is N.
- Requested N &gt; 1, illegal: requested N remains stored and visible, while effective runtime copies clamps to 1; no partial/orphan runtime chain is published.
- N → K (K &gt; current): append independent trailing copy state by **cloning the last** copy’s structure and parameters/weights. Runtime serial links are generated only when the request is active.
- N → J (J &lt; current): drop trailing copy state safely and attach runtime output to the new last copy when active.
- Copy validity is recomputed from current graph state. An inactive request **reactivates automatically** when the user repairs lane counts, the through-path, or paired shapes; the user does not have to re-enter N.
- The editor shows one editable group block. Runtime copies are materialized invisibly for publication/execution rather than as duplicate canvas boxes.

## UI

- Expose requested N on the group using ImGui controls consistent with other node properties.
- When inactive, show requested N and **effective 1**, followed by a persistent actionable reason:
  - unequal/missing declared Group Input and Group Output lanes;
  - no Group Input-to-Group Output path; or
  - the incompatible output/input lane number and both shape values.
- For a shape failure, highlight candidate shape-driving properties upstream of the affected Group Output lane (for example `channels`, `features`, `latent_size`, `stride`, or `n_band`). Where supported, the hint may suggest using `in` to preserve incoming width.
- Diagnostics MUST NOT automatically alter model properties, links, hub lane counts, or requested N. They identify user-editable repair points only.

## Persistence & library

- Persist requested `copies` and all independent per-copy values/artifacts in `GraphDocument`. Effective runtime copies is derived and is not a replacement for the request.
- Box-library save/insert of a group includes requested N, boundary hubs/lanes, and all copy state (insert still forces groups collapsed per FR-012a).
- On legacy project load or box import, groups without hubs gain Group Input and Group Output hubs. Previously inferred interface pins become explicit lanes and existing external links are rerouted through them before copy legality is evaluated.

## Interaction with other features

| Feature | Rule |
|---------|------|
| Collapse | Presentation only; the active effective count remains in the DSP graph |
| Freeze | Selection expansion includes freezable members of **all** copies; per-member freeze |
| Interface editing | Hub lane counts are user-authored; connected lanes cannot be removed |
| Property editing | Re-validates copies and can automatically reactivate requested N; never auto-adjusts another property |
| Ungroup | Boundary hubs are spliced out and removed with their owner; ordinary members lift to the parent/root with valid links preserved |

## Non-Goals (v1)

- Shared-weight repeats
- Parallel fan-out / merge of copies
- Automatic residual bypass around the whole stack
- Automatic model-parameter tuning to make a copy request legal

## Implementation anchors

- `OpenYourBox/Source/graph/GraphTypes.h` (`copies` on `GraphGroup`)
- `OpenYourBox/Source/graph/NodeGraph.cpp` (setCopies / materialize / chain validation)
- `OpenYourBox/Source/graph/NodeRenderer.cpp` (N property UI)
- `specs/006-element-groups-library/data-model.md`
