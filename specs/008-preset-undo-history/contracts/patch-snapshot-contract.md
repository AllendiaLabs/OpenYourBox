# Contract: Patch Snapshot

## Purpose

Defines the shared **full plugin patch** document used by:

1. DAW/host `getStateInformation` / `setStateInformation`
2. Named user presets (Save / Save As / load)
3. Undo/redo history steps

## Logical contents

| Part | Required | Description |
|------|----------|-------------|
| Parameter state | Yes | Full APVTS (or equivalent) tree |
| Graph document | Yes | Complete `GraphDocument` (nodes, links, groups, copies, box layout) |
| Weight archive | When available | Serialized model weights matching architecture |
| Architecture hash | With weights | Reject mismatched weight blobs; fall back to rebuild |
| Randomization counter | Yes | For deterministic recall with seed |
| Last train objective | Yes | Existing metadata continuity |
| Schema version | Yes | Integer; readers must refuse unknown future major versions safely |

## Capture semantics

1. Capture current parameters + `NodeGraph::toValueTree()` (+ published weights if any), matching today’s host-state fields.
2. Snapshot must be deep enough that a round-trip yields equal graph structure, parameter values, and sonic-relevant weight/Gold state when load succeeds (SC-002).
3. External artifact paths inside the graph may be rewritten when embedding into a preset folder; apply must resolve paths again for the live instance.

## Apply semantics

1. Run on **GUI / message thread only** (never audio thread).
2. Set a restoring/`suppressHistory` flag so apply does not recursively push undo steps.
3. Replace parameter state and graph document atomically from the caller’s perspective (no half-applied graph left on failure after validation).
4. Rebuild and **atomically publish** runtime (existing publish/compile path).
5. Do not stop host transport.
6. On weight/Gold restore failure when the snapshot claimed those artifacts: **refuse** the apply for preset load (clear message; leave live patch unchanged). Host session restore may keep today’s seed/counter fallback for resilience—preset path prefers fail-closed for sonic fidelity (FR-001a / FR-013).

## Non-goals

- Partial “parameter-only” patches
- Box-library component snapshots (different contract)
- Streaming/delta encoding (v1 is full snapshots)

## Implementation anchors

- Factor helpers from `OpenYourBoxAudioProcessor::getStateInformation` / `setStateInformation`
- NEW: `OpenYourBox/Source/state/PatchSnapshot.h/.cpp`
- Consumers: processor host state, `UserPresetLibrary`, `EditHistory`
