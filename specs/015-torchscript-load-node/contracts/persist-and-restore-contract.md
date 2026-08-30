# Contract: Persist & Restore External Checkpoints

## Purpose

Define what survives save/load of graphs and presets, and how the in-memory factory registry is rehydrated.

## Persisted per TorchScript Load node

| Data | Required |
|------|----------|
| `type` = blackbox | yes |
| `blackBoxOrigin` = `external_load` | yes |
| `artifactPath` / `weightsPath` | yes (may be empty) |
| Channel overrides (and optionally last inferred) | yes when set |
| `fidelityPercent`, `compactnessReady`, compactness blobs | yes when encode/decode path used |
| Pin layout | yes (or rebuild from capabilities after prepare) |

## Not persisted as embedded bytes

- TorchScript module file contents (path reference only in v1).

## Restore algorithm

For each `external_load` node after graph/preset apply:

1. Restore fields and pin skeleton.
2. If `artifactPath` empty → status `empty`, dry passthrough.
3. Else if file missing → status `error`, message, silence (no prior factory).
4. Else → status `loading`, call `prepareExternalArtifact` off audio thread; on success `ready` + morph pins / refresh inferred; on failure `error` + silence.
5. Re-apply stored overrides after inferred prefills (overrides win).

## Multiple nodes, same path

- Each node restores independently.
- Registry may share one factory instance per path; clearing one node must not break others still referencing that path (refcount or leave entry until unused).

## Acceptance

- FR-007, SC-005; User Story 3 scenarios.
