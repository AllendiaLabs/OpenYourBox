# Contract: Library Tags, Filters, and Capture Kind

## Purpose

Extend Phase 3 Training Library UI and Capture Samples so unpaired audio and mapping/reconstruction share one shell.

Supersedes nothing in `specs/004-steerable-discovery-training/contracts/training-library-ui-contract.md` except entry kind and ingest paths.

## Library entry kinds

| Kind | System tag | Files | Ingest |
|------|------------|-------|--------|
| `pair` | `pair` | x + y | Capture **Pair**, pair import |
| `clip` | `unpaired` | one audio file | Capture **Single**, single-file import |

User tags are optional extra strings (schema already reserved in Phase 3).

## Objective vs selection

| Objective | Valid selection | Invalid |
|-----------|-----------------|--------|
| mapping | ≥1 `pair`, **no** `clip` | any `clip` selected → **error**, Run blocked |
| reconstruction | ≥1 `pair` and/or `clip` | empty selection; mixed SR; channel ≠ graph |

Reconstruction **uses both x and y** of each selected pair (two corpus utterances). No side picker.

## Warn and filter

When objective changes:

- Mapping: warn that unpaired clips cannot be used; filter or badge clips as ineligible; if still selected, Run error
- Reconstruction: warn if selection is empty; pairs show “x+y used”

Train summary: `N pairs · M clips · ~T minutes` with objective name.

## Capture Samples

Same menu as Phase 3, plus **Kind**:

- **Pair** — existing pairing, Clean/Processed, slave reduced UI
- **Single** — this instance only; no peer required; records input; default bypass; appends `clip`

## Import

- Existing two-file pair import unchanged
- New: import one or more single files as `clip` entries

## Preview

- Pair: x and/or y (existing)
- Clip: single preview transport

## Implementation anchors

- `OpenYourBox/Source/library/TrainingLibrary.cpp`
- `OpenYourBox/Source/ui/TrainingLibraryPanel.cpp`
- `OpenYourBox/Source/ui/CaptureSamplesPanel.cpp`
- `OpenYourBox/Source/capture/CaptureRecorder.cpp`
