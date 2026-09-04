# Contract: Training Configuration Library

## Purpose

Persist and restore general training hyperparameters and loss stage schedules in (1) a **user-level library** and (2) a **project config snapshot**, without storing architecture identity or audio corpora.

## User TrainingConfigLibrary

- Location: plugin user-data directory (alongside Training Library / presets patterns).
- Entry fields: `id`, `name`, `updated_at`, `config` (see `data-model.md` TrainingConfiguration).
- Operations from Train panel: Save as…, Load…, Rename, Delete, list.
- Reusable across projects/sessions.

## Project config snapshot

- Stored with the patch/session document.
- Same `config` field set as library entries.
- Restore applies when the project loads or via explicit “Restore project train settings”.
- Replaces prior `lastTrainObjective`-only persistence.

## Load semantics (forward compatible)

1. Apply all recognized fields.
2. Ignore unknown fields (post-update extras).
3. Missing recognized fields → product defaults.
4. Warn the user when missing fields fell back to defaults (non-blocking).

## Out of scope for a config entry

- Graph topology / node IDs (except loss ids inside a schedule are only valid after load into a graph that still contains those losses—schedules that reference missing losses fail at Start, not necessarily at config load).
- Training Library audio files.
- Destination entitlement / cloud credentials.
- Architecture mode enums (none exist).

**Note**: Example configs shipped with the product MAY document intended companion example graphs; loading a config alone does not insert nodes.

## Implementation anchors

- `OpenYourBox/Source/library/` (new `TrainingConfigLibrary` or equivalent)
- `OpenYourBox/Source/ui/TrainPanel.cpp`
- `OpenYourBox/Source/PluginProcessor.*` (project snapshot)
