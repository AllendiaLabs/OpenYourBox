# Contract: Training Library UI

## Purpose

Defines a durable in-plugin **Training Library** for sample pairs used by Phase 3 Train — suitable for v1 and extensible long-term without a standalone app.

## Placement

- Master instance only (library ownership follows Train ownership).
- Accessible from a dedicated **Library** panel/section in the plug-in window, reachable from Capture and from Train (same library, not two copies).
- Capture Samples **adds** pairs; it does not replace the library browser.

## Library Entry (pair)

Each entry represents one aligned clean/processed pair.

| Field (UI-visible) | Notes |
|--------------------|--------|
| Display name | Editable; default from filename or `Capture YYYY-MM-DD HH:MM` |
| Duration | Seconds (both sides after align) |
| Sample rate | Hz |
| Channels | Per side; warn on train if mixed rates/channels in selection |
| Source | `Capture` \| `Import` badge |
| Created | Timestamp |
| Notes | Optional free text (v1 optional, reserved) |
| Tags | Zero or more string tags (v1: support schema even if UI is minimal) |
| Selected for train | Checkbox / multi-select state |

Internal (may surface in detail/inspector): file paths for x/y, byte size, pair id.

## Core Actions (v1 — required)

1. **Browse** — scrollable list/table of entries with the fields above.
2. **Multi-select** — choose which pairs feed the next Train; Select all / Select none.
3. **Import** — add a clean+processed file pair (two files or a documented pair package); lands in library as `Import`.
4. **Capture add** — successful dual-instance record appends as `Capture` without leaving the library model.
5. **Rename** — edit display name.
6. **Delete** — remove entry (confirm); deletes associated local audio files owned by the library.
7. **Preview** — play **x** and/or **y** (transport in-plugin; does not require DAW playback). Prefer independent x/y audition buttons.
8. **Train gate** — Train enables only with ≥1 selected entry (+ copyright + armed elements).

## Long-term Actions (specify now; implement incrementally)

| Capability | Intent |
|------------|--------|
| Search / filter | By name, source, tag, duration range |
| Tags & collections | Organize large libraries without folders-only rigidity; collections are named saved filters or explicit membership sets |
| Saved train selections | Named selection sets (“Compressor set A”) reusable across sessions |
| Drag-and-drop import | Drop wav/aiff onto library panel |
| Export pair | Export selected pair(s) as files for backup/sharing |
| Duplicate | Clone entry metadata + files |
| Detail inspector | Waveform thumbnails for x/y, peak/LUFS later, path reveal |
| Disk budget | Show library disk usage; warn before huge captures |
| Sample-rate policy | On Train with mixed SR: block with clear fix, or offer resample (policy: **block with message** in v1; resample optional later) |
| Persistence | Library survives plugin/DAW relaunch (user-data store) |

## Layout Guidance (long-lived UX)

- One composition: library is a **list + detail** pattern, not a dashboard of cards.
- Primary column: selectable rows; secondary: detail/preview for the focused row.
- Capture controls can sit as a compact “Add from DAW” strip above or beside the list — recording is an **ingest** path into the library.
- Train panel shows a compact summary: `N pairs selected · ~T total minutes` with a link to open Library.

## Non-Goals

- Cloud sync / marketplace sample packs (Phase 4).
- Editing audio inside the library (trim/fade) in v1 — out of scope; re-capture or external edit + re-import.
- Slave instance library editing.

## Implementation Anchors

- Store + index persistence: `OpenYourBox/Source/library/TrainingLibrary.h`, `OpenYourBox/Source/library/TrainingLibrary.cpp`
- Copyright acknowledgment log: `OpenYourBox/Source/library/CopyrightAcknowledgment.h`
- List+detail ImGui panel: `OpenYourBox/Source/ui/TrainingLibraryPanel.cpp`
- Editor orchestration (master-only): `OpenYourBox/Source/PluginEditor.cpp`
