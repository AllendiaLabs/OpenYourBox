# Implementation Plan: Real Cloud Training

**Branch**: `018-real-cloud-training` | **Date**: 2026-08-31 | **Spec**: `specs/018-real-cloud-training/spec.md`

**Input**: Feature specification from `specs/018-real-cloud-training/spec.md`

**Status**: Design complete. Ready for `/speckit-tasks`.

## Summary

Upgrade Phase 4 cloud from fake/mock job advancement to **real remote training**: the proprietary `CloudService/` worker materializes the existing cloud job package and runs the **same mapping/reconstruction recipes** as `Backend/train_worker.py`, publishing real progress, checkpoints, and Gold-loadable final artifacts. Staging E2E (plugin → local/staging API → real train → Gold) is the Done bar; choices stay **production-safe**. **Remove** mock/fake advancement worker code. Train controls for **Local and Cloud** become **Run + Stop only** (no Pause/Resume). Worker/host crash → job `failed` with downloadable checkpoints; no auto-resume.

Technical approach (see `research.md`): finish incomplete `CloudService/` API surface; implement `train_runner` invoking shared train recipes; heartbeat/liveness that fails orphaned jobs; strip pause/resume from API + plugin + local Train UI; keep auth/entitlement/one-job/retention from `017`.

## Technical Context

**Language/Version**: C++17 (JUCE plug-in / Train UI / cloud client); Python 3.10+ (`CloudService/` API + worker; reuse `Backend/train_worker.py` recipes)

**Primary Dependencies**: JUCE 8 HTTP/threads; Dear ImGui Train panel; FastAPI (or equivalent ASGI) for `CloudService/api`; PyTorch via existing local train stack on the worker host; multipart job ingest already shaped by `CloudJobPackage`

**Storage**: File-backed (or equivalent) job/corpus/artifact store under `CLOUD_DATA_DIR` for staging; same retention metadata as `017` (30-day sliding corpus). No new VST-side schema beyond removing Pause/Resume UX and any mock-mode settings.

**Testing**: Python tests for job state machine (queued→running→succeeded|failed|stopped), Stop, crash→failed, real short-train smoke (CPU OK), entitlement/one-job gates without fake-success worker; C++ tests for Run/Stop-only UI/control wiring and cloud client without pause/resume verbs; DAW/staging scenarios in `quickstart.md`

**Target Platform**: Desktop AU/VST3 (macOS first); staging CloudService on operator machine (CPU or GPU); production host later (not required for Done)

**Project Type**: Desktop audio plug-in + proprietary cloud training service (not a user-facing standalone app)

**Performance Goals**: Progress poll ≥1 / 5 s while online (SC-004); 60 FPS UI; zero audio-thread HTTP/train I/O; short guided trains acceptable for acceptance (reduced `total_steps`)

**Constraints**: Always real training (no mock success worker); staging Done + production-safe; Run/Stop only; crash→failed; Local ungated; Cloud gated by `017` account/entitlement rules; Gold auto-load submitter-only; CPU fallback allowed without fake success

**Scale/Scope**: One active cloud job per account; complete incomplete `CloudService/` stubs; remove Pause/Resume from local+cloud Train surfaces; no public production deploy required

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- `Single Interface, Decoupled Compute`: **PASS** — VST remains sole training UI; remote worker is detached compute.
- `Dual-Engine Execution Model`: **PASS** — success still yields Gold TorchScript; live path unchanged until intentional load.
- `Manual Granular Freeze Policy`: **PASS** — success auto-load / optional checkpoint load remain intentional; Stop ≠ success.
- `Shape Integrity & Legal Constraints`: **PASS** — same pre-submit gates; copyright stays local.
- `Zero Audio Allocations / Non-Blocking Audio Thread`: **PASS** — package/upload/poll/download/Stop remain off `processBlock`.
- `Phase 4 Local vs cloud access`: **PASS** — Local ungated; Cloud still account + entitlement.
- `Phase 4 Cloud Training Backend Proprietary`: **PASS** — `CloudService/` proprietary; recipes shared for parity with open local worker.
- `Latency <5 ms Gold / <7 ms live`: **PASS** — unchanged until intentional load.

**Post-Design Re-Check**: **PASS**. Complexity: completing proprietary cloud worker is constitutionally expected for Phase 4 (justified in Complexity Tracking).

## Project Structure

### Documentation (this feature)

```text
specs/018-real-cloud-training/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── cloud-train-api.md          # deltas vs 017 (Stop-only, crash fail, no mock)
│   ├── cloud-worker.md             # real train_runner obligations
│   └── cloud-train-plugin-ux.md    # Run/Stop-only Train surface
├── checklists/
│   └── requirements.md
└── tasks.md                        # /speckit-tasks (not this command)
```

### Source Code (repository root)

```text
OpenYourBox/Source/
├── train/
│   ├── TrainCoordinator.*       # remove pause/resume; Stop for local+cloud
│   ├── CloudTrainClient.*       # drop pause/resume verbs; keep stop/poll/download
│   ├── CloudJobPackage.*        # reuse (017)
│   └── CloudSettings.*          # staging URL overrides (017)
├── ui/
│   ├── TrainPanel.*             # Run + Stop only (hide Pause/Resume)
│   └── CloudSettingsPanel.*     # staging link (unchanged intent)
└── PluginEditor.* / PluginProcessor.*

Backend/
└── train_worker.py              # recipe source of truth; Stop command already exists

CloudService/                    # complete stubs; remove mock advancement
├── api/
│   ├── app.py                   # ASGI app wiring
│   ├── auth*.py / entitlement*  # keep 017 gates
│   ├── jobs.py                  # submit/list/get/stop; NO pause/resume
│   ├── artifacts.py             # checkpoints + final download
│   ├── retention.py             # 30-day sweeper
│   └── state.py                 # active = queued|running (no paused)
├── worker/
│   ├── train_runner.py          # materialize package → invoke recipes → publish
│   └── (delete/empty mock_worker advancement)
├── storefront/                  # staging link helpers (not fake train success)
├── requirements.txt
└── tests/                       # gates, stop, crash-fail, short real train smoke

Tests/
└── CloudTrainClientTests.cpp    # extend: no pause/resume client surface
```

**Structure Decision**: Extend the incomplete `017` `CloudService/` + Train stack; do not add a second UI app. Real training is a worker + API completion problem; plugin changes are control-surface simplification (Run/Stop) and removal of pause/resume cloud verbs.

## Phase 0: Research — Complete

Resolved in `research.md`: recipe invoke strategy, process supervision/crash→failed, Stop mapping, mock removal, test strategy without fake-success worker, staging URL overrides, production-safe defaults.

## Phase 1: Design — Complete

Delivered: `data-model.md`, `contracts/cloud-train-api.md`, `contracts/cloud-worker.md`, `contracts/cloud-train-plugin-ux.md`, `quickstart.md`.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Proprietary `CloudService/` worker (extra deployable) | Constitution Phase 4; real GPU/CPU train cannot run in the VST audio process | In-process train would block audio and violate decoupled compute |
| Completing full API stubs in-repo vs external-only host | Staging E2E Done bar requires a runnable real-training service the plugin can hit | External-only host is not available yet; empty stubs cannot produce real Gold |

## Execution Notes

Suggested implementation order (for `/speckit-tasks`):

1. Remove Pause/Resume from TrainPanel + TrainCoordinator + CloudTrainClient (local + cloud)  
2. Complete `CloudService` API (`app`, `jobs`, `artifacts`, retention wiring) with Stop-only control; active = queued|running  
3. Implement `train_runner` (materialize package → call shared recipes → stream progress/checkpoints/final `.pt`)  
4. Worker liveness: orphaned/crashed runs → `failed`; keep published checkpoints  
5. Delete mock/fake advancement worker paths; rewrite tests to use fixtures/short real jobs  
6. Staging README + `cloud.xml` override quickstart; short mapping + reconstruction smoke  
7. Negative paths: Stop, induced fail, crash-fail, one-job-per-account, Local still works  

## Agent handoff

- Spec: `specs/018-real-cloud-training/spec.md`  
- Plan: this file  
- Research / data-model / contracts / quickstart: same directory  
- Baseline UX/API from `specs/017-cloud-training/` (inherit unless this feature supersedes)  
- Next command: `/speckit-tasks`
