# Implementation Plan: Cloud Training

**Branch**: `017-cloud-training` | **Date**: 2026-08-31 | **Spec**: `specs/017-cloud-training/spec.md`

**Input**: Feature specification from `specs/017-cloud-training/spec.md`

**Status**: Design complete. Ready for `/speckit-tasks`.

Next: `/speckit-tasks` (then `/speckit-implement`).

## Summary

Ship **Phase 4 cloud training (pre-credits)**: from the existing Train / Library shell, users set destination **Cloud**, authenticate with a **beta API token**, submit the same mapping/reconstruction job package to a proprietary remote GPU service, monitor/control token-wide, download checkpoints, and **auto-load Gold only on the submitting instance**. Local Train stays unchanged and works without a token. WordPress, credits, and marketplace are out of scope.

Technical approach (see `research.md`): HTTPS JSON job API + poll; GPU workers reuse `train_worker` recipes; C++ `CloudTrainClient` off audio thread; soft **2 GiB** upload warning; corpus retention **30 days from last use** with extend-on-reuse; **one active cloud job per token**.

## Technical Context

**Language/Version**: C++17 (JUCE plug-in / Train UI / cloud client); Python 3 (existing local `train_worker`; cloud API + GPU worker service)

**Primary Dependencies**: JUCE 8 (HTTP, PropertiesFile, threads), Dear ImGui Train panel; PyTorch on GPU workers (same as local train); HTTPS JSON control plane (implementation may use a small Python web framework in the proprietary cloud service tree)

**Storage**: Local — masked API token + last cloud base URL override + submitter job ids in user settings / plugin state; downloaded checkpoints/artifacts under existing train artifact dirs. Remote — job records, corpus blobs, checkpoints, final `.pt` in object/file storage with retention metadata

**Testing**: C++ unit/contract tests for client request shaping, destination gating, soft-warn math, submitter auto-load flag; Python tests for API auth, one-job-per-token, retention extend/expiry, job state machine; optional mock HTTP server for plugin integration; DAW scenarios in `quickstart.md`

**Target Platform**: Desktop AU/VST3 (macOS first); cloud API + GPU worker on Linux (dev may use CPU mock worker)

**Project Type**: Single desktop audio plug-in + proprietary cloud training service (not a user-facing standalone app)

**Performance Goals**: 60 FPS UI during upload/poll; zero audio-thread allocations / no train or HTTP on `processBlock`; status updates ≥1 / 5 s while online (SC-003); submit path from ready setup &lt; 2 minutes excluding huge uploads (SC-001)

**Constraints**: VST-only end-user UI; Local|Cloud in same Train panel; no WordPress/credits; one job/instance and one cloud job/token; submitter-only success auto-load; copyright gate before Cloud Run; TLS + Bearer token; soft upload warn @ 2 GiB default

**Scale/Scope**: Beta tokens; single active remote job per token; corpora from existing Training Library selections; retention 30-day sliding window

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- `Single Interface, Decoupled Compute`: **PASS** — VST remains sole UI; cloud is detached remote compute, not a second user app or terminal workflow.
- `Dual-Engine Execution Model`: **PASS** — success still yields Gold TorchScript BlackBox; Blue live path unchanged during remote train.
- `Manual Granular Freeze Policy`: **PASS** — cloud success auto-load is explicit completion (same class as local train auto-load); optional checkpoint load is user action; Stop ≠ success.
- `Shape Integrity & Legal Constraints`: **PASS** — same library/objective gates before upload; copyright acknowledgment still required locally.
- `Zero Audio Allocations / Non-Blocking Audio Thread`: **PASS** — package/upload/poll/download/control on background or message thread only.
- `Phase 4 Cloud Training Backend Proprietary`: **PASS** — new proprietary service tree; recipes shared conceptually with open train worker for parity.
- `Latency &lt;5 ms Gold / &lt;7 ms live`: **PASS** — unchanged; cloud does not alter live graph until intentional load.

**Post-Design Re-Check**: **PASS**. Complexity: proprietary cloud service is constitutionally expected for Phase 4 (not a violation).

## Project Structure

### Documentation (this feature)

```text
specs/017-cloud-training/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── cloud-train-api.md
│   ├── cloud-train-plugin-ux.md
│   └── cloud-job-package.md
├── checklists/
│   └── requirements.md
└── tasks.md                    # /speckit-tasks (not this command)
```

### Source Code (repository root)

```text
OpenYourBox/Source/
├── train/
│   ├── TrainCoordinator.*          # destination local|cloud fork; busy gates
│   ├── CloudTrainClient.*          # HTTPS job API client (off audio thread)
│   └── CloudSettings.*             # token + base URL persistence (masked)
├── ui/
│   ├── TrainPanel.*                # Local|Cloud destination; soft size warn; cloud status
│   └── (settings / prefs UI)       # API token entry
├── library/                        # selected corpus byte sum for soft warn
└── PluginEditor.* / PluginProcessor.*  # wire client; submitter auto-load only

Backend/
└── train_worker.py                 # unchanged local path; recipes reused by cloud worker

CloudService/                       # proprietary (new)
├── api/                            # auth, jobs, corpus, artifacts, retention sweeper
├── worker/                         # pull job → materialize package → train recipe → publish
└── tests/

Tests/
├── CloudTrainClientTests.cpp       # (or equivalent)
└── test_cloud_api.py               # one-job-per-token, retention, state machine
```

**Structure Decision**: Extend OpenYourBox Train stack for the client; add a separate proprietary `CloudService/` tree for API + GPU worker. Do not introduce a user-facing standalone training application.

## Phase 0: Research — Complete

Resolved in `research.md`: REST+poll, recipe reuse, client architecture, Bearer token, concurrency, retention, 2 GiB soft warn, packaging, endpoint override, security notes.

## Phase 1: Design — Complete

Delivered: `data-model.md`, `contracts/cloud-train-api.md`, `contracts/cloud-train-plugin-ux.md`, `contracts/cloud-job-package.md`, `quickstart.md`.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Proprietary `CloudService/` tree (extra deployable) | Constitution Phase 4 cloud backend; GPU jobs cannot run inside the VST process | Embedding GPU train in-process would block audio and violate decoupled compute |

## Execution Notes

Suggested implementation order (for `/speckit-tasks`):

1. Cloud settings (token/base URL) + Train destination UI + soft-warn  
2. Job package builder (reuse local request shaping) + upload progress  
3. Cloud API skeleton (auth, one-job-per-token, job state) + mock worker  
4. Plugin client poll/control + status mapping into Train panel  
5. Checkpoint/artifact download; submitter-only success auto-load  
6. Retention sweeper + corpus_id reuse path  
7. Real GPU worker wired to train recipes; quickstart scenarios  
8. Negative paths: auth, concurrency, offline reconnect, non-submitter manual load  

## Agent handoff

- Spec: `specs/017-cloud-training/spec.md`  
- Plan: this file  
- Research / data-model / contracts / quickstart: same directory  
- Next command: `/speckit-tasks`
