# Implementation Plan: Cloud Training

**Branch**: `017-cloud-training` | **Date**: 2026-08-31 | **Spec**: `specs/017-cloud-training/spec.md`

**Input**: Feature specification from `specs/017-cloud-training/spec.md`

**Status**: Design complete (platform account + entitlement). Ready for `/speckit-tasks`.

Next: `/speckit-tasks` (then `/speckit-implement`).

## Summary

Ship **Phase 4 official cloud training**: from the existing Train / Library shell, users set destination **Cloud**, **link a platform customer account** (WordPress storefront), and submit only when **active credit/purchase entitlement** covers the job. Same mapping/reconstruction package goes to a proprietary remote GPU service; monitor/control is **account-wide**; download checkpoints; **auto-load Gold only on the submitting instance**. Local Train stays unchanged and works **without** a platform account. Purchases and balance live on the storefront (plugin may open it); in-plugin checkout and marketplace selling stay out of scope.

Technical approach (see `research.md`): HTTPS JSON job API + poll; GPU workers reuse `train_worker` recipes; C++ `CloudTrainClient` off audio thread; storefront-backed account link + entitlement check at submit; soft **2 GiB** upload warning; corpus retention **30 days from last use** with extend-on-reuse; **one active cloud job per platform customer account**.

## Technical Context

**Language/Version**: C++17 (JUCE plug-in / Train UI / cloud client); Python 3 (existing local `train_worker`; cloud API + GPU worker service); WordPress storefront as account/commerce surface (integration via cloud API, not in-process PHP)

**Primary Dependencies**: JUCE 8 (HTTP, PropertiesFile, threads, URL launch), Dear ImGui Train panel; PyTorch on GPU workers (same as local train); HTTPS JSON control plane (Python web framework in proprietary `CloudService/`); storefront identity + entitlement verified server-side

**Storage**: Local — masked linked-session credentials + last cloud base URL override + submitter job ids + optional entitlement cache in user settings / plugin state; downloaded checkpoints/artifacts under existing train artifact dirs. Remote — customer identity, entitlement ledger (or storefront sync), job records, corpus blobs, checkpoints, final `.pt` with retention metadata

**Testing**: C++ unit/contract tests for client request shaping, destination gating, account/entitlement refusal paths, soft-warn math, submitter auto-load flag; Python tests for auth, entitlement gate, one-job-per-account, retention extend/expiry, job state machine; optional mock HTTP + mock storefront entitlement for plugin integration; DAW scenarios in `quickstart.md`

**Target Platform**: Desktop AU/VST3 (macOS first); cloud API + GPU worker on Linux (dev may use CPU mock worker); WordPress storefront reachable for link and purchase flows

**Project Type**: Single desktop audio plug-in + proprietary cloud training service (not a user-facing standalone app)

**Performance Goals**: 60 FPS UI during upload/poll; zero audio-thread allocations / no train or HTTP on `processBlock`; status updates ≥1 / 5 s while online (SC-003); submit path from ready setup &lt; 2 minutes excluding huge uploads and excluding storefront purchase time (SC-001)

**Constraints**: VST-only end-user UI; Local|Cloud in same Train panel; official Cloud Run requires linked platform customer account + sufficient entitlement; Local MUST NOT require account; one job/instance and one cloud job/account; submitter-only success auto-load; copyright gate before Cloud Run; TLS; soft upload warn @ 2 GiB default; no in-plugin checkout

**Scale/Scope**: Platform customers with storefront entitlement; single active remote job per account; corpora from existing Training Library selections; retention 30-day sliding window

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- `Single Interface, Decoupled Compute`: **PASS** — VST remains sole training UI; storefront is account/commerce only (opened externally); cloud is detached remote compute.
- `Dual-Engine Execution Model`: **PASS** — success still yields Gold TorchScript BlackBox; Blue live path unchanged during remote train.
- `Manual Granular Freeze Policy`: **PASS** — cloud success auto-load is explicit completion (same class as local train auto-load); optional checkpoint load is user action; Stop ≠ success.
- `Shape Integrity & Legal Constraints`: **PASS** — same library/objective gates before upload; copyright acknowledgment still required locally.
- `Zero Audio Allocations / Non-Blocking Audio Thread`: **PASS** — package/upload/poll/download/control/account link I/O on background or message thread only.
- `Phase 4 Local vs cloud access`: **PASS** — Local train/freeze ungated; official cloud requires authenticated platform customer account + entitlement at submit.
- `Phase 4 Cloud Training Backend Proprietary`: **PASS** — proprietary service tree; billing/entitlement integration server-side; recipes shared conceptually with open train worker for parity.
- `Latency &lt;5 ms Gold / &lt;7 ms live`: **PASS** — unchanged; cloud does not alter live graph until intentional load.

**Post-Design Re-Check**: **PASS**. Complexity: proprietary cloud service + storefront entitlement sync are constitutionally expected for Phase 4 (not a violation).

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
│   └── CloudSettings.*             # linked session + base URL + storefront URL (masked)
├── ui/
│   ├── TrainPanel.*                # Local|Cloud; entitlement/auth errors; soft size warn
│   └── (settings / prefs UI)       # link / disconnect account; open storefront
├── library/                        # selected corpus byte sum for soft warn
└── PluginEditor.* / PluginProcessor.*  # wire client; submitter auto-load only

Backend/
└── train_worker.py                 # unchanged local path; recipes reused by cloud worker

CloudService/                       # proprietary (new)
├── api/                            # auth (customer session), entitlement, jobs, corpus, artifacts, retention
├── worker/                         # pull job → materialize package → train recipe → publish
├── storefront/                     # WordPress customer + entitlement sync (server-side)
└── tests/

Tests/
├── CloudTrainClientTests.cpp       # (or equivalent)
└── test_cloud_api.py               # auth, entitlement, one-job-per-account, retention, state machine
```

**Structure Decision**: Extend OpenYourBox Train stack for the client; add a separate proprietary `CloudService/` tree for API + GPU worker + storefront entitlement verification. Do not introduce a user-facing standalone training application. Do not embed WordPress checkout inside the VST.

## Phase 0: Research — Complete

Resolved in `research.md`: REST+poll, recipe reuse, client architecture, platform account link, entitlement gate, concurrency, retention, 2 GiB soft warn, packaging, endpoint override, security notes.

## Phase 1: Design — Complete

Delivered: `data-model.md`, `contracts/cloud-train-api.md`, `contracts/cloud-train-plugin-ux.md`, `contracts/cloud-job-package.md`, `quickstart.md`.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Proprietary `CloudService/` tree (extra deployable) | Constitution Phase 4 cloud backend; GPU jobs cannot run inside the VST process | Embedding GPU train in-process would block audio and violate decoupled compute |
| Server-side storefront entitlement sync | Constitution: purchases/balance on WordPress; cloud submit must check entitlement | In-plugin checkout would violate VST-only commerce boundary and FR-002a |

## Execution Notes

Suggested implementation order (for `/speckit-tasks`):

1. Cloud settings (account link / disconnect / open storefront / base URL) + Train destination UI + soft-warn  
2. Entitlement status probe + Cloud Run gating (auth vs entitlement messages)  
3. Job package builder (reuse local request shaping) + upload progress  
4. Cloud API skeleton (customer auth, entitlement, one-job-per-account, job state) + mock worker  
5. Plugin client poll/control + status mapping into Train panel  
6. Checkpoint/artifact download; submitter-only success auto-load  
7. Retention sweeper + corpus_id reuse path  
8. Real GPU worker wired to train recipes; quickstart scenarios  
9. Negative paths: unlink, insufficient entitlement, concurrency, offline reconnect, non-submitter manual load  

## Agent handoff

- Spec: `specs/017-cloud-training/spec.md`  
- Plan: this file  
- Research / data-model / contracts / quickstart: same directory  
- Next command: `/speckit-tasks` (regenerate — prior `tasks.md` is stale vs platform-account slice)
