# Contract: Instance Pairing & Capture Samples

## Purpose

Defines in-plugin dual-instance discovery, master/slave roles, Clean/Processed assignment, synchronized input recording, and capture-set ownership.

## Participants

| Role | Meaning |
|------|---------|
| Master | Instance that initiated Capture Samples pairing; owns **training library**, Train UI, auto-load |
| Slave | Paired peer; reduced capture menu only |
| Clean | Records this instance’s **audio input** as **x** (dry) |
| Processed | Records this instance’s **audio input** as **y** (wet from upstream DAW) |

Master/slave is independent of Clean/Processed.

## Discovery

Idle loaded instances are **not** discoverable. Each instance listens on loopback always, but writes a registry advertisement only while the user is searching (`Search for instances`). Closing the editor does not stop a search already in progress (pairing lives on the processor).

1. User clicks **Search for instances** on both instances → each advertises on the localhost discovery registry (`searching: true`).
2. While searching, the panel lists other searching peers by a short instance id (not the full UUID) with a **Connect** button.
3. User clicks **Connect** → that instance becomes master, the peer becomes slave → `syncState = paired`. Both retract advertisements.
4. **Stop searching** retracts the advertisement without pairing. If nobody else is searching, the UI explains that both sides must search.

## Control Messages (logical)

| Message | Direction | Payload (conceptual) |
|---------|-----------|----------------------|
| `hello` / `pair` | M↔S | `sessionId`, `instanceId`, `pluginVersion` |
| `set_capture_role` | M→S or local | `clean` \| `processed` |
| `set_bypass` | M→S / local | `enabled: bool` (default true) |
| `record_start` | M→S | `pairId`, `sampleRate`, sync epoch |
| `record_stop` | M→S | `pairId` |
| `clip_ready` | S→M | `pairId`, `role`, `path` or transfer ref |
| `unpair` / `peer_lost` | either | reason |

Transport: localhost IPC (`InterprocessConnection` / TCP loopback) plus file paths for clip audio. Never on the audio thread for connection setup or file I/O completion handling (audio thread only appends to preallocated capture rings).

## Recording Rules

- Both roles capture **input** taps (pre-graph).
- Default **bypass** passthrough while capture session active; user may disable.
- Start/stop user-gated; **no maximum duration**.
- On stop, master assembles one Sample Pair from Clean x + Processed y; partial/desync discards.
- Peer disconnect mid-record → abort take; show unpaired.

## Slave UI Scope

Slave Capture menu MAY show: pair status, assigned role, bypass toggle, record armed/indicator as directed by master.  
Slave MUST NOT show: full capture-set management, copyright Train gate workflow, Run/Pause/Stop Train panel, auto-load controls.

## Capture Set / Training Library

- On successful stop, master **adds** one Sample Pair to the **Training Library** (source `capture`).
- Train uses **user-selected** library entries (see `training-library-ui-contract.md`), not “all captures” implicitly.
- Train gate: ≥1 selected pair + copyright ack + ≥1 armed trainable element.

## Runtime Guarantees

- Capture ring writes use preallocated buffers; no audio-thread allocations.
- Bypass/flag changes prepared on message thread; audio reads atomic/flag.
- Monitoring with default bypass leaves DAW-heard signal unaffected by graph processing.

## Implementation Anchors

- Pairing types: `OpenYourBox/Source/capture/CapturePairing.h`
- Discovery + `juce::InterprocessConnectionServer` / `juce::InterprocessConnection`: `OpenYourBox/Source/capture/CapturePairing.cpp`
- Preallocated input-ring recorder: `OpenYourBox/Source/capture/CaptureRecorder.cpp`
- Capture bypass flag: `OpenYourBox/Source/PluginProcessor.cpp` (`processBlock` passthrough)
- Master/slave Capture UI: `OpenYourBox/Source/ui/CaptureSamplesPanel.cpp`
