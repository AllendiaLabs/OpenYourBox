<!--
Sync Impact Report
- Version change: 1.2.0 → 1.3.0
- Modified sections:
  - Phase 4 "Marketplace & Cloud": official cloud training requires authenticated
    platform customer account and storefront-managed credits; local train/freeze
    remain account-free; VST remains the only training UI
  - Licensing Strategy: clarify open core vs paid hosted cloud; cloud backend
    remains proprietary for the hosted service; marketplace commission unchanged
- Added sections: None
- Removed sections: None
- Deferred TODOs: None
-->

# OpenYourBox Constitution

## Core Principles

### I. Single Interface, Decoupled Compute

The VST is the sole user interface. There is no standalone application.

- **VST (C++ / JUCE)**: Hosts the node-graph editor, the parameter assignment UI, and the inference engine.
- **Backend Worker (Python)**: Runs as a detached background process managed by the VST. It handles dataset preparation, training, and model compilation.
- **The Absolute Law**: The background worker MUST NEVER block, stall, or interfere with the real-time audio thread. If training is running, the VST MUST continue to process audio with the previously loaded model seamlessly.

Any deviation that forces the user to open a terminal, run a Python script, or use a separate executable is a violation of this principle and MUST be rejected.

### II. Dual-Engine Execution Model (Live + Frozen)

To reconcile extreme flexibility with maximum performance, the VST operates two distinct execution engines that coexist simultaneously:

**A. The Live Modular Engine (C++ / `torch::nn`)**
- Executes non-frozen nodes dynamically.
- Allows real-time bypassing, layer insertion/deletion, and random weight manipulation (RONN-style glitch).
- All nodes are color-coded **Blue** in the UI.

**B. The Frozen BlackBox Engine (LibTorch / TorchScript)**
- Executes selected subgraphs compiled into TorchScript `.pt` files.
- Kernel Fusion enables optimal performance (reduced latency) and lower RAM usage.
- All frozen nodes are color-coded **Gold** with a 🔒 icon in the UI.

### III. Manual Granular Freeze Policy (V1.0)

Auto-freeze (automatic background compilation) is explicitly deferred to a future version. V1.0 relies on explicit user intent.

- **Manual Freeze**: Users right-click a selection of nodes and choose "Freeze Selection". The VST sends the subgraph to the Python backend, which returns a compiled `.pt` file. The VST atomically swaps the selected nodes for a single "BlackBox" node (Gold) without stopping audio.
- **Manual Unfreeze**: Right-clicking a frozen BlackBox and selecting "Unfreeze" reverts it to a modular state (Blue) for further editing.
- **Visual Feedback**: A clear status bar or modal window MUST display progress ("Compiling..."). On completion, the node MUST visibly flash and change color.

### IV. Shape Integrity & Legal Constraints (UI-Enforced)

The UI is the primary gatekeeper against illegal operations and legal liability:

- **Live Shape Inference**: The moment a user connects an incompatible port (e.g., 64 channels to a 32-channel input), the cable MUST turn Red and refuse to connect. A tooltip MUST explain the dimension mismatch.
- **Copyright Acknowledgment**: Before the first training session, a blocking modal dialog MUST appear: "I certify that all audio samples captured for this training are my original work or royalty-free." The "Train" button MUST remain grayed out until this box is checked. A local log of the acknowledgment MUST be stored.

## Architectural Mandate

| Component | Technology | Justification |
|-----------|-----------|---------------|
| VST Shell | C++17 / JUCE | Industry standard for plugin development |
| VST UI (Node Editor) | Dear ImGui / imgui-node-editor | GPU-accelerated, low-latency rendering embedded inside JUCE |
| Live Engine (Non-Frozen) | LibTorch (C++ API) | Dynamically instantiates layers via `torch::nn` modules |
| Frozen Engine (BlackBox) | LibTorch (C++ API) | Loads and executes TorchScript `.pt` files via `torch::jit::load()` |
| Backend Worker (Training/Compile) | Python / PyTorch | Spawned as child process; JSON architecture via local IPC |
| Model Serialization | TorchScript (`.pt`) | Strictly enforced bridge between Python backend and C++ VST |

**Data Pipeline (The Flow)**:
1. **Design**: User drags blocks inside the VST window. Shape inference runs instantly.
2. **Freeze**: User selects nodes → Right-click → "Freeze Selection". VST serializes subgraph to JSON and sends to Python backend.
3. **Compile**: Python converts JSON to PyTorch code, performs dummy forward pass to trace/script the module, exports `blackbox.pt`.
4. **Atomic Swap**: VST loads `blackbox.pt` via LibTorch on a background thread. At the next audio buffer, atomically replaces modular nodes with a single BlackBox inference node (Gold).
5. **Unfreeze**: User right-clicks Gold node → "Unfreeze". VST discards `.pt` file, reloads original modular nodes.

**Memory & Performance Management**:
- **Live Nodes**: Weights stored in standard `torch::Tensor` objects. Randomization (RONN) modifies tensors in-place on GUI thread, then atomically swaps to audio thread.
- **Frozen BlackBox**: Weights are immutable. Inference is a single `forward()` call per buffer leveraging kernel fusion.
- **Zero Audio Allocations**: Absolutely zero memory allocations (`new`/`malloc`) are permitted on the real-time audio thread. All node swaps MUST be prepared on the GUI thread and applied via atomic pointers.

## Phased Rollout Strategy

**Phase 1: "The Live Player & RONN"**
- Goal: Working VST that generates sound instantly with the Live Modular Engine.
- Model: RONN (Randomized Overdrive Neural Networks) entirely in C++ via `torch::nn`. Zero training required.
- Features: Stable VST3/AU plugin; Blue modular nodes for depth, kernel size, channels; real-time weight randomization; no Python backend required.

**Phase 2: "The Embedded Builder & Manual Freeze"**
- Goal: Integrate node graph and Python backend for compiling frozen nodes.
- Features: Full node palette; Freeze via right-click; Python backend listens for JSON, constructs `torch.nn.Module`, traces it, returns `.pt`; progress spinner; shape violation UI.

**Phase 2.2: "Signal Analysis & Expressive Input Controls"**
- Goal: Extend the embedded graph builder with per-element analysis views and richer parameter input modalities beyond inline text fields.
- Features:
  - **Per-Element Visualization Graphs**: Each element MUST expose analysis views showing the cumulative sound transformation up to that point in the graph (e.g., transfer function, frequency response, phase response, and related analysis plots). **All channels or feature dimensions** at the analysis point MUST be displayed on the same plots with distinguishable styling — stereo left/right is the common audio case, but latent or high-dimensional feature spaces (any channel count) MUST be supported equally.
  - **Activation & TCN Gain Control**: Activation-function and TCN elements MUST expose a gain parameter that controls the slope of the nonlinearity, enabling real-time shaping of transfer-characteristic steepness.
  - **Knob Inputs**: Knob Input MUST be a graph source element (like Audio Input) supplying runtime conditioning signals, combinable via Merge and connectable to processing element inputs.
  - **XY Trackpad**: XY Trackpad MUST be a graph source element supplying two-axis runtime conditioning (e.g., c0/c1), combinable via Merge and connectable to processing element inputs.

**Phase 3: "Steerable Discovery & Training"**
- Goal: Enable training of conditional models.
- Features: "Capture Samples" button; "Train" button sends architecture + samples to backend; trained model auto-loaded as Gold Frozen BlackBox.

**Phase 4: "Marketplace & Cloud"**
- Goal: Monetization.
- Marketplace: In-VST browser for downloading Gold models. Commission: 20%.
- Cloud Training: Optional paid hosted service returning `.pt` files. Job submit, status, and artifact load remain inside the VST (no standalone cloud app).
- **Local vs cloud access**: Local training, freeze, and inference MUST remain fully usable without a platform customer account. Official cloud training MUST require an authenticated platform customer account (WordPress storefront) with an active credit or purchase entitlement before a remote job is accepted. Account creation, purchases, and credit balance live on the storefront; the VST links credentials and consumes entitlement at submit time.

## Performance Benchmarks (Non-Negotiable)

| Metric | Target |
|--------|--------|
| Latency (Frozen/Gold, 256-sample buffer) | < 5 ms on standard Intel i7 |
| Latency (Live/Blue, 256-sample buffer) | < 7 ms on standard Intel i7 |
| Compilation Time (< 10 layers) | < 2 seconds |
| Atomic Swap Duration | < 100 ms (double-buffered) |
| UI Responsiveness | 60 FPS during full backend CPU load |

## Legal & Governance Constraints

**Copyright Infringement Shield**:
- The blocking modal disclaimer is legally binding. Acknowledgment is stored locally (never sent to servers).
- Marketplace Moderation: Any model containing copyrighted material MUST be removed without prior notice.

**Licensing Strategy**:
- Core Source Code (VST, Live Engine, Freeze logic, local train worker): **Apache 2.0**
- ML Forge Derived Code: MUST retain original MIT copyright notice in NOTICE file.
- Cloud Training Backend (hosted API, workers, billing/entitlement integration): **Proprietary**
- Open core licensing does not imply free hosted compute; the paid cloud product is the official remote training service.
- Marketplace Models: Creator retains IP; platform takes 20% commission.

## Governance

This constitution serves as the single source of truth for all technical decisions in OpenYourBox.

- This document supersedes all other practices and design documents when conflicts arise.
- Amendments MUST be documented with version bump rationale, approved by project maintainers, and include a migration plan for affected code.
- All code reviews MUST verify compliance with these principles.
- Complexity MUST be justified against the core principle of zero-allocation real-time audio.

**Ultimate Non-Negotiables**:
1. The VST is the only interface. No standalone app.
2. Manual freeze only in V1.0 (auto-freeze is a future feature).
3. Blue = Live (glitchable, randomizable). Gold = Frozen (optimized, stable).
4. Zero audio-thread allocations. All graph modifications prepared on GUI thread and swapped atomically.
5. Local train/freeze MUST NOT require platform login; official cloud jobs MUST.

**Version**: 1.3.0 | **Ratified**: 2026-08-19 | **Last Amended**: 2026-08-31
