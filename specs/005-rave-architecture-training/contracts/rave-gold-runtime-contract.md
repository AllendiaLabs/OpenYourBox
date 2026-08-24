# Contract: RAVE Gold Runtime

## Purpose

Live and frozen execution of trained/frozen RAVE subgraphs: methods, fidelity, Unfreeze, checkpoints.

## Gold node ports

| Port | Domain | Use |
|------|--------|-----|
| audio in | audio | `forward` and `encode` |
| audio out | audio | `forward` and `decode` |
| latent out | latent | `encode` tap |
| latent in | latent | `decode` drive |

Default host wiring uses **forward** (audio in → audio out). User may wire latent in/out without Unfreeze.

## Methods (TorchScript)

- `forward(x)` = decode(fidelity(encode(x)))
- `encode(x)` = latent after fidelity crop
- `decode(z)` = audio (expects latent **after** or **before** crop consistently; v1: encode applies fidelity, decode expects already-cropped-or-full matching export)

Export MUST document whether `decode` expects full-width z or compact z. v1: **full-width** z with fidelity applied inside encode/forward; decode applies fidelity if incoming width is full latent, or accepts compact if tagged — prefer **always full-width latent ports** (128-ch) with fidelity internal, so graph shapes stay stable when sweeping fidelity.

## Fidelity

- User control 0–100 on Gold and on live bottleneck
- Does not mutate weight files
- Same thread rules as Knob/XY: GUI sets target, audio reads smoothed or atomic scalar (no alloc)

## Auto-load / Unfreeze / checkpoints

- Success reconstruction: replace **armed** chain with this Gold (Phase 3 arm rules)
- Unfreeze: restore Blue RAVE elements + trained tensors + current fidelity
- Hear-while-training optional load: same off-thread prepare + atomic swap as mapping checkpoints; Stop is not success

## Delay

Gold/layout MUST expose causal delay (samples and ms) in the node detail. Does not need to meet 5 ms constitution row (see plan Complexity Tracking).

## Implementation anchors

- `OpenYourBox/Source/graph/NodeGraph.cpp` (auto-load / Unfreeze)
- `OpenYourBox/Source/dsp/LiveGraphEngine.cpp`
- `Backend/train_worker.py` / `Backend/freeze_worker.py` export
