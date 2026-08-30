# Quickstart: RAVE Prior Mix & Insert Catalog

Validate `016-rave-prior-mix` end-to-end inside the OpenYourBox editor (and C++ tests where noted).

## Prerequisites

- Build OpenYourBox + `LiveGraphEngineTests`
- A RAVE-capable graph: either
  - learned OYB RAVE Gold with encode/decode, or
  - TorchScript Load pointed at a checkpoint with `encode` / `decode`
- Optional: at least one User Library entry with a nested group (for catalog checks)

## 1. Prior mix morph (SC-001, SC-002)

1. Place Audio In → RAVE-capable box → Audio Out.
2. Leave bias/scale disconnected; set `priorMix = 0`; play audio — reconstruction / forward path.
3. Sweep `priorMix` toward `1` — continuous change, no host restart.
4. At `priorMix = 1`, mute or replace audio input with silence/noise while holding bias/scale fixed — output must not track the audio input (encoder skipped).

**Expect**: Smooth morph; full prior independent of audio input.

## 2. Bias / scale steering (SC-003, SC-004)

1. Confirm pins: **no latent in**; **bias** and **scale** inputs; **latent** out present.
2. Disconnected bias/scale at `priorMix = 0` and `1` — box runs (neutral 0 / 1).
3. Connect constants or Knobs: non-zero bias and non-unity scale — hear / see change on audio and latent out.
4. Attempt illegal channel width on bias/scale — connection refused with shape feedback.

**Expect**: Defaults work unwired; steering audible; bad shapes blocked.

## 3. Latent out = effective `z` (SC-005)

1. Tap latent out into analysis or a latent-legal sink.
2. Change `priorMix`, bias, and scale.
3. Confirm the tap tracks decode-driving values (including at full prior).

**Expect**: Latent out is not a stale encode-only μ stream.

## 4. Automated engine checks

```bash
# From repo build directory (adjust target name as in your CMake presets)
ctest -R LiveGraphEngineTests --output-on-failure
```

Cover at least: priorMix lerp + sample path; encoder not called at full prior; bias 0 / scale 1 defaults; latent out equals decode input; pin surface without latent-in.

## 5. Right-click catalog (SC-006, SC-007)

1. Right-click a free pin → **Add**: every left-Factory type that is attach-eligible appears; new factory types (e.g. recent LSTM / TorchScript Load / DDSP) are present.
2. Right-click a cable → **Insert**: same Factory completeness (link-eligible filter).
3. Expand **User Library** in those menus; insert a root entry and a nested subpart.
4. With empty user library, Factory list still complete.

**Expect**: Menus match current Factory catalog; library hierarchy insert works.

## References

- Runtime: `contracts/rave-prior-mix-runtime-contract.md`
- Pins: `contracts/bias-scale-pin-surface-contract.md`
- Menus: `contracts/right-click-insert-catalog-contract.md`
- Model: `data-model.md`
