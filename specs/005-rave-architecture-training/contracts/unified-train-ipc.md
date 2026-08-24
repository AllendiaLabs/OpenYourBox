# Contract: Unified Train Worker IPC (RAVE extension)

## Purpose

Extend Phase 3 `train_steerable` IPC (`specs/004-steerable-discovery-training/contracts/train-worker-ipc.md`) with reconstruction objective, stages, corpus clips, and RAVE export. **Same** ChildProcess, JSON lines, Run/Pause/Stop.

## Start Request (additions)

Existing fields remain. Additions:

```json
{
  "request_id": "string",
  "operation": "train_steerable",
  "command": "start",
  "graph_fragment": {},
  "armed_element_ids": [],
  "capture_set": {
    "pairs": [
      { "pair_id": "p1", "x_path": "/abs/x.wav", "y_path": "/abs/y.wav", "kind": "pair" }
    ],
    "clips": [
      { "clip_id": "c1", "path": "/abs/clip.wav", "kind": "clip" }
    ]
  },
  "train_options": {
    "objective": "mapping",
    "optimizer": "adam",
    "total_steps": 2500,
    "checkpoint_interval": 50,
    "export_checkpoints": true,
    "reconstruction": {
      "stage1_steps": 1000000,
      "stage2_steps": 1000000,
      "spectral_windows": [2048, 1024, 512, 256, 128],
      "kl_warmup_steps": 20000,
      "kl_beta_start": 1e-6,
      "kl_beta_end": 5e-2
    }
  }
}
```

`objective` MUST be `mapping` or `reconstruction`.

## Corpus assembly (plugin before send)

- **mapping**: `pairs` only; if any selected library `clip` → do not start; error in UI
- **reconstruction**: flatten selected pairs to two clips (x and y) plus selected `clips`; all channel counts MUST match graph audio width; mixed SR forbidden

## Reconstruction worker rules

- Build module from `graph_fragment` including PQMF, rate conv, bottleneck, noise
- Stage 1 then stage 2 as `research.md` Decision 6
- Discriminators never loaded in the plug-in
- Causal ops only
- Progress events include `stage`: `representation` | `quality`
- Hear-while-training: existing checkpoint export; plugin optional load (same as mapping)
- Success: TorchScript with methods `forward`, `encode`, `decode` plus compactness buffers
- `stopped` / `failure`: no success auto-load (optional prior checkpoint remains if user loaded it)

## Mapping worker rules

Unchanged Phase 3 recipe when `objective` is `mapping`.

## Reconstruction graph gate (plugin)

Refuse start if no variational bottleneck on armed path or decode path missing, with `error_message` suitable for the Train panel.

## Implementation anchors

- `OpenYourBox/Source/train/TrainCoordinator.cpp`
- `OpenYourBox/Source/ui/TrainPanel.cpp`
- `Backend/train_worker.py`
- `Tests/test_train_worker.py`
