# Contract: Generalized Train Worker IPC

## Purpose

Replace architecture-branded `train_steerable` + `objective` mapping|reconstruction with a single **`train_graph`** start schema consumed by local `Backend/train_worker.py` and `CloudService/worker/train_runner.py`. Preserve ChildProcess / JSON-lines Run·Pause·Stop·progress protocol from 004/005/018.

## Start request

```json
{
  "schema_version": 2,
  "request_id": "string",
  "operation": "train_graph",
  "command": "start",
  "active_data_loader_id": 12,
  "armed_element_ids": [3, 5, 8],
  "graph_fragment": {
    "elements": [
      {
        "id": 12,
        "type": "data_loader",
        "label": "Data Loader",
        "properties": [
          { "key": "output_count", "value": 3 }
        ],
        "outputs": [
          { "pin_index": 0, "label": "input" },
          { "pin_index": 1, "label": "target" },
          { "pin_index": 2, "label": "cond" }
        ]
      },
      {
        "id": 20,
        "type": "loss",
        "label": "MR-STFT",
        "properties": [
          { "key": "loss_type", "value": "mr_stft" }
        ]
      }
    ],
    "connections": [
      {
        "source_element_id": 12,
        "source_pin_index": 0,
        "destination_element_id": 3,
        "destination_pin_index": 0,
        "kind": "data_loader"
      },
      {
        "source_element_id": 8,
        "source_pin_index": 0,
        "destination_element_id": 20,
        "destination_pin_index": 0,
        "kind": "loss_prediction"
      },
      {
        "source_element_id": 12,
        "source_pin_index": 1,
        "destination_element_id": 20,
        "destination_pin_index": 1,
        "kind": "loss_target"
      }
    ],
    "io_boundary": {
      "audio_inputs": [],
      "audio_outputs": [],
      "conditioning_inputs": []
    }
  },
  "data_loader_bindings": {
    "12": {
      "0": {
        "kind": "audio_list",
        "items": [
          { "path": "/abs/a.wav", "library_id": "…" }
        ]
      },
      "1": {
        "kind": "audio_list",
        "items": [
          { "path": "/abs/b.wav", "library_id": "…" }
        ]
      },
      "2": {
        "kind": "constant_scalar",
        "value": 0.5,
        "example_count": 1
      }
    }
  },
  "loss_schedule": {
    "stages": [
      {
        "name": "representation",
        "steps": 1000000,
        "losses": [
          { "loss_node_id": 21, "weight": 1.0 },
          { "loss_node_id": 22, "weight": 0.1 }
        ],
        "freeze_element_ids": []
      },
      {
        "name": "quality",
        "steps": 1000000,
        "losses": [
          { "loss_node_id": 21, "weight": 1.0 },
          { "loss_node_id": 23, "weight": 1.0 },
          { "loss_node_id": 24, "weight": 10.0 }
        ],
        "freeze_element_ids": [3, 5, 8]
      }
    ]
  },
  "train_options": {
    "optimizer": "adam",
    "device": "auto",
    "learning_rate": 0.001,
    "generator_lr": 0.001,
    "discriminator_lr": 0.0001,
    "batch_size": 8,
    "segment_length": 65536,
    "checkpoint_interval": 50,
    "export_checkpoints": false,
    "rf_aware_crop": true,
    "host_input_channels": 2,
    "host_io_mode": "stereo",
    "adam_beta1": 0.5,
    "adam_beta2": 0.9,
    "lr_decay_end_factor": 0.1,
    "update_discriminator_every": 2,
    "phase_mangle_prob": 0.8,
    "dequantize_bits": 16,
    "mlflow": { "enabled": false }
  }
}
```

### Single-stage shorthand

If `loss_schedule.stages` is empty or omitted, the worker MUST run one stage for `train_options.total_steps` (required in that case) using every validly wired loss at weight **1.0**. Stage entries that omit `weight` MUST also default to **1.0**. Loss element properties MUST NOT be consulted for weight.

## Worker rules

1. Build trainable module from `graph_fragment` processing elements on the data-loader path; apply grads only to `armed_element_ids` minus each stage’s `freeze_element_ids` (rebuild optimizer at stage boundaries).
2. Passthrough on-path nodes participate in forward without updates.
3. Batch from active Data Loader bindings; zip connected outputs by example index.
4. Evaluate active stage losses; optimize weighted sum (plus train-only discriminator steps when adversarial losses are active).
5. Discriminators MUST NOT be exported into the plug-in artifact.
6. Progress events SHOULD include `stage` name/index when a multi-stage schedule is used.
7. Success artifact: TorchScript suitable for Gold auto-load; include `encode`/`decode` when the trained graph exposes that structure (bottleneck present)—without requiring an objective flag.
8. `stopped` / `failure`: no success auto-load.

## Plugin preflight (must refuse before send)

Documented in `train-panel-generalized-ux.md` and `data-model.md` (active loader, equal-count, feeds, losses, arm).

## Removed / ignored

- `train_options.objective`
- Required nested `train_options.reconstruction` object as mode carrier
- Global `capture_set` as the sole material source (bindings replace it)

## Implementation anchors

- `OpenYourBox/Source/graph/NodeGraph.cpp` — package assembly
- `OpenYourBox/Source/PluginEditor.cpp` — `handleTrainRun`
- `OpenYourBox/Source/train/TrainCoordinator.cpp`
- `OpenYourBox/Source/train/CloudJobPackage.cpp`
- `Backend/train_worker.py`
- `CloudService/worker/train_runner.py`
- `Tests/test_train_worker.py`
