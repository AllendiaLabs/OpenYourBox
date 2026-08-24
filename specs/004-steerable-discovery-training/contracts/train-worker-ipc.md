# Contract: Train Worker IPC

## Purpose

Local request/response/stream contract between the master plug-in (`TrainCoordinator`) and the detached Python train worker for steerable NAfx training and TorchScript export.

## Process Model

- Same pattern as freeze: `juce::ChildProcess`, worker materialized from embedded resources when needed, JSON lines on stdin/stdout (or request file + stdout events — implementation may match freeze’s file+stdout style).
- GUI/background thread only; never audio thread.
- Audio continues on **prior** model until successful auto-load swap.

## Start Request

```json
{
  "request_id": "string",
  "operation": "train_steerable",
  "command": "start",
  "graph_fragment": {
    "elements": [],
    "connections": [],
    "io_boundary": {
      "audio_inputs": [],
      "audio_outputs": [],
      "conditioning_inputs": []
    }
  },
  "armed_element_ids": ["element-1"],
  "capture_set": {
    "pairs": [
      { "pair_id": "p1", "x_path": "/abs/x.wav", "y_path": "/abs/y.wav", "source": "import" }
    ]
  },
  "train_options": {
    "optimizer": "adam",
    "loss": {
      "type": "multiresolution_stft",
      "fft_sizes": [32, 128, 512, 2048],
      "win_lengths": [32, 128, 512, 2048],
      "hop_sizes": [16, 64, 256, 1024]
    },
    "steer_conditioning": 0.0,
    "learning_rate_schedule": [
      { "until_fraction": 0.80, "lr": 1e-3 },
      { "until_fraction": 0.95, "lr": 1e-4 },
      { "until_fraction": 1.00, "lr": 1e-5 }
    ],
    "total_steps": 2500,
    "segment_length": 228308,
    "rf_aware_crop": true,
    "mlflow": {
      "enabled": true,
      "tracking_uri": "http://127.0.0.1:5000",
      "experiment": "openyourbox",
      "name": "",
      "tags": ["train", "steerable"]
    }
  }
}
```

## Control Commands

```json
{ "request_id": "string", "operation": "train_steerable", "command": "pause" }
```

```json
{ "request_id": "string", "operation": "train_steerable", "command": "resume" }
```

```json
{ "request_id": "string", "operation": "train_steerable", "command": "stop" }
```

## Progress Event (streamed)

```json
{
  "request_id": "string",
  "status": "running",
  "step": 120,
  "total_steps": 2500,
  "loss": 0.042,
  "learning_rate": 0.001
}
```

Allowed `status` on events: `running` | `paused` | `stopping`.

## Success Response

```json
{
  "request_id": "string",
  "status": "success",
  "artifact_path": "/abs/path/trained_blackbox.pt",
  "blackbox_metadata": {
    "origin": "train_autoload",
    "display_name": "Trained Steerable",
    "ports": [],
    "shape_signature": {},
    "conditioning": true,
    "baseline_metrics": {
      "train_steps": 2500,
      "final_loss": 0.01
    }
  }
}
```

## Failure / Stopped Response

```json
{
  "request_id": "string",
  "status": "failure",
  "error_message": "Human-readable failure summary"
}
```

```json
{
  "request_id": "string",
  "status": "stopped",
  "step": 400,
  "message": "Stopped by user"
}
```

## Request Rules

- `graph_fragment` is the **armed trainable** subgraph snapshot at Run (not live-edited thereafter).
- `steer_conditioning` MUST be `0.0` (ca = 0) for v1 recipe.
- `capture_set.pairs` MUST be non-empty readable paths from the **selected** training-library entries (capture and/or file import).
- Control sources MUST NOT appear as trainable armed elements in the fragment.
- When `rf_aware_crop` is true, each step MUST include receptive-field context before the target segment of `segment_length` samples (or shorter if audio is limited).
- Loss MUST use the specified multiresolution STFT sizes (not an unspecified STFT variant).
- When `train_options.mlflow.enabled` is true, the worker SHOULD log params, metrics, and artifacts to `tracking_uri` using the MLflow Tracking REST API (`/api/2.0/mlflow`). REST failures MUST NOT fail training.

## Response Rules

- `success` → plugin prepares TorchScript load off-thread, then atomically replaces armed chain with Gold BlackBox; Knob/XY remain.
- `stopped` / `failure` → **no** model swap; prior model remains.
- `request_id` echoed on all messages.

## Runtime Guarantees

- Worker MUST NOT be invoked from the audio thread.
- Pause/Resume preserve optimizer state for the same job when supported by worker implementation.
- Stop ends the job without writing a replacement live model.
- UI may show loss updates from progress events (≥ ~1 Hz when events available).

## Implementation Anchors

- Coordinator (ChildProcess + JSON lines, freeze-style): `OpenYourBox/Source/train/TrainCoordinator.h`, `OpenYourBox/Source/train/TrainCoordinator.cpp`
- Armed subgraph snapshot: `OpenYourBox/Source/graph/NodeGraph.cpp` (`createTrainRequest`)
- Worker recipe + TorchScript export: `Backend/train_worker.py`
- Embedded packaging: `CMakeLists.txt` (`juce_add_binary_data` alongside `freeze_worker.py`)
- Python recipe tests: `Tests/test_train_worker.py`
