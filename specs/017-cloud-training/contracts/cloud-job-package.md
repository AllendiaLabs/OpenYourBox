# Contract: Cloud Job Package

**Feature**: `017-cloud-training`  
**Applies to**: Plugin packager + `CloudService` ingest + GPU worker

## Purpose

Define what is uploaded/referenced when destination is Cloud so the remote worker can run the **same** recipes as local Train (`specs/005-rave-architecture-training/contracts/unified-train-ipc.md`).

## Manifest (`manifest.json`)

```json
{
  "schema_version": 1,
  "client": {
    "plugin_version": "string",
    "client_instance_id": "string",
    "host_label": "optional"
  },
  "operation": "train_steerable",
  "graph_fragment": {},
  "armed_element_ids": [],
  "capture_set": {
    "pairs": [
      { "pair_id": "p1", "x_name": "x.wav", "y_name": "y.wav", "kind": "pair" }
    ],
    "clips": [
      { "clip_id": "c1", "name": "clip.wav", "kind": "clip" }
    ]
  },
  "train_options": {
    "objective": "mapping",
    "optimizer": "adam",
    "total_steps": 2500,
    "checkpoint_interval": 50,
    "export_checkpoints": true,
    "reconstruction": {}
  },
  "corpus_id": null
}
```

Notes:

- `capture_set` file names refer to multipart parts (or paths inside a corpus archive), **not** absolute local paths.
- When reusing retention, set `corpus_id` and omit file parts (or send empty capture files); server resets `last_used_at`.
- `train_options` / objective rules match unified train IPC (mapping vs reconstruction eligibility enforced **in plugin before upload** and re-checked server-side).

## Multipart upload (new corpus)

Parts:

| Part | Content |
|------|---------|
| `manifest` | `manifest.json` |
| `file:<name>` | Audio bytes for each referenced name |

Plugin computes `totalBytes` for soft warn before send.

## Server materialization

1. Validate auth + one-job-per-token.  
2. Store files under new `corpus_id` **or** resolve existing `corpus_id`.  
3. Rewrite capture paths to worker-local absolute paths.  
4. Invoke train recipe equivalent to local start request.  
5. Stream/publish progress, checkpoints, final artifact.

## Compatibility

Worker MUST accept graph elements and objectives supported by current local `train_worker.py`. Unknown element types → `validation_failed` before long GPU burn when possible.

## Implementation anchors

- Plugin: package builder beside `TrainCoordinator` start-request assembly  
- `CloudService/api` ingest  
- `CloudService/worker` path rewrite + recipe invoke  
