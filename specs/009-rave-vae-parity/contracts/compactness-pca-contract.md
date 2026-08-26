# Contract: Compactness PCA & Validation Split

## Purpose

Define train/validation splitting, PCA timing, buffer layout, and fidelity gating for RAVE reconstruction — extending spec 005 Gold compactness.

## Corpus split (job start)

When `objective = reconstruction` and Run starts:

1. Flatten selected library paths to clip list (pairs → x + y per spec 005)
2. Shuffle with **seed 42**
3. Assign **~2%** to validation, remainder to train
4. Cap validation list at **1000** paths (`max_residual`)
5. Persist split in job snapshot for reproducibility

**Stage 1**: minibatches draw audio segments from **train paths only**.

If validation set is empty but train non-empty: skip PCA at end, `compactness.ready = false`, emit warning in train event.

## PCA procedure (representation stage end)

Trigger: last optimization step of stage 1, **before** stage 2 begins.

1. `module.eval()`
2. For each validation path (segmented like training):
   - Load audio segment
   - `μ = encode_to_bottleneck(features)` — **μ only**, no sampling
   - Append flattened rows `[time × batch, latent_size]`
3. Stack all rows → matrix `M`
4. `mean = M.mean(0)`
5. SVD on `(M - mean)` → singular values `s`, right singular vectors `V`
6. `cumulative[i] = sum(s[0:i+1]) / sum(s)`
7. Store `mean`, `V`, `cumulative` on module + export buffers
8. Set `compactness.ready = true` in metadata **only if** `M.shape[0] >= latent_size` (else fallback per spec)

**Do not** run PCA on training minibatches or sampled `z`.

## Fidelity application

Given fidelity percent `f` ∈ [0,100]:

```
target = f / 100
keep = min { k | cumulative[k-1] >= target }  (1-indexed, min 1)
```

Apply in PCA basis (existing `applyFidelity` semantics):
- Keep first `keep` components
- Replace dropped dims with N(0,1) noise
- Project back to latent space; add `mean`

Inactive when `compactness.ready == false` → keep = latent_size.

## Train IPC events

Extend running/success events:

```json
{
  "compactness": {
    "ready": false,
    "validation_segments": 0,
    "status": "not_ready"
  }
}
```

On PCA success at stage-1 end:

```json
{
  "compactness": {
    "ready": true,
    "validation_segments": 42,
    "status": "ready"
  }
}
```

Mid-stage-1 checkpoints: always `ready: false`.

## Gold / Unfreeze

**TorchScript buffers** (required on success export):
- `latent_mean`: `[latent_size]`
- `latent_pca`: `[latent_size, latent_size]`
- `cumulative_variance`: `[latent_size]`

**Unfreeze**: VST copies buffers onto the Blue `variationalBottleneck` node:
- `compactnessReady = true`
- tensors loaded into live prepare path
- current `fidelity` preserved

## Hear-while-training

Checkpoint load before stage-1 PCA completion:
- Live μ-only encode (variational-bottleneck contract)
- Fidelity control **inactive** (`status: not_ready`)

## Implementation anchors

- `Backend/train_worker.py`: `split_reconstruction_corpus()`, validation loop, `compute_compactness()`
- `OpenYourBox/Source/dsp/VariationalBottleneck.cpp`: `applyFidelity`
- `OpenYourBox/Source/dsp/LiveGraphEngine.cpp`: Gold + live compactness tensors
- `OpenYourBox/Source/graph/NodeGraph.cpp`: Unfreeze compactness copy
- `OpenYourBox/Source/ui/TrainPanel.*`: compactness status display

## Tests

- Split reproducibility (seed 42, 1000 cap)
- Train minibatch never draws from val paths
- PCA rows sourced from μ only (mock: sampled z would differ)
- Checkpoint before stage-1 end → `ready: false`
- Success export → Unfreeze → live `compactnessReady`
