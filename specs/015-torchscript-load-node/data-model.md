# Data Model: TorchScript Checkpoint Loader Node

## Entities

### TorchScript Load Element (graph node)

Specialization of Gold `NodeType::blackBox` with `BlackBoxOrigin::externalLoad`.

| Field | Type / meaning | Notes |
|-------|----------------|-------|
| `id` | Node id | Existing |
| `type` | `blackBox` | Palette label “TorchScript Load” |
| `state` | `frozenGold` when ready; may remain Gold-styled while empty/error | Not Blue / not randomizable |
| `blackBoxOrigin` | `externalLoad` | New enum value; Unfreeze disabled |
| `label` | string | Default “TorchScript Load”; may show filename stem when loaded |
| `artifactPath` | string | Absolute path to `.pt` / `.pth`; empty when never chosen / cleared |
| `weightsPath` | string | Same as artifact for provenance; optional mirror of today’s BlackBox pattern |
| `loadStatus` | enum: `empty` \| `loading` \| `ready` \| `error` | User-visible; may be derived + cached message |
| `loadErrorMessage` | string | Short recoverable reason when `error` |
| `inferredInputChannels` | int | Last successful probe input width |
| `inferredOutputChannels` | int | From probe `size(1)` |
| `inferredLatentChannels` | int | From encode probe when encode/decode; else 0 |
| `overrideInputChannels` | optional int | When set, wins for shape check |
| `overrideOutputChannels` | optional int | When set, wins |
| `overrideLatentChannels` | optional int | When set and latent pins exist, wins |
| `hasEncodeDecode` | bool | From factory after load |
| `acceptsConditioning` | bool | From factory after load |
| `fidelityPercent` | float | Existing; meaningful when encode/decode + compactness |
| `compactnessReady` | bool | Existing |
| Compactness buffers | tensors / blobs | Existing `latentMean` / `latentPca` / `cumulativeVariance` |
| Pins | audio in/out; optional `latent` in/out; optional `control` in | Morph on load/clear |

**Relationships**: Zero or one published `FrozenBlackBoxFactory` in the processor registry keyed by `artifactPath`. Multiple nodes may share the same path (independent elements; unloading one must not unregister if others still reference — refcount or leave registry entry until unused).

### Checkpoint Reference

| Field | Meaning |
|-------|---------|
| Absolute filesystem path | Persisted string |
| Runnable TorchScript artifact | Must `jit::load` and pass forward (or conditioned) probe |

### Channel Override Set

| Field | Meaning |
|-------|---------|
| Effective input / output / latent | `override ?? inferred` |
| Reset | Clears overrides → effective = inferred |

### Load Status (lifecycle)

```text
empty ──choose file──► loading ──success──► ready
  ▲                      │
  │                      └──fail──► error
  │                                   │
  └──clear◄───────────────────────────┘
ready ──choose other / reload──► loading ──… (retain prior factory until swap/clear)
error ──choose other──► loading
```

**Audio policy by status** (when no prior ready factory retained):

| Status | Audio |
|--------|--------|
| `empty` | Dry passthrough main in → out |
| `loading` (no prior) | Continue prior policy: empty→passthrough, error→silence |
| `ready` | Factory forward / encode-decode / conditioned |
| `error` (no prior) | Silence |

## Validation Rules

1. Path must be an existing regular file (not a directory) with accepted extensions (at least `.pt`, `.pth`).
2. Load failure → `error` + message; keep prior ready factory if any.
3. Shape checking uses effective channel counts; illegal cables refused with mismatch feedback.
4. If inference fails and overrides incomplete → incomplete-shape / error; connections not treated as legal until valid overrides exist.
5. Control pin present only if `acceptsConditioning`; latent pins only if `hasEncodeDecode`.
6. Fidelity editable only if encode/decode and `compactnessReady`; otherwise hidden/disabled with explanation.
7. Unfreeze forbidden for `externalLoad`.
8. Clearing path → `empty`, remove optional pins, clear factory association for this node, dry passthrough.

## Persistence Mapping (ValueTree / preset)

| Spec field | Property / child |
|------------|------------------|
| origin | `blackBoxOrigin` = `"external_load"` |
| path | `artifactPath`, `weightsPath` |
| overrides | e.g. `overrideInputChannels`, `overrideOutputChannels`, `overrideLatentChannels` (−1 or absent = no override) |
| inferred (optional cache) | `inferredInputChannels`, … |
| fidelity / compactness | existing `fidelityPercent`, `compactnessReady`, blobs |
| status | may be recomputed on restore; persist last `loadErrorMessage` optional |

On restore: re-run prepare when file exists; else `error` + silence.
