# Contract: Bias / Scale Pin Surface

**Feature**: `016-rave-prior-mix`  
**Applies to**: `NodeGraph::applyExternalLoadSurface` and train-autoload RAVE Gold pin setup; compile/shape checks in `LiveGraphEngine`

## Pin surface (encode/decode ready)

| Pin | Kind | Label / detection | Required |
|-----|------|-------------------|----------|
| Audio in | input | existing | yes |
| Audio out | output | existing | yes |
| Control | input | when conditioning advertised | optional |
| Bias | input | `isBiasPin` | present; may be disconnected |
| Scale | input | `isScalePin` | present; may be disconnected |
| Latent | output | `isLatentPin` | yes |

**Forbidden**: Latent **input** pin on RAVE-capable boxes in scope.

## Defaults

| Pin | Disconnected value |
|-----|--------------------|
| Bias | 0 |
| Scale | 1 |

## Shape checking

- Bias and scale MUST match **effective latent channel** count (after overrides / compactness width rules already used for latent ports).
- Time-alignment rules follow existing latent-domain cable conventions.
- Illegal connections: refuse + tooltip (same class as other shape violations).

## Property: `priorMix`

| Key | Kind | Range | Default | Gold-editable |
|-----|------|-------|---------|---------------|
| `priorMix` | real | `[0, 1]` | `0` | yes (with `fidelity`) |

Persisted with the node. Forward-only TorchScript Load nodes MUST NOT show `priorMix` or bias/scale/latent-out.

## Latent output

Latent out MUST equal the sampled `z` consumed by `decode` for that buffer (see runtime contract).

## Compile mapping

`CompiledElement` MUST expose indices (or equivalent) to gather bias and scale separately from audio and Control; remove `latentInputIndex` decode-from-wired-z shortcut for these boxes.
