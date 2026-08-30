# Data Model: RAVE Prior Mix & Insert Catalog

## Entities

### RAVE-capable box (`GraphNode` blackBox)

| Field / facet | Type / range | Notes |
|---------------|--------------|-------|
| `blackBoxOrigin` | `trainAutoload` \| `externalLoad` (and encode/decode ready) | In-scope origins when `hasEncodeDecode` |
| `priorMix` | float `[0, 1]`, default `0` | Full forward → full prior; `NodeProperty` key `priorMix` |
| `fidelityPercent` | existing | Unchanged; applied before prior-mix on encoder distribution when compactness ready |
| Pins | audio in/out; optional Control; **bias** in; **scale** in; **latent** out | No latent **input** |
| Compactness buffers | existing | Unchanged |

**Validation**:
- `priorMix` clamped to `[0, 1]` on set (same refuse-or-clamp policy as other real props in this codebase for fidelity-class controls — prefer clamp-to-range consistent with fidelity).
- Bias/scale connections must match effective latent channel width (and time rules of latent-domain cables).
- Forward-only external load: no `priorMix` / bias / scale / latent-out surface.

### Prior mix (property)

| Attribute | Value |
|-----------|--------|
| Key | `priorMix` |
| Kind | `PropertyKind::real` |
| Range | `[0, 1]` |
| Default | `0` |
| Editable on Gold | Yes (like `fidelity`) |
| Persistence | ValueTree with node / preset |

### Bias pin

| Attribute | Value |
|-----------|--------|
| Role | Input |
| Domain | Latent-compatible tensor |
| Disconnected default | `0` (all channels) |
| Effect | Added to post-mix mean before sample |

### Scale pin

| Attribute | Value |
|-----------|--------|
| Role | Input |
| Domain | Latent-compatible tensor |
| Disconnected default | `1` (all channels) |
| Effect | Multiplies post-mix spread before sample |

### Effective sampled latent

| Attribute | Value |
|-----------|--------|
| Produced | After mix → bias/scale → sample |
| Consumers | `decode`; latent **output** pin / `latentOutputs` tap |
| Shape | `[batch, latentChannels, time]` matching decoder |

### Factory palette catalog (shared)

| Attribute | Value |
|-----------|--------|
| Contents | Categorized `PaletteItem` list (effects, neural, layers, sources, …) |
| Consumers | Left Factory tree; Pin Add menu; Link Insert menu |
| Rule | Single source of truth; intentional omit list for non-palette types preserved |

### Right-click insert catalog view

| Attribute | Value |
|-----------|--------|
| Factory section | Categories from shared catalog, filtered by attach/insert rules |
| User Library section | Folders → entries → expandable snapshot members |
| Insert actions | Factory → `attachNodeToPin` / `insertNodeOnLink` / place; Library → `UserBoxLibrary::insertBox` (+ optional wire) |

## Relationships

```text
RAVE-capable box
  ├── owns priorMix property
  ├── owns bias pin ──(optional cable)──► latent-domain source
  ├── owns scale pin ──(optional cable)──► latent-domain source
  ├── publishes effective sampled latent ──► latent out pin
  └── decode(effective z) ──► audio out

Shared Factory catalog ──► left Factory + Pin Add + Link Insert
UserBoxLibrary ──► left User Library panel + right-click User Library submenu
```

## State transitions (runtime, per buffer)

```text
α = priorMix
if α ≈ 1:
  skip encode; (μ_e, σ_e) = (0, 1)
else:
  encode → (μ_e, σ_e) [σ_e may fallback to 1]
  optional fidelity on encoder distribution
(μ, σ) = lerp((μ_e, σ_e), (0, 1), α)
(μ, σ) = (μ + bias, σ ⊙ scale)
z = μ + σ ⊙ ε
latent_out = z
audio_out = decode(z)
```

## Persistence

| Data | Persist? |
|------|----------|
| `priorMix` | Yes |
| Bias/scale cables | Yes (normal graph connections) |
| Factory catalog | Code / shared module (not user data) |
| User library | Existing on-disk library (unchanged) |
| Old latent-in cables | Not migrated |
