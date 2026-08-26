# Contract: Variational Bottleneck (Reference Parity)

## Purpose

Define the variational bottleneck head geometry, parameterization, and sampling rules shared by the **live C++ engine**, **train worker**, and **freeze worker**.

## Head geometry

| Parameter | Value | Configurable |
|-----------|-------|--------------|
| Groups | 2 (group 0 = μ, group 1 = variance pre-act) | No |
| Default kernel | 5 | Yes (`kernel_size` property) |
| Dilation | 1 | No (v1) |
| Padding | Causal left `(kernel_size − 1)` | Derived |
| Bias | False (match reference) | No |

**Channel rules**
- Input `C_in` MUST be even; each group receives `C_in / 2` channels
- Output `C_out = latent_size`; each group produces `latent_size / 2` channels
- `latent_size` MUST be even

**Weight shape (PyTorch grouped Conv1d)**
- `[latent_size, C_in / 2, kernel_size]` with `groups=2`

## Forward modes

### Training (worker, stage 1 only)

```
μ, v_pre = grouped_causal_conv(features)   # split output halves
σ² = softplus(v_pre) + eps
z = μ + exp(0.5 * log(σ²)) * ε,  ε ~ N(0,I)
return z
```

KL (per batch, mean over dims/time): standard unit-Gaussian closed form using μ and log σ².

### Eval / live / Gold / validation pass

```
μ, v_pre = grouped_causal_conv(features)
return μ    # no ε; v_pre unused on forward output (still computed if needed for diagnostics)
```

**Live engine MUST NOT call sampling**, including when reconstruction training runs in the background.

## Properties (graph JSON / ValueTree)

```json
{
  "type": "variational_bottleneck",
  "properties": [
    { "key": "latent_size", "value": 128 },
    { "key": "kernel_size", "value": 5 },
    { "key": "fidelity", "value": 99 }
  ]
}
```

Layouts MUST set `kernel_size: 5` on insert.

## Shape gate (UI)

When connecting or arming, if upstream `channels % 2 != 0`:

> Variational bottleneck requires an even channel count (grouped mean/variance head).

## Breaking change

Implementations MUST NOT accept or emit legacy dual 1×1 conv weights. Graphs prepared before this feature require re-init/retrain.

## Implementation anchors

- `OpenYourBox/Source/dsp/VariationalBottleneck.cpp`
- `OpenYourBox/Source/dsp/LiveGraphEngine.cpp` (`variationalBottleneck` case)
- `Backend/train_worker.py` (`VariationalBottleneckLayer`)
- `Backend/freeze_worker.py` (shared builder or duplicated layer)
- `OpenYourBox/Source/graph/NodeGraph.cpp` (property + gate)
- `OpenYourBox/Source/graph/RaveLayouts.cpp` (defaults)

## Tests

- Even/odd channel gate
- Default kernel 5 on layout insert
- Worker train forward ≠ eval forward (stochastic vs deterministic)
- Live forward equals eval μ path for same weights/input
