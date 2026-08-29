# Contract: Freeze / Train Type Registry

## Purpose

Ensure the seven new element types are first-class citizens of the JSON → PyTorch → TorchScript pipeline used by Freeze and Train.

## Type strings

Must match C++ `nodeTypeName` / `nodeTypeFromName`:

| C++ / UI | JSON `type` |
|----------|-------------|
| Reverb | `reverb` |
| ExpDecayReverb | `exp_decay_reverb` |
| FilteredNoiseReverb | `filtered_noise_reverb` |
| FIRFilter | `fir_filter` |
| ModDelay | `mod_delay` |
| LSTM | `lstm` |
| RNN | `rnn` |

## Registration sites (all required)

1. `Backend/freeze_worker.py` — `build_module` (and multi-input path if Reverb `ir` is wired)
2. `Backend/train_worker.py` — `build_module` and any DAG builder used for multi-input graphs
3. C++ known-type persist gate (`isKnownPersistedNodeType`)
4. Live engine compile/process (Blue path; freeze consumes exported `.pt`)

## Fragment expectations

```json
{
  "elements": [
    {
      "type": "exp_decay_reverb",
      "properties": [
        { "key": "gain", "float_value": 1.0 },
        { "key": "decay", "float_value": 4.0 },
        { "key": "reverb_length", "value": 48000 },
        { "key": "add_dry", "value": 1 }
      ]
    }
  ],
  "connections": []
}
```

- Property encoding follows existing `{ key, value | float_value | string_value }` conventions.
- Trainable IR / magnitudes / recurrent weights participate in training parameter sets when armed.
- Unsupported legacy workers that omit a type must fail with a clear error listing the unknown `type` (no silent skip).

## Reverb multi-input

When the freeze/train fragment includes a connection into the Reverb `ir` port, the built module MUST accept that second tensor (or baked external IR) and apply `ir_blend` consistently with live. Empty IR handling in offline workers should match live fallback semantics or fail the job with an explicit error before swapping Gold nodes—never leave the live graph half-applied.

## Gain on recurrent cells

Document and implement consistently: prefer baking Activation/TCN-style `gain` into the scripted forward so Gold matches Blue for frozen LSTM/RNN (improve on Activation’s current live-only gain gap where feasible for these new types).

## Tests

- `Tests/test_freeze_worker.py`: build + script each new type (minimal graph).
- `Tests/test_train_worker.py`: parameter presence for weighted types; forward shape for LSTM/RNN (uni/bi).
- C++ live tests remain source of truth for buffer carry and soft warnings.
