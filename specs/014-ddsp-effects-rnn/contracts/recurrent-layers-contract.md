# Contract: Recurrent Layers (LSTM / RNN)

## Purpose

Define single-layer LSTM and RNN graph elements: parameters, shapes, hidden-state lifecycle, and freeze/train expectations.

## Types

| Persist string | Display | Layer count |
|----------------|---------|-------------|
| `lstm` | LSTM | Exactly 1 |
| `rnn` | RNN | Exactly 1 |

Depth is achieved only by stacking/grouping multiple elements.

## Properties

| Key | Default | Notes |
|-----|---------|-------|
| `hidden_size` | project default ≥ 1 | Editable |
| `bidirectional` | `false` | Toggle |
| `bias` | `true` | Learnable bias on/off |
| `activation` | Tanh | Same menu as Activation/TCN: ReLU, Sigmoid, Tanh, LeakyReLU, PReLU |
| `negative_slope` | `0.01` | Visible iff LeakyReLU; bounds `[0, 1]` |
| `gain` | `1.0` | Same pre-nonlinearity gain as Activation/TCN (`[0.1, 10]`) |

**Absent**: `num_layers`, `input_size`.

## Shape contract

| Mode | Output time | Output channels |
|------|-------------|-----------------|
| Unidirectional | Equal to input time length | `hidden_size` |
| Bidirectional | Equal to input time length | `2 * hidden_size` (forward ‖ backward) |

- Input feature size = upstream output channel count (inferred).
- Illegal or missing connections → refuse with Shape Integrity message.

## In-cell activation

- Chosen activation **replaces** the cell’s primary nonlinearity (RNN nonlinearity; LSTM candidate/primary tanh slot), not a post-layer-only activation.
- `gain` multiplies the argument to that nonlinearity (Activation/TCN semantics).
- LSTM gates remain sigmoid.

## Hidden-state lifecycle

| Event | Hidden / cell state |
|-------|---------------------|
| Consecutive live buffers, stable graph | **Carry** |
| Graph rebuild, port reconnect, freeze or unfreeze swap | **Reset** |
| Re-init / randomize weights on that element | **Reset** |

State is runtime-only (not required in project save); correctness is behavioral across buffers within a session.

## Freeze / train

- Legal shapes → Freeze Selection succeeds; Gold BlackBox replaces selection without stopping UI.
- Train workflows that accept modular graphs accept these types.
- Custom cell implementations in Python must match live audible/parameter semantics for scripted export.

## Acceptance Checks

- Menu lists LSTM and RNN under Neural / Sequence.
- Two stacked unidirectional LSTMs deepen recurrence without a `num_layers` control.
- Bidirectional on doubles output channels vs hidden size.
- Buffer-to-buffer carry verified in live tests; reset verified after rebuild/randomize.
