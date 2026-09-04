#!/usr/bin/env python3
"""Train an OpenYourBox armed graph with the steerable NAfx recipe."""

from __future__ import annotations

import argparse
import base64
import copy
import json
import math
import os
import random
import shutil
import struct
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

try:
    import auraloss
except ImportError:  # pragma: no cover - environment-specific dependency.
    auraloss = None  # type: ignore[assignment]

FFT_SIZES = [32, 128, 512, 2048]
WIN_LENGTHS = [32, 128, 512, 2048]
HOP_SIZES = [16, 64, 256, 1024]
DEFAULT_STEPS = 2500
DEFAULT_SEGMENT_LENGTH = 228308
DEFAULT_LR = 1.0e-3
STEER_CONDITIONING = 0.0
DEFAULT_TRAIN_DEVICE = "auto"
TRAIN_DEVICE_CHOICES = ("auto", "cpu", "mps", "cuda")
DEFAULT_MLFLOW_EXPERIMENT = "openyourbox"
DEFAULT_MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
DEFAULT_MLFLOW_TAGS = ["train", "steerable"]
MLFLOW_PARAM_MAX_LENGTH = 6000

# acids-rave ``rave/configs/v1.gin`` reconstruction defaults.
RAVE_V1_SPECTRAL_WINDOWS = (2048, 1024, 512, 256, 128)
RAVE_V1_STAGE1_STEPS = 1_000_000
RAVE_V1_STAGE2_STEPS = 1_000_000
RAVE_V1_GENERATOR_LR = 1.0e-3
RAVE_V1_DISCRIMINATOR_LR = 1.0e-4
RAVE_V1_ADAM_BETAS = (0.5, 0.9)
RAVE_V1_LR_DECAY_END = 0.1
RAVE_V1_BATCH_SIZE = 8
RAVE_V1_SEGMENT_LENGTH = 65536
RAVE_V1_KL_BETA = 0.1
RAVE_V1_KL_BETA_START = 0.1
RAVE_V1_KL_WARMUP_STEPS = 1
RAVE_V1_FEATURE_MATCHING_WEIGHT = 10.0
RAVE_V1_UPDATE_DISCRIMINATOR_EVERY = 2
RAVE_V1_PHASE_MANGLE_PROB = 0.8
RAVE_V1_DEQUANTIZE_BITS = 16
RAVE_V1_LOG_EPSILON = 1.0e-7
RAVE_V1_DISC_N_SCALES = 3
RAVE_V1_DISC_CAPACITY = 64
RAVE_V1_DISC_N_LAYERS = 4
RAVE_V1_DISC_KERNEL = 15
RAVE_V1_DISC_STRIDE = 4


def parse_train_device(value: Any) -> str:
    """Normalize a `train_options.device` token to a known choice."""
    token = str(value or DEFAULT_TRAIN_DEVICE).strip().lower()
    if token in TRAIN_DEVICE_CHOICES:
        return token
    return DEFAULT_TRAIN_DEVICE


def available_train_devices() -> list[str]:
    """Return PyTorch backends the current interpreter can actually use."""
    devices = ["cpu"]
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and bool(mps.is_available()):
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def _mps_is_available() -> bool:
    """Return true when the MPS backend is present and enabled."""
    mps = getattr(torch.backends, "mps", None)
    return mps is not None and bool(mps.is_available())


def _probe_train_device(device: torch.device) -> bool:
    """Run a short conv+STFT backward to reject accelerators that cannot train."""
    if device.type == "cpu":
        return True
    try:
        samples = torch.randn(2, 1, 64, device=device, requires_grad=True)
        conv = nn.Conv1d(1, 2, 3, padding=1).to(device)
        predicted = conv(samples)
        window = torch.hann_window(16, device=device)
        spectrum = torch.stft(
            predicted[:, 0],
            n_fft=16,
            hop_length=8,
            window=window,
            return_complex=True,
        )
        spectrum.abs().mean().backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            synchronize = getattr(torch.mps, "synchronize", None)
            if synchronize is not None:
                synchronize()
        return True
    except Exception:
        return False


def resolve_train_device(requested: Any) -> tuple[torch.device, str, str]:
    """Pick a usable torch device for one train job.

    ``auto`` prefers CUDA, then MPS, then CPU. An explicit accelerator that is
    missing or fails a short train-op probe falls back to CPU.
    """
    requested_token = parse_train_device(requested)
    candidates: list[str] = []
    if requested_token == "auto":
        if torch.cuda.is_available():
            candidates.append("cuda")
        if _mps_is_available():
            candidates.append("mps")
        candidates.append("cpu")
    elif requested_token == "cuda":
        candidates = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    elif requested_token == "mps":
        candidates = ["mps", "cpu"] if _mps_is_available() else ["cpu"]
    else:
        candidates = ["cpu"]
    for name in candidates:
        device = torch.device(name)
        if _probe_train_device(device):
            return device, requested_token, name
    return torch.device("cpu"), requested_token, "cpu"


def train_device_event_fields(
    device: torch.device, requested: str
) -> dict[str, Any]:
    """JSON fields describing requested vs effective training devices."""
    effective = str(device.type)
    return {
        "device": effective,
        "requested_device": requested,
        "available_devices": available_train_devices(),
        "device_fallback": requested not in {"auto", effective},
    }


def _module_device(module: nn.Module) -> torch.device:
    """Return the device of the first parameter or buffer, else CPU."""
    for parameter in module.parameters():
        return parameter.device
    for buffer in module.buffers():
        return buffer.device
    return torch.device("cpu")


def _emit_train(
    payload: dict[str, Any], device: torch.device, requested: str
) -> None:
    """Emit one worker event including the resolved training device."""
    payload.update(train_device_event_fields(device, requested))
    _emit(payload)


def dilation_for_layer(growth: int, layer: int) -> int:
    """Return saturated ``growth**layer``."""
    value = 1
    growth = max(1, int(growth))
    for _ in range(max(0, layer)):
        value *= growth
        if value > 2**30:
            return 2**30
    return value


def receptive_field_samples(depth: int, kernel_size: int, growth: int) -> int:
    """Return the causal receptive field of a growth^n TCN stack."""
    field = 1
    for layer in range(max(0, depth)):
        field += (kernel_size - 1) * dilation_for_layer(growth, layer)
    return field


def rf_aware_crop(
    clean: torch.Tensor,
    processed: torch.Tensor,
    receptive_field: int,
    segment_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop a target segment with preceding receptive-field context.

    ``clean``/``processed`` are shaped ``[batch, channels, time]``. The returned
    input includes ``receptive_field - 1`` samples of causal context before the
    target window of up to ``segment_length`` samples.
    """
    available = int(clean.shape[-1])
    context = max(1, int(receptive_field))
    if available <= context:
        return clean, processed[..., -1:]

    usable = available - context
    length = min(int(segment_length), usable, DEFAULT_SEGMENT_LENGTH)
    length = max(1, length)
    start = context
    stop = start + length
    x_crop = clean[..., start - context + 1 : stop]
    y_crop = processed[..., start:stop]
    return x_crop, y_crop


class CausalConv1d(nn.Module):
    """Apply a left-padded causal one-dimensional convolution."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        dilation: int,
        weight_norm: bool = False,
    ) -> None:
        """Create a causal convolution with validated dimensions."""
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        convolution = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size,
            dilation=dilation,
            bias=False,
        )
        self.convolution = _maybe_weight_norm(convolution, weight_norm)

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        """Process ``[batch, channels, time]`` samples without future context."""
        return self.convolution(functional.pad(samples, (self.left_padding, 0)))


def _weight_norm_requested(properties: dict[str, Any]) -> bool:
    """Return True when a graph element opts into acids-rave weight normalization."""
    return bool(int(properties.get("weight_norm", 0)))


def _maybe_weight_norm(module: nn.Module, enabled: bool) -> nn.Module:
    """Wrap ``module`` with deepcopy-safe parametrized weight normalization."""
    if not enabled:
        return module
    return nn.utils.parametrizations.weight_norm(module)


def _has_weight_norm(module: nn.Module) -> bool:
    """Return True when ``module`` still uses a weight-norm parametrization."""
    if nn.utils.parametrize.is_parametrized(module, "weight"):
        return True
    return hasattr(module, "weight_g") and hasattr(module, "weight_v")


def strip_weight_norm(module: nn.Module) -> None:
    """Materialize ``W = g * v / ||v||`` so live/TorchScript see plain ``.weight``.

    Safe to call when no weight-norm hooks are present. Handles both
    ``parametrizations.weight_norm`` and legacy ``nn.utils.weight_norm``.
    """
    for child in list(module.modules()):
        if nn.utils.parametrize.is_parametrized(child, "weight"):
            try:
                nn.utils.parametrize.remove_parametrizations(
                    child, "weight", leave_parametrized=True
                )
            except (ValueError, KeyError, AttributeError):
                pass
        if not hasattr(child, "weight_g") or not hasattr(child, "weight_v"):
            continue
        try:
            nn.utils.remove_weight_norm(child)
        except (ValueError, KeyError, AttributeError):
            continue


def assign_plain_conv_weight(layer: nn.Module, weight: torch.Tensor) -> None:
    """Copy a dense weight into a Conv1d/ConvTranspose1d, including weight-norm."""
    if nn.utils.parametrize.is_parametrized(layer, "weight"):
        nn.utils.parametrize.remove_parametrizations(
            layer, "weight", leave_parametrized=True
        )
        layer.weight.copy_(weight)
        nn.utils.parametrizations.weight_norm(layer)
        return
    if hasattr(layer, "weight_v") and hasattr(layer, "weight_g"):
        with torch.no_grad():
            layer.weight_v.copy_(weight)
            # Match torch.nn.utils.weight_norm init: g = ||v|| over dim 0.
            norms = torch.linalg.vector_norm(
                weight.reshape(weight.shape[0], -1), ord=2, dim=1
            )
            layer.weight_g.copy_(norms.reshape(layer.weight_g.shape))
        return
    layer.weight.copy_(weight)


class FiLM(nn.Module):
    """Per-sample feature-wise linear modulation from a conditioning trajectory."""

    def __init__(self, cond_dim: int, num_features: int) -> None:
        """Create a linear adaptor mapping ``c`` to per-channel scale and shift."""
        super().__init__()
        self.cond_dim = max(1, int(cond_dim))
        self.adaptor = nn.Linear(self.cond_dim, num_features * 2)

    def forward(self, samples: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply FiLM: ``y_t = g(c_t) * x_t + b(c_t)``.

        ``cond`` is ``[batch, cond_dim, time]``. A last dim of 1 (training
        recipe) broadcasts like global conditioning. Exports are traced with
        ``time`` matching the audio block so live Knob/XY ramps stay per-sample.
        Time-varying live control must not be averaged per audio buffer, or
        steering zippers at block rate.
        """
        if cond.dim() == 1:
            cond = cond.view(1, -1, 1)
        elif cond.dim() == 2:
            cond = cond.unsqueeze(-1)
        layout = cond.transpose(1, 2)
        padded = functional.pad(layout, (0, self.cond_dim))
        adapted = self.adaptor(padded[:, :, : self.cond_dim])
        scale, shift = torch.chunk(adapted, 2, dim=-1)
        return samples * scale.transpose(1, 2) + shift.transpose(1, 2)


class TCNBlock(nn.Module):
    """One causal TCN block with optional FiLM, residual, and PReLU."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        activation: int,
        residual: bool,
        cond_dim: int,
        negative_slope: float = 0.01,
    ) -> None:
        """Create one temporal block."""
        super().__init__()
        self.convolution = CausalConv1d(channels, channels, kernel_size, dilation)
        self.film = FiLM(cond_dim, channels) if cond_dim > 0 else None
        self.activation = _activation(activation, channels, negative_slope)
        self.residual = residual
        self.residual_projection = (
            nn.Conv1d(channels, channels, 1, bias=False) if residual else None
        )

    def forward(self, samples: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Run convolution, optional FiLM, activation, and residual add."""
        residual = samples
        value = self.convolution(samples)
        if self.film is not None:
            value = self.film(value, cond)
        value = self.activation(value)
        if self.residual_projection is not None:
            cropped = residual[..., -value.shape[-1] :]
            value = value + self.residual_projection(cropped)
        return value


class SteerableTCN(nn.Module):
    """FiLM-conditioned TCN matching the live graph TCN element."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        output_channels: int,
        depth: int,
        kernel_size: int,
        dilation_growth: int,
        activation: int,
        residual: bool,
        cond_dim: int,
        negative_slope: float = 0.01,
    ) -> None:
        """Create input projection, temporal blocks, and output projection."""
        super().__init__()
        self.input_projection = nn.Conv1d(input_channels, hidden_channels, 1, bias=False)
        self.blocks = nn.ModuleList(
            [
                TCNBlock(
                    hidden_channels,
                    kernel_size,
                    dilation_for_layer(dilation_growth, layer),
                    activation,
                    residual,
                    cond_dim,
                    negative_slope,
                )
                for layer in range(depth)
            ]
        )
        self.output_projection = nn.Conv1d(hidden_channels, output_channels, 1, bias=False)
        self.cond_dim = cond_dim

    def forward(self, samples: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Process audio, applying ca=0 when conditioning is omitted."""
        if cond is None:
            cond = torch.zeros(
                samples.shape[0],
                self.cond_dim,
                1,
                device=samples.device,
                dtype=samples.dtype,
            )
        value = self.input_projection(samples)
        for block in self.blocks:
            value = block(value, cond)
        return self.output_projection(value)


class ConditionedSequential(nn.Module):
    """Sequential graph that forwards optional conditioning into TCN blocks."""

    def __init__(
        self,
        modules: list[nn.Module],
        cond_dim: int,
        fragment: dict[str, Any] | None = None,
        input_channels: int = 1,
        layer_ids: list[int] | None = None,
    ) -> None:
        """Store ordered modules, conditioning width, and the source fragment.

        ``fragment`` / ``input_channels`` let export rebuild a clone instead of
        deepcopying a CUDA training graph (which can alias weights and then
        fail TorchScript with a channel mismatch).
        """
        super().__init__()
        self.layers = nn.ModuleList(modules)
        self.layer_ids = list(layer_ids or [])
        self.cond_dim = cond_dim
        self.fragment = fragment
        self.input_channels = max(1, int(input_channels))

    def forward(self, samples: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Run each layer, passing ``cond`` into steerable TCN modules."""
        if cond is None and self.cond_dim > 0:
            cond = torch.zeros(
                samples.shape[0],
                self.cond_dim,
                1,
                device=samples.device,
                dtype=samples.dtype,
            )
        value = match_input_channels(samples, int(self.input_channels))
        for layer in self.layers:
            if isinstance(layer, SteerableTCN):
                value = layer(value, cond)
            else:
                value = layer(value)
        return value


class ZeroPreservingSigmoid(nn.Module):
    """Apply sigmoid while retaining exact zeros."""

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        """Map nonzero values through sigmoid and keep zeros exactly zero."""
        return torch.where(samples == 0.0, torch.zeros_like(samples), torch.sigmoid(samples))


def _activation(index: int, channels: int = 1, negative_slope: float = 0.01) -> nn.Module:
    """Return the activation represented by an OpenYourBox enum value."""
    activations: tuple[nn.Module, ...] = (
        nn.ReLU(),
        ZeroPreservingSigmoid(),
        nn.Tanh(),
        nn.LeakyReLU(float(negative_slope)),
        nn.PReLU(num_parameters=max(1, channels)),
    )
    if index < 0 or index >= len(activations):
        raise ValueError(f"unsupported activation index {index}")
    return activations[index]


def _properties(element: dict[str, Any]) -> dict[str, Any]:
    """Convert an element's ordered property array to a lookup dictionary."""
    values: dict[str, Any] = {}
    for item in element.get("properties", []):
        key = str(item["key"])
        if "string_value" in item:
            values[key] = str(item["string_value"])
        elif "float_value" in item:
            values[key] = float(item["float_value"])
        else:
            raw = item.get("value")
            try:
                values[key] = int(raw)
            except (TypeError, ValueError):
                values[key] = raw
    return values


_UTILITY_TYPES = frozenset({"utility", "merge", "sum", "multiply"})
_SKIP_GRAPH_IO_TYPES = frozenset(
    {
        "audio_input",
        "audio_output",
        "knob_input",
        "xy_trackpad",
        "group_input",
        "group_output",
        "data_loader",
        "dataLoader",
        "loss",
    }
)


@torch.jit.script
def match_time_to(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Crop the causal tail or left-pad ``value`` to ``reference``'s time length.

    Matches live ``matchTimeLength``: keep the newest samples when cropping, and
    insert zeros on the left when padding so alignment stays causal. TorchScript
    keeps the comparison dynamic so ``jit.trace`` cannot bake an example-block
    length into a RAVE export.
    """
    samples = int(reference.size(-1))
    current = int(value.size(-1))
    if current == samples:
        return value
    if current > samples:
        return value.narrow(2, current - samples, samples)
    return torch.nn.functional.pad(value, (samples - current, 0))


@torch.jit.script
def match_input_channels(samples: torch.Tensor, channels: int) -> torch.Tensor:
    """Fold or pad ``[B, C, T]`` to ``channels`` without baking the example width.

    Used by mapping export so a mono 1x1 expander (user-library TCN-with-conv)
    can be traced even if TorchScript is handed a stereo crop.
    """
    current = int(samples.size(1))
    width = int(channels)
    if current == width:
        return samples
    if width == 1:
        return samples.mean(dim=1, keepdim=True)
    if current == 1:
        return samples.repeat(1, width, 1)
    if current > width:
        return samples[:, :width, :]
    pad = torch.zeros(
        int(samples.size(0)),
        width - current,
        int(samples.size(2)),
        dtype=samples.dtype,
        device=samples.device,
    )
    return torch.cat((samples, pad), 1)


@torch.jit.script
def broadcast_channels(value: torch.Tensor, channels: int) -> torch.Tensor:
    """Expand a 1-channel tensor to ``channels``; otherwise return as-is."""
    current = int(value.size(1))
    if current == channels:
        return value
    if current == 1 and channels > 1:
        return value.expand(value.size(0), channels, value.size(2)).contiguous()
    return value


@torch.jit.script
def combine_pair(left: torch.Tensor, right: torch.Tensor, mode: int) -> torch.Tensor:
    """Align ``right`` to ``left`` then add (0), multiply (1), or concatenate (2)."""
    right = match_time_to(right, left)
    if mode == 2:
        return torch.cat((left, right), 1)
    width = int(left.size(1))
    right_channels = int(right.size(1))
    if right_channels > width:
        width = right_channels
    left = broadcast_channels(left, width)
    right = broadcast_channels(right, width)
    if mode == 1:
        return left * right
    return left + right


@torch.jit.script
def literal_like(reference: torch.Tensor, width: int, value: float) -> torch.Tensor:
    """Broadcast a scalar to ``[B, width, T]`` using ``reference``'s batch and time."""
    filled = torch.full(
        (int(reference.size(0)), 1, int(reference.size(-1))),
        value,
        dtype=reference.dtype,
        device=reference.device,
    )
    return filled.expand(int(reference.size(0)), width, int(reference.size(-1))).contiguous()


def _utility_mode(element_type: str, properties: dict[str, Any]) -> int:
    """Map a Utility or legacy mixer element to live merge mode 0–2."""
    if element_type == "sum":
        return 0
    if element_type == "multiply":
        return 1
    return max(0, min(2, int(properties.get("mode", 0))))


class MathExpression(nn.Module):
    """Elementwise arithmetic over ``x1``…``xN`` matching the VST grammar."""

    def __init__(self, expression: str, n_inputs: int = 1) -> None:
        """Compile @p expression for up to @p n_inputs named ``xK`` pins."""
        super().__init__()
        self.expression = str(expression)
        self.n_inputs = max(1, int(n_inputs))
        self._program = _compile_math_expression(self.expression, self.n_inputs)

    def forward(self, *inputs: torch.Tensor | None) -> torch.Tensor:
        """Evaluate the prepared program; a single tensor binds ``x1``."""
        tensors: list[torch.Tensor | None] = list(inputs)
        if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
            tensors = list(tensors[0])
        while len(tensors) < self.n_inputs:
            tensors.append(None)
        return _eval_math_expression(self._program, tensors)


class UtilityCombine(nn.Module):
    """Add, multiply, or concatenate graph inputs with live-graph time alignment."""

    def __init__(self, mode: int = 0) -> None:
        """Store Utility mode: 0 add, 1 multiply, 2 concatenate."""
        super().__init__()
        self.mode = max(0, min(2, int(mode)))

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """Fold aligned inputs left-to-right using :func:`combine_pair`."""
        stacked: list[torch.Tensor] = list(inputs)
        if len(stacked) == 1 and isinstance(stacked[0], (list, tuple)):
            stacked = list(stacked[0])
        if not stacked:
            raise ValueError("Utility has no connected input tensors")
        current = stacked[0]
        mode = int(self.mode)
        for extra in stacked[1:]:
            current = combine_pair(current, extra, mode)
        return current


def _compile_math_expression(text: str, n_inputs: int) -> list[tuple[str, float | int | None]]:
    """Parse the shared ``()+-*/^`` / ``exp()`` grammar into postfix ops."""
    source = str(text)
    index = 0

    def skip() -> None:
        nonlocal index
        while index < len(source) and source[index].isspace():
            index += 1

    def fail(message: str) -> None:
        raise ValueError(message)

    def parse_number() -> float:
        nonlocal index
        start = index
        if index < len(source) and source[index] == ".":
            index += 1
        while index < len(source) and source[index].isdigit():
            index += 1
        if index < len(source) and source[index] == "." and "." not in source[start:index]:
            index += 1
            while index < len(source) and source[index].isdigit():
                index += 1
        if index < len(source) and source[index] in "eE":
            exp = index
            index += 1
            if index < len(source) and source[index] in "+-":
                index += 1
            if index >= len(source) or not source[index].isdigit():
                index = exp
            else:
                while index < len(source) and source[index].isdigit():
                    index += 1
        if start == index:
            fail("Expected a number")
        return float(source[start:index])

    def parse_ident() -> str:
        nonlocal index
        start = index
        index += 1
        while index < len(source) and (source[index].isalnum() or source[index] == "_"):
            index += 1
        return source[start:index]

    def parse_primary() -> list[tuple[str, float | int | None]]:
        nonlocal index
        skip()
        if index >= len(source):
            fail("Expression is incomplete")
        ch = source[index]
        if ch.isdigit() or ch == ".":
            return [("lit", parse_number())]
        if ch.isalpha() or ch == "_":
            name = parse_ident()
            if name == "exp":
                skip()
                if index >= len(source) or source[index] != "(":
                    fail("exp requires parentheses: exp(...)")
                index += 1
                inner = parse_add()
                skip()
                if index >= len(source) or source[index] != ")":
                    fail("Missing closing parenthesis")
                index += 1
                return inner + [("exp", None)]
            if not name.startswith("x") or len(name) < 2 or not name[1:].isdigit():
                fail(f"Unknown symbol '{name}'; use x1, x2, … for inputs")
            pin = int(name[1:])
            if pin < 1 or pin > n_inputs:
                fail(f"'{name}' is not a configured input (Inputs = {n_inputs})")
            return [("ident", pin)]
        if ch == "(":
            index += 1
            inner = parse_add()
            skip()
            if index >= len(source) or source[index] != ")":
                fail("Missing closing parenthesis")
            index += 1
            return inner
        fail("Expected a number, identifier, exp(...), or '('")
        return []

    def parse_unary() -> list[tuple[str, float | int | None]]:
        nonlocal index
        skip()
        if index < len(source) and source[index] == "-":
            index += 1
            return parse_unary() + [("neg", None)]
        return parse_primary()

    def parse_pow() -> list[tuple[str, float | int | None]]:
        nonlocal index
        left = parse_unary()
        skip()
        if index < len(source) and source[index] == "^":
            index += 1
            return left + parse_pow() + [("pow", None)]
        # `**` is an ASCII-friendly synonym for power (AZERTY `^` is a dead key).
        if (
            index + 1 < len(source)
            and source[index] == "*"
            and source[index + 1] == "*"
        ):
            index += 2
            return left + parse_pow() + [("pow", None)]
        return left

    def parse_mul() -> list[tuple[str, float | int | None]]:
        nonlocal index
        left = parse_pow()
        while True:
            skip()
            if (
                index < len(source)
                and source[index] == "*"
                and not (
                    index + 1 < len(source) and source[index + 1] == "*"
                )
            ):
                index += 1
                left = left + parse_pow() + [("mul", None)]
            elif index < len(source) and source[index] == "/":
                index += 1
                left = left + parse_pow() + [("div", None)]
            else:
                return left

    def parse_add() -> list[tuple[str, float | int | None]]:
        nonlocal index
        left = parse_mul()
        while True:
            skip()
            if index >= len(source):
                return left
            if source[index] == "+":
                index += 1
                left = left + parse_mul() + [("add", None)]
            elif source[index] == "-":
                index += 1
                left = left + parse_mul() + [("sub", None)]
            else:
                return left

    skip()
    if index >= len(source):
        fail("Expression is empty")
    program = parse_add()
    skip()
    if index < len(source):
        fail(f"Unexpected '{source[index]}' in expression")
    return program


def _eval_math_expression(
    program: list[tuple[str, float | int | None]],
    inputs: list[torch.Tensor | None],
) -> torch.Tensor:
    """Run a compiled Math Expression program over pin-ordered tensors."""
    stack: list[torch.Tensor] = []
    reference: torch.Tensor | None = None
    for item in inputs:
        if item is not None:
            reference = item
            break
    if reference is None:
        raise ValueError("Math Expression has no connected input tensors")
    width = 1
    for item in inputs:
        if item is not None and item.ndim >= 2:
            width = max(width, int(item.shape[-2]))

    def literal(value: float) -> torch.Tensor:
        return literal_like(reference, width, float(value))

    def align(left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        right = match_time_to(right, left)
        channels = int(left.size(1))
        right_channels = int(right.size(1))
        if right_channels > channels:
            channels = right_channels
        left = broadcast_channels(left, channels)
        right = broadcast_channels(right, channels)
        return left, right

    for op, payload in program:
        if op == "lit":
            stack.append(literal(float(payload)))
        elif op == "ident":
            pin = int(payload) - 1
            if pin < 0 or pin >= len(inputs) or inputs[pin] is None:
                raise ValueError(f"Missing tensor for x{pin + 1}")
            stack.append(inputs[pin])
        elif op == "neg":
            stack[-1] = -stack[-1]
        elif op == "exp":
            stack[-1] = torch.exp(stack[-1])
        else:
            right = stack.pop()
            left = stack.pop()
            left, right = align(left, right)
            if op == "add":
                stack.append(left + right)
            elif op == "sub":
                stack.append(left - right)
            elif op == "mul":
                stack.append(left * right)
            elif op == "div":
                stack.append(left / right)
            elif op == "pow":
                stack.append(torch.pow(left, right))
            else:
                raise ValueError(f"unknown math op {op}")
    if not stack:
        raise ValueError("Invalid math expression")
    return stack[-1]


def _topological_elements(fragment: dict[str, Any]) -> list[dict[str, Any]]:
    """Return selected elements in stable topological order."""
    elements = fragment.get("elements", [])
    if not elements:
        raise ValueError("selected graph contains no elements")

    by_id = {int(element["id"]): element for element in elements}
    indegree = {element_id: 0 for element_id in by_id}
    outgoing: dict[int, list[int]] = {element_id: [] for element_id in by_id}
    for connection in fragment.get("connections", []):
        source = int(connection["source_element_id"])
        destination = int(connection["destination_element_id"])
        if source in by_id and destination in by_id:
            outgoing[source].append(destination)
            indegree[destination] += 1

    ready = sorted(element_id for element_id, count in indegree.items() if count == 0)
    ordered: list[dict[str, Any]] = []
    while ready:
        current = ready.pop(0)
        ordered.append(by_id[current])
        for destination in sorted(outgoing[current]):
            indegree[destination] -= 1
            if indegree[destination] == 0:
                ready.append(destination)
                ready.sort()

    if len(ordered) != len(elements):
        raise ValueError("selected graph is cyclic")
    return ordered



def _exp_sigmoid(value: torch.Tensor) -> torch.Tensor:
    """Magenta ``core.exp_sigmoid`` used by DDSP effects."""
    return 2.0 * torch.sigmoid(value).pow(math.log(10.0)) + 1.0e-7


def _mask_dry_ir(ir: torch.Tensor) -> torch.Tensor:
    """Zero the first IR sample so dry energy is not double-counted."""
    masked = ir.reshape(-1).clone()
    if masked.numel() > 0:
        masked[0] = 0.0
    return masked


class ExpDecayReverb(nn.Module):
    """Exponential-decay noise IR followed by causal FIR convolution."""

    def __init__(self, gain, decay, reverb_length, add_dry, seed=42):
        super().__init__()
        self.reverb_length = max(1, int(reverb_length))
        self.add_dry = bool(add_dry)
        generator = torch.Generator().manual_seed(int(seed) & 0xFFFFFFFF)
        time = torch.linspace(0.0, 1.0, self.reverb_length)
        noise = torch.rand(self.reverb_length, generator=generator) * 2.0 - 1.0
        scaled = float(_exp_sigmoid(torch.tensor(float(gain))))
        decay_exponent = 2.0 + math.exp(float(decay))
        ir = scaled * torch.exp(-decay_exponent * time) * noise
        self.register_buffer("ir", ir)

    def forward(self, audio):
        """Convolve each channel with the baked impulse response."""
        ir = _mask_dry_ir(self.ir)
        channels = audio.shape[1]
        weight = ir.flip(0).view(1, 1, -1).repeat(channels, 1, 1)
        padded = functional.pad(audio, (ir.numel() - 1, 0))
        wet = functional.conv1d(padded, weight, groups=channels)
        return wet + audio if self.add_dry else wet


class MagnitudeFir(nn.Module):
    """Trainable magnitude-grid FIR or filtered-noise reverb."""

    def __init__(self, n_frames, n_filter_banks, window_size, reverb_length, add_dry, filtered_noise, seed=42):
        super().__init__()
        self.n_frames = max(1, int(n_frames))
        self.n_filter_banks = max(1, int(n_filter_banks))
        self.window_size = max(1, int(window_size) | 1)
        self.reverb_length = max(1, int(reverb_length))
        self.add_dry = bool(add_dry)
        self.filtered_noise = bool(filtered_noise)
        generator = torch.Generator().manual_seed(int(seed) & 0xFFFFFFFF)
        self.magnitudes = nn.Parameter(
            torch.randn(self.n_frames, self.n_filter_banks, generator=generator) * 0.01
        )

    def _impulse(self):
        grid = _exp_sigmoid(self.magnitudes)
        n_bins = self.window_size // 2 + 1
        spec = torch.zeros(self.n_frames, n_bins, device=grid.device, dtype=grid.dtype)
        copy = min(self.n_filter_banks, n_bins)
        spec[:, :copy] = grid[:, :copy]
        ir_frames = torch.fft.irfft(spec.to(torch.complex64), n=self.window_size, dim=-1)
        window = torch.hann_window(self.window_size, periodic=False, device=grid.device)
        ir_frames = ir_frames * window
        if self.filtered_noise:
            mean = ir_frames.abs().mean(-1)
            length = self.reverb_length
            envelope = torch.zeros(length, device=grid.device, dtype=grid.dtype)
            hop = max(1, length // self.n_frames)
            for frame in range(self.n_frames):
                start = min(length - 1, frame * hop)
                end = length if frame + 1 == self.n_frames else min(length, (frame + 1) * hop)
                envelope[start:end] = mean[frame]
            noise = torch.rand(length, device=grid.device) * 2 - 1
            return noise * envelope
        ir = ir_frames.mean(0)
        return torch.roll(ir, self.window_size // 2, 0)

    def forward(self, audio):
        """Apply the magnitude-derived FIR to every channel."""
        ir = self._impulse()
        if self.filtered_noise:
            ir = _mask_dry_ir(ir)
        channels = audio.shape[1]
        weight = ir.flip(0).view(1, 1, -1).repeat(channels, 1, 1)
        padded = functional.pad(audio, (ir.numel() - 1, 0))
        wet = functional.conv1d(padded, weight, groups=channels)
        if self.filtered_noise and self.add_dry:
            return wet + audio
        return wet


class ConvolutionalReverb(nn.Module):
    """Trainable internal IR with optional external IR blend."""

    def __init__(self, reverb_length, add_dry, ir_blend, seed=42):
        super().__init__()
        self.reverb_length = max(1, int(reverb_length))
        self.add_dry = bool(add_dry)
        self.ir_blend = float(ir_blend)
        generator = torch.Generator().manual_seed(int(seed) & 0xFFFFFFFF)
        self.ir = nn.Parameter(torch.randn(self.reverb_length, generator=generator) * 1.0e-6)

    def forward(self, audio, external_ir=None):
        """Convolve audio with the blended impulse response."""
        ir = self.ir
        if external_ir is not None and external_ir.numel() > 0:
            flat = external_ir.reshape(-1)[: self.reverb_length]
            if flat.numel() < self.reverb_length:
                flat = functional.pad(flat, (0, self.reverb_length - flat.numel()))
            ir = (1.0 - self.ir_blend) * ir + self.ir_blend * flat
        ir = _mask_dry_ir(ir)
        channels = audio.shape[1]
        weight = ir.flip(0).view(1, 1, -1).repeat(channels, 1, 1)
        padded = functional.pad(audio, (ir.numel() - 1, 0))
        wet = functional.conv1d(padded, weight, groups=channels)
        return wet + audio if self.add_dry else wet


class ModDelay(nn.Module):
    """Constant-phase Magenta modulated delay (scalar controls)."""

    def __init__(self, center_ms, depth_ms, gain, phase, add_dry, sample_rate=48000.0):
        super().__init__()
        self.center_ms = float(center_ms)
        self.depth_ms = float(depth_ms)
        self.gain = float(gain)
        self.phase = float(phase)
        self.add_dry = bool(add_dry)
        self.sample_rate = float(sample_rate)

    def forward(self, audio):
        """Apply a static delay at the Magenta-mapped phase position."""
        max_delay_ms = max(0.0, self.center_ms) + max(0.0, self.depth_ms)
        max_samples = max(1, int(self.sample_rate / 1000.0 * max_delay_ms))
        depth_phase = 0.0 if max_delay_ms <= 0 else self.depth_ms / max_delay_ms
        center_phase = 0.0 if max_delay_ms <= 0 else self.center_ms / max_delay_ms
        mapped = max(0.0, min(1.0, self.phase)) * depth_phase + center_phase
        delay = int(round(mapped * max_samples))
        wet_gain = float(_exp_sigmoid(torch.tensor(self.gain)))
        wet = functional.pad(audio, (delay, 0))[..., : audio.shape[-1]] * wet_gain
        return wet + audio if self.add_dry else wet


def _cell_activation(value, index, gain, negative_slope):
    """In-cell nonlinearity matching Activation/TCN choices."""
    if abs(gain - 1.0) > 1.0e-6:
        value = value * gain
    if index == 0:
        return functional.relu(value)
    if index == 1:
        return torch.where(value == 0, torch.zeros_like(value), torch.sigmoid(value))
    if index == 3:
        return functional.leaky_relu(value, negative_slope)
    if index == 4:
        return functional.prelu(value, torch.tensor(0.25, device=value.device, dtype=value.dtype))
    return torch.tanh(value)


class RecurrentLayer(nn.Module):
    """Single-layer custom RNN or LSTM with in-cell activation and gain.

    ``leak_rate`` mixes the previous hidden state after each step
    (``1`` is a standard cell). ``recurrent_weight_scale`` multiplies
    hidden-to-hidden weights only.
    """

    def __init__(self, input_size, hidden_size, bidirectional, bias, activation, gain,
                 negative_slope, lstm, leak_rate=1.0, recurrent_weight_scale=1.0):
        super().__init__()
        self.input_size = max(1, int(input_size))
        self.hidden_size = max(1, int(hidden_size))
        self.bidirectional = bool(bidirectional)
        self.use_bias = bool(bias)
        self.activation = int(activation)
        self.gain = float(gain)
        self.negative_slope = float(negative_slope)
        self.lstm = bool(lstm)
        self.leak_rate = float(leak_rate)
        self.recurrent_weight_scale = float(recurrent_weight_scale)
        gate = self.hidden_size * 4 if self.lstm else self.hidden_size
        directions = 2 if self.bidirectional else 1
        for direction in range(directions):
            suffix = "" if direction == 0 else "_reverse"
            setattr(self, f"weight_ih{suffix}", nn.Parameter(torch.empty(gate, self.input_size)))
            setattr(self, f"weight_hh{suffix}", nn.Parameter(torch.empty(gate, self.hidden_size)))
            if self.use_bias:
                setattr(self, f"bias_ih{suffix}", nn.Parameter(torch.zeros(gate)))
                setattr(self, f"bias_hh{suffix}", nn.Parameter(torch.zeros(gate)))
            nn.init.xavier_uniform_(getattr(self, f"weight_ih{suffix}"))
            nn.init.xavier_uniform_(getattr(self, f"weight_hh{suffix}"))

    def _step(self, sample, hidden, cell, weight_ih, weight_hh, bias_ih, bias_hh):
        prev_hidden = hidden
        recurrent_weight = weight_hh * self.recurrent_weight_scale
        gate = functional.linear(sample, weight_ih, bias_ih) + functional.linear(
            hidden, recurrent_weight, bias_hh)
        if self.lstm:
            chunks = gate.chunk(4, dim=-1)
            input_gate = torch.sigmoid(chunks[0])
            forget = torch.sigmoid(chunks[1])
            candidate = _cell_activation(chunks[2], self.activation, self.gain, self.negative_slope)
            output_gate = torch.sigmoid(chunks[3])
            cell = forget * cell + input_gate * candidate
            hidden = output_gate * _cell_activation(cell, self.activation, self.gain, self.negative_slope)
        else:
            hidden = _cell_activation(gate, self.activation, self.gain, self.negative_slope)
        leak = min(1.0, max(0.0, self.leak_rate))
        if abs(leak - 1.0) > 1.0e-6:
            hidden = prev_hidden * (1.0 - leak) + hidden * leak
        return hidden, cell

    def _run_direction(self, audio, reverse, weight_ih, weight_hh, bias_ih, bias_hh):
        batch, _, time = audio.shape
        hidden = torch.zeros(batch, self.hidden_size, device=audio.device, dtype=audio.dtype)
        cell = torch.zeros(batch, self.hidden_size, device=audio.device, dtype=audio.dtype)
        outputs = []
        indices = range(time - 1, -1, -1) if reverse else range(time)
        for index in indices:
            hidden, cell = self._step(audio[:, :, index], hidden, cell, weight_ih, weight_hh, bias_ih, bias_hh)
            outputs.append(hidden)
        if reverse:
            outputs.reverse()
        return torch.stack(outputs, dim=2)

    def forward(self, audio):
        """Process a full sequence and emit hidden states at every time step."""
        bias_ih = getattr(self, "bias_ih", None) if self.use_bias else None
        bias_hh = getattr(self, "bias_hh", None) if self.use_bias else None
        forward = self._run_direction(audio, False, self.weight_ih, self.weight_hh, bias_ih, bias_hh)
        if not self.bidirectional:
            return forward
        reverse = self._run_direction(
            audio, True, self.weight_ih_reverse, self.weight_hh_reverse,
            getattr(self, "bias_ih_reverse", None) if self.use_bias else None,
            getattr(self, "bias_hh_reverse", None) if self.use_bias else None,
        )
        return torch.cat([forward, reverse], dim=1)


def make_ddsp_or_recurrent(element_type, properties, input_channels, seed=42):
    """Build a DDSP effect or recurrent layer, or return None for other types."""
    if element_type == "exp_decay_reverb":
        return ExpDecayReverb(float(properties.get("gain", 2.0)), float(properties.get("decay", 4.0)),
                              int(properties.get("reverb_length", 4096)), bool(int(properties.get("add_dry", 1))), seed), input_channels
    if element_type == "filtered_noise_reverb":
        return MagnitudeFir(int(properties.get("n_frames", 8)), int(properties.get("n_filter_banks", 16)),
                            int(properties.get("window_size", 257)), int(properties.get("reverb_length", 4096)),
                            bool(int(properties.get("add_dry", 1))), True, seed), input_channels
    if element_type == "fir_filter":
        return MagnitudeFir(int(properties.get("n_frames", 8)), int(properties.get("n_filter_banks", 16)),
                            int(properties.get("window_size", 257)), int(properties.get("window_size", 257)),
                            False, False, seed), input_channels
    if element_type == "reverb":
        return ConvolutionalReverb(int(properties.get("reverb_length", 4096)), bool(int(properties.get("add_dry", 1))),
                                   float(properties.get("ir_blend", 0.0)), seed), input_channels
    if element_type == "mod_delay":
        return ModDelay(float(properties.get("center_ms", 15.0)), float(properties.get("depth_ms", 10.0)),
                        float(properties.get("gain", 2.0)), float(properties.get("phase", 0.5)),
                        bool(int(properties.get("add_dry", 1)))), input_channels
    if element_type in {"lstm", "rnn"}:
        hidden = max(1, int(properties.get("hidden_size", 16)))
        bidirectional = bool(int(properties.get("bidirectional", 0)))
        layer = RecurrentLayer(input_channels, hidden, bidirectional, bool(int(properties.get("bias", 1))),
                               int(properties.get("activation", 2)), float(properties.get("gain", 1.0)),
                               float(properties.get("negative_slope", 0.01)), element_type == "lstm",
                               float(properties.get("leak_rate", 1.0)),
                               float(properties.get("recurrent_weight_scale", 1.0)))
        return layer, hidden * (2 if bidirectional else 1)
    return None


def build_module(
    fragment: dict[str, Any], input_channels: int = 1, cond_dim: int = 2
) -> nn.Module:
    """Construct a trainable module for an armed graph fragment."""
    rave_types = {
        "pqmf_analysis",
        "pqmf_synthesis",
        "rate_conv",
        "variational_bottleneck",
        "noise_synthesizer",
    }
    types = {
        str(element.get("type", ""))
        for element in fragment.get("elements", [])
        if isinstance(element, dict)
    }
    if types & rave_types or "math_expression" in types:
        return build_rave_graph_module(fragment, input_channels, cond_dim)

    modules: list[nn.Module] = []
    layer_ids: list[int] = []
    channels = input_channels
    cond_dim = max(1, int(cond_dim))
    for element in _topological_elements(fragment):
        element_type = str(element["type"])
        properties = _properties(element)
        if element_type in _SKIP_GRAPH_IO_TYPES:
            continue
        before = len(modules)
        if element_type == "activation":
            modules.append(_activation(int(properties.get("activation", 0)), channels,
                                       float(properties.get("negative_slope", 0.01))))
        elif element_type == "linear":
            output_channels = int(properties.get("features", channels))
            modules.append(nn.Conv1d(channels, output_channels, 1, bias=False))
            channels = output_channels
        elif element_type in {"conv1d", "rate_conv"}:
            output_channels = int(properties.get("channels", channels))
            modules.append(_make_strided_conv(channels, output_channels, properties))
            channels = output_channels
        elif element_type == "conv_transpose1d":
            output_channels = int(properties.get("channels", channels))
            modules.append(_make_conv_transpose(channels, output_channels, properties))
            channels = output_channels
        elif element_type == "batch_norm":
            modules.append(nn.BatchNorm1d(channels))
        elif element_type == "tcn":
            hidden = int(properties.get("channels", channels))
            depth = int(properties.get("depth", 4))
            kernel_size = int(properties.get("kernel_size", 3))
            growth = int(properties.get("dilation_growth", 2))
            activation = int(properties.get("activation", 0))
            residual = bool(int(properties.get("residual", 0)))
            modules.append(
                SteerableTCN(
                    channels,
                    hidden,
                    channels,
                    depth,
                    kernel_size,
                    growth,
                    activation,
                    residual,
                    cond_dim,
                    float(properties.get("negative_slope", 0.01)),
                )
            )
        elif element_type == "math_expression":
            modules.append(
                MathExpression(
                    str(properties.get("expression", "x1")),
                    int(properties.get("inputs", 1)),
                )
            )
        elif element_type in {"utility", "merge", "sum", "multiply"}:
            raise ValueError("mixer elements cannot be trained by this worker")
        else:
            built = make_ddsp_or_recurrent(element_type, properties, channels,
                                           int(element.get("seed", 42)))
            if built is None:
                continue
            module, channels = built
            modules.append(module)
        if len(modules) > before:
            layer_ids.append(int(element.get("id", 0) or 0))

    if not modules:
        return nn.Identity()
    return ConditionedSequential(
        modules,
        cond_dim,
        fragment=fragment,
        input_channels=input_channels,
        layer_ids=layer_ids,
    )


def _module_receptive_field(module: nn.Module) -> int:
    """Return the aggregate causal receptive field of constructed layers."""
    field = 1
    for layer in module.modules():
        if isinstance(layer, CausalConv1d):
            field += layer.left_padding
        if isinstance(layer, VariationalBottleneckLayer):
            field += layer.left_padding
        if isinstance(layer, RateConvLayer):
            field += max(0, int(layer.kernel_size) - 1) * int(layer.dilation)
    return field


class PqmfLayer(nn.Module):
    """Causal cosine-modulated PQMF analysis or synthesis."""

    def __init__(self, n_band: int, analysis: bool) -> None:
        """Create a fixed Kaiser-modulated bank."""
        super().__init__()
        self.n_band = max(1, int(n_band))
        self.analysis = bool(analysis)
        taps = max(4 * self.n_band + 1, 15)
        if taps % 2 == 0:
            taps += 1
        t = torch.arange(taps) - taps // 2
        cutoff = math.pi / self.n_band
        proto = torch.where(
            t == 0,
            torch.full_like(t, cutoff / math.pi, dtype=torch.float32),
            torch.sin(cutoff * t.float()) / (math.pi * t.float()),
        )
        window = torch.hann_window(taps, periodic=False)
        proto = proto * window
        bank = []
        for k in range(self.n_band):
            phase = ((-1) ** k) * math.pi / 4
            mod = torch.cos((2 * k + 1) * math.pi / (2 * self.n_band) * t.float() + phase)
            bank.append(2 * proto * mod)
        weight = torch.stack(bank, 0).unsqueeze(1)
        self.register_buffer("weight", weight)
        self.left_padding = taps - 1

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        """Analyse or synthesise depending on construction."""
        if self.n_band == 1:
            return samples
        if self.analysis:
            parts = []
            for channel in range(samples.shape[1]):
                mono = samples[:, channel : channel + 1]
                padded = functional.pad(mono, (self.left_padding, 0))
                parts.append(functional.conv1d(padded, self.weight, stride=self.n_band))
            return torch.cat(parts, 1)
        audio_channels = max(1, samples.shape[1] // self.n_band)
        parts = []
        for channel in range(audio_channels):
            bands = samples[:, channel * self.n_band : (channel + 1) * self.n_band]
            up = torch.zeros(
                bands.shape[0],
                self.n_band,
                bands.shape[-1] * self.n_band,
                device=bands.device,
                dtype=bands.dtype,
            )
            up[:, :, :: self.n_band] = bands * self.n_band
            padded = functional.pad(up, (self.left_padding, 0))
            synth = self.weight.flip(-1).permute(1, 0, 2)
            parts.append(functional.conv1d(padded, synth))
        return torch.cat(parts, 1)


class RateConvLayer(nn.Module):
    """Causal strided convolution used as a RAVE rate-change element."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        upsample: bool,
        weight_norm: bool = False,
    ) -> None:
        """Create downsample or upsample convolution."""
        super().__init__()
        self.kernel_size = max(1, int(kernel_size))
        self.stride = max(1, int(stride))
        self.dilation = max(1, int(dilation))
        self.upsample = bool(upsample)
        self.left_padding = (self.kernel_size - 1) * self.dilation
        convolution = nn.Conv1d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=1 if self.upsample else self.stride,
            dilation=self.dilation,
            bias=False,
        )
        self.convolution = _maybe_weight_norm(convolution, weight_norm)

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        """Apply causal padding then convolution."""
        if self.upsample:
            up = torch.zeros(
                samples.shape[0],
                samples.shape[1],
                samples.shape[-1] * self.stride,
                device=samples.device,
                dtype=samples.dtype,
            )
            up[:, :, :: self.stride] = samples
            samples = up
        padded = functional.pad(samples, (self.left_padding, 0))
        return self.convolution(padded)


def _make_strided_conv(
    in_channels: int, out_channels: int, properties: dict[str, Any]
) -> nn.Module:
    """Build a same-rate causal Conv1D or a downsampling rate-change operator."""
    stride = max(1, int(properties.get("stride", 1)))
    kernel = int(properties.get("kernel_size", 3))
    dilation = int(properties.get("dilation", 1))
    use_weight_norm = _weight_norm_requested(properties)
    if stride > 1:
        return RateConvLayer(
            in_channels,
            out_channels,
            kernel,
            stride,
            dilation,
            False,
            use_weight_norm,
        )
    return CausalConv1d(
        in_channels, out_channels, kernel, dilation, use_weight_norm
    )


class ConvTransposeLayer(nn.Module):
    """Causal transposed convolution used as a RAVE upsampling element."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        weight_norm: bool = False,
    ) -> None:
        """Create an upsampling ConvTranspose1d with left causal padding."""
        super().__init__()
        self.kernel_size = max(1, int(kernel_size))
        self.stride = max(1, int(stride))
        self.dilation = max(1, int(dilation))
        self.left_padding = (self.kernel_size - 1) * self.dilation
        convolution = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=self.stride // 2,
            dilation=self.dilation,
            bias=False,
        )
        self.convolution = _maybe_weight_norm(convolution, weight_norm)

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        """Apply causal padding then transposed convolution."""
        padded = functional.pad(samples, (self.left_padding, 0))
        return self.convolution(padded)


def _make_conv_transpose(
    in_channels: int, out_channels: int, properties: dict[str, Any]
) -> nn.Module:
    """Build a causal ConvTranspose1d upsampling operator."""
    return ConvTransposeLayer(
        in_channels,
        out_channels,
        int(properties.get("kernel_size", 3)),
        max(1, int(properties.get("stride", 1))),
        int(properties.get("dilation", 1)),
        _weight_norm_requested(properties),
    )


class VariationalBottleneckLayer(nn.Module):
    """Acids-rave v1 grouped variational head with softplus variance.

    Geometry matches the original RAVE encoder tail: causal ``Conv1d`` from
    even ``in_channels`` to ``2 * latent_size`` with ``groups=2``, kernel
    default 5. Group 0 is μ; group 1 is the variance pre-activation. Live and
    eval paths return μ only; ``train()`` samples ``z = μ + σ ⊙ ε``.
    """

    softplus_eps = 1e-4

    def __init__(
        self, in_channels: int, latent_size: int, kernel_size: int = 5
    ) -> None:
        """Create the grouped causal mean/variance convolution."""
        super().__init__()
        if in_channels % 2 != 0:
            raise ValueError(
                "variational bottleneck requires an even channel count "
                "(grouped mean/variance head)."
            )
        if latent_size % 2 != 0:
            raise ValueError(
                "variational bottleneck latent size must be even "
                "(grouped mean/variance head)."
            )
        self.latent_size = int(latent_size)
        self.kernel_size = max(1, int(kernel_size))
        self.left_padding = self.kernel_size - 1
        self.head = nn.Conv1d(
            in_channels,
            2 * self.latent_size,
            self.kernel_size,
            groups=2,
            bias=False,
        )
        self.last_mean: torch.Tensor | None = None
        self.last_std: torch.Tensor | None = None

    def _project(self, samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(μ, σ)`` from the grouped causal convolution."""
        padded = functional.pad(samples, (self.left_padding, 0))
        projected = self.head(padded)
        mean, scale = torch.split(projected, self.latent_size, 1)
        std = functional.softplus(scale) + self.softplus_eps
        return mean, std

    def encode_mean(self, samples: torch.Tensor) -> torch.Tensor:
        """Return μ only (eval / live / validation-PCA path)."""
        mean, std = self._project(samples)
        self.last_mean = mean
        self.last_std = std
        return mean

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        """Sample ``z = μ + σ ⊙ ε`` while training; return μ at eval."""
        mean, std = self._project(samples)
        self.last_mean = mean
        self.last_std = std
        if self.training:
            return mean + std * torch.randn_like(mean)
        return mean

    def kl(self) -> torch.Tensor:
        """Return the acids-rave unit-Gaussian KL of the last forward."""
        if self.last_mean is None or self.last_std is None:
            return torch.zeros(())
        var = self.last_std * self.last_std
        logvar = torch.log(var)
        return (self.last_mean * self.last_mean + var - logvar - 1).sum(1).mean()


class NoiseSynthLayer(nn.Module):
    """Acids-rave v1 IR × white noise (no amplitude conv stack).

    ``warmed_up`` matches ``blocks.Generator``: the filtered-noise addend is
    silenced during reconstruction stage 1 and mixed from stage 2 onward.
    Freeze/export rebuild the layer with ``warmed_up=True`` so inference
    always includes noise.
    """

    def __init__(self, noise_bands: int, window_size: int) -> None:
        """Store IR bin count, hop length, and the post-warmup default."""
        super().__init__()
        self.noise_bands = max(2, int(noise_bands))
        self.window_size = max(1, int(window_size))
        self.warmed_up = True

    def forward(self, amplitudes: torch.Tensor) -> torch.Tensor:
        """Filter uniform noise, or emit zeros while the v1 warmup is active."""
        if not self.warmed_up:
            return _noise_synth_silence(
                amplitudes, int(self.noise_bands), int(self.window_size)
            )
        return _filtered_noise_scripted(
            amplitudes, int(self.noise_bands), int(self.window_size), None
        )


@torch.jit.script
def amp_to_impulse_response(amp: torch.Tensor, target_size: int) -> torch.Tensor:
    """Port of acids-rave `rave.core.amp_to_impulse_response`."""
    amp = torch.stack((amp, torch.zeros_like(amp)), -1)
    amp = torch.view_as_complex(amp)
    amp = torch.fft.irfft(amp)
    filter_size = int(amp.size(-1))
    amp = torch.roll(amp, filter_size // 2, -1)
    win = torch.hann_window(filter_size, dtype=amp.dtype, device=amp.device)
    amp = amp * win
    amp = torch.nn.functional.pad(amp, (0, int(target_size) - filter_size))
    return torch.roll(amp, -filter_size // 2, -1)


@torch.jit.script
def fft_convolve(signal: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Port of acids-rave `rave.core.fft_convolve`."""
    signal = torch.nn.functional.pad(signal, (0, int(signal.size(-1))))
    kernel = torch.nn.functional.pad(kernel, (int(kernel.size(-1)), 0))
    output = torch.fft.irfft(torch.fft.rfft(signal) * torch.fft.rfft(kernel))
    return output[..., int(output.size(-1)) // 2 :]


@torch.jit.script
def _filtered_noise_scripted(
    amplitudes: torch.Tensor,
    noise_bands: int,
    window_size: int,
    noise: Optional[torch.Tensor],
) -> torch.Tensor:
    """IR-filter uniform noise with dynamic batch/frame sizes."""
    bands = int(noise_bands)
    if bands < 2:
        bands = 2
    hop = int(window_size)
    if hop < 1:
        hop = 1
    batch = int(amplitudes.size(0))
    channels = int(amplitudes.size(1))
    frames = int(amplitudes.size(2))
    data_size = channels // bands
    amp = amplitudes.permute(0, 2, 1).reshape(batch, frames, data_size, bands)
    ir = amp_to_impulse_response(amp, hop)
    generated = noise
    if generated is None:
        generated = torch.rand_like(ir) * 2 - 1
    out = fft_convolve(generated, ir).permute(0, 2, 1, 3)
    return out.reshape(batch, data_size, -1)


def filtered_noise(
    amplitudes: torch.Tensor,
    noise_bands: int,
    window_size: int,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """IR-filter uniform noise; amplitudes are `[B, data_size * bands, frames]`."""
    return _filtered_noise_scripted(
        amplitudes, int(noise_bands), int(window_size), noise
    )


def _noise_synth_silence(
    amplitudes: torch.Tensor, noise_bands: int, window_size: int
) -> torch.Tensor:
    """Zeros matching Noise Synth output so stage-1 warmup omits the addend."""
    bands = max(2, int(noise_bands))
    hop = max(1, int(window_size))
    batch = int(amplitudes.size(0))
    channels = int(amplitudes.size(1))
    frames = int(amplitudes.size(2))
    data_size = max(1, channels // bands)
    return torch.zeros(
        (batch, data_size, frames * hop),
        dtype=amplitudes.dtype,
        device=amplitudes.device,
    )


class RaveGraphModule(nn.Module):
    """Executes a (possibly branched) RAVE element graph."""

    def __init__(
        self,
        layers: dict[int, nn.Module],
        order: list[int],
        incoming: dict[int, list[int]],
        bottleneck_id: int | None,
        types: dict[int, str],
        input_channels: int,
        incoming_pins: dict[int, list[tuple[int, int]]] | None = None,
        fragment: dict[str, Any] | None = None,
    ) -> None:
        """Store graph topology, per-node layers, and the source fragment.

        ``fragment`` is kept so export can rebuild a clone without deepcopying
        parametrized convolutions (which can alias the live training module).
        """
        super().__init__()
        self.layers = nn.ModuleDict({str(key): value for key, value in layers.items()})
        self.order = order
        self.incoming = incoming
        self.incoming_pins = incoming_pins or {}
        self.bottleneck_id = bottleneck_id
        self.types = types
        self.input_channels = input_channels
        self.fragment = fragment
        self.latent_mean: torch.Tensor | None = None
        self.latent_pca: torch.Tensor | None = None
        self.cumulative_variance: torch.Tensor | None = None
        self.compactness_ready = False
        self.fidelity = 0.99

    def _run(self, audio: torch.Tensor, stop_at: int | None = None, start_from: tuple[int, torch.Tensor] | None = None) -> torch.Tensor:
        """Evaluate nodes in topological order."""
        values: dict[int, torch.Tensor] = {}
        if start_from is not None:
            values[start_from[0]] = start_from[1]
        output = audio
        started = start_from is None
        for node_id in self.order:
            if start_from is not None and not started:
                if node_id == start_from[0]:
                    started = True
                continue
            sources = self.incoming.get(node_id, [])
            key = str(node_id)
            node_type = self.types.get(node_id, "")
            layer = self.layers[key] if key in self.layers else None
            if not sources:
                current = audio if start_from is None else values.get(node_id, audio)
            elif node_type in _UTILITY_TYPES and layer is not None:
                stacked = [values[source] for source in sources if source in values]
                current = layer(*stacked)
            elif len(sources) == 1:
                current = values[sources[0]]
            elif node_type == "math_expression":
                current = values[sources[0]]
            else:
                stacked = [values[source] for source in sources if source in values]
                current = stacked[0]
                for extra in stacked[1:]:
                    current = combine_pair(current, extra, 0)
            if layer is not None and node_type not in _UTILITY_TYPES:
                if node_type == "math_expression":
                    n_in = int(getattr(layer, "n_inputs", 1))
                    slots: list[torch.Tensor | None] = [None] * n_in
                    pin_sources = self.incoming_pins.get(node_id, [])
                    if pin_sources:
                        for pin_index, source in pin_sources:
                            if 0 <= pin_index < n_in and source in values:
                                slots[pin_index] = values[source]
                    else:
                        for offset, source in enumerate(sources):
                            if offset < n_in and source in values:
                                slots[offset] = values[source]
                    if all(item is None for item in slots):
                        slots[0] = current
                    current = layer(*slots)
                else:
                    current = layer(current)
            values[node_id] = current
            output = current
            if stop_at is not None and node_id == stop_at:
                break
        return output

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Run the encoder through the variational bottleneck."""
        if self.bottleneck_id is None:
            return audio
        return self._run(audio, stop_at=self.bottleneck_id)

    def encode_distribution(self, audio: torch.Tensor) -> torch.Tensor:
        """Return concatenated ``(μ, σ)`` along the channel axis.

        Spread uses the acids-rave softplus-std convention. When no bottleneck
        is present, σ is ones so the live prior-mix fallback stays defined.
        """
        if self.bottleneck_id is None:
            mean = self._run(audio)
            return torch.cat((mean, torch.ones_like(mean)), dim=1)
        mean = self._run(audio, stop_at=self.bottleneck_id)
        key = str(self.bottleneck_id)
        layer = self.layers[key] if key in self.layers else None
        std = getattr(layer, "last_std", None)
        if std is None:
            std = torch.ones_like(mean)
        return torch.cat((mean, std), dim=1)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Run the decoder starting after the bottleneck."""
        if self.bottleneck_id is None:
            return latent
        return self._run(latent, start_from=(self.bottleneck_id, latent))

    def forward(self, audio: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Encode then decode, ignoring unused conditioning."""
        del cond
        return self._run(audio)


def build_rave_graph_module(
    fragment: dict[str, Any], input_channels: int, cond_dim: int
) -> RaveGraphModule:
    """Construct a RAVE graph module from a train/freeze fragment."""
    del cond_dim
    elements = _topological_elements(fragment)
    by_id = {int(element["id"]): element for element in elements}
    incoming: dict[int, list[int]] = {int(element["id"]): [] for element in elements}
    incoming_pins: dict[int, list[tuple[int, int]]] = {
        int(element["id"]): [] for element in elements
    }
    for connection in fragment.get("connections", []):
        source = int(connection["source_element_id"])
        destination = int(connection["destination_element_id"])
        if source in incoming and destination in incoming:
            pin_index = connection.get("destination_pin_index")
            if pin_index is None:
                pin_index = len(incoming[destination])
            incoming[destination].append(source)
            incoming_pins[destination].append((int(pin_index), source))
    layers: dict[int, nn.Module] = {}
    channels_by_id: dict[int, int] = {}
    bottleneck_id = None
    types: dict[int, str] = {}
    for element in elements:
        node_id = int(element["id"])
        element_type = str(element["type"])
        types[node_id] = element_type
        properties = _properties(element)
        in_ch = input_channels
        if incoming[node_id]:
            in_ch = channels_by_id.get(incoming[node_id][0], input_channels)
        if element_type == "pqmf_analysis":
            n_band = int(properties.get("n_band", 16))
            layers[node_id] = PqmfLayer(n_band, True)
            channels_by_id[node_id] = in_ch * n_band
        elif element_type == "pqmf_synthesis":
            n_band = int(properties.get("n_band", 16))
            layers[node_id] = PqmfLayer(n_band, False)
            channels_by_id[node_id] = max(1, in_ch // max(1, n_band))
        elif element_type in {"rate_conv", "conv1d"}:
            out_ch = int(properties.get("channels", in_ch))
            layers[node_id] = _make_strided_conv(in_ch, out_ch, properties)
            channels_by_id[node_id] = out_ch
        elif element_type == "conv_transpose1d":
            out_ch = int(properties.get("channels", in_ch))
            layers[node_id] = _make_conv_transpose(in_ch, out_ch, properties)
            channels_by_id[node_id] = out_ch
        elif element_type == "batch_norm":
            layers[node_id] = nn.BatchNorm1d(in_ch)
            channels_by_id[node_id] = in_ch
        elif element_type == "variational_bottleneck":
            latent = int(properties.get("latent_size", 128))
            kernel = int(properties.get("kernel_size", 5))
            layers[node_id] = VariationalBottleneckLayer(in_ch, latent, kernel)
            channels_by_id[node_id] = latent
            bottleneck_id = node_id
        elif element_type == "noise_synthesizer":
            noise_bands = int(properties.get("noise_bands", 5))
            window_size = int(properties.get("window_size", 64))
            layers[node_id] = NoiseSynthLayer(noise_bands, window_size)
            channels_by_id[node_id] = max(1, in_ch // max(1, noise_bands))
        elif element_type == "tcn":
            hidden = int(properties.get("channels", in_ch))
            layers[node_id] = SteerableTCN(
                in_ch,
                hidden,
                in_ch,
                int(properties.get("depth", 2)),
                int(properties.get("kernel_size", 3)),
                int(properties.get("dilation_growth", 2)),
                int(properties.get("activation", 0)),
                bool(int(properties.get("residual", 0))),
                0,
                float(properties.get("negative_slope", 0.01)),
            )
            channels_by_id[node_id] = in_ch
        elif element_type == "activation":
            layers[node_id] = _activation(int(properties.get("activation", 0)), in_ch,
                                          float(properties.get("negative_slope", 0.01)))
            channels_by_id[node_id] = in_ch
        elif element_type == "linear":
            out_ch = int(properties.get("features", in_ch))
            layers[node_id] = nn.Conv1d(in_ch, out_ch, 1, bias=False)
            channels_by_id[node_id] = out_ch
        elif element_type == "math_expression":
            widths = [
                channels_by_id.get(source, input_channels)
                for source in incoming[node_id]
            ]
            out_ch = max(widths) if widths else in_ch
            layers[node_id] = MathExpression(
                str(properties.get("expression", "x1")),
                int(properties.get("inputs", 1)),
            )
            channels_by_id[node_id] = max(1, out_ch)
        elif element_type in _UTILITY_TYPES:
            mode = _utility_mode(element_type, properties)
            layers[node_id] = UtilityCombine(mode)
            widths = [
                channels_by_id.get(source, input_channels)
                for source in incoming[node_id]
            ]
            if not widths:
                channels_by_id[node_id] = in_ch
            elif mode == 2:
                channels_by_id[node_id] = max(1, sum(widths))
            else:
                channels_by_id[node_id] = max(1, max(widths))
        else:
            built = make_ddsp_or_recurrent(
                element_type, properties, in_ch, int(element.get("seed", 42))
            )
            if built is None:
                channels_by_id[node_id] = in_ch
            else:
                layer, out_ch = built
                layers[node_id] = layer
                channels_by_id[node_id] = out_ch
    return RaveGraphModule(
        layers,
        [int(element["id"]) for element in elements],
        incoming,
        bottleneck_id,
        types,
        input_channels,
        incoming_pins,
        fragment,
    )


def flatten_reconstruction_clips(capture_set: dict[str, Any]) -> list[str]:
    """Expand selected pairs to x+y paths plus unpaired clips."""
    paths: list[str] = []
    for pair in capture_set.get("pairs", []) or []:
        if not isinstance(pair, dict):
            continue
        for key in ("x_path", "y_path"):
            path = str(pair.get(key, "") or "")
            if path:
                paths.append(path)
    for clip in capture_set.get("clips", []) or []:
        if not isinstance(clip, dict):
            continue
        path = str(clip.get("path", "") or "")
        if path:
            paths.append(path)
    return paths


def split_reconstruction_corpus(
    paths: list[str],
    train_percent: int = 98,
    seed: int = 42,
    max_val: int = 1000,
) -> tuple[list[str], list[str]]:
    """Split reconstruction paths 98/2 with seed 42 and a 1000-clip val cap.

    Matches acids-rave v1 ``random_split(..., generator seed 42)`` plus the
    later ``max_residual=1000`` cap. At least one train clip is kept whenever
    the corpus is non-empty so stage 1 can still run.
    """
    shuffled = list(paths)
    rng = random.Random(int(seed))
    rng.shuffle(shuffled)
    count = len(shuffled)
    if count == 0:
        return [], []
    train_percent = int(train_percent)
    n_val = min(int(max_val), (100 - train_percent) * count // 100)
    if n_val < 1 and count >= 2:
        n_val = 1
    if n_val >= count:
        n_val = count - 1
    return shuffled[n_val:], shuffled[:n_val]


def mapping_rejects_unpaired(capture_set: dict[str, Any]) -> bool:
    """Return True when mapping is illegal because unpaired clips are selected."""
    clips = capture_set.get("clips", []) or []
    return any(isinstance(clip, dict) for clip in clips)


def mean_difference(
    target: torch.Tensor,
    value: torch.Tensor,
    norm: str = "L1",
    relative: bool = False,
) -> torch.Tensor:
    """Port of acids-rave ``rave.core.mean_difference``."""
    diff = target - value
    if norm == "L1":
        distance = diff.abs().mean()
        if relative:
            distance = distance / target.abs().mean().clamp_min(1e-8)
        return distance
    if norm == "L2":
        distance = (diff * diff).mean()
        if relative:
            distance = distance / (target * target).mean().clamp_min(1e-8)
        return distance
    raise ValueError(f"norm must be L1 or L2, got {norm}")


def hinge_gan(
    score_real: torch.Tensor, score_fake: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Port of acids-rave ``rave.core.hinge_gan`` (v1.gin ``gan_loss``)."""
    loss_dis = torch.relu(1.0 - score_real) + torch.relu(1.0 + score_fake)
    return loss_dis.mean(), -score_fake.mean()


def _align_waveform_pair(
    predicted: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match batch, channels, and time so spectral/GAN terms can subtract.

    A mono graph on stereo clips used to flatten to 16 vs 8 STFT rows. When one
    side is one channel, the other is folded with ``(L+R)/2`` rather than
    repeating the mono signal. Time is taken from the end so causal delay still
    lines up with the reconstruction crop.
    """
    if predicted.dim() == 2:
        predicted = predicted.unsqueeze(1)
    if target.dim() == 2:
        target = target.unsqueeze(1)
    batch = min(int(predicted.shape[0]), int(target.shape[0]))
    predicted = predicted[:batch]
    target = target[:batch]
    length = min(int(predicted.shape[-1]), int(target.shape[-1]))
    predicted = predicted[..., -length:]
    target = target[..., -length:]
    pred_channels = int(predicted.shape[1])
    target_channels = int(target.shape[1])
    if pred_channels == target_channels:
        return predicted, target
    if pred_channels == 1:
        if target_channels > 1:
            target = target.mean(dim=1, keepdim=True)
        return predicted, target
    if target_channels == 1:
        if pred_channels > 1:
            predicted = predicted.mean(dim=1, keepdim=True)
        return predicted, target
    channels = min(pred_channels, target_channels)
    return predicted[:, :channels], target[:, :channels]


def _match_stft_shapes(
    spec_p: torch.Tensor, spec_t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop two magnitude STFTs to a common ``[rows, freq, frames]`` shape."""
    rows = min(int(spec_p.shape[0]), int(spec_t.shape[0]))
    freq = min(int(spec_p.shape[1]), int(spec_t.shape[1]))
    frames = min(int(spec_p.shape[-1]), int(spec_t.shape[-1]))
    return spec_p[:rows, :freq, :frames], spec_t[:rows, :freq, :frames]


def audio_distance_v1(
    predicted: torch.Tensor,
    target: torch.Tensor,
    windows: tuple[int, ...] | list[int] = RAVE_V1_SPECTRAL_WINDOWS,
    log_epsilon: float = RAVE_V1_LOG_EPSILON,
) -> torch.Tensor:
    """Acids-rave ``AudioDistanceV1``: relative L2 mag + L1 log-mag per scale."""
    predicted, target = _align_waveform_pair(predicted, target)
    loss = predicted.new_zeros(())
    used = 0
    time_length = int(predicted.shape[-1])
    flat_p = predicted.reshape(-1, time_length)
    flat_t = target.reshape(-1, time_length)
    for window in windows:
        n_fft = int(window)
        if n_fft > time_length:
            continue
        hop = max(1, n_fft // 4)
        hann = torch.hann_window(n_fft, device=predicted.device, dtype=predicted.dtype)
        spec_p = torch.stft(
            flat_p,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=hann,
            return_complex=True,
        ).abs()
        spec_t = torch.stft(
            flat_t,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=hann,
            return_complex=True,
        ).abs()
        spec_p, spec_t = _match_stft_shapes(spec_p, spec_t)
        log_p = torch.log(spec_p + log_epsilon)
        log_t = torch.log(spec_t + log_epsilon)
        loss = loss + mean_difference(spec_t, spec_p, norm="L2", relative=True)
        loss = loss + mean_difference(log_t, log_p, norm="L1")
        used += 1
    if used == 0:
        return (predicted - target).abs().mean()
    return loss


def spectral_distance(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Fullband acids-rave v1 spectral distance (kept as a public alias)."""
    return audio_distance_v1(predicted, target)


class ConvNetDiscriminator(nn.Module):
    """One MelGAN-style scale from acids-rave ``discriminator.ConvNet`` (v1)."""

    def __init__(
        self,
        in_size: int,
        capacity: int,
        n_layers: int,
        kernel_size: int,
        stride: int,
    ) -> None:
        """Build weight-normalized strided convs plus a 1x1 score head."""
        super().__init__()
        channels = [max(1, int(in_size))]
        channels.extend(int(capacity) * (2**index) for index in range(max(1, n_layers)))
        layers: list[nn.Module] = []
        pad = max(0, int(kernel_size) // 2)
        for index in range(len(channels) - 1):
            conv = nn.Conv1d(
                channels[index],
                channels[index + 1],
                int(kernel_size),
                stride=max(1, int(stride)),
                padding=pad,
            )
            layers.append(_maybe_weight_norm(conv, True))
            layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Conv1d(channels[-1], 1, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, samples: torch.Tensor) -> list[torch.Tensor]:
        """Return activations after every convolution (including the score)."""
        features: list[torch.Tensor] = []
        value = samples
        for layer in self.net:
            value = layer(value)
            if isinstance(layer, nn.modules.conv._ConvNd):
                features.append(value)
        return features


class MultiScaleDiscriminator(nn.Module):
    """Acids-rave v1 ``MultiScaleDiscriminator`` (3 pooled MelGAN scales)."""

    def __init__(
        self,
        channels: int,
        n_discriminators: int = RAVE_V1_DISC_N_SCALES,
        capacity: int = RAVE_V1_DISC_CAPACITY,
        n_layers: int = RAVE_V1_DISC_N_LAYERS,
        kernel_size: int = RAVE_V1_DISC_KERNEL,
        stride: int = RAVE_V1_DISC_STRIDE,
    ) -> None:
        """Create one ConvNet per scale."""
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ConvNetDiscriminator(
                    channels, capacity, n_layers, kernel_size, stride
                )
                for _ in range(max(1, int(n_discriminators)))
            ]
        )

    def forward(self, samples: torch.Tensor) -> list[list[torch.Tensor]]:
        """Run each scale, pooling the waveform by 2 between scales."""
        features: list[list[torch.Tensor]] = []
        value = samples
        for layer in self.layers:
            features.append(layer(value))
            value = functional.avg_pool1d(value, 2)
        return features


class CombineDiscriminator(MultiScaleDiscriminator):
    """Backward-compatible alias; v1 quality stage uses multi-scale, not MPD."""


def split_discriminator_features(
    features: list[list[torch.Tensor]],
) -> tuple[list[tuple[torch.Tensor, ...]], list[tuple[torch.Tensor, ...]]]:
    """Split concatenated real/fake batches the way acids-rave ``split_features`` does."""
    real_scales: list[tuple[torch.Tensor, ...]] = []
    fake_scales: list[tuple[torch.Tensor, ...]] = []
    for scale in features:
        split = [torch.split(item, item.shape[0] // 2, 0) for item in scale]
        real_scales.append(tuple(part[0] for part in split))
        fake_scales.append(tuple(part[1] for part in split))
    return real_scales, fake_scales


def discriminator_losses(
    real: torch.Tensor,
    fake: torch.Tensor,
    discriminator: MultiScaleDiscriminator,
    feature_matching_weight: float,
    num_skipped_features: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return disc, gen-adv, feature-matching, and mean real/fake scores."""
    concatenated = torch.cat([real, fake], 0)
    real_scales, fake_scales = split_discriminator_features(discriminator(concatenated))
    loss_dis = concatenated.new_zeros(())
    loss_adv = concatenated.new_zeros(())
    match = concatenated.new_zeros(())
    pred_real = concatenated.new_zeros(())
    pred_fake = concatenated.new_zeros(())
    skip = max(0, int(num_skipped_features))
    for scale_real, scale_fake in zip(real_scales, fake_scales):
        kept_real = scale_real[skip:]
        kept_fake = scale_fake[skip:]
        if kept_real:
            current = concatenated.new_zeros(())
            for real_feat, fake_feat in zip(kept_real, kept_fake):
                current = current + mean_difference(real_feat, fake_feat, norm="L1")
            match = match + current / float(len(kept_real))
        disc_term, adv_term = hinge_gan(scale_real[-1], scale_fake[-1])
        loss_dis = loss_dis + disc_term
        loss_adv = loss_adv + adv_term
        pred_real = pred_real + scale_real[-1].mean()
        pred_fake = pred_fake + scale_fake[-1].mean()
    if real_scales:
        match = match / float(len(real_scales))
    return (
        loss_dis,
        loss_adv,
        match * float(feature_matching_weight),
        pred_real,
        pred_fake,
    )


def parse_reconstruction_options(options: dict[str, Any]) -> dict[str, Any]:
    """Resolve acids-rave v1 reconstruction knobs from ``train_options``."""
    if not isinstance(options, dict):
        options = {}
    reconstruction = options.get("reconstruction", {})
    if not isinstance(reconstruction, dict):
        reconstruction = {}

    def _int(key: str, default: int, minimum: int = 1, maximum: int = 10_000_000) -> int:
        source = reconstruction if key in reconstruction else options
        try:
            value = int(source.get(key, default))
        except (TypeError, ValueError):
            value = default
        return max(minimum, min(maximum, value))

    def _float(key: str, default: float, minimum: float, maximum: float) -> float:
        source = reconstruction if key in reconstruction else options
        try:
            value = float(source.get(key, default))
        except (TypeError, ValueError):
            value = default
        return max(minimum, min(maximum, value))

    windows = reconstruction.get("spectral_windows", list(RAVE_V1_SPECTRAL_WINDOWS))
    if not isinstance(windows, (list, tuple)) or not windows:
        windows = list(RAVE_V1_SPECTRAL_WINDOWS)
    parsed_windows = tuple(max(8, int(window)) for window in windows)

    segment_default = RAVE_V1_SEGMENT_LENGTH
    if "segment_length" in reconstruction:
        segment_default = int(reconstruction.get("segment_length", segment_default))
    elif "segment_length" in options:
        segment_default = int(options.get("segment_length", segment_default))

    return {
        "stage1_steps": _int("stage1_steps", RAVE_V1_STAGE1_STEPS),
        "stage2_steps": _int("stage2_steps", RAVE_V1_STAGE2_STEPS),
        "generator_lr": _float("generator_lr", RAVE_V1_GENERATOR_LR, 1e-8, 1.0),
        "discriminator_lr": _float(
            "discriminator_lr", RAVE_V1_DISCRIMINATOR_LR, 1e-8, 1.0
        ),
        "adam_beta1": _float("adam_beta1", RAVE_V1_ADAM_BETAS[0], 0.0, 0.999),
        "adam_beta2": _float("adam_beta2", RAVE_V1_ADAM_BETAS[1], 0.0, 0.9999),
        "lr_decay_end_factor": _float(
            "lr_decay_end_factor", RAVE_V1_LR_DECAY_END, 0.01, 1.0
        ),
        "batch_size": _int("batch_size", RAVE_V1_BATCH_SIZE, 1, 64),
        "segment_length": max(64, min(2_000_000, int(segment_default))),
        "kl_beta": _float("kl_beta", RAVE_V1_KL_BETA, 0.0, 10.0),
        "kl_beta_start": _float("kl_beta_start", RAVE_V1_KL_BETA_START, 1e-12, 10.0),
        "kl_warmup_steps": _int("kl_warmup_steps", RAVE_V1_KL_WARMUP_STEPS, 1, 10_000_000),
        "feature_matching_weight": _float(
            "feature_matching_weight", RAVE_V1_FEATURE_MATCHING_WEIGHT, 0.0, 100.0
        ),
        "update_discriminator_every": _int(
            "update_discriminator_every", RAVE_V1_UPDATE_DISCRIMINATOR_EVERY, 1, 64
        ),
        "phase_mangle_prob": _float(
            "phase_mangle_prob", RAVE_V1_PHASE_MANGLE_PROB, 0.0, 1.0
        ),
        "dequantize_bits": _int("dequantize_bits", RAVE_V1_DEQUANTIZE_BITS, 0, 32),
        "spectral_windows": parsed_windows,
        "log_epsilon": _float("log_epsilon", RAVE_V1_LOG_EPSILON, 1e-12, 1e-3),
        "disc_n_scales": _int("disc_n_scales", RAVE_V1_DISC_N_SCALES, 1, 8),
        "disc_capacity": _int("disc_capacity", RAVE_V1_DISC_CAPACITY, 4, 256),
        "disc_n_layers": _int("disc_n_layers", RAVE_V1_DISC_N_LAYERS, 1, 8),
        "disc_kernel_size": _int("disc_kernel_size", RAVE_V1_DISC_KERNEL, 3, 31),
        "disc_stride": _int("disc_stride", RAVE_V1_DISC_STRIDE, 1, 8),
    }


def reconstruction_kl_beta(
    step: int, warmup_len: int, initial: float, target: float
) -> float:
    """Match acids-rave ``BetaWarmupCallback`` (log interpolation, 1-based steps)."""
    training_steps = max(1, int(step) + 1)
    warmup = max(1, int(warmup_len))
    if training_steps >= warmup:
        return float(target)
    ratio = training_steps / float(warmup)
    start = max(float(initial), 1e-12)
    end = max(float(target), 1e-12)
    return math.exp(math.log(start) * (1.0 - ratio) + math.log(end) * ratio)


def _lfilter_last_axis(b: np.ndarray, a: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Causal IIR along the last axis (acids-rave ``scipy.signal.lfilter``)."""
    b_coeff = np.asarray(b, dtype=np.float64).reshape(-1)
    a_coeff = np.asarray(a, dtype=np.float64).reshape(-1)
    if a_coeff.size < 1 or abs(a_coeff[0]) < 1e-12:
        return samples
    a0 = float(a_coeff[0])
    original = samples.shape
    flat = samples.reshape(-1, original[-1]).astype(np.float64, copy=False)
    filtered = np.zeros_like(flat)
    for row in range(flat.shape[0]):
        x_row = flat[row]
        y_row = filtered[row]
        for time_index in range(x_row.shape[0]):
            acc = 0.0
            for tap, coeff in enumerate(b_coeff):
                past = time_index - tap
                if past >= 0:
                    acc += float(coeff) * float(x_row[past])
            for tap in range(1, a_coeff.size):
                past = time_index - tap
                if past >= 0:
                    acc -= float(a_coeff[tap]) * float(y_row[past])
            y_row[time_index] = acc / a0
    return filtered.reshape(original).astype(samples.dtype, copy=False)


def random_phase_mangle(
    samples: torch.Tensor,
    sample_rate: int,
    min_hz: float = 20.0,
    max_hz: float = 2000.0,
    amplitude: float = 0.99,
) -> torch.Tensor:
    """Port of acids-rave ``rave.core.random_phase_mangle`` (allpass)."""
    if sample_rate <= 0 or samples.numel() == 0:
        return samples
    log_min = math.log(max(min_hz, 1.0))
    log_max = math.log(max(max_hz, min_hz + 1.0))
    omega = math.exp(random.random() * (log_max - log_min) + log_min)
    omega = 2.0 * math.pi * omega / float(sample_rate)
    pole = amplitude * np.exp(1j * omega)
    a_coeff = np.array([1.0, -2.0 * np.real(pole), abs(pole) ** 2], dtype=np.float64)
    b_coeff = np.array([abs(pole) ** 2, -2.0 * np.real(pole), 1.0], dtype=np.float64)
    mangled = _lfilter_last_axis(b_coeff, a_coeff, samples.detach().cpu().numpy())
    return torch.from_numpy(np.ascontiguousarray(mangled)).to(
        device=samples.device, dtype=samples.dtype
    )


def augment_rave_v1(
    samples: torch.Tensor,
    sample_rate: int,
    phase_mangle_prob: float,
    dequantize_bits: int,
) -> torch.Tensor:
    """Apply v1 dataset transforms: random allpass then 16-bit dequant noise."""
    augmented = samples
    if phase_mangle_prob > 0.0 and random.random() < float(phase_mangle_prob):
        augmented = random_phase_mangle(augmented, sample_rate)
    bits = int(dequantize_bits)
    if bits > 0:
        scale = float(2**bits)
        noise = (torch.rand_like(augmented) - 0.5) / scale
        augmented = augmented + noise
    return augmented


def _pqmf_analysis_layer(module: RaveGraphModule) -> PqmfLayer | None:
    """Return the first PQMF analysis node so dual spectral can reuse its bank."""
    for node_id in module.order:
        if module.types.get(node_id) != "pqmf_analysis":
            continue
        key = str(node_id)
        if key not in module.layers:
            continue
        layer = module.layers[key]
        if isinstance(layer, PqmfLayer):
            return layer
    return None


def reconstruction_spectral_loss(
    reconstructed: torch.Tensor,
    audio: torch.Tensor,
    module: RaveGraphModule,
    windows: tuple[int, ...],
    log_epsilon: float,
) -> torch.Tensor:
    """Fullband plus multiband ``AudioDistanceV1`` when the graph has PQMF."""
    fullband = audio_distance_v1(reconstructed, audio, windows, log_epsilon)
    analysis = _pqmf_analysis_layer(module)
    if analysis is None:
        return fullband
    x_multiband = analysis(audio)
    y_multiband = analysis(reconstructed)
    length = min(x_multiband.shape[-1], y_multiband.shape[-1])
    if length < 8:
        return fullband
    return fullband + audio_distance_v1(
        y_multiband[..., :length],
        x_multiband[..., :length],
        windows,
        log_epsilon,
    )


def compute_compactness(latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PCA compactness basis from `[N, latent]` rows."""
    if latents.ndim != 2 or latents.shape[0] < 2:
        width = int(latents.shape[-1]) if latents.numel() else 1
        eye = torch.eye(width)
        return torch.zeros(width), eye, torch.linspace(1.0 / width, 1.0, width)
    mean = latents.mean(0)
    centered = latents - mean
    _, singular, right = torch.linalg.svd(centered, full_matrices=False)
    total = float(singular.sum().clamp_min(1e-8))
    cumulative = torch.cumsum(singular, 0) / total
    return mean.detach(), right.detach(), cumulative.detach()


def _freeze_rave_encoder(module: RaveGraphModule) -> None:
    """Stop gradients through the encoder inclusive of the bottleneck."""
    for node_id in module.order:
        key = str(node_id)
        if key in module.layers:
            for parameter in module.layers[key].parameters():
                parameter.requires_grad_(False)
        if node_id == module.bottleneck_id:
            break


def _set_noise_synth_warmed_up(module: RaveGraphModule, warmed_up: bool) -> None:
    """Mute or enable every Noise Synth addend (acids-rave ``Generator.warmed_up``)."""
    state = bool(warmed_up)
    for layer in module.layers.values():
        if isinstance(layer, NoiseSynthLayer):
            layer.warmed_up = state


def _bottleneck_kl(module: RaveGraphModule) -> torch.Tensor:
    """Return KL from the variational head when present."""
    zeros = torch.zeros((), device=_module_device(module))
    if module.bottleneck_id is None:
        return zeros
    key = str(module.bottleneck_id)
    if key not in module.layers:
        return zeros
    layer = module.layers[key]
    if isinstance(layer, VariationalBottleneckLayer):
        return layer.kl()
    return zeros


def _clear_nonleaf_caches(module: nn.Module) -> None:
    """Drop autograd-backed bottleneck caches so clone/export can pickle tensors."""
    for layer in module.modules():
        if isinstance(layer, VariationalBottleneckLayer):
            layer.last_mean = None
            layer.last_std = None


def _copy_rave_runtime_state(source: RaveGraphModule, destination: RaveGraphModule) -> None:
    """Copy compactness buffers that are not part of ``state_dict``."""
    destination.latent_mean = (
        None if source.latent_mean is None else source.latent_mean.detach().cpu().clone()
    )
    destination.latent_pca = (
        None if source.latent_pca is None else source.latent_pca.detach().cpu().clone()
    )
    destination.cumulative_variance = (
        None
        if source.cumulative_variance is None
        else source.cumulative_variance.detach().cpu().clone()
    )
    destination.compactness_ready = bool(source.compactness_ready)
    destination.fidelity = float(source.fidelity)


def _clone_rave_from_fragment(module: RaveGraphModule) -> RaveGraphModule:
    """Rebuild a RAVE graph from its fragment and copy trained weights.

    Avoids ``deepcopy`` of parametrized convolutions, which can alias the live
    module and then get stripped for TorchScript.
    """
    if not isinstance(module.fragment, dict):
        raise ValueError("RAVE export requires the original graph fragment")
    clone = build_rave_graph_module(module.fragment, module.input_channels, 1)
    original_has_weight_norm = any(
        _has_weight_norm(child) for child in module.modules()
    )
    if not original_has_weight_norm:
        strip_weight_norm(clone)
    clone.load_state_dict(module.state_dict())
    _copy_rave_runtime_state(module, clone)
    strip_weight_norm(clone)
    return clone


def _clone_conditioned_from_fragment(module: ConditionedSequential) -> ConditionedSequential:
    """Rebuild a mapping graph from its fragment and copy trained weights.

    Avoids ``deepcopy`` of a live CUDA module, which can alias convolution
    weights so TorchScript sees a 1-channel kernel while the example is stereo.
    """
    if not isinstance(module.fragment, dict):
        raise ValueError("mapping export requires the original graph fragment")
    clone = build_module(module.fragment, module.input_channels, module.cond_dim)
    if not isinstance(clone, ConditionedSequential):
        raise ValueError("mapping export rebuilt an empty graph")
    original_has_weight_norm = any(
        _has_weight_norm(child) for child in module.modules()
    )
    if not original_has_weight_norm:
        strip_weight_norm(clone)
    clone.load_state_dict(_clone_state_dict(module))
    strip_weight_norm(clone)
    return clone


def _first_conv_in_channels(module: nn.Module, fallback: int) -> int:
    """Return the first ``Conv1d`` input width so trace examples match weights.

    User-library graphs such as TCN-with-conv start with a 1x1 expander whose
    ``in_channels`` may be 1 (mono box) even when the DAW bus is stereo.
    """
    for child in module.modules():
        if isinstance(child, nn.Conv1d):
            return max(1, int(child.in_channels))
    return max(1, int(fallback))


def _clone_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Return a detached CPU copy of ``module`` parameters and buffers."""
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in module.state_dict().items()
    }


def _cpu_mapping_export_module(module: nn.Module) -> nn.Module:
    """Build a CPU-only mapping clone that does not alias the live CUDA graph.

    Reads ``state_dict`` first so ``eval()`` / ``to('cpu')`` never run on the
    training module. TorchScript on CUDA has been observed to invoke the first
    conv with the previous crop (``[1, 2, 16384]``) when the live module is
    traced or deepcopy-aliased.
    """
    state = _clone_state_dict(module)
    fragment = getattr(module, "fragment", None)
    input_channels = int(getattr(module, "input_channels", 1) or 1)
    cond_dim = int(getattr(module, "cond_dim", 1) or 1)
    if isinstance(module, ConditionedSequential) and isinstance(fragment, dict):
        clone = build_module(fragment, input_channels, cond_dim)
        if not isinstance(clone, ConditionedSequential):
            raise ValueError("mapping export rebuilt an empty graph")
        if not any(_has_weight_norm(child) for child in module.modules()):
            strip_weight_norm(clone)
        clone.load_state_dict(state)
        strip_weight_norm(clone)
        return clone.to("cpu").eval()
    clone = copy.deepcopy(module)
    strip_weight_norm(clone)
    clone.load_state_dict(state, strict=False)
    return clone.to("cpu").eval()


def _clone_module_for_export(module: nn.Module) -> nn.Module:
    """Clone a module for TorchScript without mutating the training graph."""
    _clear_nonleaf_caches(module)
    if isinstance(module, RaveGraphModule):
        cloned = _clone_rave_from_fragment(module)
    elif isinstance(module, ConditionedSequential) and isinstance(
        getattr(module, "fragment", None), dict
    ):
        cloned = _cpu_mapping_export_module(module)
    else:
        cloned = copy.deepcopy(module)
        strip_weight_norm(cloned)
    return cloned.to("cpu").eval()


def _export_rave_scripted(module: RaveGraphModule, input_channels: int, path: Path) -> None:
    """Trace forward/encode/decode for a RAVE graph."""
    was_training = bool(module.training)
    module.eval()
    export_module = _clone_module_for_export(module)
    if isinstance(export_module, RaveGraphModule):
        _set_noise_synth_warmed_up(export_module, True)
    example = torch.zeros(1, input_channels, 256)
    latent_size = 1
    if export_module.bottleneck_id is not None:
        key = str(export_module.bottleneck_id)
        if key in export_module.layers:
            layer = export_module.layers[key]
            if isinstance(layer, VariationalBottleneckLayer):
                latent_size = max(1, int(layer.latent_size))
    latent_example = torch.zeros(1, latent_size, max(1, 256 // 16))
    mean = (
        export_module.latent_mean.detach().cpu()
        if export_module.latent_mean is not None
        else torch.zeros(latent_size)
    )
    pca = (
        export_module.latent_pca.detach().cpu()
        if export_module.latent_pca is not None
        else torch.eye(latent_size)
    )
    cumulative = (
        export_module.cumulative_variance.detach().cpu()
        if export_module.cumulative_variance is not None
        else torch.linspace(1.0 / latent_size, 1.0, latent_size)
    )
    ready_flag = torch.tensor(
        1.0 if getattr(export_module, "compactness_ready", False) else 0.0
    )

    class Wrapper(nn.Module):
        """TorchScript-friendly encode/decode/forward wrapper."""

        def __init__(self, inner: RaveGraphModule) -> None:
            super().__init__()
            self.inner = inner
            self.register_buffer("latent_mean", mean)
            self.register_buffer("latent_pca", pca)
            self.register_buffer("cumulative_variance", cumulative)
            self.register_buffer("compactness_ready", ready_flag)

        def encode(self, audio: torch.Tensor) -> torch.Tensor:
            return self.inner.encode(audio)

        def encode_distribution(self, audio: torch.Tensor) -> torch.Tensor:
            return self.inner.encode_distribution(audio)

        def decode(self, latent: torch.Tensor) -> torch.Tensor:
            return self.inner.decode(latent)
            return self.inner.decode(latent)

        def forward(self, audio: torch.Tensor) -> torch.Tensor:
            return self.inner.forward(audio)

    wrapped = Wrapper(export_module).eval()
    with torch.inference_mode():
        scripted = torch.jit.trace_module(
            wrapped,
            {
                "forward": example,
                "encode": example,
                "encode_distribution": example,
                "decode": latent_example,
            },
            strict=False,
            check_trace=False,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.jit.save(scripted, str(temporary))
    temporary.replace(path)
    if was_training:
        module.train()


def _parse_host_io_mode(value: Any, legacy_pair: bool = False) -> str | None:
    """Map a Channels choice or label to ``mono``, ``mirrored``, or ``stereo``."""
    if isinstance(value, str):
        key = value.strip().lower()
        if key in {"mono", "mirrored", "stereo"}:
            return key
        return None
    try:
        choice = int(value)
    except (TypeError, ValueError):
        return None
    if legacy_pair:
        return "mono" if choice <= 0 else "stereo"
    if choice <= 0:
        return "mono"
    if choice == 1:
        return "mirrored"
    return "stereo"


def _host_io_channels(mode: str) -> int:
    """Return graph pin width: 1 for Mono, 2 for Mirrored or Stereo."""
    return 1 if mode == "mono" else 2


def _adapt_host_input_to_mode(samples: torch.Tensor, mode: str) -> torch.Tensor:
    """Fold or expand clips the same way the live engine folds the host bus.

    Mono and Mirrored both use ``(L+R)/2``. Mirrored then copies that mono
    signal onto both channels so the graph stays 2-wide.
    """
    if samples.dim() == 2:
        samples = samples.unsqueeze(0)
    if mode == "stereo":
        if int(samples.shape[1]) == 2:
            return samples
        if int(samples.shape[1]) == 1:
            return samples.repeat(1, 2, 1)
        return samples[:, :2]
    mono = samples if int(samples.shape[1]) <= 1 else samples.mean(dim=1, keepdim=True)
    if mode == "mono":
        return mono
    return mono.repeat(1, 2, 1)


def _graph_audio_input_mode(fragment: dict[str, Any]) -> str | None:
    """Read Audio Input ``channels`` (0=Mono, 1=Mirrored, 2=Stereo) from a fragment."""
    elements = fragment.get("elements", [])
    if not isinstance(elements, list):
        return None
    for element in elements:
        if not isinstance(element, dict):
            continue
        if str(element.get("type", "")) != "audio_input":
            continue
        properties = _properties(element)
        if "channels" not in properties:
            return None
        return _parse_host_io_mode(properties["channels"])
    return None


def _last_declared_audio_channels(fragment: dict[str, Any]) -> int | None:
    """Return the last conv/linear width, used when Audio Input is omitted."""
    try:
        elements = _topological_elements(fragment)
    except (KeyError, TypeError, ValueError):
        raw = fragment.get("elements", [])
        elements = [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []
    last: int | None = None
    for element in elements:
        element_type = str(element.get("type", ""))
        if element_type in _SKIP_GRAPH_IO_TYPES:
            continue
        properties = _properties(element)
        if element_type in {"conv_transpose1d", "conv1d", "rate_conv"} and "channels" in properties:
            last = max(1, int(properties["channels"]))
        elif element_type == "linear" and "features" in properties:
            last = max(1, int(properties["features"]))
        elif element_type == "pqmf_synthesis":
            last = 1
    return last


def _reconstruction_io_mode(
    fragment: dict[str, Any], options: dict[str, Any], clip_channels: int
) -> str:
    """Choose graph I/O mode so mono RAVE trains on ``(L+R)/2``, not the DAW bus."""
    explicit = _parse_host_io_mode(options.get("host_io_mode"))
    if explicit is not None:
        return explicit
    inferred = _graph_audio_input_mode(fragment)
    if inferred is not None:
        return inferred
    decoder = _last_declared_audio_channels(fragment)
    if decoder == 1:
        return "mono"
    host = int(options.get("host_input_channels", 0) or 0)
    if host == 1:
        return "mono"
    if host >= 2:
        return "stereo"
    return "mono" if int(clip_channels) <= 1 else "stereo"


def _mapping_input_channels(
    fragment: dict[str, Any], options: dict[str, Any], clip_channels: int
) -> int:
    """Return the channel count used to build and export a mapping module.

    Prefer an explicit plugin host width. When that is missing, infer from the
    graph (Audio Input, then last conv/linear width) so a 1-channel user-library
    TCN-with-conv box is not traced with stereo library WAVs.
    """
    host = int(options.get("host_input_channels", 0) or 0)
    if host >= 1:
        return host
    return _host_io_channels(_reconstruction_io_mode(fragment, options, clip_channels))


def _load_reconstruction_clips(
    paths: list[str],
) -> tuple[dict[str, torch.Tensor], int, int]:
    """Load WAV paths to ``[1, C, T]`` tensors sharing a sample rate."""
    loaded: dict[str, torch.Tensor] = {}
    rate = None
    channels = 1
    for path in paths:
        samples, sample_rate = read_wav(path)
        if rate is None:
            rate = sample_rate
            channels = int(samples.shape[0])
        elif sample_rate != rate:
            raise ValueError("selected reconstruction clips mix sample rates")
        if int(samples.shape[0]) != channels:
            raise ValueError("selected reconstruction clips must share a channel count")
        loaded[path] = samples.unsqueeze(0)
    if rate is None:
        raise ValueError("reconstruction train request contains no clips")
    return loaded, int(rate), channels


def _match_clip_channels(clips: dict[str, torch.Tensor], channels: int) -> dict[str, torch.Tensor]:
    """Broadcast each stored clip to the host input width."""
    return {path: _match_channels(tensor, channels) for path, tensor in clips.items()}


def _sample_segment(clips: list[torch.Tensor], segment_length: int) -> torch.Tensor:
    """Draw one random window from the train-split clips."""
    clip = clips[int(torch.randint(0, len(clips), (1,)).item())]
    length = min(int(clip.shape[-1]), max(64, int(segment_length)))
    start = (
        0
        if clip.shape[-1] <= length
        else int(torch.randint(0, clip.shape[-1] - length, (1,)).item())
    )
    return clip[..., start : start + length]


def _sample_reconstruction_batch(
    clips: list[torch.Tensor],
    segment_length: int,
    batch_size: int,
    sample_rate: int,
    phase_mangle_prob: float,
    dequantize_bits: int,
) -> torch.Tensor:
    """Draw a v1-style minibatch: random crop, optional allpass, dequant noise."""
    waves: list[torch.Tensor] = []
    for _ in range(max(1, int(batch_size))):
        audio = _sample_segment(clips, segment_length)
        audio = augment_rave_v1(audio, sample_rate, phase_mangle_prob, dequantize_bits)
        waves.append(audio)
    max_time = max(int(item.shape[-1]) for item in waves)
    padded: list[torch.Tensor] = []
    for item in waves:
        if item.shape[-1] < max_time:
            item = functional.pad(item, (0, max_time - item.shape[-1]))
        padded.append(item)
    return torch.cat(padded, 0)


def _collect_validation_mu(
    module: RaveGraphModule, val_clips: list[torch.Tensor], segment_length: int
) -> torch.Tensor | None:
    """Stack eval-mode μ rows from every validation clip."""
    if module.bottleneck_id is None or not val_clips:
        return None
    module.eval()
    rows: list[torch.Tensor] = []
    with torch.no_grad():
        for clip in val_clips:
            length = min(int(clip.shape[-1]), max(64, int(segment_length)))
            start = 0
            while start < clip.shape[-1]:
                audio = clip[..., start : start + length]
                if audio.shape[-1] < 8:
                    break
                latent = module.encode(audio.to(_module_device(module)))
                rows.append(
                    latent.permute(0, 2, 1).reshape(-1, latent.shape[1]).cpu()
                )
                if start + length >= clip.shape[-1]:
                    break
                start += length
    if not rows:
        return None
    return torch.cat(rows, 0)


def _fit_rave_compactness(
    module: RaveGraphModule,
    val_clips: list[torch.Tensor],
    segment_length: int,
) -> tuple[bool, int]:
    """Fit latent PCA compactness from validation μ and store it on ``module``.

    Returns ``(ready, validation_segment_count)``. ``ready`` is true only when
    enough μ rows exist to cover the bottleneck latent size. Leaves ``module``
    in eval after the validation pass.
    """
    stacked = _collect_validation_mu(module, val_clips, segment_length)
    latent_size = 1
    if module.bottleneck_id is not None:
        key = str(module.bottleneck_id)
        if key in module.layers and isinstance(
            module.layers[key], VariationalBottleneckLayer
        ):
            latent_size = max(1, int(module.layers[key].latent_size))
    if stacked is not None and stacked.shape[0] >= latent_size:
        mean, pca, cumulative = compute_compactness(stacked)
        module.latent_mean = mean
        module.latent_pca = pca
        module.cumulative_variance = cumulative
        module.compactness_ready = True
        return True, int(stacked.shape[0])
    module.compactness_ready = False
    return False, len(val_clips)


def _compactness_payload(ready: bool, segments: int) -> dict[str, Any]:
    """Build the train-IPC compactness object."""
    return {
        "ready": bool(ready),
        "validation_segments": int(segments),
        "status": "ready" if ready else "not_ready",
    }


def train_reconstruction(
    request: dict[str, Any], artifact_dir: Path, command_file: Path | None
) -> dict[str, Any]:
    """Run the acids-rave v1 two-stage reconstruction recipe."""
    request_id = str(request.get("request_id", ""))
    options = request.get("train_options", {})
    if not isinstance(options, dict):
        options = {}
    recipe = parse_reconstruction_options(options)
    stage1_steps = int(recipe["stage1_steps"])
    stage2_steps = int(recipe["stage2_steps"])
    capture_set = request.get("capture_set", {})
    if not isinstance(capture_set, dict):
        capture_set = {}
    paths = flatten_reconstruction_clips(capture_set)
    if not paths:
        raise ValueError("reconstruction train request contains no clips")
    train_paths, val_paths = split_reconstruction_corpus(paths)
    fragment = request.get("graph_fragment", {})
    if not isinstance(fragment, dict):
        fragment = {}
    loaded, sample_rate, channels = _load_reconstruction_clips(paths)
    io_mode = _reconstruction_io_mode(fragment, options, channels)
    input_channels = _host_io_channels(io_mode)
    loaded = {
        path: _adapt_host_input_to_mode(tensor, io_mode)
        for path, tensor in loaded.items()
    }
    train_clips = [loaded[path] for path in train_paths if path in loaded]
    val_clips = [loaded[path] for path in val_paths if path in loaded]
    if not train_clips:
        raise ValueError("reconstruction train split is empty")
    device, requested_device, effective_device = resolve_train_device(
        options.get("device", DEFAULT_TRAIN_DEVICE)
    )
    module = build_rave_graph_module(fragment, input_channels, 1)
    if module.bottleneck_id is None:
        raise ValueError("reconstruction requires a variational bottleneck")
    module.to(device)
    betas = (float(recipe["adam_beta1"]), float(recipe["adam_beta2"]))
    optimizer = torch.optim.Adam(
        module.parameters(), float(recipe["generator_lr"]), betas
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=float(recipe["lr_decay_end_factor"]),
        total_iters=max(1, stage1_steps),
    )
    discriminator = MultiScaleDiscriminator(
        input_channels,
        int(recipe["disc_n_scales"]),
        int(recipe["disc_capacity"]),
        int(recipe["disc_n_layers"]),
        int(recipe["disc_kernel_size"]),
        int(recipe["disc_stride"]),
    )
    discriminator.to(device)
    disc_opt = torch.optim.Adam(
        discriminator.parameters(), float(recipe["discriminator_lr"]), betas
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = (artifact_dir / f"{request_id}.pt").resolve()
    checkpoint_path = (artifact_dir / f"{request_id}.ckpt.pt").resolve()
    graph_path = (artifact_dir / f"{request_id}.graph.json").resolve()
    export_checkpoints = bool(options.get("export_checkpoints", False))
    checkpoint_interval = max(0, int(options.get("checkpoint_interval", 50)))
    total_steps = stage1_steps + stage2_steps
    windows = tuple(int(window) for window in recipe["spectral_windows"])
    log_epsilon = float(recipe["log_epsilon"])
    length = max(64, int(recipe["segment_length"]))
    tracker = start_mlflow_tracker(
        request,
        extra_config={
            "objective": "reconstruction",
            "stage1_steps": stage1_steps,
            "stage2_steps": stage2_steps,
            "total_steps": total_steps,
            "generator_lr": float(recipe["generator_lr"]),
            "discriminator_lr": float(recipe["discriminator_lr"]),
            "adam_beta1": betas[0],
            "adam_beta2": betas[1],
            "lr_decay_end_factor": float(recipe["lr_decay_end_factor"]),
            "batch_size": int(recipe["batch_size"]),
            "segment_length": length,
            "kl_beta": float(recipe["kl_beta"]),
            "kl_beta_start": float(recipe["kl_beta_start"]),
            "kl_warmup_steps": int(recipe["kl_warmup_steps"]),
            "feature_matching_weight": float(recipe["feature_matching_weight"]),
            "update_discriminator_every": int(recipe["update_discriminator_every"]),
            "phase_mangle_prob": float(recipe["phase_mangle_prob"]),
            "dequantize_bits": int(recipe["dequantize_bits"]),
            "spectral_windows": list(windows),
            "host_input_channels": input_channels,
            "host_io_mode": io_mode,
            "input_channels": input_channels,
            "clip_count": len(paths),
            "recipe": "acids-rave-v1",
            "device": effective_device,
            "requested_device": requested_device,
        },
    )
    try:
        graph_path.write_text(
            json.dumps(request.get("graph_fragment", {}), indent=2),
            encoding="utf-8",
        )
        tracker.save(str(graph_path))
    except OSError:
        graph_path = None  # type: ignore[assignment]
    last_loss = 0.0
    best_loss = math.inf
    best_state = _clone_state_dict(module)
    step = 0
    paused = False
    compactness_ready = False
    validation_segments = len(val_clips)
    last_adv = 0.0
    last_match = 0.0
    last_dis = 0.0
    last_kl = 0.0
    last_beta = float(recipe["kl_beta"])
    try:
        _emit_train(
            {
                "request_id": request_id,
                "status": "running",
                "step": 0,
                "total_steps": total_steps,
                "stage": "representation",
                "objective": "reconstruction",
                "loss": last_loss,
                "compactness": _compactness_payload(
                    compactness_ready, validation_segments
                ),
            },
            device,
            requested_device,
        )
        while step < total_steps:
            command = _read_command(command_file)
            if command == "stop":
                tracker.update_summary(
                    {
                        "success": False,
                        "status": "stopped",
                        "step": step,
                        "best_loss": best_loss if math.isfinite(best_loss) else last_loss,
                        "final_loss": last_loss,
                    }
                )
                _emit_train(
                    {
                        "request_id": request_id,
                        "status": "stopped",
                        "step": step,
                        "message": "Stopped by user",
                    },
                    device,
                    requested_device,
                )
                return {
                    "request_id": request_id,
                    "status": "stopped",
                    "step": step,
                    "message": "Stopped by user",
                    **train_device_event_fields(device, requested_device),
                }
            if command == "pause":
                paused = True
            elif command == "resume":
                paused = False
            stage = "representation" if step < stage1_steps else "quality"
            compactness = _compactness_payload(compactness_ready, validation_segments)
            if paused:
                _emit_train(
                    {
                        "request_id": request_id,
                        "status": "paused",
                        "step": step,
                        "total_steps": total_steps,
                        "stage": stage,
                        "objective": "reconstruction",
                        "loss": last_loss,
                        "compactness": compactness,
                    },
                    device,
                    requested_device,
                )
                time.sleep(0.05)
                continue
            module.train()
            _set_noise_synth_warmed_up(module, stage != "representation")
            audio = _sample_reconstruction_batch(
                train_clips,
                length,
                int(recipe["batch_size"]),
                sample_rate,
                float(recipe["phase_mangle_prob"]),
                int(recipe["dequantize_bits"]),
            ).to(device)
            reconstructed = module(audio)
            reconstructed = _match_channels(reconstructed, input_channels)
            reconstructed, audio = _align_waveform_pair(reconstructed, audio)
            spec = reconstruction_spectral_loss(
                reconstructed, audio, module, windows, log_epsilon
            )
            kl_term = _bottleneck_kl(module)
            last_kl = float(kl_term.detach().item())
            if stage == "representation":
                last_beta = reconstruction_kl_beta(
                    step,
                    int(recipe["kl_warmup_steps"]),
                    float(recipe["kl_beta_start"]),
                    float(recipe["kl_beta"]),
                )
                optimizer.zero_grad()
                (spec + last_beta * kl_term).backward()
                optimizer.step()
                scheduler.step()
                if step == stage1_steps - 1:
                    _freeze_rave_encoder(module)
                    compactness_ready, validation_segments = _fit_rave_compactness(
                        module, val_clips, length
                    )
                    module.train()
            else:
                quality_index = step - stage1_steps
                update_every = int(recipe["update_discriminator_every"])
                discriminator.train()
                if quality_index % update_every == 0:
                    disc_loss, _, _, pred_real, pred_fake = discriminator_losses(
                        audio.detach(),
                        reconstructed.detach(),
                        discriminator,
                        float(recipe["feature_matching_weight"]),
                    )
                    disc_opt.zero_grad()
                    disc_loss.backward()
                    disc_opt.step()
                    last_dis = float(disc_loss.detach().item())
                    last_adv = 0.0
                    last_match = 0.0
                    del pred_real, pred_fake
                else:
                    _, adv, match, _pred_real, _pred_fake = discriminator_losses(
                        audio.detach(),
                        reconstructed,
                        discriminator,
                        float(recipe["feature_matching_weight"]),
                    )
                    optimizer.zero_grad()
                    (spec + adv + match).backward()
                    optimizer.step()
                    scheduler.step()
                    last_adv = float(adv.detach().item())
                    last_match = float(match.detach().item())
            last_loss = float(spec.detach().item())
            if last_loss < best_loss:
                best_loss = last_loss
                best_state = _clone_state_dict(module)
            step += 1
            resolved_best = best_loss if math.isfinite(best_loss) else last_loss
            event: dict[str, Any] = {
                "request_id": request_id,
                "status": "running",
                "step": step,
                "total_steps": total_steps,
                "stage": stage,
                "objective": "reconstruction",
                "loss": last_loss,
                "best_loss": resolved_best,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "compactness": _compactness_payload(
                    compactness_ready, validation_segments
                ),
            }
            if export_checkpoints and checkpoint_interval and step % checkpoint_interval == 0:
                _export_rave_scripted(module, input_channels, checkpoint_path)
                event["artifact_path"] = str(checkpoint_path)
                tracker.save(str(checkpoint_path))
            metrics = {
                "loss": last_loss,
                "best_loss": resolved_best,
                "learning_rate": event["learning_rate"],
                "kl": last_kl,
                "kl_beta": last_beta,
                "stage": 1.0 if stage == "representation" else 2.0,
            }
            if stage == "quality":
                metrics["adversarial"] = last_adv
                metrics["feature_matching"] = last_match
                metrics["loss_dis"] = last_dis
            tracker.log(metrics, step=step)
            _emit_train(event, device, requested_device)
        module.load_state_dict(best_state)
        if stage1_steps > 0:
            _freeze_rave_encoder(module)
            compactness_ready, validation_segments = _fit_rave_compactness(
                module, val_clips, length
            )
        _export_rave_scripted(module, input_channels, artifact_path)
        tracker.save(str(artifact_path))
        resolved_best = best_loss if math.isfinite(best_loss) else last_loss
        tracker.update_summary(
            {
                "success": True,
                "status": "success",
                "best_loss": resolved_best,
                "final_loss": last_loss,
                "train_steps": total_steps,
                "compactness_ready": compactness_ready,
            }
        )
        result: dict[str, Any] = {
            "request_id": request_id,
            "status": "success",
            "step": step,
            "total_steps": total_steps,
            "loss": last_loss,
            "best_loss": resolved_best,
            "artifact_path": str(artifact_path),
            "objective": "reconstruction",
            "has_encode_decode": True,
            "compactness": _compactness_payload(compactness_ready, validation_segments),
            **train_device_event_fields(device, requested_device),
            "blackbox_metadata": {
                "origin": "train_autoload",
                "display_name": "Trained RAVE",
                "shape_signature": {
                    "input_channels": input_channels,
                    "output_channels": input_channels,
                },
                "receptive_field_samples": _module_receptive_field(module),
                "conditioning": False,
                "methods": ["forward", "encode", "decode"],
                "compactness": _compactness_payload(
                    compactness_ready, validation_segments
                ),
            },
        }
        if tracker.url:
            result["mlflow_url"] = tracker.url
        return result
    except Exception:
        tracker.update_summary({"success": False, "status": "failure", "step": step})
        raise
    finally:
        tracker.finish()



def _parse_wav_fmt(fmt: bytes) -> tuple[int, int, int, int]:
    """Return ``(audio_format, channels, sample_rate, bits_per_sample)``."""
    if len(fmt) < 16:
        raise ValueError("WAV fmt chunk is truncated")
    audio_format, channels, sample_rate, _byte_rate, _align, bits = struct.unpack_from(
        "<HHIIHH", fmt, 0
    )
    if audio_format == 0xFFFE and len(fmt) >= 26:
        audio_format = struct.unpack_from("<H", fmt, 24)[0]
    return int(audio_format), int(channels), int(sample_rate), int(bits)


def _decode_wav_samples(payload: bytes, channels: int, bits: int, ieee: bool) -> torch.Tensor:
    """Decode interleaved PCM or IEEE-float samples to ``[channels, time]``."""
    if channels < 1:
        raise ValueError("WAV channel count must be positive")
    if ieee:
        if bits != 32:
            raise ValueError("only 32-bit float WAV is supported")
        samples = torch.frombuffer(bytearray(payload), dtype=torch.float32).clone()
    elif bits == 16:
        samples = (
            torch.frombuffer(bytearray(payload), dtype=torch.int16).clone().to(torch.float32)
            / 32768.0
        )
    elif bits == 32:
        samples = (
            torch.frombuffer(bytearray(payload), dtype=torch.int32).clone().to(torch.float32)
            / 2147483648.0
        )
    else:
        raise ValueError(f"unsupported WAV bit depth {bits}")
    leftover = samples.numel() % channels
    if leftover:
        samples = samples[: samples.numel() - leftover]
    return samples.view(-1, channels).transpose(0, 1).contiguous()


def read_wav(path: str) -> tuple[torch.Tensor, int]:
    """Load a WAV file as ``[channels, time]`` float32 without torchaudio."""
    raw = Path(path).read_bytes()
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError(f"{path} is not a WAV file")
    offset = 12
    fmt: bytes | None = None
    data: bytes | None = None
    while offset + 8 <= len(raw):
        chunk_id = raw[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", raw, offset + 4)[0]
        start = offset + 8
        stop = start + chunk_size
        payload = raw[start:stop]
        offset = stop + (chunk_size & 1)
        if chunk_id == b"fmt ":
            fmt = payload
        elif chunk_id == b"data":
            data = payload
            break
    if fmt is None or data is None:
        raise ValueError(f"{path} is missing fmt or data chunks")
    audio_format, channels, sample_rate, bits = _parse_wav_fmt(fmt)
    ieee = audio_format == 3
    if audio_format not in {1, 3}:
        raise ValueError(f"unsupported WAV format {audio_format}")
    return _decode_wav_samples(data, channels, bits, ieee), sample_rate


def _match_channels(samples: torch.Tensor, channels: int) -> torch.Tensor:
    """Repeat, average, or crop ``[batch, channels, time]`` to ``channels``."""
    current = int(samples.shape[1])
    channels = max(1, int(channels))
    if current == channels:
        return samples
    if channels == 1:
        return samples.mean(dim=1, keepdim=True)
    if current == 1:
        return samples.repeat(1, channels, 1)
    if current > channels:
        return samples[:, :channels, :]
    pad = torch.zeros(
        samples.shape[0],
        channels - current,
        samples.shape[-1],
        dtype=samples.dtype,
        device=samples.device,
    )
    return torch.cat([samples, pad], dim=1)


def _blackbox_metadata(
    input_channels: int,
    output_channels: int,
    cond_dim: int,
    receptive_field: int,
    baseline_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return train-artifact metadata so the host loads `(audio, cond)` forward."""
    metadata: dict[str, Any] = {
        "origin": "train_autoload",
        "display_name": "Trained Steerable",
        "ports": [],
        "shape_signature": {
            "input_channels": int(input_channels),
            "output_channels": int(output_channels),
        },
        "conditioning": True,
        "cond_dim": int(max(1, cond_dim)),
        "receptive_field_samples": int(receptive_field),
    }
    if baseline_metrics is not None:
        metadata["baseline_metrics"] = baseline_metrics
    return metadata


def _export_scripted(
    module: nn.Module, input_channels: int, cond_dim: int, path: Path
) -> None:
    """Trace, freeze, and atomically save a conditioned TorchScript module.

    Mapping graphs are cloned onto CPU from the fragment before tracing so the
    live CUDA crop cannot leak into TorchScript as ``[1, 2, 16384]``.
    """
    was_training = bool(module.training)
    if isinstance(module, ConditionedSequential):
        export_module = _cpu_mapping_export_module(module)
    else:
        module.eval()
        export_module = _clone_module_for_export(module)
    example_samples = 256
    traced_channels = _first_conv_in_channels(
        export_module, int(getattr(module, "input_channels", input_channels) or input_channels)
    )
    example_x = torch.zeros(
        1, traced_channels, example_samples, dtype=torch.float32, device="cpu"
    )
    example_c = torch.zeros(
        1, max(1, cond_dim), example_samples, dtype=torch.float32, device="cpu"
    )
    export_module = export_module.to("cpu").eval()
    with torch.no_grad():
        _ = export_module(example_x, example_c)
        scripted = torch.jit.trace(
            export_module,
            (example_x, example_c),
            strict=False,
            check_trace=False,
        )
        scripted = torch.jit.freeze(scripted)
        output = scripted(example_x, example_c)
        if output.dim() != 3:
            raise ValueError("trained artifact returned an invalid tensor rank")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.jit.save(scripted, str(temporary_path))
    temporary_path.replace(path)
    if was_training:
        module.train()


def _load_pair(path_x: str, path_y: str) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Load an aligned x/y WAV pair as ``[1, channels, time]`` tensors."""
    x, sr_x = read_wav(path_x)
    y, sr_y = read_wav(path_y)
    if sr_x != sr_y:
        raise ValueError("training pairs must share one sample rate")
    length = min(x.shape[-1], y.shape[-1])
    channels = min(x.shape[0], y.shape[0])
    x = x[:channels, :length].unsqueeze(0)
    y = y[:channels, :length].unsqueeze(0)
    return x, y, int(sr_x)


def _read_command(path: Path | None) -> str:
    """Return the latest command verb from the coordinator command file."""
    if path is None or not path.exists():
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(payload.get("command", "")).lower()


def _emit(payload: dict[str, Any]) -> None:
    """Write one JSON event line to standard output."""
    print(json.dumps(payload), flush=True)


def mlflow_config_from_request(
    request: dict[str, Any], extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Build an MLflow params dict from a train request."""
    options = request.get("train_options", {})
    if not isinstance(options, dict):
        options = {}
    fragment = request.get("graph_fragment", {})
    if not isinstance(fragment, dict):
        fragment = {}
    capture = request.get("capture_set", {})
    if not isinstance(capture, dict):
        capture = {}
    elements = fragment.get("elements", [])
    if not isinstance(elements, list):
        elements = []
    pairs = capture.get("pairs", [])
    if not isinstance(pairs, list):
        pairs = []
    clips = capture.get("clips", [])
    if not isinstance(clips, list):
        clips = []
    loss = options.get("loss", {})
    if not isinstance(loss, dict):
        loss = {}
    config: dict[str, Any] = {
        "request_id": str(request.get("request_id", "")),
        "objective": str(options.get("objective", "mapping") or "mapping"),
        "optimizer": str(options.get("optimizer", "adam")),
        "total_steps": int(options.get("total_steps", DEFAULT_STEPS)),
        "learning_rate": float(options.get("learning_rate", DEFAULT_LR)),
        "segment_length": int(options.get("segment_length", DEFAULT_SEGMENT_LENGTH)),
        "checkpoint_interval": int(options.get("checkpoint_interval", 50)),
        "export_checkpoints": bool(options.get("export_checkpoints", False)),
        "host_input_channels": int(options.get("host_input_channels", 0)),
        "host_io_mode": str(options.get("host_io_mode", "") or ""),
        "cond_dim": int(options.get("cond_dim", 2)),
        "rf_aware_crop": bool(options.get("rf_aware_crop", True)),
        "steer_conditioning": float(options.get("steer_conditioning", STEER_CONDITIONING)),
        "device": parse_train_device(options.get("device", DEFAULT_TRAIN_DEVICE)),
        "loss": str(loss.get("type", "multiresolution_stft")),
        "element_count": len(elements),
        "element_types": [
            str(element.get("type", ""))
            for element in elements
            if isinstance(element, dict)
        ],
        "pair_count": len(pairs),
        "clip_count": len(clips),
    }
    reconstruction = options.get("reconstruction", {})
    if isinstance(reconstruction, dict) and reconstruction:
        config["reconstruction"] = reconstruction
    if extra:
        config.update(extra)
    return config


def _now_ms() -> int:
    """Return the current Unix time in milliseconds."""
    return int(time.time() * 1000)


def _param_value(value: Any) -> str:
    """Serialize one MLflow param value and truncate to the REST limit."""
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, (dict, list)):
        text = json.dumps(value, separators=(",", ":"))
    else:
        text = str(value)
    return text[:MLFLOW_PARAM_MAX_LENGTH]


def _mlflow_http(
    method: str,
    url: str,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 10.0,
) -> tuple[int, bytes]:
    """Send one HTTP request to an MLflow tracking server.

    Tests may replace this function to avoid a live server.
    """
    request = urllib.request.Request(url, data=body, method=method)
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read()
    except urllib.error.HTTPError as error:
        payload = error.read() if error.fp is not None else b""
        return int(error.code), payload


def _file_uri_to_path(uri: str) -> Path:
    """Convert a ``file:`` artifact URI to a local path."""
    parsed = urllib.parse.urlparse(uri)
    return Path(urllib.parse.unquote(parsed.path))


class MlflowTracker:
    """Optional MLflow run via the Tracking Server REST API."""

    def __init__(self) -> None:
        """Create an inactive tracker until ``init`` succeeds."""
        self._active = False
        self._tracking_uri = ""
        self._api_root = ""
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        self._run_id = ""
        self._experiment_id = ""
        self._artifact_uri = ""
        self._terminal_status = "FINISHED"
        self.url = ""

    @property
    def active(self) -> bool:
        """True after a successful REST ``runs/create``."""
        return self._active

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        query: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """POST or GET JSON under ``/api/2.0/mlflow``."""
        url = f"{self._api_root}/{path.lstrip('/')}"
        if query:
            url = f"{url}?{urllib.parse.urlencode(query)}"
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        status, raw = _mlflow_http(method, url, body, self._headers)
        if not raw:
            return status, {}
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return status, {}
        if not isinstance(decoded, dict):
            return status, {}
        return status, decoded

    def _log_batch(
        self,
        *,
        metrics: list[dict[str, Any]] | None = None,
        params: list[dict[str, str]] | None = None,
        tags: list[dict[str, str]] | None = None,
    ) -> None:
        """Call ``2.0/mlflow/runs/log-batch``."""
        if not self._active or not self._run_id:
            return
        payload: dict[str, Any] = {"run_id": self._run_id}
        if metrics:
            payload["metrics"] = metrics
        if params:
            payload["params"] = params
        if tags:
            payload["tags"] = tags
        self._request("POST", "runs/log-batch", payload)

    def init(
        self,
        *,
        tracking_uri: str,
        experiment: str,
        name: str = "",
        tags: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Resolve the experiment, create a run, and log params over REST."""
        origin = tracking_uri.strip().rstrip("/")
        if not origin.lower().startswith(("http://", "https://")):
            return
        self._tracking_uri = origin
        self._api_root = f"{origin}/api/2.0/mlflow"
        headers = {"Content-Type": "application/json"}
        token = os.environ.get("MLFLOW_TRACKING_TOKEN", "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        else:
            username = os.environ.get("MLFLOW_TRACKING_USERNAME", "")
            password = os.environ.get("MLFLOW_TRACKING_PASSWORD", "")
            if username:
                secret = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode(
                    "ascii"
                )
                headers["Authorization"] = f"Basic {secret}"
        self._headers = headers
        experiment_name = experiment or DEFAULT_MLFLOW_EXPERIMENT
        try:
            status, body = self._request(
                "GET",
                "experiments/get-by-name",
                query={"experiment_name": experiment_name},
            )
            experiment_id = ""
            if status == 200:
                experiment_id = str(body.get("experiment", {}).get("experiment_id", ""))
            if not experiment_id:
                status, body = self._request(
                    "POST",
                    "experiments/create",
                    {"name": experiment_name},
                )
                experiment_id = str(body.get("experiment_id", ""))
                if not experiment_id and status != 200:
                    status, body = self._request(
                        "GET",
                        "experiments/get-by-name",
                        query={"experiment_name": experiment_name},
                    )
                    experiment_id = str(
                        body.get("experiment", {}).get("experiment_id", "")
                    )
            if not experiment_id:
                return
            run_tags: list[dict[str, str]] = []
            if name:
                run_tags.append({"key": "mlflow.runName", "value": name})
            for tag in tags or DEFAULT_MLFLOW_TAGS:
                tag_name = str(tag).strip()
                if tag_name:
                    run_tags.append({"key": tag_name, "value": "true"})
            status, body = self._request(
                "POST",
                "runs/create",
                {
                    "experiment_id": experiment_id,
                    "start_time": _now_ms(),
                    "tags": run_tags,
                },
            )
            info = body.get("run", {}).get("info", {}) if status == 200 else {}
            if not isinstance(info, dict):
                info = {}
            run_id = str(info.get("run_id") or info.get("run_uuid") or "")
            if not run_id:
                return
            self._run_id = run_id
            self._experiment_id = experiment_id
            self._artifact_uri = str(info.get("artifact_uri", "") or "")
            self._active = True
            self._terminal_status = "FINISHED"
            self.url = f"{origin}/#/experiments/{experiment_id}/runs/{run_id}"
            if config:
                self._log_batch(
                    params=[
                        {"key": str(key), "value": _param_value(value)}
                        for key, value in config.items()
                    ]
                )
        except Exception:
            self._active = False
            self._run_id = ""
            self.url = ""

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """POST numeric metrics to ``runs/log-batch``."""
        if not self._active:
            return
        timestamp = _now_ms()
        resolved_step = 0 if step is None else int(step)
        entries: list[dict[str, Any]] = []
        for key, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            numeric = float(value)
            if not math.isfinite(numeric):
                continue
            entries.append(
                {
                    "key": str(key),
                    "value": numeric,
                    "timestamp": timestamp,
                    "step": resolved_step,
                }
            )
        if not entries:
            return
        try:
            self._log_batch(metrics=entries)
        except Exception:
            return

    def update_summary(self, summary: dict[str, Any]) -> None:
        """Log summary fields as tags and metrics; map status to the run end state."""
        if not self._active:
            return
        status_value = str(summary.get("status", "")).lower()
        if status_value == "stopped":
            self._terminal_status = "KILLED"
        elif status_value == "failure":
            self._terminal_status = "FAILED"
        elif status_value == "success":
            self._terminal_status = "FINISHED"
        timestamp = _now_ms()
        tags: list[dict[str, str]] = []
        metrics: list[dict[str, Any]] = []
        for key, value in summary.items():
            if isinstance(value, bool):
                tags.append({"key": str(key), "value": "true" if value else "false"})
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric = float(value)
                if math.isfinite(numeric):
                    metrics.append(
                        {
                            "key": str(key),
                            "value": numeric,
                            "timestamp": timestamp,
                            "step": 0,
                        }
                    )
            else:
                tags.append({"key": str(key), "value": _param_value(value)})
        try:
            self._log_batch(metrics=metrics or None, tags=tags or None)
        except Exception:
            return

    def save(self, path: str) -> None:
        """Upload ``path`` through the artifacts REST proxy or a ``file:`` URI."""
        if not self._active or not path:
            return
        source = Path(path)
        if not source.is_file():
            return
        try:
            if self._artifact_uri.lower().startswith("file:"):
                destination_dir = _file_uri_to_path(self._artifact_uri)
                destination_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination_dir / source.name)
                return
            artifact_root = self._artifact_uri.rstrip("/")
            if artifact_root.lower().startswith(("http://", "https://")):
                url = f"{artifact_root}/{source.name}"
            else:
                url = (
                    f"{self._tracking_uri}/api/2.0/mlflow-artifacts/artifacts/"
                    f"{self._run_id}/{source.name}"
                )
            headers = {
                key: value
                for key, value in self._headers.items()
                if key.lower() != "content-type"
            }
            headers["Content-Type"] = "application/octet-stream"
            _mlflow_http("PUT", url, source.read_bytes(), headers)
        except Exception:
            return

    def finish(self) -> None:
        """POST ``runs/update`` with the terminal status and drop the client."""
        if not self._active or not self._run_id:
            self._active = False
            return
        try:
            self._request(
                "POST",
                "runs/update",
                {
                    "run_id": self._run_id,
                    "status": self._terminal_status,
                    "end_time": _now_ms(),
                },
            )
        except Exception:
            pass
        self._active = False
        self._run_id = ""
        self.url = ""


def start_mlflow_tracker(
    request: dict[str, Any], extra_config: dict[str, Any] | None = None
) -> MlflowTracker:
    """Start a tracker when ``train_options.mlflow.enabled`` is true."""
    tracker = MlflowTracker()
    options = request.get("train_options", {})
    mlflow_opts = options.get("mlflow", {}) if isinstance(options, dict) else {}
    if isinstance(mlflow_opts, bool):
        enabled = mlflow_opts
        mlflow_opts = {}
    elif not isinstance(mlflow_opts, dict):
        return tracker
    else:
        enabled = bool(mlflow_opts.get("enabled", False))
    if not enabled:
        return tracker
    tags = mlflow_opts.get("tags", DEFAULT_MLFLOW_TAGS)
    if not isinstance(tags, list):
        tags = list(DEFAULT_MLFLOW_TAGS)
    tracking_uri = str(mlflow_opts.get("tracking_uri", "") or "").strip()
    if not tracking_uri:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if not tracking_uri:
        tracking_uri = DEFAULT_MLFLOW_TRACKING_URI
    experiment = str(mlflow_opts.get("experiment", "") or "").strip()
    if not experiment:
        experiment = str(mlflow_opts.get("project", "") or "").strip()
    tracker.init(
        tracking_uri=tracking_uri,
        experiment=experiment or DEFAULT_MLFLOW_EXPERIMENT,
        name=str(mlflow_opts.get("name", "") or ""),
        tags=[str(tag) for tag in tags],
        config=mlflow_config_from_request(request, extra_config),
    )
    return tracker


def _loss_nodes_from_fragment(fragment: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Index Loss elements in the train fragment by node id."""
    nodes: dict[int, dict[str, Any]] = {}
    for element in fragment.get("elements", []) or []:
        if not isinstance(element, dict):
            continue
        if str(element.get("type", "")) != "loss":
            continue
        nodes[int(element["id"])] = element
    return nodes


def _loss_type_of(element: dict[str, Any]) -> str:
    """Return the catalog token for a Loss node."""
    properties = _properties(element)
    raw = properties.get("loss_type", "mr_stft")
    mapping = {
        0: "mr_stft",
        1: "spectral_distance",
        2: "kl",
        3: "adversarial",
        4: "feature_matching",
        "0": "mr_stft",
        "1": "spectral_distance",
        "2": "kl",
        "3": "adversarial",
        "4": "feature_matching",
    }
    return str(mapping.get(raw, raw) or "mr_stft")


def _zip_data_loader_bindings(
    request: dict[str, Any],
) -> tuple[list[str], list[str], list[float]]:
    """Zip active Data Loader audio lists (and optional scalar) by example index."""
    bindings = request.get("data_loader_bindings", {})
    if not isinstance(bindings, dict) or not bindings:
        return [], [], []
    loader_id = str(request.get("active_data_loader_id", "") or "")
    per_loader = bindings.get(loader_id) if loader_id else None
    if not isinstance(per_loader, dict):
        first = next(iter(bindings.values()), None)
        per_loader = first if isinstance(first, dict) else {}
    audio_lists: list[list[str]] = []
    scalars: list[float] = []
    for key in sorted(per_loader, key=lambda item: int(item) if str(item).isdigit() else 0):
        output = per_loader[key]
        if not isinstance(output, dict):
            continue
        kind = str(output.get("kind", "audio_list"))
        if kind == "constant_scalar":
            scalars.append(float(output.get("value", 0.0)))
            continue
        paths: list[str] = []
        for item in output.get("items", []) or []:
            if isinstance(item, dict) and item.get("path"):
                paths.append(str(item["path"]))
        audio_lists.append(paths)
    if len(audio_lists) < 2:
        return [], [], scalars
    count = min(len(audio_lists[0]), len(audio_lists[1]))
    if count < 1:
        return [], [], scalars
    return audio_lists[0][:count], audio_lists[1][:count], scalars


def _resolve_loss_stages(
    request: dict[str, Any], fragment: dict[str, Any], total_steps: int
) -> list[dict[str, Any]]:
    """Build concrete stages from the schedule or every wired Loss node."""
    loss_nodes = _loss_nodes_from_fragment(fragment)
    schedule = request.get("loss_schedule", {})
    stages: list[dict[str, Any]] = []
    if isinstance(schedule, dict):
        raw_stages = schedule.get("stages", [])
        if isinstance(raw_stages, list):
            stages = [stage for stage in raw_stages if isinstance(stage, dict)]
    if stages:
        return stages
    if not loss_nodes:
        raise ValueError(
            "loss_schedule is empty; wire Loss nodes or add stages before train_graph"
        )
    return [
        {
            "name": "train",
            "steps": total_steps,
            "losses": [
                {"loss_node_id": node_id, "weight": 1.0}
                for node_id, element in loss_nodes.items()
            ],
        }
    ]


def _apply_train_graph_grads(
    module: nn.Module,
    armed: set[int],
    freeze: set[int],
    learning_rate: float,
) -> torch.optim.Optimizer:
    """Enable grads for armed∖freeze layers and rebuild Adam on that set."""
    if isinstance(module, ConditionedSequential) and module.layer_ids:
        for layer, element_id in zip(module.layers, module.layer_ids):
            train = element_id in armed and element_id not in freeze
            for param in layer.parameters():
                param.requires_grad_(train)
    trainable = [param for param in module.parameters() if param.requires_grad]
    if not trainable:
        raise ValueError(
            "armed_element_ids / freeze_element_ids left no trainable parameters"
        )
    return torch.optim.Adam(trainable, learning_rate)


def train_graph(
    request: dict[str, Any], artifact_dir: Path, command_file: Path | None
) -> dict[str, Any]:
    """Train the graph fragment from Data Loader bindings and a loss schedule."""
    request_id = str(request.get("request_id", ""))
    options = request.get("train_options", {})
    if not isinstance(options, dict):
        options = {}
    fragment = request.get("graph_fragment", {})
    if not isinstance(fragment, dict):
        fragment = {}
    total_steps = max(1, int(options.get("total_steps", DEFAULT_STEPS)))
    stages = _resolve_loss_stages(request, fragment, total_steps)
    scheduled_steps = sum(max(1, int(stage.get("steps", 1))) for stage in stages)
    if scheduled_steps > 0:
        total_steps = scheduled_steps

    x_paths, y_paths, _scalars = _zip_data_loader_bindings(request)
    capture_set = request.get("capture_set", {})
    if not isinstance(capture_set, dict):
        capture_set = {}
    pairs = capture_set.get("pairs", []) if not x_paths else []
    if x_paths and y_paths:
        x_batch, y_batch, _sample_rate = _load_pair(x_paths[0], y_paths[0])
        for x_path, y_path in zip(x_paths[1:], y_paths[1:]):
            extra_x, extra_y, _ = _load_pair(x_path, y_path)
            x_batch = torch.cat([x_batch, extra_x], dim=-1)
            y_batch = torch.cat([y_batch, extra_y], dim=-1)
    elif pairs:
        x_batch, y_batch, _sample_rate = _load_pair(pairs[0]["x_path"], pairs[0]["y_path"])
        for pair in pairs[1:]:
            extra_x, extra_y, _ = _load_pair(pair["x_path"], pair["y_path"])
            x_batch = torch.cat([x_batch, extra_x], dim=-1)
            y_batch = torch.cat([y_batch, extra_y], dim=-1)
    else:
        raise ValueError("train_graph requires data_loader_bindings or capture pairs")

    cond_dim = max(1, int(options.get("cond_dim", 2)))
    input_channels = _mapping_input_channels(fragment, options, int(x_batch.shape[1]))
    x_batch = _match_channels(x_batch, input_channels)
    y_batch = _match_channels(y_batch, input_channels)
    device, requested_device, effective_device = resolve_train_device(
        options.get("device", DEFAULT_TRAIN_DEVICE)
    )
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)
    module = build_module(fragment, input_channels, cond_dim)
    module.to(device)
    armed = {int(item) for item in (request.get("armed_element_ids") or [])}
    if auraloss is None:
        raise RuntimeError("auraloss is required for mr_stft training")
    loss_fn = auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=list(options.get("loss", {}).get("fft_sizes", FFT_SIZES)),
        win_lengths=list(options.get("loss", {}).get("win_lengths", WIN_LENGTHS)),
        hop_sizes=list(options.get("loss", {}).get("hop_sizes", HOP_SIZES)),
    )
    loss_fn.to(device)
    learning_rate = float(options.get("learning_rate", DEFAULT_LR))
    segment_length = max(1, int(options.get("segment_length", DEFAULT_SEGMENT_LENGTH)))
    cond = torch.full(
        (1, cond_dim, 1),
        float(STEER_CONDITIONING),
        dtype=torch.float32,
        device=device,
    )
    module.train()
    last_loss = 0.0
    best_loss = math.inf
    best_state = {
        name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()
    }
    step = 0
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = (artifact_dir / f"{request_id}.pt").resolve()
    paused = False
    optimizer = _apply_train_graph_grads(module, armed, set(), learning_rate)
    for stage_index, stage in enumerate(stages):
        stage_name = str(stage.get("name") or f"stage{stage_index + 1}")
        stage_steps = max(1, int(stage.get("steps", 1)))
        freeze = {
            int(item)
            for item in (stage.get("freeze_element_ids") or [])
            if str(item).strip() != ""
        }
        optimizer = _apply_train_graph_grads(module, armed, freeze, learning_rate)
        for _ in range(stage_steps):
            command = _read_command(command_file)
            if command == "stop":
                return {
                    "request_id": request_id,
                    "status": "stopped",
                    "step": step,
                    "total_steps": total_steps,
                    "stage": stage_name,
                    "loss": last_loss,
                }
            if command == "pause":
                paused = True
            elif command == "resume":
                paused = False
            if paused:
                continue
            crop = min(segment_length, int(x_batch.shape[-1]), int(y_batch.shape[-1]))
            start = 0 if x_batch.shape[-1] <= crop else torch.randint(
                0, x_batch.shape[-1] - crop + 1, (1,)
            ).item()
            x_crop = x_batch[..., start : start + crop]
            y_crop = y_batch[..., start : start + crop]
            predicted = module(x_crop, cond) if cond_dim else module(x_crop)
            if isinstance(predicted, tuple):
                predicted = predicted[0]
            stage_loss = torch.zeros((), device=device)
            for loss_ref in stage.get("losses", []) or []:
                if not isinstance(loss_ref, dict):
                    continue
                node_id = int(loss_ref.get("loss_node_id", 0) or 0)
                element = _loss_nodes_from_fragment(fragment).get(node_id)
                weight = float(loss_ref.get("weight", 1.0))
                loss_type = _loss_type_of(element) if element else "mr_stft"
                if loss_type in {"mr_stft", "spectral_distance"}:
                    stage_loss = stage_loss + weight * loss_fn(predicted, y_crop)
                elif loss_type == "kl":
                    stage_loss = stage_loss + weight * (predicted.pow(2).mean() * 0.0)
                else:
                    stage_loss = stage_loss + weight * loss_fn(predicted, y_crop)
            optimizer.zero_grad(set_to_none=True)
            stage_loss.backward()
            optimizer.step()
            last_loss = float(stage_loss.detach().cpu())
            if last_loss < best_loss:
                best_loss = last_loss
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in module.state_dict().items()
                }
            step += 1
            _emit_train(
                {
                    "request_id": request_id,
                    "status": "running",
                    "step": step,
                    "total_steps": total_steps,
                    "stage": stage_name,
                    "stage_index": stage_index,
                    "loss": last_loss,
                    "best_loss": best_loss,
                    "learning_rate": learning_rate,
                },
                device,
                requested_device,
            )
    module.load_state_dict(best_state)
    module.eval()
    cpu_module = module.cpu()
    example = torch.zeros(1, int(input_channels), 64)
    cond_example = torch.zeros(1, cond_dim, 1) if cond_dim else None
    if cond_example is None:
        scripted = torch.jit.trace(cpu_module, example, strict=False)
    else:
        scripted = torch.jit.trace(cpu_module, (example, cond_example), strict=False)
    scripted.save(str(artifact_path))
    return {
        "request_id": request_id,
        "status": "success",
        "step": step,
        "total_steps": total_steps,
        "loss": last_loss,
        "best_loss": best_loss,
        "artifact_path": str(artifact_path),
        "device": effective_device,
        "requested_device": requested_device,
    }


def train_request(request: dict[str, Any], artifact_dir: Path, command_file: Path | None) -> dict[str, Any]:
    """Run architecture-agnostic train_graph, or legacy objective recipes."""
    request_id = str(request.get("request_id", ""))
    operation = str(request.get("operation", "") or "")
    if operation not in {"train_graph", "train_steerable"} or not request_id:
        raise ValueError("invalid train request envelope")

    options = request.get("train_options", {})
    if not isinstance(options, dict):
        options = {}
    if operation == "train_graph":
        return train_graph(request, artifact_dir, command_file)
    objective = str(options.get("objective", "mapping") or "mapping").strip().lower()
    if objective == "reconstruction":
        return train_reconstruction(request, artifact_dir, command_file)
    capture_set = request.get("capture_set", {})
    if not isinstance(capture_set, dict):
        capture_set = {}
    if mapping_rejects_unpaired(capture_set):
        raise ValueError("mapping cannot train unpaired clips")

    total_steps = max(1, int(options.get("total_steps", DEFAULT_STEPS)))
    segment_length = max(1, int(options.get("segment_length", DEFAULT_SEGMENT_LENGTH)))
    learning_rate = float(options.get("learning_rate", DEFAULT_LR))
    cond_dim = max(1, int(options.get("cond_dim", 2)))
    export_checkpoints = bool(options.get("export_checkpoints", False))
    checkpoint_interval = max(0, int(options.get("checkpoint_interval", 50)))
    pairs = request.get("capture_set", {}).get("pairs", [])
    if not pairs:
        raise ValueError("train request contains no selected library pairs")

    x_batch, y_batch, _sample_rate = _load_pair(pairs[0]["x_path"], pairs[0]["y_path"])
    for pair in pairs[1:]:
        extra_x, extra_y, _ = _load_pair(pair["x_path"], pair["y_path"])
        if extra_x.shape[1] != x_batch.shape[1]:
            raise ValueError("selected pairs must share a channel count")
        x_batch = torch.cat([x_batch, extra_x], dim=-1)
        y_batch = torch.cat([y_batch, extra_y], dim=-1)

    fragment = request.get("graph_fragment", {})
    if not isinstance(fragment, dict):
        fragment = {}
    input_channels = _mapping_input_channels(fragment, options, int(x_batch.shape[1]))
    x_batch = _match_channels(x_batch, input_channels)
    y_batch = _match_channels(y_batch, input_channels)
    device, requested_device, effective_device = resolve_train_device(
        options.get("device", DEFAULT_TRAIN_DEVICE)
    )
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)
    module = build_module(fragment, input_channels, cond_dim)
    module.to(device)
    receptive_field = _module_receptive_field(module)
    if auraloss is None:
        raise RuntimeError("auraloss is required for the multiresolution STFT loss")
    loss_fn = auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=list(options.get("loss", {}).get("fft_sizes", FFT_SIZES)),
        win_lengths=list(options.get("loss", {}).get("win_lengths", WIN_LENGTHS)),
        hop_sizes=list(options.get("loss", {}).get("hop_sizes", HOP_SIZES)),
    )
    loss_fn.to(device)
    optimizer = torch.optim.Adam(module.parameters(), learning_rate)
    milestones = [int(total_steps * 0.80), int(total_steps * 0.95)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
    cond = torch.full(
        (1, cond_dim, 1),
        float(STEER_CONDITIONING),
        dtype=torch.float32,
        device=device,
    )

    tracker = start_mlflow_tracker(
        request,
        extra_config={
            "receptive_field_samples": receptive_field,
            "input_channels": input_channels,
            "total_steps": total_steps,
            "learning_rate": learning_rate,
            "segment_length": segment_length,
            "cond_dim": cond_dim,
            "host_input_channels": input_channels,
            "pair_count": len(pairs),
            "device": effective_device,
            "requested_device": requested_device,
        },
    )

    module.train()
    last_loss = 0.0
    best_loss = math.inf
    best_state = {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}
    step = 0
    paused = False
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = (artifact_dir / f"{request_id}.ckpt.pt").resolve()
    artifact_path = (artifact_dir / f"{request_id}.pt").resolve()
    graph_path = (artifact_dir / f"{request_id}.graph.json").resolve()
    try:
        graph_path.write_text(
            json.dumps(request.get("graph_fragment", {}), indent=2),
            encoding="utf-8",
        )
        tracker.save(str(graph_path))
    except OSError:
        graph_path = None  # type: ignore[assignment]
    try:
        _emit_train(
            {
                "request_id": request_id,
                "status": "running",
                "step": 0,
                "total_steps": total_steps,
                "stage": "mapping",
                "objective": "mapping",
                "loss": last_loss,
            },
            device,
            requested_device,
        )
        while step < total_steps:
            command = _read_command(command_file)
            if command == "stop":
                tracker.update_summary(
                    {
                        "success": False,
                        "status": "stopped",
                        "step": step,
                        "best_loss": best_loss if math.isfinite(best_loss) else last_loss,
                        "final_loss": last_loss,
                    }
                )
                _emit_train(
                    {
                        "request_id": request_id,
                        "status": "stopped",
                        "step": step,
                        "message": "Stopped by user",
                    },
                    device,
                    requested_device,
                )
                return {
                    "request_id": request_id,
                    "status": "stopped",
                    "step": step,
                    "message": "Stopped by user",
                    **train_device_event_fields(device, requested_device),
                }
            if command == "pause":
                paused = True
            elif command == "resume":
                paused = False
            if paused:
                _emit_train(
                    {
                        "request_id": request_id,
                        "status": "paused",
                        "step": step,
                        "total_steps": total_steps,
                        "loss": last_loss,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    },
                    device,
                    requested_device,
                )
                time.sleep(0.05)
                continue

            optimizer.zero_grad()
            x_crop, y_crop = rf_aware_crop(
                x_batch, y_batch, receptive_field, segment_length
            )
            predicted = module(x_crop, cond)
            if predicted.shape[-1] != y_crop.shape[-1]:
                predicted = predicted[..., -y_crop.shape[-1] :]
            loss = loss_fn(predicted, y_crop)
            loss.backward()
            optimizer.step()
            scheduler.step()
            last_loss = float(loss.item())
            step += 1
            if last_loss < best_loss:
                best_loss = last_loss
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in module.state_dict().items()
                }
            event: dict[str, Any] = {
                "request_id": request_id,
                "status": "running",
                "step": step,
                "total_steps": total_steps,
                "stage": "mapping",
                "objective": "mapping",
                "loss": last_loss,
                "best_loss": best_loss if math.isfinite(best_loss) else last_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            if (
                export_checkpoints
                and checkpoint_interval > 0
                and step % checkpoint_interval == 0
            ):
                _export_scripted(module, input_channels, cond_dim, checkpoint_path)
                module.train()
                event["artifact_path"] = str(checkpoint_path)
                event["blackbox_metadata"] = _blackbox_metadata(
                    _first_conv_in_channels(module, input_channels),
                    input_channels,
                    cond_dim,
                    receptive_field,
                )
            tracker.log(
                {
                    "loss": last_loss,
                    "best_loss": event["best_loss"],
                    "learning_rate": event["learning_rate"],
                    "step": step,
                },
                step=step,
            )
            _emit_train(event, device, requested_device)

        module.load_state_dict(best_state)
        _export_scripted(module, input_channels, cond_dim, artifact_path)
        tracker.save(str(artifact_path))
        probe_channels = _first_conv_in_channels(module, input_channels)
        example_x = torch.zeros(1, probe_channels, 256, device=device)
        example_c = torch.zeros(1, cond_dim, 1, device=device)
        with torch.inference_mode():
            output = module(example_x, example_c)
        resolved_best = best_loss if math.isfinite(best_loss) else last_loss
        tracker.update_summary(
            {
                "success": True,
                "status": "success",
                "best_loss": resolved_best,
                "final_loss": last_loss,
                "train_steps": total_steps,
            }
        )
        result: dict[str, Any] = {
            "request_id": request_id,
            "status": "success",
            "objective": "mapping",
            "has_encode_decode": False,
            "artifact_path": str(artifact_path),
            **train_device_event_fields(device, requested_device),
            "blackbox_metadata": _blackbox_metadata(
                probe_channels,
                int(output.size(1)),
                cond_dim,
                receptive_field,
                {
                    "train_steps": total_steps,
                    "final_loss": last_loss,
                    "best_loss": resolved_best,
                },
            ),
        }
        if tracker.url:
            result["mlflow_url"] = tracker.url
        return result
    except Exception:
        tracker.update_summary({"success": False, "status": "failure", "step": step})
        raise
    finally:
        tracker.finish()


def main() -> int:
    """Run the command-line worker and stream JSON events."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--command-file", type=Path, default=None)
    arguments = parser.parse_args()

    request_id = ""
    try:
        request = json.loads(arguments.request.read_text(encoding="utf-8"))
        request_id = str(request.get("request_id", ""))
        response = train_request(request, arguments.artifact_dir, arguments.command_file)
        print(json.dumps(response), flush=True)
        return 0 if response.get("status") == "success" else 0
    except Exception as error:  # Worker boundary must return user-facing errors.
        print(
            json.dumps(
                {
                    "request_id": request_id,
                    "status": "failure",
                    "error_message": f"{error}\n{traceback.format_exc()}",
                }
            ),
            flush=True,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
