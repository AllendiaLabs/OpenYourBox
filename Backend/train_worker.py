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
from typing import Any

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
DEFAULT_MLFLOW_EXPERIMENT = "openyourbox"
DEFAULT_MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
DEFAULT_MLFLOW_TAGS = ["train", "steerable"]
MLFLOW_PARAM_MAX_LENGTH = 6000


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
    """Wrap ``module`` with ``nn.utils.weight_norm`` when enabled."""
    if not enabled:
        return module
    return nn.utils.weight_norm(module)


def strip_weight_norm(module: nn.Module) -> None:
    """Materialize ``W = g * v / ||v||`` so live/TorchScript see plain ``.weight``.

    Safe to call when no weight-norm hooks are present. Walks every submodule
    that still exposes ``weight_g`` / ``weight_v``.
    """
    for child in list(module.modules()):
        if not hasattr(child, "weight_g") or not hasattr(child, "weight_v"):
            continue
        try:
            nn.utils.remove_weight_norm(child)
        except (ValueError, KeyError, AttributeError):
            continue


def assign_plain_conv_weight(layer: nn.Module, weight: torch.Tensor) -> None:
    """Copy a dense weight into a Conv1d/ConvTranspose1d, including weight-norm."""
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

    def __init__(self, modules: list[nn.Module], cond_dim: int) -> None:
        """Store ordered modules and the conditioning width."""
        super().__init__()
        self.layers = nn.ModuleList(modules)
        self.cond_dim = cond_dim

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
        value = samples
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
        if "float_value" in item:
            values[key] = float(item["float_value"])
        else:
            values[key] = int(item["value"])
    return values


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
    if types & rave_types:
        return build_rave_graph_module(fragment, input_channels, cond_dim)

    modules: list[nn.Module] = []
    channels = input_channels
    cond_dim = max(1, int(cond_dim))
    for element in _topological_elements(fragment):
        element_type = str(element["type"])
        properties = _properties(element)
        if element_type in {"audio_input", "audio_output", "knob_input", "xy_trackpad"}:
            continue
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
        elif element_type in {"utility", "merge", "sum", "multiply"}:
            raise ValueError("mixer elements cannot be trained by this worker")

    return ConditionedSequential(modules, cond_dim) if modules else nn.Identity()


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
    """Filtered-noise addend."""

    def __init__(self, in_channels: int, noise_bands: int) -> None:
        """Create a 1x1 amplitude projector."""
        super().__init__()
        self.projector = nn.Conv1d(in_channels, max(1, noise_bands), 1, bias=False)

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        """Return noise mixed back to the input width."""
        bands = torch.sigmoid(self.projector(samples))
        mixed = (bands * torch.randn_like(bands)).mean(1, keepdim=True)
        return mixed.expand_as(samples)


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
    ) -> None:
        """Store graph topology and per-node layers."""
        super().__init__()
        self.layers = nn.ModuleDict({str(key): value for key, value in layers.items()})
        self.order = order
        self.incoming = incoming
        self.bottleneck_id = bottleneck_id
        self.types = types
        self.input_channels = input_channels
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
            if not sources:
                current = audio if start_from is None else values.get(node_id, audio)
            elif len(sources) == 1:
                current = values[sources[0]]
            else:
                stacked = [values[source] for source in sources if source in values]
                current = stacked[0]
                for extra in stacked[1:]:
                    length = min(current.shape[-1], extra.shape[-1])
                    current = current[..., :length] + extra[..., :length]
            key = str(node_id)
            if key in self.layers:
                current = self.layers[key](current)
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
    for connection in fragment.get("connections", []):
        source = int(connection["source_element_id"])
        destination = int(connection["destination_element_id"])
        if source in incoming and destination in incoming:
            incoming[destination].append(source)
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
            layers[node_id] = NoiseSynthLayer(in_ch, int(properties.get("noise_bands", 5)))
            channels_by_id[node_id] = in_ch
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
        else:
            channels_by_id[node_id] = in_ch
    return RaveGraphModule(
        layers,
        [int(element["id"]) for element in elements],
        incoming,
        bottleneck_id,
        types,
        input_channels,
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


def spectral_distance(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Multi-resolution log-magnitude distance used by reconstruction stage 1."""
    loss = predicted.new_zeros(())
    used = 0
    time_length = int(predicted.shape[-1])
    for window in (2048, 1024, 512, 256, 128):
        if window > time_length:
            continue
        hop = max(1, window // 4)
        spec_p = torch.stft(
            predicted.reshape(-1, predicted.shape[-1]),
            n_fft=window,
            hop_length=hop,
            win_length=window,
            return_complex=True,
        ).abs()
        spec_t = torch.stft(
            target.reshape(-1, target.shape[-1]),
            n_fft=window,
            hop_length=hop,
            win_length=window,
            return_complex=True,
        ).abs()
        loss = loss + (spec_p.add(1e-7).log() - spec_t.add(1e-7).log()).abs().mean()
        used += 1
    if used == 0:
        return (predicted - target).abs().mean()
    return loss


class CombineDiscriminator(nn.Module):
    """Lightweight train-only hinge discriminator."""

    def __init__(self, channels: int) -> None:
        """Create a small strided conv stack."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, 16, 15, stride=1, padding=7),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(16, 32, 15, stride=4, padding=7),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 1, 5, stride=1, padding=2),
        )

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        """Return per-sample logits."""
        return self.net(samples)

    def features(self, samples: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Return logits plus intermediate activations for feature matching."""
        collected: list[torch.Tensor] = []
        value = samples
        for layer in self.net:
            value = layer(value)
            if isinstance(layer, nn.LeakyReLU):
                collected.append(value)
        return value, collected


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


def _bottleneck_kl(module: RaveGraphModule) -> torch.Tensor:
    """Return KL from the variational head when present."""
    if module.bottleneck_id is None:
        return torch.zeros(())
    key = str(module.bottleneck_id)
    if key not in module.layers:
        return torch.zeros(())
    layer = module.layers[key]
    if isinstance(layer, VariationalBottleneckLayer):
        return layer.kl()
    return torch.zeros(())


def _export_rave_scripted(module: RaveGraphModule, input_channels: int, path: Path) -> None:
    """Trace forward/encode/decode for a RAVE graph."""
    module.eval()
    export_module = copy.deepcopy(module)
    strip_weight_norm(export_module)
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
        export_module.latent_mean
        if export_module.latent_mean is not None
        else torch.zeros(latent_size)
    )
    pca = (
        export_module.latent_pca
        if export_module.latent_pca is not None
        else torch.eye(latent_size)
    )
    cumulative = (
        export_module.cumulative_variance
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

        def decode(self, latent: torch.Tensor) -> torch.Tensor:
            return self.inner.decode(latent)

        def forward(self, audio: torch.Tensor) -> torch.Tensor:
            return self.inner.forward(audio)

    wrapped = Wrapper(export_module).eval()
    with torch.inference_mode():
        scripted = torch.jit.trace_module(
            wrapped,
            {"forward": example, "encode": example, "decode": latent_example},
            strict=False,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.jit.save(scripted, str(temporary))
    temporary.replace(path)


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
                latent = module.encode(audio)
                rows.append(
                    latent.permute(0, 2, 1).reshape(-1, latent.shape[1]).cpu()
                )
                if start + length >= clip.shape[-1]:
                    break
                start += length
    if not rows:
        return None
    return torch.cat(rows, 0)


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
    """Run the two-stage RAVE reconstruction recipe."""
    request_id = str(request.get("request_id", ""))
    options = request.get("train_options", {})
    reconstruction = options.get("reconstruction", {}) if isinstance(options, dict) else {}
    if not isinstance(reconstruction, dict):
        reconstruction = {}
    stage1_steps = max(1, int(reconstruction.get("stage1_steps", 1_000_000)))
    stage2_steps = max(1, int(reconstruction.get("stage2_steps", 1_000_000)))
    capture_set = request.get("capture_set", {})
    if not isinstance(capture_set, dict):
        capture_set = {}
    paths = flatten_reconstruction_clips(capture_set)
    if not paths:
        raise ValueError("reconstruction train request contains no clips")
    train_paths, val_paths = split_reconstruction_corpus(paths)
    loaded, _rate, channels = _load_reconstruction_clips(paths)
    input_channels = max(1, int(options.get("host_input_channels", channels)))
    loaded = _match_clip_channels(loaded, input_channels)
    train_clips = [loaded[path] for path in train_paths if path in loaded]
    val_clips = [loaded[path] for path in val_paths if path in loaded]
    if not train_clips:
        raise ValueError("reconstruction train split is empty")
    module = build_rave_graph_module(request["graph_fragment"], input_channels, 1)
    if module.bottleneck_id is None:
        raise ValueError("reconstruction requires a variational bottleneck")
    optimizer = torch.optim.Adam(module.parameters(), 1e-3)
    discriminator = CombineDiscriminator(input_channels)
    disc_opt = torch.optim.Adam(discriminator.parameters(), 1e-4)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = (artifact_dir / f"{request_id}.pt").resolve()
    checkpoint_path = (artifact_dir / f"{request_id}.ckpt.pt").resolve()
    export_checkpoints = bool(options.get("export_checkpoints", False))
    checkpoint_interval = max(0, int(options.get("checkpoint_interval", 50)))
    kl_warmup = max(1, int(reconstruction.get("kl_warmup_steps", 20000)))
    total_steps = stage1_steps + stage2_steps
    last_loss = 0.0
    step = 0
    paused = False
    compactness_ready = False
    validation_segments = len(val_clips)
    while step < total_steps:
        command = _read_command(command_file)
        if command == "stop":
            _emit({"request_id": request_id, "status": "stopped", "step": step, "message": "Stopped by user"})
            return {"request_id": request_id, "status": "stopped", "step": step, "message": "Stopped by user"}
        if command == "pause":
            paused = True
        elif command == "resume":
            paused = False
        stage = "representation" if step < stage1_steps else "quality"
        compactness = _compactness_payload(compactness_ready, validation_segments)
        if paused:
            _emit(
                {
                    "request_id": request_id,
                    "status": "paused",
                    "step": step,
                    "total_steps": total_steps,
                    "stage": stage,
                    "objective": "reconstruction",
                    "loss": last_loss,
                    "compactness": compactness,
                }
            )
            time.sleep(0.05)
            continue
        length = max(64, int(options.get("segment_length", 8192)))
        if stage == "representation":
            module.train()
            audio = _sample_segment(train_clips, length)
        else:
            audio = _sample_segment(train_clips, length)
        reconstructed = module(audio)
        if reconstructed.shape[-1] != audio.shape[-1]:
            reconstructed = reconstructed[..., -audio.shape[-1] :]
            audio = audio[..., -reconstructed.shape[-1] :]
        spec = spectral_distance(reconstructed, audio)
        if stage == "representation":
            kl_beta = 1e-6 + (5e-2 - 1e-6) * min(1.0, step / float(kl_warmup))
            optimizer.zero_grad()
            (spec + kl_beta * _bottleneck_kl(module)).backward()
            optimizer.step()
            if step == stage1_steps - 1:
                _freeze_rave_encoder(module)
                stacked = _collect_validation_mu(module, val_clips, length)
                latent_size = 1
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
                    compactness_ready = True
                    validation_segments = int(stacked.shape[0])
                else:
                    module.compactness_ready = False
                    compactness_ready = False
                module.train()
        else:
            real_score, _real_feats = discriminator.features(audio)
            fake_score, _ = discriminator.features(reconstructed.detach())
            disc_loss = torch.relu(1.0 - real_score).mean() + torch.relu(1.0 + fake_score).mean()
            disc_opt.zero_grad()
            disc_loss.backward()
            disc_opt.step()
            gen_score, fake_feats = discriminator.features(reconstructed)
            adv = torch.relu(1.0 - gen_score).mean()
            match = reconstructed.new_zeros(())
            _, real_feats_live = discriminator.features(audio.detach())
            for fake_feat, real_feat in zip(fake_feats, real_feats_live):
                match = match + (fake_feat - real_feat.detach()).abs().mean()
            optimizer.zero_grad()
            (spec + adv + match).backward()
            optimizer.step()
        last_loss = float(spec.item())
        step += 1
        event: dict[str, Any] = {
            "request_id": request_id,
            "status": "running",
            "step": step,
            "total_steps": total_steps,
            "stage": stage,
            "objective": "reconstruction",
            "loss": last_loss,
            "best_loss": last_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "compactness": _compactness_payload(compactness_ready, validation_segments),
        }
        if export_checkpoints and checkpoint_interval and step % checkpoint_interval == 0:
            _export_rave_scripted(module, input_channels, checkpoint_path)
            if stage == "representation":
                module.train()
            event["artifact_path"] = str(checkpoint_path)
        _emit(event)
    _export_rave_scripted(module, input_channels, artifact_path)
    return {
        "request_id": request_id,
        "status": "success",
        "step": step,
        "total_steps": total_steps,
        "loss": last_loss,
        "best_loss": last_loss,
        "artifact_path": str(artifact_path),
        "objective": "reconstruction",
        "has_encode_decode": True,
        "compactness": _compactness_payload(compactness_ready, validation_segments),
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
            "compactness": _compactness_payload(compactness_ready, validation_segments),
        },
    }



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

    The example control tensor matches the audio time length so freeze does not
    specialise FiLM to a single global value per call.
    """
    module.eval()
    export_module = copy.deepcopy(module)
    strip_weight_norm(export_module)
    example_samples = 256
    example_x = torch.zeros(1, input_channels, example_samples)
    example_c = torch.zeros(1, max(1, cond_dim), example_samples)
    with torch.inference_mode():
        scripted = torch.jit.trace(export_module, (example_x, example_c), strict=False)
        scripted = torch.jit.freeze(scripted)
        output = scripted(example_x, example_c)
        if output.dim() != 3:
            raise ValueError("trained artifact returned an invalid tensor rank")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.jit.save(scripted, str(temporary_path))
    temporary_path.replace(path)


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
    loss = options.get("loss", {})
    if not isinstance(loss, dict):
        loss = {}
    config: dict[str, Any] = {
        "request_id": str(request.get("request_id", "")),
        "optimizer": str(options.get("optimizer", "adam")),
        "total_steps": int(options.get("total_steps", DEFAULT_STEPS)),
        "learning_rate": float(options.get("learning_rate", DEFAULT_LR)),
        "segment_length": int(options.get("segment_length", DEFAULT_SEGMENT_LENGTH)),
        "checkpoint_interval": int(options.get("checkpoint_interval", 50)),
        "export_checkpoints": bool(options.get("export_checkpoints", False)),
        "host_input_channels": int(options.get("host_input_channels", 0)),
        "cond_dim": int(options.get("cond_dim", 2)),
        "rf_aware_crop": bool(options.get("rf_aware_crop", True)),
        "steer_conditioning": float(options.get("steer_conditioning", STEER_CONDITIONING)),
        "loss": str(loss.get("type", "multiresolution_stft")),
        "element_count": len(elements),
        "element_types": [
            str(element.get("type", ""))
            for element in elements
            if isinstance(element, dict)
        ],
        "pair_count": len(pairs),
    }
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


def train_request(request: dict[str, Any], artifact_dir: Path, command_file: Path | None) -> dict[str, Any]:
    """Run the fixed steerable NAfx recipe and export TorchScript on success."""
    request_id = str(request.get("request_id", ""))
    if request.get("operation") != "train_steerable" or not request_id:
        raise ValueError("invalid train request envelope")

    options = request.get("train_options", {})
    if not isinstance(options, dict):
        options = {}
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

    input_channels = int(options.get("host_input_channels", x_batch.shape[1]))
    input_channels = max(1, input_channels)
    x_batch = _match_channels(x_batch, input_channels)
    y_batch = _match_channels(y_batch, input_channels)
    module = build_module(request["graph_fragment"], input_channels, cond_dim)
    receptive_field = _module_receptive_field(module)
    if auraloss is None:
        raise RuntimeError("auraloss is required for the multiresolution STFT loss")
    loss_fn = auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=list(options.get("loss", {}).get("fft_sizes", FFT_SIZES)),
        win_lengths=list(options.get("loss", {}).get("win_lengths", WIN_LENGTHS)),
        hop_sizes=list(options.get("loss", {}).get("hop_sizes", HOP_SIZES)),
    )
    optimizer = torch.optim.Adam(module.parameters(), learning_rate)
    milestones = [int(total_steps * 0.80), int(total_steps * 0.95)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
    cond = torch.full((1, cond_dim, 1), float(STEER_CONDITIONING), dtype=torch.float32)

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
                _emit(
                    {
                        "request_id": request_id,
                        "status": "stopped",
                        "step": step,
                        "message": "Stopped by user",
                    }
                )
                return {
                    "request_id": request_id,
                    "status": "stopped",
                    "step": step,
                    "message": "Stopped by user",
                }
            if command == "pause":
                paused = True
            elif command == "resume":
                paused = False
            if paused:
                _emit(
                    {
                        "request_id": request_id,
                        "status": "paused",
                        "step": step,
                        "total_steps": total_steps,
                        "loss": last_loss,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }
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
                    input_channels,
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
            _emit(event)

        module.load_state_dict(best_state)
        _export_scripted(module, input_channels, cond_dim, artifact_path)
        tracker.save(str(artifact_path))
        example_x = torch.zeros(1, input_channels, 256)
        example_c = torch.zeros(1, cond_dim, 1)
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
            "blackbox_metadata": _blackbox_metadata(
                input_channels,
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
