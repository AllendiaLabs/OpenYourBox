#!/usr/bin/env python3
"""Compile an OpenYourBox selected graph into a local TorchScript artifact."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import struct
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as functional

_UINT64_MASK = (1 << 64) - 1
_RAVE_TYPES = {
    "pqmf_analysis",
    "pqmf_synthesis",
    "rate_conv",
    "variational_bottleneck",
    "noise_synthesizer",
}


def _load_train_worker():
    """Load the sibling train worker so freeze can reuse the RAVE graph builder."""
    path = Path(__file__).with_name("train_worker.py")
    spec = importlib.util.spec_from_file_location("openyourbox_train_worker", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not locate train_worker.py beside freeze_worker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _float32(value: float) -> float:
    """Round one Python float exactly to IEEE-754 binary32."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _split_mix_64(state: int) -> tuple[int, int]:
    """Advance the C++ SplitMix64 sequence and return state plus output."""
    state = (state + 0x9E3779B97F4A7C15) & _UINT64_MASK
    value = state
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return state, value ^ (value >> 31)


def _make_weight(
    output_channels: int,
    input_channels: int,
    kernel_size: int,
    state: int,
) -> tuple[torch.Tensor, int]:
    """Reproduce one deterministic LiveGraphEngine weight tensor."""
    scale = _float32(math.sqrt(6.0 / max(1, input_channels * kernel_size)))
    values: list[float] = []
    for _ in range(output_channels * input_channels * kernel_size):
        state, bits = _split_mix_64(state)
        unit = float(bits >> 11) / float(1 << 53)
        signed = _float32(unit * 2.0 - 1.0)
        values.append(_float32(signed * scale))
    return (
        torch.tensor(values, dtype=torch.float32).reshape(
            output_channels, input_channels, kernel_size
        ),
        state,
    )


def _make_linear_weight(
    output_features: int, input_features: int, state: int
) -> tuple[torch.Tensor, int]:
    """Reproduce one deterministic LiveGraphEngine Linear weight tensor."""
    scale = _float32(math.sqrt(6.0 / max(1, input_features)))
    values: list[float] = []
    for _ in range(output_features * input_features):
        state, bits = _split_mix_64(state)
        unit = float(bits >> 11) / float(1 << 53)
        signed = _float32(unit * 2.0 - 1.0)
        values.append(_float32(signed * scale))
    return (
        torch.tensor(values, dtype=torch.float32).reshape(
            output_features, input_features
        ),
        state,
    )


def _make_bias(features: int, state: int) -> tuple[torch.Tensor, int]:
    """Reproduce one deterministic LiveGraphEngine Linear bias tensor."""
    scale = _float32(math.sqrt(6.0 / max(1, features)))
    values: list[float] = []
    for _ in range(features):
        state, bits = _split_mix_64(state)
        unit = float(bits >> 11) / float(1 << 53)
        signed = _float32(unit * 2.0 - 1.0)
        values.append(_float32(signed * scale))
    return torch.tensor(values, dtype=torch.float32), state


def _assign_deterministic_weights(module: nn.Module, seed: int) -> None:
    """Copy the live C++ engine's seeded weights into one weighted element."""
    train_worker = _load_train_worker()
    state = (seed & 0xFFFF_FFFF) ^ 0xA0761D6478BD642F
    with torch.no_grad():
        for layer in module.modules():
            if not isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
                continue
            # Skip modules that only expose a computed weight_norm view without
            # owning the underlying Conv parameters (parent wrappers).
            if not any(
                name == "weight" or name.startswith("weight_")
                for name, _ in layer.named_parameters(recurse=False)
            ):
                continue
            weight, state = _make_weight(
                layer.out_channels,
                layer.in_channels // getattr(layer, "groups", 1),
                layer.kernel_size[0],
                state,
            )
            train_worker.assign_plain_conv_weight(layer, weight)


def _assign_tcn_weights(module: "SteerableTCN", seed: int) -> None:
    """Copy live TCN conv then FiLM weights in LiveGraphEngine seed order."""
    state = (seed & 0xFFFF_FFFF) ^ 0xA0761D6478BD642F
    with torch.no_grad():
        projection, state = _make_weight(
            module.input_projection.out_channels,
            module.input_projection.in_channels,
            1,
            state,
        )
        module.input_projection.weight.copy_(projection)
        for block in module.blocks:
            conv = block.convolution.convolution
            weight, state = _make_weight(
                conv.out_channels, conv.in_channels, conv.kernel_size[0], state
            )
            conv.weight.copy_(weight)
        output, state = _make_weight(
            module.output_projection.out_channels,
            module.output_projection.in_channels,
            1,
            state,
        )
        module.output_projection.weight.copy_(output)
        if module.cond_dim < 1:
            return
        for block in module.blocks:
            if block.film is None:
                continue
            film_out = block.film.adaptor.out_features
            weight, state = _make_linear_weight(film_out, module.cond_dim, state)
            bias, state = _make_bias(film_out, state)
            block.film.adaptor.weight.copy_(weight)
            block.film.adaptor.bias.copy_(bias)


class CausalConv1d(nn.Module):
    """Apply a left-padded causal one-dimensional convolution."""

    def __init__(
        self, input_channels: int, output_channels: int, kernel_size: int, dilation: int
    ) -> None:
        """Create a causal convolution with validated dimensions."""
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.convolution = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size,
            dilation=dilation,
            bias=False,
        )

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        """Process ``[batch, channels, time]`` samples without future context."""
        return self.convolution(functional.pad(samples, (self.left_padding, 0)))


class ChannelLinear(nn.Module):
    """Apply a linear projection independently at every audio sample."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        """Create a per-sample channel projection."""
        super().__init__()
        self.projection = nn.Conv1d(input_channels, output_channels, 1, bias=False)

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        """Project the channel axis of an audio tensor."""
        return self.projection(samples)


class ZeroPreservingSigmoid(nn.Module):
    """Apply sigmoid while retaining the audio engine's exact-zero contract."""

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        """Map nonzero values through sigmoid and keep zeros exactly zero."""
        return torch.where(
            samples == 0.0, torch.zeros_like(samples), torch.sigmoid(samples)
        )


def dilation_for_layer(growth: int, layer: int) -> int:
    """Return saturated ``growth**layer`` matching the live TCN."""
    value = 1
    growth = max(1, int(growth))
    for _ in range(max(0, layer)):
        value *= growth
        if value > 2**30:
            return 2**30
    return value


class FiLM(nn.Module):
    """Per-sample feature-wise linear modulation from a conditioning trajectory."""

    def __init__(self, cond_dim: int, num_features: int) -> None:
        """Create a linear adaptor mapping ``c`` to per-channel scale and shift."""
        super().__init__()
        self.cond_dim = max(1, int(cond_dim))
        self.adaptor = nn.Linear(self.cond_dim, num_features * 2)

    def forward(self, samples: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply FiLM: ``y_t = g(c_t) * x_t + b(c_t)``."""
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
    """One causal TCN block with optional FiLM and identity residual."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        activation: int,
        residual: bool,
        cond_dim: int,
    ) -> None:
        """Create one temporal block matching the live TCN element."""
        super().__init__()
        self.convolution = CausalConv1d(channels, channels, kernel_size, dilation)
        self.film = FiLM(cond_dim, channels) if cond_dim > 0 else None
        self.activation = _activation(activation)
        self.residual = residual

    def forward(self, samples: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Run convolution, optional FiLM, activation, and residual add."""
        residual = samples
        value = self.convolution(samples)
        if self.film is not None:
            value = self.film(value, cond)
        value = self.activation(value)
        if self.residual:
            value = value + residual[..., -value.shape[-1] :]
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
                )
                for layer in range(depth)
            ]
        )
        self.output_projection = nn.Conv1d(hidden_channels, output_channels, 1, bias=False)
        self.cond_dim = max(0, int(cond_dim))

    def forward(self, samples: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Process audio, applying ca=0 when conditioning is omitted."""
        if cond is None:
            cond = torch.zeros(
                samples.shape[0],
                max(1, self.cond_dim),
                1,
                device=samples.device,
                dtype=samples.dtype,
            )
        value = self.input_projection(samples)
        for block in self.blocks:
            value = block(value, cond)
        return self.output_projection(value)


class ConditionedSequential(nn.Module):
    """Sequential graph that forwards live control into TCN blocks."""

    def __init__(self, modules: list[nn.Module], cond_dim: int) -> None:
        """Store ordered modules and the conditioning width."""
        super().__init__()
        self.layers = nn.ModuleList(modules)
        self.cond_dim = max(1, int(cond_dim))

    def forward(self, samples: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Run each layer, passing ``cond`` into steerable TCN modules."""
        if cond is None:
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


def _activation(index: int) -> nn.Module:
    """Return the activation represented by an OpenYourBox enum value."""
    activations: tuple[nn.Module, ...] = (
        nn.ReLU(),
        ZeroPreservingSigmoid(),
        nn.Tanh(),
        nn.LeakyReLU(0.01),
        nn.PReLU(),
    )
    if index < 0 or index >= len(activations):
        raise ValueError(f"unsupported activation index {index}")
    return activations[index]


def _properties(element: dict[str, Any]) -> dict[str, int]:
    """Convert an element's ordered property array to a lookup dictionary."""
    return {
        str(item["key"]): int(item["value"])
        for item in element.get("properties", [])
    }


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
    if any(len(destinations) > 1 for destinations in outgoing.values()):
        types = {str(element.get("type", "")) for element in elements}
        if not (types & _RAVE_TYPES):
            raise ValueError("branched selected graphs are not supported by this worker")
    return ordered


def build_module(
    fragment: dict[str, Any], input_channels: int = 1, cond_dim: int = 0
) -> nn.Module:
    """Construct a module for a fragment with the specified input channels.

    When ``cond_dim`` is positive, TCN elements keep a live Control pin: the
    returned module is ``(audio, cond)`` so Knob/XY still steer after freeze.
    """
    types = {
        str(element.get("type", ""))
        for element in fragment.get("elements", [])
        if isinstance(element, dict)
    }
    if types & _RAVE_TYPES:
        return _load_train_worker().build_rave_graph_module(
            fragment, input_channels, cond_dim
        )

    modules: list[nn.Module] = []
    channels = input_channels
    cond_dim = max(0, int(cond_dim))
    has_conditioned_tcn = False
    for element in _topological_elements(fragment):
        element_type = str(element["type"])
        properties = _properties(element)
        seed = int(element.get("seed", 42))

        if element_type in {"audio_input", "audio_output", "knob_input", "xy_trackpad"}:
            continue
        if element_type == "activation":
            modules.append(_activation(properties.get("activation", 0)))
        elif element_type == "linear":
            output_channels = properties.get("features", channels)
            module = ChannelLinear(channels, output_channels)
            _assign_deterministic_weights(module, seed)
            modules.append(module)
            channels = output_channels
        elif element_type in {"conv1d", "rate_conv"}:
            output_channels = properties.get("channels", channels)
            module = _load_train_worker()._make_strided_conv(
                channels,
                output_channels,
                properties,
            )
            _assign_deterministic_weights(module, seed)
            modules.append(module)
            channels = output_channels
        elif element_type == "conv_transpose1d":
            output_channels = properties.get("channels", channels)
            module = _load_train_worker()._make_conv_transpose(
                channels,
                output_channels,
                properties,
            )
            _assign_deterministic_weights(module, seed)
            modules.append(module)
            channels = output_channels
        elif element_type == "batch_norm":
            modules.append(nn.BatchNorm1d(channels))
        elif element_type == "tcn":
            hidden = properties.get("channels", channels)
            depth = properties.get("depth", 1)
            kernel_size = properties.get("kernel_size", 3)
            growth = properties.get("dilation_growth", 2)
            activation = properties.get("activation", 0)
            residual = bool(properties.get("residual", 0))
            module = SteerableTCN(
                channels,
                hidden,
                channels,
                depth,
                kernel_size,
                growth,
                activation,
                residual,
                cond_dim,
            )
            _assign_tcn_weights(module, seed)
            modules.append(module)
            if cond_dim > 0:
                has_conditioned_tcn = True
        elif element_type in {"utility", "merge", "sum", "multiply"}:
            raise ValueError(
                "mixer elements cannot be frozen by the linear freeze worker"
            )
        elif element_type in _RAVE_TYPES:
            return _load_train_worker().build_rave_graph_module(
                fragment, input_channels, cond_dim
            )

    if not modules:
        return nn.Identity()
    if has_conditioned_tcn:
        return ConditionedSequential(modules, cond_dim)
    return nn.Sequential(*modules)


def _receptive_field(module: nn.Module) -> int:
    """Return the aggregate causal receptive field of a sequential graph."""
    field = 1
    for layer in module.modules():
        padding = getattr(layer, "left_padding", None)
        if padding is not None:
            field += int(padding)
    return field


def compile_request(request: dict[str, Any], artifact_dir: Path) -> dict[str, Any]:
    """Compile one validated manual-freeze request and return its response."""
    request_id = str(request.get("request_id", ""))
    if request.get("operation") != "freeze_selection" or not request_id:
        raise ValueError("invalid freeze request envelope")

    options = request.get("compile_options", {})
    input_channels = int(options.get("host_input_channels", 2))
    output_channels = int(options.get("host_output_channels", input_channels))
    example_samples = int(options.get("example_samples", 256))
    if not 1 <= input_channels <= 1024 or not 1 <= output_channels <= 1024:
        raise ValueError("freeze artifact channel counts must be between 1 and 1024")
    if example_samples < 1 or example_samples > 1 << 20:
        raise ValueError("invalid freeze example block size")

    started = time.perf_counter()
    cond_dim = max(0, int(options.get("cond_dim", 0)))
    if not bool(options.get("conditioning", cond_dim > 0)):
        cond_dim = 0
    module = build_module(request["graph_fragment"], input_channels, cond_dim).eval()
    receptive_field = _receptive_field(module)
    example = torch.zeros(1, input_channels, example_samples)
    train_worker = None
    rave_type = False
    try:
        train_worker = _load_train_worker()
        rave_type = isinstance(module, train_worker.RaveGraphModule)
    except Exception:
        rave_type = False
    if train_worker is not None:
        train_worker.strip_weight_norm(module)
    conditioned = isinstance(module, ConditionedSequential)
    has_encode_decode = bool(rave_type and getattr(module, "bottleneck_id", None) is not None)
    with torch.inference_mode():
        if has_encode_decode and train_worker is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = (artifact_dir / f"{request_id}.pt").resolve()
            train_worker._export_rave_scripted(module, input_channels, artifact_path)
            # VariationalBottleneckLayer lives in train_worker (acids-rave v1
            # grouped head, softplus variance, eval μ-only).
            scripted = torch.jit.load(str(artifact_path))
            output = scripted(example)
        elif conditioned:
            example_c = torch.zeros(1, max(1, cond_dim), example_samples)
            scripted = torch.jit.trace(module, (example, example_c), strict=False)
            scripted = torch.jit.freeze(scripted)
            output = scripted(example, example_c)
        else:
            scripted = torch.jit.trace(module, example, strict=True)
            scripted = torch.jit.freeze(scripted)
            output = scripted(example)
        if output.dim() != 3 or output.size(1) != output_channels:
            raise ValueError(
                "compiled artifact output channels do not match the host configuration"
            )
        for _ in range(2):
            output = scripted(example, example_c) if conditioned else scripted(example)
        latency_started = time.perf_counter()
        for _ in range(8):
            if conditioned:
                scripted(example, example_c)
            else:
                scripted(example)
        latency_ms = (time.perf_counter() - latency_started) * 1000.0 / 8.0

    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = (artifact_dir / f"{request_id}.pt").resolve()
    if not has_encode_decode:
        temporary_path = artifact_path.with_suffix(".pt.tmp")
        torch.jit.save(scripted, str(temporary_path))
        temporary_path.replace(artifact_path)
    compile_ms = (time.perf_counter() - started) * 1000.0
    methods = ["forward", "encode", "decode"] if has_encode_decode else ["forward"]
    return {
        "request_id": request_id,
        "status": "success",
        "artifact_path": str(artifact_path),
        "blackbox_metadata": {
            "display_name": "Frozen Selection",
            "ports": [],
            "shape_signature": {
                "input_channels": input_channels,
                "output_channels": int(output.size(1)),
            },
            "receptive_field_samples": receptive_field,
            "conditioning": conditioned,
            "cond_dim": int(cond_dim) if conditioned else 0,
            "has_encode_decode": has_encode_decode,
            "methods": methods,
            "baseline_metrics": {
                "compile_time_ms": compile_ms,
                "estimated_latency_ms": latency_ms,
            },
        },
    }


def main() -> int:
    """Run the command-line worker and emit exactly one JSON response."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    arguments = parser.parse_args()

    request_id = ""
    try:
        request = json.loads(arguments.request.read_text(encoding="utf-8"))
        request_id = str(request.get("request_id", ""))
        response = compile_request(request, arguments.artifact_dir)
        print(json.dumps(response), flush=True)
        return 0
    except Exception as error:  # Worker boundary must return user-facing errors.
        print(
            json.dumps(
                {
                    "request_id": request_id,
                    "status": "failure",
                    "error_message": str(error),
                }
            ),
            flush=True,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
