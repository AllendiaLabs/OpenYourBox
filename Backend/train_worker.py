#!/usr/bin/env python3
"""Train an OpenYourBox armed graph with the steerable NAfx recipe."""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
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
    ) -> None:
        """Create one temporal block."""
        super().__init__()
        self.convolution = CausalConv1d(channels, channels, kernel_size, dilation)
        self.film = FiLM(cond_dim, channels) if cond_dim > 0 else None
        self.activation = _activation(activation, channels)
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


def _activation(index: int, channels: int = 1) -> nn.Module:
    """Return the activation represented by an OpenYourBox enum value."""
    activations: tuple[nn.Module, ...] = (
        nn.ReLU(),
        ZeroPreservingSigmoid(),
        nn.Tanh(),
        nn.LeakyReLU(0.01),
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
    modules: list[nn.Module] = []
    channels = input_channels
    cond_dim = max(1, int(cond_dim))
    for element in _topological_elements(fragment):
        element_type = str(element["type"])
        properties = _properties(element)
        if element_type in {"audio_input", "audio_output", "knob_input", "xy_trackpad"}:
            continue
        if element_type == "activation":
            modules.append(_activation(int(properties.get("activation", 0)), channels))
        elif element_type == "linear":
            output_channels = int(properties.get("features", channels))
            modules.append(nn.Conv1d(channels, output_channels, 1, bias=False))
            channels = output_channels
        elif element_type == "conv1d":
            output_channels = int(properties.get("channels", channels))
            modules.append(
                CausalConv1d(
                    channels,
                    output_channels,
                    int(properties.get("kernel_size", 3)),
                    int(properties.get("dilation", 1)),
                )
            )
            channels = output_channels
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
                )
            )
        elif element_type in {"merge", "sum", "multiply"}:
            raise ValueError("mixer elements cannot be trained by this worker")

    return ConditionedSequential(modules, cond_dim) if modules else nn.Identity()


def _module_receptive_field(module: nn.Module) -> int:
    """Return the aggregate causal receptive field of constructed layers."""
    field = 1
    for layer in module.modules():
        if isinstance(layer, CausalConv1d):
            field += layer.left_padding
    return field


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
    example_samples = 256
    example_x = torch.zeros(1, input_channels, example_samples)
    example_c = torch.zeros(1, max(1, cond_dim), example_samples)
    with torch.inference_mode():
        scripted = torch.jit.trace(module, (example_x, example_c), strict=False)
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
