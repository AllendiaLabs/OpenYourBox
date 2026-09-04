"""Pytest fixtures for the cloud control plane (no fake-success worker)."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from CloudService.api.state import STORE


def tiny_wav_bytes(samples: int = 2048) -> bytes:
    """Return a mono 44.1 kHz float32 WAV large enough for short recipes."""
    values = [0.05] * int(samples)
    payload = struct.pack("<" + "f" * len(values), *values)
    fmt = struct.pack("<HHIIHH", 3, 1, 44100, 44100 * 4, 4, 32)
    riff_size = 4 + (8 + len(fmt)) + (8 + len(payload))
    return (
        b"RIFF"
        + struct.pack("<I", riff_size)
        + b"WAVEfmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", len(payload))
        + payload
    )


def mapping_graph_fragment() -> dict:
    """Return a tiny TCN fragment accepted by ``train_worker.build_module``."""
    return {
        "elements": [
            {
                "id": 1,
                "type": "tcn",
                "properties": [
                    {"key": "channels", "value": 4},
                    {"key": "depth", "value": 2},
                    {"key": "kernel_size", "value": 3},
                    {"key": "dilation_growth", "value": 2},
                    {"key": "residual", "value": 1},
                    {"key": "activation", "value": 4},
                ],
            }
        ],
        "connections": [],
    }


def reconstruction_graph_fragment() -> dict:
    """Return a tiny encode/decode fragment for reconstruction smoke tests."""
    return {
        "elements": [
            {
                "id": 1,
                "type": "rate_conv",
                "properties": [
                    {"key": "channels", "value": 4},
                    {"key": "kernel_size", "value": 3},
                    {"key": "stride", "value": 2},
                    {"key": "dilation", "value": 1},
                ],
            },
            {
                "id": 2,
                "type": "variational_bottleneck",
                "properties": [
                    {"key": "latent_size", "value": 4},
                    {"key": "kernel_size", "value": 5},
                ],
            },
            {
                "id": 3,
                "type": "conv_transpose1d",
                "properties": [
                    {"key": "channels", "value": 1},
                    {"key": "kernel_size", "value": 3},
                    {"key": "stride", "value": 2},
                    {"key": "dilation", "value": 1},
                ],
            },
        ],
        "connections": [
            {"source_element_id": 1, "destination_element_id": 2},
            {"source_element_id": 2, "destination_element_id": 3},
        ],
    }


def mapping_manifest(
    total_steps: int = 8,
    checkpoint_interval: int = 4,
    export_checkpoints: bool = False,
) -> dict:
    """Return a mapping ``train_steerable`` manifest for tests."""
    return {
        "schema_version": 1,
        "operation": "train_steerable",
        "client": {"plugin_version": "test", "client_instance_id": "inst-a"},
        "graph_fragment": mapping_graph_fragment(),
        "train_options": {
            "objective": "mapping",
            "total_steps": total_steps,
            "checkpoint_interval": checkpoint_interval,
            "export_checkpoints": export_checkpoints,
            "segment_length": 256,
            "host_input_channels": 1,
            "device": "cpu",
            "mlflow": {"enabled": False},
            "loss": {
                "fft_sizes": [32, 64],
                "win_lengths": [32, 64],
                "hop_sizes": [8, 16],
            },
        },
        "capture_set": {
            "pairs": [
                {"pair_id": "p1", "x_name": "x.wav", "y_name": "y.wav", "kind": "pair"}
            ]
        },
    }


def reconstruction_manifest() -> dict:
    """Return a reconstruction ``train_steerable`` manifest for tests."""
    return {
        "schema_version": 1,
        "operation": "train_steerable",
        "client": {"plugin_version": "test", "client_instance_id": "inst-a"},
        "graph_fragment": reconstruction_graph_fragment(),
        "train_options": {
            "objective": "reconstruction",
            "host_input_channels": 1,
            "segment_length": 128,
            "device": "cpu",
            "export_checkpoints": False,
            "checkpoint_interval": 0,
            "mlflow": {"enabled": False},
            "reconstruction": {
                "stage1_steps": 1,
                "stage2_steps": 1,
                "kl_warmup_steps": 1,
                "batch_size": 1,
                "phase_mangle_prob": 0.0,
                "dequantize_bits": 0,
                "disc_n_scales": 1,
                "disc_capacity": 8,
                "disc_n_layers": 1,
            },
        },
        "capture_set": {
            "pairs": [
                {"pair_id": "p1", "x_name": "x.wav", "y_name": "y.wav", "kind": "pair"}
            ],
            "clips": [],
        },
    }


def wav_files(manifest: dict) -> dict:
    """Build a multipart ``files`` dict with tiny valid WAVs for ``manifest``."""
    wav = tiny_wav_bytes()
    files = {
        "manifest": (None, json.dumps(manifest), "application/json"),
        "file:x.wav": ("x.wav", wav, "application/octet-stream"),
        "file:y.wav": ("y.wav", wav, "application/octet-stream"),
    }
    return files


def link_headers(client: TestClient, customer_id: str = "cust-1") -> dict[str, str]:
    """Complete the staging device-code link and return Bearer headers."""
    start = client.post("/v1/auth/link/start")
    body = start.json()
    client.post(
        "/mock/link/complete",
        json={"user_code": body["user_code"], "customer_id": customer_id},
    )
    token = client.post("/v1/auth/link/token", json={"device_code": body["device_code"]})
    return {"Authorization": f"Bearer {token.json()['access_token']}"}


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Unauthenticated TestClient with the real worker disabled (gates/Stop)."""
    monkeypatch.setenv("CLOUD_AUTO_WORKER", "0")
    monkeypatch.setenv("CLOUD_ALLOW_ANONYMOUS", "0")
    monkeypatch.setenv("CLOUD_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CLOUD_API_PUBLIC_URL", "http://testserver")
    STORE.reset()
    from CloudService.api.app import create_app

    with TestClient(create_app()) as test_client:
        yield test_client
    STORE.reset()


@pytest.fixture
def live_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """TestClient that claims queued jobs and runs real recipes."""
    monkeypatch.setenv("CLOUD_AUTO_WORKER", "1")
    monkeypatch.setenv("CLOUD_ALLOW_ANONYMOUS", "0")
    monkeypatch.setenv("CLOUD_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CLOUD_API_PUBLIC_URL", "http://testserver")
    monkeypatch.setenv("CLOUD_HEARTBEAT_TIMEOUT_SECONDS", "120")
    STORE.reset()
    from CloudService.api.app import create_app

    with TestClient(create_app()) as test_client:
        yield test_client
    STORE.reset()
