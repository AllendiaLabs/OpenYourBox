"""Focused tests for the detached steerable train worker recipe."""

from __future__ import annotations

import importlib.util
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path

import torch

WORKER_PATH = Path(__file__).parents[1] / "Backend" / "train_worker.py"
SPEC = importlib.util.spec_from_file_location("train_worker", WORKER_PATH)
assert SPEC is not None and SPEC.loader is not None
train_worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train_worker)


class TrainWorkerRecipeTests(unittest.TestCase):
    """Validate recipe constants and RF-aware cropping helpers."""

    def test_recipe_constants_match_steerable_nafx(self) -> None:
        """FFT, hop, step, and segment defaults must match the specification."""
        self.assertEqual(train_worker.FFT_SIZES, [32, 128, 512, 2048])
        self.assertEqual(train_worker.WIN_LENGTHS, [32, 128, 512, 2048])
        self.assertEqual(train_worker.HOP_SIZES, [16, 64, 256, 1024])
        self.assertEqual(train_worker.DEFAULT_STEPS, 2500)
        self.assertEqual(train_worker.DEFAULT_SEGMENT_LENGTH, 228308)
        self.assertEqual(train_worker.STEER_CONDITIONING, 0.0)
        self.assertEqual(train_worker.DEFAULT_TRAIN_DEVICE, "auto")
        self.assertEqual(
            train_worker.TRAIN_DEVICE_CHOICES, ("auto", "cpu", "mps", "cuda")
        )

    def test_available_train_devices_includes_cpu(self) -> None:
        """CPU is always a valid training backend."""
        devices = train_worker.available_train_devices()
        self.assertIn("cpu", devices)
        self.assertEqual(devices[0], "cpu")

    def test_resolve_train_device_cpu(self) -> None:
        """An explicit CPU request must stay on CPU."""
        device, requested, effective = train_worker.resolve_train_device("cpu")
        self.assertEqual(requested, "cpu")
        self.assertEqual(effective, "cpu")
        self.assertEqual(device.type, "cpu")
        fields = train_worker.train_device_event_fields(device, requested)
        self.assertFalse(fields["device_fallback"])
        self.assertEqual(fields["device"], "cpu")

    def test_resolve_train_device_unknown_uses_auto(self) -> None:
        """Unknown tokens fall back to auto resolution."""
        device, requested, effective = train_worker.resolve_train_device("tpu")
        self.assertEqual(requested, "auto")
        self.assertIn(effective, train_worker.available_train_devices())
        self.assertEqual(device.type, effective)

    def test_resolve_train_device_cuda_falls_back_when_missing(self) -> None:
        """CUDA requests must land on CPU when the backend is absent."""
        if torch.cuda.is_available():
            self.skipTest("cuda is available")
        device, requested, effective = train_worker.resolve_train_device("cuda")
        self.assertEqual(requested, "cuda")
        self.assertEqual(effective, "cpu")
        self.assertEqual(device.type, "cpu")
        fields = train_worker.train_device_event_fields(device, requested)
        self.assertTrue(fields["device_fallback"])

    def test_resolve_train_device_mps_stays_when_available(self) -> None:
        """An explicit MPS request must keep MPS when the backend trains."""
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            self.skipTest("mps is unavailable")
        device, requested, effective = train_worker.resolve_train_device("mps")
        self.assertEqual(requested, "mps")
        self.assertIn(effective, {"mps", "cpu"})
        self.assertEqual(device.type, effective)
        fields = train_worker.train_device_event_fields(device, requested)
        if effective == "mps":
            self.assertFalse(fields["device_fallback"])
        else:
            self.assertTrue(fields["device_fallback"])

    def test_resolve_train_device_mps_falls_back_when_missing(self) -> None:
        """MPS requests must land on CPU when the backend is absent."""
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            self.skipTest("mps is available")
        device, requested, effective = train_worker.resolve_train_device("mps")
        self.assertEqual(requested, "mps")
        self.assertEqual(effective, "cpu")
        fields = train_worker.train_device_event_fields(device, requested)
        self.assertTrue(fields["device_fallback"])

    def test_dilation_growth_schedule(self) -> None:
        """Layer dilations follow G^n."""
        self.assertEqual(train_worker.dilation_for_layer(2, 0), 1)
        self.assertEqual(train_worker.dilation_for_layer(2, 3), 8)
        self.assertEqual(train_worker.dilation_for_layer(8, 2), 64)
        self.assertEqual(train_worker.dilation_for_layer(10, 1), 10)

    def test_rf_aware_crop_includes_context(self) -> None:
        """Each crop must prepend receptive-field context before the target."""
        clean = torch.arange(1000, dtype=torch.float32).view(1, 1, -1)
        processed = clean.clone()
        x_crop, y_crop = train_worker.rf_aware_crop(clean, processed, 16, 32)
        self.assertEqual(y_crop.shape[-1], 32)
        self.assertEqual(x_crop.shape[-1], 32 + 16 - 1)
        self.assertTrue(torch.equal(y_crop, processed[..., 16 : 48]))

    def test_build_module_accepts_film_tcn(self) -> None:
        """Armed TCN fragments construct a conditioned module."""
        fragment = {
            "elements": [
                {
                    "id": 1,
                    "type": "tcn",
                    "properties": [
                        {"key": "channels", "value": 4},
                        {"key": "depth", "value": 2},
                        {"key": "kernel_size", "value": 3},
                        {"key": "dilation_growth", "value": 8},
                        {"key": "residual", "value": 1},
                        {"key": "activation", "value": 4},
                    ],
                }
            ],
            "connections": [],
        }
        module = train_worker.build_module(fragment, 1)
        audio = torch.zeros(1, 1, 64)
        cond = torch.zeros(1, 2, 1)
        output = module(audio, cond)
        self.assertEqual(tuple(output.shape), (1, 1, 64))

    def test_film_trace_accepts_mismatched_control_width(self) -> None:
        """Host validation used to probe cond_dim=1 models with a 2-wide tensor."""
        import tempfile

        fragment = {
            "elements": [
                {
                    "id": 1,
                    "type": "tcn",
                    "properties": [
                        {"key": "channels", "value": 16},
                        {"key": "depth", "value": 1},
                        {"key": "kernel_size", "value": 3},
                        {"key": "dilation_growth", "value": 2},
                        {"key": "residual", "value": 0},
                        {"key": "activation", "value": 0},
                    ],
                }
            ],
            "connections": [],
        }
        module = train_worker.build_module(fragment, 1, cond_dim=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            train_worker._export_scripted(module, 1, 1, path)
            loaded = torch.jit.load(str(path))
            audio = torch.zeros(1, 1, 256)
            probed = loaded(audio, torch.zeros(1, 2, 1))
            native = loaded(audio, torch.zeros(1, 1, 1))
            ramped = loaded(audio, torch.zeros(1, 1, 256))
        self.assertEqual(tuple(probed.shape), (1, 1, 256))
        self.assertEqual(tuple(native.shape), (1, 1, 256))
        self.assertEqual(tuple(ramped.shape), (1, 1, 256))

    def test_film_keeps_xy_as_two_dimensional_vector(self) -> None:
        """FiLM must not collapse concatenated XY into the scalar mean x+y."""
        film = train_worker.FiLM(2, 4)
        with torch.no_grad():
            film.adaptor.weight.zero_()
            film.adaptor.bias.zero_()
            film.adaptor.weight[0, 0] = 1.0
            film.adaptor.weight[1, 1] = 1.0
        samples = torch.ones(1, 4, 1)
        x_only = torch.tensor([[[1.0], [0.0]]])
        y_only = torch.tensor([[[0.0], [1.0]]])
        out_x = film(samples, x_only)
        out_y = film(samples, y_only)
        self.assertFalse(torch.allclose(out_x, out_y))

    def test_film_is_per_sample_not_block_mean(self) -> None:
        """A ramped control must modulate each sample, not the buffer-mean."""
        film = train_worker.FiLM(1, 1)
        with torch.no_grad():
            film.adaptor.weight.fill_(1.0)
            film.adaptor.bias.zero_()
        samples = torch.ones(1, 1, 2)
        cond = torch.tensor([[[0.0, 1.0]]])
        out = film(samples, cond)
        self.assertAlmostEqual(out[0, 0, 0].item(), 0.0, places=5)
        self.assertAlmostEqual(out[0, 0, 1].item(), 2.0, places=5)

    def test_exported_film_preserves_time_varying_control(self) -> None:
        """Traced checkpoints must modulate a ramp per sample, not as one value."""
        fragment = {
            "elements": [
                {
                    "id": 1,
                    "type": "tcn",
                    "properties": [
                        {"key": "channels", "value": 4},
                        {"key": "depth", "value": 1},
                        {"key": "kernel_size", "value": 3},
                        {"key": "dilation_growth", "value": 2},
                        {"key": "residual", "value": 0},
                        {"key": "activation", "value": 0},
                    ],
                }
            ],
            "connections": [],
        }
        module = train_worker.build_module(fragment, 1, cond_dim=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            train_worker._export_scripted(module, 1, 1, path)
            loaded = torch.jit.load(str(path))
            audio = torch.ones(1, 1, 32)
            ramp = torch.linspace(0.0, 1.0, 32).view(1, 1, 32)
            out_ramp = loaded(audio, ramp)
            out_zero = loaded(audio, torch.zeros(1, 1, 32))
            out_one = loaded(audio, torch.ones(1, 1, 32))
            out_mid = loaded(audio, torch.full((1, 1, 32), 0.5))
            out_scalar = loaded(audio, torch.zeros(1, 1, 1))
        self.assertEqual(tuple(out_ramp.shape), (1, 1, 32))
        self.assertEqual(tuple(out_scalar.shape), (1, 1, 32))
        self.assertFalse(torch.allclose(out_ramp, out_zero, atol=1.0e-4))
        self.assertFalse(torch.allclose(out_ramp, out_one, atol=1.0e-4))
        self.assertFalse(torch.allclose(out_ramp, out_mid, atol=1.0e-4))
        self.assertFalse(torch.allclose(out_ramp[..., :1], out_ramp[..., -1:], atol=1.0e-4))

    def test_checkpoint_metadata_requires_conditioning(self) -> None:
        """Hear-while-training checkpoints must advertise the (audio, cond) signature."""
        metadata = train_worker._blackbox_metadata(2, 2, 2, 16)
        self.assertTrue(metadata["conditioning"])
        self.assertEqual(metadata["cond_dim"], 2)
        self.assertEqual(metadata["shape_signature"]["input_channels"], 2)

    def test_scripted_checkpoint_requires_cond_argument(self) -> None:
        """Traced train artifacts require cond; the host must not call forward(audio)."""
        fragment = {
            "elements": [
                {
                    "id": 1,
                    "type": "tcn",
                    "properties": [
                        {"key": "channels", "value": 4},
                        {"key": "depth", "value": 1},
                        {"key": "kernel_size", "value": 3},
                        {"key": "dilation_growth", "value": 2},
                        {"key": "residual", "value": 0},
                        {"key": "activation", "value": 0},
                    ],
                }
            ],
            "connections": [],
        }
        module = train_worker.build_module(fragment, 1, cond_dim=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            train_worker._export_scripted(module, 1, 2, path)
            loaded = torch.jit.load(str(path))
            audio = torch.zeros(1, 1, 256)
            with self.assertRaises(RuntimeError):
                loaded(audio)
            output = loaded(audio, torch.zeros(1, 2, 1))
        self.assertEqual(tuple(output.shape), (1, 1, 256))
        """Training pairs are broadcast to the live host channel count."""
        mono = torch.zeros(1, 1, 8)
        stereo = train_worker._match_channels(mono, 2)
        self.assertEqual(tuple(stereo.shape), (1, 2, 8))


    def test_read_wav_ieee_float_without_torchaudio(self) -> None:
        """Library WAVs must load with the stdlib RIFF parser."""
        import tempfile

        samples = torch.tensor(
            [[0.0, 0.5, -0.25, 0.125], [0.1, -0.1, 0.0, -0.5]], dtype=torch.float32
        )
        channels, _length = samples.shape
        interleaved = samples.transpose(0, 1).contiguous().view(-1).tolist()
        payload = struct.pack("<" + "f" * len(interleaved), *interleaved)
        fmt = struct.pack("<HHIIHH", 3, channels, 44100, 44100 * channels * 4, channels * 4, 32)
        riff_size = 4 + (8 + len(fmt)) + (8 + len(payload))
        wav = (
            b"RIFF"
            + struct.pack("<I", riff_size)
            + b"WAVE"
            + b"fmt "
            + struct.pack("<I", len(fmt))
            + fmt
            + b"data"
            + struct.pack("<I", len(payload))
            + payload
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pair.wav"
            path.write_bytes(wav)
            loaded, sample_rate = train_worker.read_wav(str(path))
        self.assertEqual(sample_rate, 44100)
        self.assertEqual(tuple(loaded.shape), (2, 4))
        self.assertTrue(torch.allclose(loaded, samples, atol=1.0e-6))


class TrainWorkerMlflowTests(unittest.TestCase):
    """Validate MLflow Tracking Server REST init/config/log/save/summary/finish."""

    def test_config_includes_recipe_and_graph_summary(self) -> None:
        """Config mirrors hyperparameters plus a compact graph fingerprint."""
        config = train_worker.mlflow_config_from_request(
            {
                "request_id": "r1",
                "graph_fragment": {"elements": [{"type": "tcn"}, {"type": "prelu"}]},
                "capture_set": {"pairs": [{}, {}]},
                "train_options": {
                    "total_steps": 10,
                    "learning_rate": 1.0e-3,
                    "segment_length": 1024,
                    "optimizer": "adam",
                },
            }
        )
        self.assertEqual(config["request_id"], "r1")
        self.assertEqual(config["total_steps"], 10)
        self.assertEqual(config["element_types"], ["tcn", "prelu"])
        self.assertEqual(config["pair_count"], 2)
        self.assertEqual(config["optimizer"], "adam")
        self.assertEqual(config["device"], "auto")

    def test_disabled_tracker_does_not_call_rest(self) -> None:
        """Logging stays opt-in so CI and offline trains do not hit MLflow."""

        def _boom(*args, **kwargs):
            raise AssertionError("MLflow REST should not be called")

        original = train_worker._mlflow_http
        train_worker._mlflow_http = _boom  # type: ignore[method-assign]
        try:
            tracker = train_worker.start_mlflow_tracker({"train_options": {}})
            self.assertFalse(tracker.active)
            tracker = train_worker.start_mlflow_tracker(
                {
                    "train_options": {
                        "mlflow": {"enabled": False, "experiment": "x"}
                    }
                }
            )
            self.assertFalse(tracker.active)
        finally:
            train_worker._mlflow_http = original

    def test_tracker_mirrors_mlflow_rest_lifecycle(self) -> None:
        """get-or-create experiment → create run → log-batch → artifact → update."""

        class FakeHttp:
            def __init__(self) -> None:
                self.calls: list[tuple[str, str, dict | None]] = []

            def __call__(self, method, url, body=None, headers=None, timeout=10.0):
                payload = None
                if body and (headers or {}).get("Content-Type") == "application/json":
                    payload = json.loads(body.decode("utf-8"))
                self.calls.append((method, url, payload))
                if "experiments/get-by-name" in url:
                    return 200, json.dumps(
                        {"experiment": {"experiment_id": "7"}}
                    ).encode()
                if url.endswith("/experiments/create"):
                    return 200, json.dumps({"experiment_id": "7"}).encode()
                if url.endswith("/runs/create"):
                    return 200, json.dumps(
                        {
                            "run": {
                                "info": {
                                    "run_id": "abc",
                                    "run_uuid": "abc",
                                    "experiment_id": "7",
                                    "artifact_uri": "http://127.0.0.1:5000/api/2.0/mlflow-artifacts/artifacts/abc",
                                }
                            }
                        }
                    ).encode()
                if url.endswith("/runs/log-batch") or url.endswith("/runs/update"):
                    return 200, b"{}"
                if method == "PUT":
                    return 200, b""
                return 500, b"{}"

        fake = FakeHttp()
        original = train_worker._mlflow_http
        train_worker._mlflow_http = fake  # type: ignore[method-assign]
        try:
            tracker = train_worker.start_mlflow_tracker(
                {
                    "request_id": "r1",
                    "graph_fragment": {"elements": [{"type": "tcn"}]},
                    "train_options": {
                        "total_steps": 3,
                        "mlflow": {
                            "enabled": True,
                            "tracking_uri": "http://127.0.0.1:5000",
                            "experiment": "openyourbox",
                            "name": "run-1",
                            "tags": ["train", "steerable"],
                        },
                    },
                }
            )
            self.assertTrue(tracker.active)
            self.assertEqual(
                tracker.url,
                "http://127.0.0.1:5000/#/experiments/7/runs/abc",
            )
            create = next(call for call in fake.calls if call[1].endswith("/runs/create"))
            self.assertEqual(create[2]["experiment_id"], "7")
            self.assertIn(
                {"key": "mlflow.runName", "value": "run-1"}, create[2]["tags"]
            )
            param_batch = next(
                call
                for call in fake.calls
                if call[1].endswith("/runs/log-batch") and call[2] and "params" in call[2]
            )
            param_keys = {item["key"] for item in param_batch[2]["params"]}
            self.assertIn("total_steps", param_keys)
            tracker.log({"loss": 0.5, "learning_rate": 1.0e-3}, step=1)
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as handle:
                handle.write(b"ckpt")
                artifact = handle.name
            try:
                tracker.save(artifact)
            finally:
                os.unlink(artifact)
            tracker.update_summary({"success": True, "status": "success"})
            tracker.finish()
        finally:
            train_worker._mlflow_http = original
        metric_batch = next(
            call
            for call in fake.calls
            if call[1].endswith("/runs/log-batch")
            and call[2]
            and "metrics" in call[2]
            and any(item["key"] == "loss" for item in call[2]["metrics"])
        )
        self.assertEqual(metric_batch[2]["metrics"][0]["value"], 0.5)
        self.assertEqual(metric_batch[2]["metrics"][0]["step"], 1)
        self.assertTrue(any(call[0] == "PUT" for call in fake.calls))
        update = next(call for call in fake.calls if call[1].endswith("/runs/update"))
        self.assertEqual(update[2]["status"], "FINISHED")
        self.assertFalse(tracker.active)

    def test_init_failure_does_not_block_train(self) -> None:
        """A down tracking server must leave the tracker inactive."""

        def _fail(*args, **kwargs):
            raise RuntimeError("connection refused")

        original = train_worker._mlflow_http
        train_worker._mlflow_http = _fail  # type: ignore[method-assign]
        try:
            tracker = train_worker.MlflowTracker()
            tracker.init(
                tracking_uri="http://127.0.0.1:5000", experiment="openyourbox"
            )
            self.assertFalse(tracker.active)
            tracker.log({"loss": 1.0}, step=1)
            tracker.finish()
        finally:
            train_worker._mlflow_http = original

    def test_tracking_uri_falls_back_to_env(self) -> None:
        """Empty tracking_uri uses MLFLOW_TRACKING_URI."""
        seen: list[str] = []

        def _capture(method, url, body=None, headers=None, timeout=10.0):
            seen.append(url)
            if "experiments/get-by-name" in url:
                return 200, json.dumps(
                    {"experiment": {"experiment_id": "1"}}
                ).encode()
            if url.endswith("/runs/create"):
                return 200, json.dumps(
                    {"run": {"info": {"run_id": "xyz", "experiment_id": "1"}}}
                ).encode()
            return 200, b"{}"

        original_http = train_worker._mlflow_http
        original_uri = os.environ.get("MLFLOW_TRACKING_URI")
        train_worker._mlflow_http = _capture  # type: ignore[method-assign]
        os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow.example:5000"
        try:
            tracker = train_worker.start_mlflow_tracker(
                {"train_options": {"mlflow": {"enabled": True, "experiment": "demo"}}}
            )
            self.assertTrue(tracker.active)
            self.assertTrue(
                any(url.startswith("http://mlflow.example:5000/") for url in seen)
            )
        finally:
            train_worker._mlflow_http = original_http
            if original_uri is None:
                os.environ.pop("MLFLOW_TRACKING_URI", None)
            else:
                os.environ["MLFLOW_TRACKING_URI"] = original_uri

    def test_flatten_reconstruction_uses_pair_x_and_y(self) -> None:
        """Reconstruction corpus must include both sides of each selected pair."""
        paths = train_worker.flatten_reconstruction_clips(
            {
                "pairs": [{"x_path": "/tmp/x.wav", "y_path": "/tmp/y.wav"}],
                "clips": [{"path": "/tmp/clip.wav"}],
            }
        )
        self.assertEqual(paths, ["/tmp/x.wav", "/tmp/y.wav", "/tmp/clip.wav"])

    def test_mapping_rejects_unpaired_clips(self) -> None:
        """Mapping must refuse a capture set that includes unpaired clips."""
        self.assertTrue(
            train_worker.mapping_rejects_unpaired({"clips": [{"path": "/tmp/a.wav"}]})
        )
        self.assertFalse(train_worker.mapping_rejects_unpaired({"pairs": [{}], "clips": []}))

    def test_objective_dispatch_mapping_rejects_clips(self) -> None:
        """train_request mapping path must error before loading pairs when clips exist."""
        with self.assertRaisesRegex(ValueError, "unpaired"):
            train_worker.train_request(
                {
                    "request_id": "map-reject",
                    "operation": "train_steerable",
                    "train_options": {"objective": "mapping"},
                    "capture_set": {"pairs": [], "clips": [{"path": "/tmp/a.wav"}]},
                    "graph_fragment": {"elements": [], "connections": []},
                },
                Path("/tmp"),
                None,
            )

    def test_reconstruction_two_stage_and_encode_decode_export(self) -> None:
        """Short reconstruction must emit both stages and export encode/decode."""

        def _write_wav(path: Path) -> None:
            samples = [0.05] * 256
            payload = struct.pack("<" + "f" * len(samples), *samples)
            fmt = struct.pack("<HHIIHH", 3, 1, 44100, 44100 * 4, 4, 32)
            riff_size = 4 + (8 + len(fmt)) + (8 + len(payload))
            path.write_bytes(
                b"RIFF"
                + struct.pack("<I", riff_size)
                + b"WAVEfmt "
                + struct.pack("<I", len(fmt))
                + fmt
                + b"data"
                + struct.pack("<I", len(payload))
                + payload
            )

        fragment = {
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
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "clip.wav"
            _write_wav(wav)
            events: list[dict] = []
            original_emit = train_worker._emit

            def _capture(event):
                events.append(event)
                original_emit(event)

            train_worker._emit = _capture  # type: ignore[method-assign]
            try:
                result = train_worker.train_request(
                    {
                        "request_id": "rave-short",
                        "operation": "train_steerable",
                        "train_options": {
                            "objective": "reconstruction",
                            "host_input_channels": 1,
                            "segment_length": 128,
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
                            "pairs": [{"x_path": str(wav), "y_path": str(wav)}],
                            "clips": [],
                        },
                        "graph_fragment": fragment,
                    },
                    Path(tmp),
                    None,
                )
            finally:
                train_worker._emit = original_emit
            self.assertEqual(result["status"], "success")
            self.assertTrue(result["has_encode_decode"])
            self.assertIn(result.get("device"), train_worker.available_train_devices())
            self.assertEqual(result.get("requested_device"), "auto")
            stages = {event.get("stage") for event in events}
            self.assertIn("representation", stages)
            self.assertIn("quality", stages)
            for event in events:
                self.assertIn(event.get("device"), train_worker.available_train_devices())
                self.assertEqual(event.get("requested_device"), "auto")
            loaded = torch.jit.load(result["artifact_path"])
            audio = torch.zeros(1, 1, 256)
            encoded = loaded.encode(audio)
            decoded = loaded.decode(encoded)
            forwarded = loaded.forward(audio)
            self.assertEqual(encoded.dim(), 3)
            self.assertEqual(decoded.dim(), 3)
            self.assertEqual(forwarded.dim(), 3)
            compactness = result.get("compactness", {})
            self.assertIn("ready", compactness)
            self.assertIn("status", compactness)

    def test_reconstruction_exports_best_not_last_weights(self) -> None:
        """Final RAVE artifact must restore the lowest spectral-loss state_dict."""

        def _write_wav(path: Path) -> None:
            samples = [0.05] * 256
            payload = struct.pack("<" + "f" * len(samples), *samples)
            fmt = struct.pack("<HHIIHH", 3, 1, 44100, 44100 * 4, 4, 32)
            riff_size = 4 + (8 + len(fmt)) + (8 + len(payload))
            path.write_bytes(
                b"RIFF"
                + struct.pack("<I", riff_size)
                + b"WAVEfmt "
                + struct.pack("<I", len(fmt))
                + fmt
                + b"data"
                + struct.pack("<I", len(payload))
                + payload
            )

        fragment = {
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
        scheduled = [1.0, 0.1, 0.8, 0.8]
        call_index = {"n": 0}
        snapshots: list[dict[str, torch.Tensor]] = []
        exported: dict[str, torch.Tensor] = {}
        original_loss = train_worker.reconstruction_spectral_loss
        original_clone = train_worker._clone_state_dict
        original_export = train_worker._export_rave_scripted

        def _scheduled_loss(reconstructed, audio, module, windows, log_epsilon):
            del audio, module, windows, log_epsilon
            value = scheduled[min(call_index["n"], len(scheduled) - 1)]
            call_index["n"] += 1
            return reconstructed.sum() * 0.0 + reconstructed.new_tensor(value)

        def _tracking_clone(module):
            snap = original_clone(module)
            snapshots.append({name: tensor.clone() for name, tensor in snap.items()})
            return snap

        def _tracking_export(module, input_channels, path):
            exported.update(original_clone(module))
            original_export(module, input_channels, path)

        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "clip.wav"
            _write_wav(wav)
            train_worker.reconstruction_spectral_loss = _scheduled_loss
            train_worker._clone_state_dict = _tracking_clone
            train_worker._export_rave_scripted = _tracking_export
            try:
                result = train_worker.train_request(
                    {
                        "request_id": "rave-best",
                        "operation": "train_steerable",
                        "train_options": {
                            "objective": "reconstruction",
                            "device": "cpu",
                            "host_input_channels": 1,
                            "segment_length": 128,
                            "reconstruction": {
                                "stage1_steps": 3,
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
                            "pairs": [{"x_path": str(wav), "y_path": str(wav)}],
                            "clips": [],
                        },
                        "graph_fragment": fragment,
                    },
                    Path(tmp),
                    None,
                )
            finally:
                train_worker.reconstruction_spectral_loss = original_loss
                train_worker._clone_state_dict = original_clone
                train_worker._export_rave_scripted = original_export
            self.assertEqual(result["status"], "success")
            self.assertGreaterEqual(len(snapshots), 2)
            best = snapshots[-1]
            self.assertTrue(exported)
            self.assertEqual(set(exported), set(best))
            for name in best:
                self.assertTrue(
                    torch.equal(exported[name], best[name]),
                    msg=f"exported {name} did not match best-loss snapshot",
                )
            self.assertAlmostEqual(float(result["best_loss"]), 0.1)
            self.assertAlmostEqual(float(result["loss"]), 0.8)

    def test_noise_synth_warmup_mutes_addend(self) -> None:
        """Stage-1 warmup must zero the noise addend and block amplitude grads."""
        layer = train_worker.NoiseSynthLayer(5, 16)
        self.assertTrue(layer.warmed_up)
        amplitudes = torch.randn(2, 10, 3, requires_grad=True)
        muted_layer = train_worker.NoiseSynthLayer(5, 16)
        muted_layer.warmed_up = False
        muted = muted_layer(amplitudes)
        self.assertEqual(tuple(muted.shape), (2, 2, 48))
        self.assertTrue(torch.equal(muted, torch.zeros_like(muted)))
        self.assertFalse(muted.requires_grad)
        self.assertIsNone(amplitudes.grad)
        torch.manual_seed(0)
        live = layer(amplitudes.detach())
        self.assertEqual(tuple(live.shape), (2, 2, 48))
        self.assertFalse(torch.allclose(live, torch.zeros_like(live)))

    def test_reconstruction_toggles_noise_synth_warmup(self) -> None:
        """Representation mutes Noise Synth; quality and export enable it."""

        def _write_wav(path: Path) -> None:
            samples = [0.05] * 256
            payload = struct.pack("<" + "f" * len(samples), *samples)
            fmt = struct.pack("<HHIIHH", 3, 1, 44100, 44100 * 4, 4, 32)
            riff_size = 4 + (8 + len(fmt)) + (8 + len(payload))
            path.write_bytes(
                b"RIFF"
                + struct.pack("<I", riff_size)
                + b"WAVEfmt "
                + struct.pack("<I", len(fmt))
                + fmt
                + b"data"
                + struct.pack("<I", len(payload))
                + payload
            )

        fragment = {
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
                {
                    "id": 4,
                    "type": "conv1d",
                    "properties": [
                        {"key": "channels", "value": 2},
                        {"key": "kernel_size", "value": 1},
                        {"key": "stride", "value": 1},
                        {"key": "dilation", "value": 1},
                    ],
                },
                {
                    "id": 5,
                    "type": "noise_synthesizer",
                    "properties": [
                        {"key": "noise_bands", "value": 2},
                        {"key": "window_size", "value": 2},
                    ],
                },
                {
                    "id": 6,
                    "type": "utility",
                    "properties": [
                        {"key": "mode", "value": 0},
                        {"key": "inputs", "value": 2},
                    ],
                },
            ],
            "connections": [
                {"source_element_id": 1, "destination_element_id": 2},
                {"source_element_id": 2, "destination_element_id": 3},
                {"source_element_id": 2, "destination_element_id": 4},
                {"source_element_id": 4, "destination_element_id": 5},
                {
                    "source_element_id": 3,
                    "destination_element_id": 6,
                    "destination_pin_index": 0,
                },
                {
                    "source_element_id": 5,
                    "destination_element_id": 6,
                    "destination_pin_index": 1,
                },
            ],
        }
        flags: list[bool] = []
        original_set = train_worker._set_noise_synth_warmed_up

        def _spy(module, warmed_up):
            flags.append(bool(warmed_up))
            original_set(module, warmed_up)

        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "clip.wav"
            _write_wav(wav)
            train_worker._set_noise_synth_warmed_up = _spy  # type: ignore[method-assign]
            try:
                result = train_worker.train_request(
                    {
                        "request_id": "rave-noise-warmup",
                        "operation": "train_steerable",
                        "train_options": {
                            "objective": "reconstruction",
                            "host_input_channels": 1,
                            "segment_length": 128,
                            "reconstruction": {
                                "stage1_steps": 2,
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
                            "pairs": [{"x_path": str(wav), "y_path": str(wav)}],
                            "clips": [],
                        },
                        "graph_fragment": fragment,
                    },
                    Path(tmp),
                    None,
                )
            finally:
                train_worker._set_noise_synth_warmed_up = original_set
            self.assertEqual(result["status"], "success")
            self.assertGreaterEqual(len(flags), 3)
            self.assertEqual(flags[:3], [False, False, True])
            self.assertTrue(all(flags[3:]))
            loaded = torch.jit.load(result["artifact_path"])
            audio = torch.zeros(1, 1, 256)
            with torch.inference_mode():
                forwarded = loaded.forward(audio)
            self.assertEqual(forwarded.dim(), 3)
            self.assertEqual(int(forwarded.size(0)), 1)
            self.assertGreaterEqual(int(forwarded.size(2)), 1)

    def test_v1_defaults_and_losses(self) -> None:
        """Reconstruction knobs and losses must follow acids-rave v1.gin."""
        recipe = train_worker.parse_reconstruction_options({})
        self.assertEqual(recipe["stage1_steps"], 1_000_000)
        self.assertEqual(recipe["stage2_steps"], 1_000_000)
        self.assertEqual(recipe["kl_beta"], 0.1)
        self.assertEqual(recipe["kl_beta_start"], 0.1)
        self.assertEqual(recipe["kl_warmup_steps"], 1)
        self.assertEqual(recipe["feature_matching_weight"], 10.0)
        self.assertEqual(recipe["update_discriminator_every"], 2)
        self.assertEqual(recipe["batch_size"], 8)
        self.assertEqual(recipe["segment_length"], 65536)
        self.assertEqual(recipe["adam_beta1"], 0.5)
        self.assertEqual(recipe["adam_beta2"], 0.9)
        self.assertAlmostEqual(
            train_worker.reconstruction_kl_beta(0, 1, 0.1, 0.1), 0.1
        )
        disc, gen = train_worker.hinge_gan(-torch.ones(4), torch.ones(4))
        self.assertGreater(float(disc), 0.0)
        self.assertAlmostEqual(float(gen), -1.0)
        predicted = torch.randn(1, 1, 256)
        target = torch.randn(1, 1, 256)
        distance = train_worker.audio_distance_v1(predicted, target)
        self.assertGreater(float(distance), 0.0)
        same = train_worker.audio_distance_v1(target, target)
        self.assertLess(float(same), float(distance))
        stereo = torch.randn(8, 2, 256)
        mono = torch.randn(8, 1, 256)
        mixed = train_worker.audio_distance_v1(mono, stereo)
        self.assertTrue(torch.isfinite(mixed))
        self.assertGreater(float(mixed), 0.0)
        left = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        folded = train_worker._adapt_host_input_to_mode(left, "mono")
        self.assertEqual(tuple(folded.shape), (1, 1, 2))
        self.assertTrue(
            torch.allclose(folded, torch.tensor([[[2.0, 3.0]]]), atol=1.0e-6)
        )
        mirrored = train_worker._adapt_host_input_to_mode(left, "mirrored")
        self.assertEqual(tuple(mirrored.shape), (1, 2, 2))
        self.assertTrue(torch.allclose(mirrored[:, 0], mirrored[:, 1]))
        stereo_host = train_worker._reconstruction_io_mode(
            {
                "elements": [
                    {
                        "id": 1,
                        "type": "audio_input",
                        "properties": [{"key": "channels", "value": 0}],
                    }
                ]
            },
            {"host_input_channels": 2},
            2,
        )
        self.assertEqual(stereo_host, "mono")
        decoder_mono = train_worker._reconstruction_io_mode(
            {
                "elements": [
                    {
                        "id": 1,
                        "type": "conv_transpose1d",
                        "properties": [{"key": "channels", "value": 1}],
                    }
                ]
            },
            {"host_input_channels": 2},
            2,
        )
        self.assertEqual(decoder_mono, "mono")
        explicit = train_worker._reconstruction_io_mode(
            {}, {"host_io_mode": "stereo", "host_input_channels": 1}, 1
        )
        self.assertEqual(explicit, "stereo")

    def test_pqmf_dual_spectral_uses_module_dict(self) -> None:
        """Dual fullband+PQMF spectral must look up analysis via ModuleDict keys."""
        fragment = {
            "elements": [
                {
                    "id": 1,
                    "type": "pqmf_analysis",
                    "properties": [{"key": "n_band", "value": 4}],
                },
                {
                    "id": 2,
                    "type": "variational_bottleneck",
                    "properties": [
                        {"key": "latent_size", "value": 2},
                        {"key": "kernel_size", "value": 5},
                    ],
                },
                {
                    "id": 3,
                    "type": "pqmf_synthesis",
                    "properties": [{"key": "n_band", "value": 4}],
                },
            ],
            "connections": [
                {"source_element_id": 1, "destination_element_id": 2},
                {"source_element_id": 2, "destination_element_id": 3},
            ],
        }
        module = train_worker.build_rave_graph_module(fragment, 1, 1)
        analysis = train_worker._pqmf_analysis_layer(module)
        self.assertIsInstance(analysis, train_worker.PqmfLayer)
        audio = torch.randn(1, 1, 64)
        reconstructed = torch.randn(1, 1, 64)
        loss = train_worker.reconstruction_spectral_loss(
            reconstructed, audio, module, (32,), 1.0e-7
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss), 0.0)

    def test_rave_export_allows_non_matching_time_length(self) -> None:
        """PQMF and conv-transpose may change T; the artifact must still load."""
        fragment = {
            "elements": [
                {
                    "id": 1,
                    "type": "pqmf_analysis",
                    "properties": [{"key": "n_band", "value": 4}],
                },
                {
                    "id": 2,
                    "type": "variational_bottleneck",
                    "properties": [
                        {"key": "latent_size", "value": 2},
                        {"key": "kernel_size", "value": 5},
                    ],
                },
                {
                    "id": 3,
                    "type": "conv_transpose1d",
                    "properties": [
                        {"key": "channels", "value": 4},
                        {"key": "kernel_size", "value": 3},
                        {"key": "stride", "value": 2},
                        {"key": "dilation", "value": 1},
                    ],
                },
                {
                    "id": 4,
                    "type": "pqmf_synthesis",
                    "properties": [{"key": "n_band", "value": 4}],
                },
            ],
            "connections": [
                {"source_element_id": 1, "destination_element_id": 2},
                {"source_element_id": 2, "destination_element_id": 3},
                {"source_element_id": 3, "destination_element_id": 4},
            ],
        }
        module = train_worker.build_rave_graph_module(fragment, 1, 1).eval()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rave.pt"
            train_worker._export_rave_scripted(module, 1, path)
            loaded = torch.jit.load(str(path))
            audio = torch.zeros(1, 1, 256)
            with torch.inference_mode():
                forwarded = loaded.forward(audio)
                encoded = loaded.encode(audio)
                mu, std = loaded.encode_distribution(audio)
            self.assertEqual(forwarded.dim(), 3)
            self.assertEqual(int(forwarded.size(0)), 1)
            self.assertGreaterEqual(int(forwarded.size(1)), 1)
            self.assertGreaterEqual(int(forwarded.size(2)), 1)
            self.assertEqual(tuple(mu.shape), tuple(encoded.shape))
            self.assertEqual(tuple(std.shape), tuple(mu.shape))
            self.assertTrue(bool(torch.all(std > 0)))

    def test_match_time_to_is_causal(self) -> None:
        """Shorter tensors left-pad; longer tensors keep the newest samples."""
        short = torch.ones(1, 2, 3)
        long = torch.arange(5, dtype=torch.float32).view(1, 1, 5)
        padded = train_worker.match_time_to(short, long)
        self.assertEqual(tuple(padded.shape), (1, 2, 5))
        self.assertTrue(torch.equal(padded[..., :2], torch.zeros(1, 2, 2)))
        cropped = train_worker.match_time_to(long, short)
        self.assertEqual(tuple(cropped.shape), (1, 1, 3))
        self.assertTrue(torch.equal(cropped, long[..., -3:]))
        added = train_worker.combine_pair(long.expand(1, 2, 5), short, 0)
        self.assertEqual(tuple(added.shape), (1, 2, 5))

    def test_rave_utility_export_accepts_host_block_sizes(self) -> None:
        """Traced Utility + noise must run at live block sizes, not only T=256."""
        fragment = {
            "elements": [
                {
                    "id": 1,
                    "type": "conv_transpose1d",
                    "properties": [
                        {"key": "channels", "value": 1},
                        {"key": "kernel_size", "value": 8},
                        {"key": "stride", "value": 4},
                        {"key": "dilation", "value": 1},
                    ],
                },
                {
                    "id": 2,
                    "type": "conv1d",
                    "properties": [
                        {"key": "channels", "value": 5},
                        {"key": "kernel_size", "value": 1},
                        {"key": "stride", "value": 1},
                        {"key": "dilation", "value": 1},
                    ],
                },
                {
                    "id": 3,
                    "type": "noise_synthesizer",
                    "properties": [
                        {"key": "noise_bands", "value": 5},
                        {"key": "window_size", "value": 16},
                    ],
                },
                {
                    "id": 4,
                    "type": "utility",
                    "properties": [
                        {"key": "mode", "value": 0},
                        {"key": "inputs", "value": 2},
                    ],
                },
            ],
            "connections": [
                {
                    "source_element_id": 1,
                    "destination_element_id": 4,
                    "destination_pin_index": 0,
                },
                {"source_element_id": 2, "destination_element_id": 3},
                {
                    "source_element_id": 3,
                    "destination_element_id": 4,
                    "destination_pin_index": 1,
                },
            ],
        }
        module = train_worker.build_rave_graph_module(fragment, 1, 1).eval()
        self.assertIsInstance(module.layers["4"], train_worker.UtilityCombine)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rave.pt"
            train_worker._export_rave_scripted(module, 1, path)
            loaded = torch.jit.load(str(path))
            for length in (64, 256, 512, 1024):
                audio = torch.zeros(1, 1, length)
                with torch.inference_mode():
                    forwarded = loaded.forward(audio)
                self.assertEqual(forwarded.dim(), 3, msg=f"T={length}")
                self.assertEqual(int(forwarded.size(0)), 1, msg=f"T={length}")
                self.assertGreaterEqual(int(forwarded.size(1)), 1, msg=f"T={length}")
                self.assertGreaterEqual(int(forwarded.size(2)), 1, msg=f"T={length}")

    def test_clone_export_survives_weight_norm(self) -> None:
        """Checkpoint export must not deepcopy weight_norm parametrizations."""
        layer = train_worker.CausalConv1d(2, 4, 3, 1, weight_norm=True)
        module = torch.nn.Sequential(layer)
        samples = torch.randn(1, 2, 32)
        out = module(samples)
        self.assertEqual(tuple(out.shape[:2]), (1, 4))
        cloned = train_worker._clone_module_for_export(module)
        self.assertTrue(train_worker._has_weight_norm(layer.convolution))
        cloned_conv = cloned[0].convolution
        self.assertFalse(train_worker._has_weight_norm(cloned_conv))
        self.assertTrue(hasattr(cloned_conv, "weight"))
        cloned_out = cloned(samples)
        self.assertEqual(tuple(cloned_out.shape), tuple(out.shape))

    def test_reconstruction_weight_norm_checkpoint_and_mlflow(self) -> None:
        """Hear-while-training export with weight_norm must succeed and log MLflow."""

        def _write_wav(path: Path) -> None:
            samples = [0.05] * 256
            payload = struct.pack("<" + "f" * len(samples), *samples)
            fmt = struct.pack("<HHIIHH", 3, 1, 44100, 44100 * 4, 4, 32)
            riff_size = 4 + (8 + len(fmt)) + (8 + len(payload))
            path.write_bytes(
                b"RIFF"
                + struct.pack("<I", riff_size)
                + b"WAVEfmt "
                + struct.pack("<I", len(fmt))
                + fmt
                + b"data"
                + struct.pack("<I", len(payload))
                + payload
            )

        fragment = {
            "elements": [
                {
                    "id": 1,
                    "type": "rate_conv",
                    "properties": [
                        {"key": "channels", "value": 4},
                        {"key": "kernel_size", "value": 3},
                        {"key": "stride", "value": 2},
                        {"key": "dilation", "value": 1},
                        {"key": "weight_norm", "value": 1},
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
                        {"key": "weight_norm", "value": 1},
                    ],
                },
            ],
            "connections": [
                {"source_element_id": 1, "destination_element_id": 2},
                {"source_element_id": 2, "destination_element_id": 3},
            ],
        }

        class FakeHttp:
            def __init__(self) -> None:
                self.calls: list[tuple[str, str, dict | None]] = []

            def __call__(self, method, url, body=None, headers=None, timeout=10.0):
                payload = None
                if body and (headers or {}).get("Content-Type") == "application/json":
                    payload = json.loads(body.decode("utf-8"))
                self.calls.append((method, url, payload))
                if "experiments/get-by-name" in url:
                    return 200, json.dumps(
                        {"experiment": {"experiment_id": "7"}}
                    ).encode()
                if url.endswith("/runs/create"):
                    return 200, json.dumps(
                        {
                            "run": {
                                "info": {
                                    "run_id": "rec",
                                    "experiment_id": "7",
                                    "artifact_uri": "http://127.0.0.1:5000/api/2.0/mlflow-artifacts/artifacts/rec",
                                }
                            }
                        }
                    ).encode()
                if url.endswith("/runs/log-batch") or url.endswith("/runs/update"):
                    return 200, b"{}"
                if method == "PUT":
                    return 200, b""
                return 200, b"{}"

        fake = FakeHttp()
        original = train_worker._mlflow_http
        train_worker._mlflow_http = fake  # type: ignore[method-assign]
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "clip.wav"
            _write_wav(wav)
            try:
                result = train_worker.train_request(
                    {
                        "request_id": "rave-wn",
                        "operation": "train_steerable",
                        "train_options": {
                            "objective": "reconstruction",
                            "host_input_channels": 1,
                            "segment_length": 128,
                            "export_checkpoints": True,
                            "checkpoint_interval": 1,
                            "reconstruction": {
                                "stage1_steps": 1,
                                "stage2_steps": 1,
                                "batch_size": 1,
                                "phase_mangle_prob": 0.0,
                                "dequantize_bits": 0,
                                "disc_n_scales": 1,
                                "disc_capacity": 8,
                                "disc_n_layers": 1,
                            },
                            "mlflow": {
                                "enabled": True,
                                "tracking_uri": "http://127.0.0.1:5000",
                                "experiment": "openyourbox",
                                "name": "rave-v1",
                                "tags": ["train", "reconstruction"],
                            },
                        },
                        "capture_set": {
                            "pairs": [{"x_path": str(wav), "y_path": str(wav)}],
                            "clips": [],
                        },
                        "graph_fragment": fragment,
                    },
                    Path(tmp),
                    None,
                )
                self.assertEqual(result["status"], "success")
                self.assertTrue(result.get("mlflow_url"))
                loaded = torch.jit.load(result["artifact_path"])
                audio = torch.zeros(1, 1, 256)
                self.assertEqual(loaded.forward(audio).dim(), 3)
            finally:
                train_worker._mlflow_http = original
        metric_batches = [
            call
            for call in fake.calls
            if call[1].endswith("/runs/log-batch")
            and call[2]
            and "metrics" in call[2]
        ]
        self.assertTrue(metric_batches)
        metric_keys = {
            item["key"]
            for batch in metric_batches
            for item in batch[2]["metrics"]
        }
        self.assertIn("loss", metric_keys)
        self.assertIn("kl", metric_keys)
        param_batch = next(
            call
            for call in fake.calls
            if call[1].endswith("/runs/log-batch") and call[2] and "params" in call[2]
        )
        param_keys = {item["key"] for item in param_batch[2]["params"]}
        self.assertIn("objective", param_keys)
        self.assertIn("recipe", param_keys)


class VariationalBottleneckParityTests(unittest.TestCase):
    """RAVE variational-head parity against acids-rave v1 rules."""

    def test_train_forward_differs_from_eval(self) -> None:
        """Worker sampling is active only in train mode."""
        torch.manual_seed(0)
        layer = train_worker.VariationalBottleneckLayer(8, 4, kernel_size=5)
        samples = torch.randn(2, 8, 16)
        layer.train()
        train_a = layer(samples)
        train_b = layer(samples)
        layer.eval()
        eval_a = layer(samples)
        eval_b = layer(samples)
        self.assertFalse(torch.equal(train_a, train_b))
        self.assertTrue(torch.equal(eval_a, eval_b))
        self.assertTrue(torch.equal(eval_a, layer.encode_mean(samples)))

    def test_split_reproducible_seed_42(self) -> None:
        """98/2 split with seed 42 is deterministic and caps validation at 1000."""
        paths = [f"clip-{index}" for index in range(200)]
        train_a, val_a = train_worker.split_reconstruction_corpus(paths)
        train_b, val_b = train_worker.split_reconstruction_corpus(paths)
        self.assertEqual(train_a, train_b)
        self.assertEqual(val_a, val_b)
        self.assertEqual(len(val_a), 4)
        self.assertEqual(len(train_a), 196)
        huge = [f"clip-{index}" for index in range(80_000)]
        _train, val_huge = train_worker.split_reconstruction_corpus(huge)
        self.assertEqual(len(val_huge), 1000)

    def test_train_sampler_excludes_val_paths(self) -> None:
        """Stage-1 sampling draws only from the train split."""
        paths = [f"p{index}" for index in range(50)]
        train_paths, val_paths = train_worker.split_reconstruction_corpus(paths)
        self.assertTrue(set(train_paths).isdisjoint(set(val_paths)))
        self.assertEqual(sorted(train_paths + val_paths), sorted(paths))

    def test_compactness_uses_linear_singular_values(self) -> None:
        """Fidelity rank uses cumulative singular values, not s²."""
        torch.manual_seed(1)
        rows = torch.randn(32, 4)
        _mean, _pca, cumulative = train_worker.compute_compactness(rows)
        self.assertEqual(tuple(cumulative.shape), (4,))
        self.assertGreaterEqual(float(cumulative[-1]), 0.999)
        keep_high = int((cumulative >= 0.9).nonzero()[0]) + 1
        keep_low = int((cumulative >= 0.5).nonzero()[0]) + 1
        self.assertGreaterEqual(keep_high, keep_low)

    def test_weight_norm_property_wraps_and_strips_for_export(self) -> None:
        """Decoder Conv1d with weight_norm trains as g/v and exports plain weight."""
        layer = train_worker.CausalConv1d(4, 8, 3, 1, weight_norm=True)
        self.assertTrue(train_worker._has_weight_norm(layer.convolution))
        plain = train_worker.CausalConv1d(4, 8, 3, 1, weight_norm=False)
        self.assertFalse(train_worker._has_weight_norm(plain.convolution))

        wrapped = train_worker.ConvTransposeLayer(8, 4, 4, 2, 1, weight_norm=True)
        module = torch.nn.Sequential(wrapped)
        train_worker.strip_weight_norm(module)
        self.assertFalse(train_worker._has_weight_norm(wrapped.convolution))
        self.assertTrue(hasattr(wrapped.convolution, "weight"))
        samples = torch.zeros(1, 8, 16)
        out = wrapped(samples)
        self.assertEqual(tuple(out.shape[:2]), (1, 4))

        props_on = {"channels": 8, "kernel_size": 3, "dilation": 1, "stride": 1, "weight_norm": 1}
        props_off = {"channels": 8, "kernel_size": 3, "dilation": 1, "stride": 1, "weight_norm": 0}
        on = train_worker._make_strided_conv(4, 8, props_on)
        off = train_worker._make_strided_conv(4, 8, props_off)
        self.assertTrue(train_worker._has_weight_norm(on.convolution))
        self.assertFalse(train_worker._has_weight_norm(off.convolution))

    def test_eval_mu_not_sampled_z(self) -> None:
        """Validation μ collection must not use reparameterized samples."""
        torch.manual_seed(2)
        layer = train_worker.VariationalBottleneckLayer(8, 4, kernel_size=5)
        samples = torch.randn(1, 8, 32)
        layer.train()
        sampled = layer(samples)
        layer.eval()
        mu = layer.encode_mean(samples)
        self.assertFalse(torch.allclose(sampled, mu))

    def test_trains_filtered_noise_and_lstm(self) -> None:
        """Trainable magnitudes and recurrent weights must appear in the module."""
        noise = train_worker.build_module(
            {
                "elements": [
                    {
                        "id": 1,
                        "type": "filtered_noise_reverb",
                        "seed": 3,
                        "properties": [
                            {"key": "n_frames", "value": 4},
                            {"key": "n_filter_banks", "value": 8},
                            {"key": "window_size", "value": 17},
                            {"key": "reverb_length", "value": 32},
                            {"key": "add_dry", "value": 1},
                        ],
                    }
                ],
                "connections": [],
            },
            2,
        )
        self.assertTrue(any(parameter.requires_grad for parameter in noise.parameters()))
        lstm = train_worker.build_module(
            {
                "elements": [
                    {
                        "id": 1,
                        "type": "lstm",
                        "seed": 3,
                        "properties": [
                            {"key": "hidden_size", "value": 4},
                            {"key": "bidirectional", "value": 0},
                            {"key": "bias", "value": 1},
                            {"key": "activation", "value": 2},
                            {"key": "gain", "float_value": 1.0},
                            {"key": "leak_rate", "float_value": 1.0},
                            {"key": "recurrent_weight_scale", "float_value": 1.0},
                        ],
                    }
                ],
                "connections": [],
            },
            2,
        )
        output = lstm(torch.randn(1, 2, 16), torch.zeros(1, 2, 16))
        self.assertEqual(tuple(output.shape), (1, 4, 16))
        zero_leak = train_worker.RecurrentLayer(2, 3, False, True, 2, 1.0, 0.01, False, 0.0, 1.0)
        silent = zero_leak(torch.randn(1, 2, 8))
        self.assertTrue(torch.allclose(silent, torch.zeros_like(silent)))


if __name__ == "__main__":
    unittest.main()
