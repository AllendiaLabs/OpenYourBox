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
                    "properties": [{"key": "latent_size", "value": 4}],
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
            stages = {event.get("stage") for event in events}
            self.assertIn("representation", stages)
            self.assertIn("quality", stages)
            loaded = torch.jit.load(result["artifact_path"])
            audio = torch.zeros(1, 1, 256)
            encoded = loaded.encode(audio)
            decoded = loaded.decode(encoded)
            forwarded = loaded.forward(audio)
            self.assertEqual(encoded.dim(), 3)
            self.assertEqual(decoded.dim(), 3)
            self.assertEqual(forwarded.dim(), 3)


if __name__ == "__main__":
    unittest.main()
