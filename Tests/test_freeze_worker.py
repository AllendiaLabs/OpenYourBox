"""Focused tests for the detached manual-freeze worker."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch


WORKER_PATH = Path(__file__).parents[1] / "Backend" / "freeze_worker.py"
SPEC = importlib.util.spec_from_file_location("freeze_worker", WORKER_PATH)
assert SPEC is not None and SPEC.loader is not None
freeze_worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(freeze_worker)


class FreezeWorkerTests(unittest.TestCase):
    """Validate artifact generation and malformed graph rejection."""

    def test_weighted_elements_reproduce_signed_seed(self) -> None:
        """Worker weights must deterministically follow each signed node seed."""
        fragment = {
            "elements": [
                {
                    "id": 1,
                    "type": "linear",
                    "seed": -42,
                    "properties": [{"key": "features", "value": 2}],
                }
            ],
            "connections": [],
        }
        first = freeze_worker.build_module(fragment, 2)
        second = freeze_worker.build_module(fragment, 2)
        self.assertTrue(
            all(
                torch.equal(left, right)
                for left, right in zip(first.parameters(), second.parameters())
            )
        )
        self.assertEqual(
            next(first.parameters()).flatten().tolist(),
            [
                -1.6598129272460938,
                0.15951856970787048,
                -0.4109762907028198,
                0.3478688597679138,
            ],
        )

    def test_compiles_connected_selection_to_loadable_artifact(self) -> None:
        """A valid selected chain should produce a callable TorchScript file."""
        request = {
            "request_id": "worker-test",
            "operation": "freeze_selection",
            "selected_element_ids": [1, 2, 3],
            "graph_fragment": {
                "elements": [
                    {"id": 1, "type": "audio_input", "seed": 1, "properties": []},
                    {
                        "id": 2,
                        "type": "conv1d",
                        "seed": -42,
                        "properties": [
                            {"key": "channels", "value": 2},
                            {"key": "kernel_size", "value": 3},
                            {"key": "dilation", "value": 1},
                        ],
                    },
                    {"id": 3, "type": "audio_output", "seed": 1, "properties": []},
                ],
                "connections": [
                    {"source_element_id": 1, "destination_element_id": 2},
                    {"source_element_id": 2, "destination_element_id": 3},
                ],
                "io_boundary": {"inputs": [], "outputs": []},
            },
            "compile_options": {
                "mode": "manual_freeze",
                "host_input_channels": 1,
                "host_output_channels": 2,
                "example_samples": 64,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            response = freeze_worker.compile_request(request, Path(directory))
            artifact = Path(response["artifact_path"])
            self.assertTrue(artifact.is_file())
            module = torch.jit.load(str(artifact))
            output = module(torch.zeros(1, 1, 64))
            self.assertEqual(tuple(output.shape), (1, 2, 64))
            self.assertEqual(
                response["blackbox_metadata"]["receptive_field_samples"], 3
            )

    def test_tcn_uses_cpp_base_dilation_schedule(self) -> None:
        """TCN layers must use baseDilation multiplied by powers of two."""
        module = freeze_worker.SteerableTCN(2, 2, 2, 3, 3, 3, 2, 0, False, 0)
        dilations = [
            block.convolution.convolution.dilation[0] for block in module.blocks
        ]
        self.assertEqual(dilations, [3, 6, 12])

    def test_sigmoid_preserves_digital_silence(self) -> None:
        """Frozen sigmoid must match the live engine's exact-zero behavior."""
        output = freeze_worker.ZeroPreservingSigmoid()(torch.zeros(1, 2, 16))
        self.assertEqual(torch.count_nonzero(output).item(), 0)

    def test_compiles_stereo_identity_for_current_host_shape(self) -> None:
        """A stereo host request must trace and retain two audio channels."""
        request = {
            "request_id": "stereo-worker-test",
            "operation": "freeze_selection",
            "selected_element_ids": [1, 2],
            "graph_fragment": {
                "elements": [
                    {"id": 1, "type": "audio_input", "properties": []},
                    {"id": 2, "type": "audio_output", "properties": []},
                ],
                "connections": [
                    {"source_element_id": 1, "destination_element_id": 2}
                ],
                "io_boundary": {"inputs": [], "outputs": []},
            },
            "compile_options": {
                "mode": "manual_freeze",
                "host_input_channels": 2,
                "host_output_channels": 2,
                "example_samples": 32,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            response = freeze_worker.compile_request(request, Path(directory))
            module = torch.jit.load(response["artifact_path"])
            output = module(torch.zeros(1, 2, 32))
            signature = response["blackbox_metadata"]["shape_signature"]
            self.assertEqual(tuple(output.shape), (1, 2, 32))
            self.assertEqual(signature, {"input_channels": 2, "output_channels": 2})

    def test_cli_keeps_json_separate_from_stderr_warnings(self) -> None:
        """CLI warnings on stderr must not corrupt its JSON stdout contract."""
        request = {
            "request_id": "cli-worker-test",
            "operation": "freeze_selection",
            "selected_element_ids": [1],
            "graph_fragment": {
                "elements": [{"id": 1, "type": "audio_input", "properties": []}],
                "connections": [],
                "io_boundary": {"inputs": [], "outputs": []},
            },
            "compile_options": {
                "mode": "manual_freeze",
                "host_input_channels": 1,
                "host_output_channels": 1,
                "example_samples": 16,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            request_path = Path(directory) / "request.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(WORKER_PATH),
                    "--request",
                    str(request_path),
                    "--artifact-dir",
                    directory,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            response = json.loads(completed.stdout.strip().splitlines()[-1])
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(response["status"], "success")

    def test_rejects_cyclic_selection(self) -> None:
        """A cycle must fail before tracing or artifact publication."""
        fragment = {
            "elements": [
                {"id": 1, "type": "activation", "properties": []},
                {"id": 2, "type": "activation", "properties": []},
            ],
            "connections": [
                {"source_element_id": 1, "destination_element_id": 2},
                {"source_element_id": 2, "destination_element_id": 1},
            ],
        }
        with self.assertRaisesRegex(ValueError, "cyclic"):
            freeze_worker.build_module(fragment)

    def test_frozen_tcn_keeps_live_control(self) -> None:
        """A TCN freeze with cond_dim must stay steerable by Knob/XY."""
        request = {
            "request_id": "film-freeze-test",
            "operation": "freeze_selection",
            "selected_element_ids": [1],
            "graph_fragment": {
                "elements": [
                    {
                        "id": 1,
                        "type": "tcn",
                        "seed": 23,
                        "properties": [
                            {"key": "channels", "value": 4},
                            {"key": "depth", "value": 1},
                            {"key": "kernel_size", "value": 3},
                            {"key": "dilation", "value": 1},
                            {"key": "dilation_growth", "value": 2},
                            {"key": "residual", "value": 0},
                            {"key": "activation", "value": 0},
                        ],
                    }
                ],
                "connections": [],
                "io_boundary": {"inputs": [], "outputs": []},
            },
            "compile_options": {
                "mode": "manual_freeze",
                "host_input_channels": 2,
                "host_output_channels": 2,
                "example_samples": 32,
                "conditioning": True,
                "cond_dim": 1,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            response = freeze_worker.compile_request(request, Path(directory))
            self.assertTrue(response["blackbox_metadata"]["conditioning"])
            self.assertEqual(response["blackbox_metadata"]["cond_dim"], 1)
            module = torch.jit.load(response["artifact_path"])
            audio = torch.ones(1, 2, 32)
            out_zero = module(audio, torch.zeros(1, 1, 32))
            out_one = module(audio, torch.ones(1, 1, 32))
        self.assertEqual(tuple(out_zero.shape), (1, 2, 32))
        self.assertFalse(torch.allclose(out_zero, out_one, atol=1.0e-5))


if __name__ == "__main__":
    unittest.main()
