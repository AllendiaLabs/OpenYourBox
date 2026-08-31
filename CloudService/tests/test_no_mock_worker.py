"""Assert the product has no fake-success advancement worker."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cloud_mock_worker_flag_is_absent() -> None:
    matches: list[str] = []
    for path in ROOT.rglob("*.py"):
        if path.name == "test_no_mock_worker.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "CLOUD_MOCK_WORKER" in text:
            matches.append(str(path.relative_to(ROOT)))
    assert matches == [], f"CLOUD_MOCK_WORKER still referenced in {matches}"


def test_no_mock_worker_module_or_tick_api() -> None:
    assert not (ROOT / "worker" / "mock_worker.py").exists()
    for path in ROOT.rglob("*.py"):
        if path.name == "test_no_mock_worker.py":
            continue
        text = path.read_text(encoding="utf-8")
        assert "MOCK_WORKER" not in text
        assert "mock_worker" not in text
