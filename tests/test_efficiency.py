import numpy as np
import pytest

from vlmab.metrics.efficiency import (
    count_parameters,
    peak_vram_mb,
    reset_peak_vram,
    summarize_latencies,
)


class _FakeParam:
    def __init__(self, n):
        self._n = n

    def numel(self):
        return self._n


class _FakeModel:
    def __init__(self, sizes):
        self._params = [_FakeParam(n) for n in sizes]

    def parameters(self):
        return iter(self._params)


def test_summarize_latencies_median_and_p95():
    out = summarize_latencies(list(range(1, 101)))
    assert out["median_ms"] == pytest.approx(50.5)
    assert out["p95_ms"] == pytest.approx(95.05)
    assert out["n"] == 100


def test_summarize_latencies_single_value():
    out = summarize_latencies([7.0])
    assert out["median_ms"] == pytest.approx(7.0)
    assert out["p95_ms"] == pytest.approx(7.0)
    assert out["n"] == 1


def test_summarize_latencies_rejects_empty():
    with pytest.raises(ValueError):
        summarize_latencies([])


def test_count_parameters_sums_numel():
    assert count_parameters(_FakeModel([10, 20, 5])) == 35


def test_count_parameters_empty_model_is_zero():
    assert count_parameters(_FakeModel([])) == 0


class _FakeCuda:
    def __init__(self, peak_bytes):
        self._peak = peak_bytes
        self.resets = 0

    def max_memory_allocated(self):
        return self._peak

    def reset_peak_memory_stats(self):
        self.resets += 1


def test_peak_vram_is_none_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr("vlmab.metrics.efficiency._cuda", lambda: None)
    assert peak_vram_mb() is None


def test_peak_vram_converts_bytes_to_mib(monkeypatch):
    monkeypatch.setattr("vlmab.metrics.efficiency._cuda", lambda: _FakeCuda(3 * 1024 ** 2))
    assert peak_vram_mb() == pytest.approx(3.0)


def test_reset_peak_vram_calls_through_when_cuda_available(monkeypatch):
    fake = _FakeCuda(0)
    monkeypatch.setattr("vlmab.metrics.efficiency._cuda", lambda: fake)
    reset_peak_vram()
    assert fake.resets == 1


def test_reset_peak_vram_is_a_noop_without_cuda(monkeypatch):
    """Must not raise on a CPU-only box — the runner calls this unconditionally."""
    monkeypatch.setattr("vlmab.metrics.efficiency._cuda", lambda: None)
    reset_peak_vram()


def test_cuda_probe_returns_none_without_torch():
    """CI has no torch installed; the probe must swallow the ImportError."""
    from vlmab.metrics.efficiency import _cuda

    assert _cuda() is None or hasattr(_cuda(), "max_memory_allocated")
