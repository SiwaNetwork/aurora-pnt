"""
Smoke-тесты: каждый ключевой модуль запускается через run_X_analysis и
создаёт необходимое число PNG/CSV.

Все тесты:
- быстрые (< 30 с каждый)
- используют tmp_results фикстуру
- проверяют что нет исключений и количество выходных файлов
"""
import os
from pathlib import Path

import pytest

from tests.conftest import assert_png_exists, assert_csv_valid


# ── Базовые расчётные модули ───────────────────────────────────────────────


@pytest.mark.smoke
def test_radiation_runs(tmp_results):
    from aurora.pnt.radiation import run_radiation_analysis
    r = run_radiation_analysis(tmp_results, "test")
    assert r is not None
    # Должно быть ≥ 3 PNG
    pngs = list(Path(tmp_results).glob("*.png"))
    assert len(pngs) >= 3, f"Expected ≥3 PNG, got {len(pngs)}"


@pytest.mark.smoke
def test_relativistic_runs(tmp_results):
    from aurora.pnt.relativistic import run_relativistic_analysis
    r = run_relativistic_analysis(tmp_results, "test")
    assert r is not None
    pngs = list(Path(tmp_results).glob("*.png"))
    assert len(pngs) >= 3


@pytest.mark.smoke
def test_troposphere_runs(tmp_results):
    from aurora.pnt.troposphere import run_troposphere_analysis
    r = run_troposphere_analysis(tmp_results, "test")
    assert r is not None
    pngs = list(Path(tmp_results).glob("*.png"))
    assert len(pngs) >= 3


@pytest.mark.smoke
def test_reliability_runs(tmp_results):
    from aurora.pnt.reliability import run_reliability_analysis
    r = run_reliability_analysis(tmp_results, "test")
    assert r is not None
    pngs = list(Path(tmp_results).glob("*.png"))
    assert len(pngs) >= 3


# ── Phase 2 modules ────────────────────────────────────────────────────────


@pytest.mark.smoke
def test_pvt_montecarlo_runs(tmp_results):
    from aurora.pnt.pvt_montecarlo import run_pvt_montecarlo_analysis
    r = run_pvt_montecarlo_analysis(tmp_results, "test")
    assert r is not None
    pngs = list(Path(tmp_results).glob("*.png"))
    assert len(pngs) >= 3


@pytest.mark.smoke
def test_dop_temporal_runs(tmp_results):
    from aurora.pnt.dop_temporal import run_dop_temporal_analysis
    r = run_dop_temporal_analysis(tmp_results, "test")
    assert r is not None
    pngs = list(Path(tmp_results).glob("*.png"))
    assert len(pngs) >= 3


@pytest.mark.smoke
def test_araim_runs(tmp_results):
    from aurora.pnt.araim import run_araim_analysis
    r = run_araim_analysis(tmp_results, "test")
    assert r is not None
    pngs = list(Path(tmp_results).glob("*.png"))
    assert len(pngs) >= 3


@pytest.mark.smoke
def test_cost_model_runs(tmp_results):
    from aurora.pnt.cost_model import run_cost_analysis
    r = run_cost_analysis(tmp_results, "test")
    assert r is not None
    # LCC должен быть в разумном диапазоне (50–200 млрд ₽)
    lcc_7y = r.get("lcc_7y", None) or r.get("LCC_7_лет", None) or 0
    if lcc_7y:
        assert 50_000 < lcc_7y < 250_000, f"LCC 7 лет = {lcc_7y} млн ₽ вне диапазона"


# ── Phase 3 modules (risks/schedule/cyber/e2e) ────────────────────────────


@pytest.mark.smoke
def test_risks_runs(tmp_results):
    from aurora.pnt.risks import run_risks_analysis
    r = run_risks_analysis(tmp_results, "test")
    assert r is not None
    pngs = list(Path(tmp_results).glob("*.png"))
    assert len(pngs) >= 3


@pytest.mark.smoke
def test_schedule_runs(tmp_results):
    from aurora.pnt.schedule import run_schedule_analysis
    r = run_schedule_analysis(tmp_results, "test")
    assert r is not None
    pngs = list(Path(tmp_results).glob("*.png"))
    assert len(pngs) >= 3


@pytest.mark.smoke
def test_cybersec_runs(tmp_results):
    from aurora.pnt.cybersec_threat import run_cybersec_analysis
    r = run_cybersec_analysis(tmp_results, "test")
    assert r is not None
    pngs = list(Path(tmp_results).glob("*.png"))
    assert len(pngs) >= 3


@pytest.mark.smoke
@pytest.mark.slow
def test_e2e_pipeline_runs(tmp_results):
    """E2E PVT pipeline — slow test (~10 с)."""
    from aurora.pnt.e2e_pipeline import run_e2e_pipeline_analysis
    r = run_e2e_pipeline_analysis(tmp_results, "test")
    assert r is not None


# ── Phase 4 modules (prototyping) ──────────────────────────────────────────


@pytest.mark.smoke
@pytest.mark.slow
def test_code_gen_runs(tmp_results):
    """Code generator — slow test (~10 с, FFT по 10к точек)."""
    from aurora.pnt.code_gen import run_code_gen_analysis
    r = run_code_gen_analysis(tmp_results, "test")
    assert r is not None


# ── Phase 5 modules (real data + validation) ──────────────────────────────


@pytest.mark.smoke
def test_real_data_runs(tmp_results):
    from aurora.pnt.real_data import run_real_data_analysis
    r = run_real_data_analysis(tmp_results, "test")
    assert r is not None


@pytest.mark.smoke
def test_validate_runs(tmp_results):
    from aurora.pnt.validate_models import run_validate_analysis
    r = run_validate_analysis(tmp_results, "test")
    assert r is not None
    pngs = list(Path(tmp_results).glob("*.png"))
    assert len(pngs) >= 3
