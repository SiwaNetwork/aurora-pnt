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


# ── Phase 6 modules (A-GNSS, точность, криптозащита) ──────────────────────────


@pytest.mark.smoke
def test_agps_server_runs(tmp_results):
    from aurora.pnt.agps_server import run_agps_server_analysis
    r = run_agps_server_analysis(tmp_results, "test")
    assert r is not None
    assert_png_exists(tmp_results, "agps_ttff_test.png")
    assert_csv_valid(tmp_results, "agps_server_test.csv")
    # A-GNSS должен ускорять холодный старт минимум на порядок
    assert r["agps"]["total_s"] < r["cold"]["total_s"]
    assert r["speedup"] > 10, f"ускорение TTFF {r['speedup']:.1f}x < 10x"


@pytest.mark.smoke
def test_accuracy_paths_runs(tmp_results):
    from aurora.pnt.accuracy_paths import run_accuracy_paths_analysis
    r = run_accuracy_paths_analysis(tmp_results, "test")
    assert r is not None
    assert_png_exists(tmp_results, "accuracy_paths_test.png")
    assert_csv_valid(tmp_results, "accuracy_paths_test.csv")
    rows = r["rows"]
    # UERE и H-95 монотонно убывают от базового к улучшенному сценарию
    ueres = [x["uere"] for x in rows]
    assert ueres == sorted(ueres, reverse=True), f"UERE не убывает: {ueres}"
    assert rows[-1]["uere"] < rows[0]["uere"]
    # фазовый PPP-RTK — отдельный «этаж» точнее любого псевдодальностного H-95
    assert r["ppp_rtk_h95"] < rows[-1]["h95"]


@pytest.mark.smoke
def test_crypto_auth_runs(tmp_results):
    from aurora.pnt.crypto_auth import run_crypto_auth_analysis
    r = run_crypto_auth_analysis(tmp_results, "test")
    assert r is not None
    assert_png_exists(tmp_results, "crypto_auth_test.png")
    assert_csv_valid(tmp_results, "crypto_auth_test.csv")
    # базовый профиль (тег 128 б, интервал 30 с): оверхед мал относительно канала
    base = next(x for x in r["rows"]
                if x["tag_bits"] == 128 and x["interval_s"] == 30)
    assert base["pct_nav"] < 10.0, f"оверхед {base['pct_nav']:.1f}% > 10% канала"
    assert base["forge_prob_log2"] == -128
    # связка ГОСТ покрывает все пять криптофункций
    assert len(r["suite"]) == 5


@pytest.mark.smoke
def test_reference_frame_runs(tmp_results):
    from aurora.pnt.reference_frame import run_reference_frame_analysis
    r = run_reference_frame_analysis(tmp_results, "test")
    assert r is not None
    assert_png_exists(tmp_results, "reference_frame_test.png")
    assert_csv_valid(tmp_results, "reference_frame_test.csv")
    # ПЗ-90.11 совмещён с ITRF2008 на см-уровне (сдвиг < 1 см)
    assert r["datum_shift_m"] < 0.01, f"датум-сдвиг {r['datum_shift_m']*1000:.1f} мм > 10 мм"
    # EOP-ошибка монотонно растёт с ошибкой прогноза UT1
    tot = [row["total_m"] for row in r["eop_rows"]]
    assert tot == sorted(tot), f"EOP-ошибка не монотонна: {tot}"


@pytest.mark.smoke
def test_tgd_dcb_runs(tmp_results):
    from aurora.pnt.tgd_dcb import run_tgd_dcb_analysis
    r = run_tgd_dcb_analysis(tmp_results, "test")
    assert r is not None
    assert_png_exists(tmp_results, "tgd_dcb_test.png")
    assert_csv_valid(tmp_results, "tgd_dcb_test.csv")
    # двухчастотный остаток TGD/DCB укладывается в часовой член бюджета (0,20 м)
    assert r["dual_uere_m"] < 0.20, f"двухчастотный вклад TGD {r['dual_uere_m']:.2f} м > 0,20 м"
    # сама задержка существенно больше остатка → учёт обязателен
    assert r["tgd_range_m"] > r["dual_uere_m"]


@pytest.mark.smoke
def test_dual_service_runs(tmp_results):
    from aurora.pnt.dual_service import run_dual_service_analysis
    r = run_dual_service_analysis(tmp_results, "test")
    assert r is not None
    assert_png_exists(tmp_results, "dual_service_test.png")
    assert_csv_valid(tmp_results, "dual_service_test.csv")
    a = r["А (открытый, RNSS)"]
    b = r["Б (защищённый, своя полоса)"]
    # Сервис А — комплаентен маске ПФП МСЭ; Сервис Б — мощнее (но в своей полосе)
    assert a["itu_ok"], f"Сервис А ПФП {a['pfd']:.1f} нарушает маску МСЭ"
    assert b["adv_db"] > a["adv_db"], "Сервис Б должен быть сильнее А"
    # Сервис Б достигает целевого преимущества ≈ +23 дБ (×~200)
    assert 21.0 <= b["adv_db"] <= 25.0, f"Сервис Б преимущество {b['adv_db']:.1f} дБ вне [21,25]"
    assert b["jam_x"] > 100, f"Сервис Б помехозащита ×{b['jam_x']:.0f} < 100"
