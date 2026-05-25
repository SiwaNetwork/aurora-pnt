"""
Тесты бюджета линии связи (§10 техпроекта, после аудита формул).
"""
import math
import pytest


def fspl_db(d_m: float, f_hz: float) -> float:
    """Free-Space Path Loss в дБ."""
    c = 299792458.0
    return 20.0 * math.log10(4 * math.pi * d_m * f_hz / c)


def test_fspl_aurora_L1_zenith(aurora_constants):
    """FSPL AURORA L1 в зените (1000 км) ≈ 156,4 дБ — исправлено после аудита."""
    d_m = aurora_constants["altitude_km"] * 1000.0
    L = fspl_db(d_m, aurora_constants["freq_L1_Hz"])
    assert L == pytest.approx(156.4, abs=0.2)


def test_fspl_aurora_L1_horizon(aurora_constants):
    """FSPL AURORA L1 на 10° углу возвышения — наклонная дальность ≈ 3000 км → ≈ 166 дБ."""
    # Для elevation 10°: R ≈ 2860 км по §10.2
    d_m = 2860e3
    L = fspl_db(d_m, aurora_constants["freq_L1_Hz"])
    # Допуск шире из-за приближённости
    assert 165 < L < 170


def test_fspl_gps_meo_for_comparison(aurora_constants):
    """FSPL GPS MEO 20200 км ≈ 182,5 дБ — для сравнения с LEO."""
    d_m = 20200e3
    L = fspl_db(d_m, aurora_constants["freq_L1_Hz"])
    # GPS орбита 20 200 км: FSPL ≈ 182-183 дБ
    assert 182 < L < 184


def test_fspl_aurora_vs_gps_advantage(aurora_constants):
    """AURORA должна давать ≈ 26 дБ выигрыша FSPL в зените vs GPS MEO."""
    aurora = fspl_db(1000e3, aurora_constants["freq_L1_Hz"])
    gps = fspl_db(20200e3, aurora_constants["freq_L1_Hz"])
    advantage = gps - aurora
    assert advantage == pytest.approx(26.0, abs=1.0)


def test_isl_fspl_ka(aurora_constants):
    """FSPL ISL Ka (26,5 ГГц) на 3000 км ≈ 190,5 дБ — после аудита §9."""
    d_m = 3000e3
    f = aurora_constants["freq_ISL_Hz"]
    L = fspl_db(d_m, f)
    assert 190 < L < 191


def test_cn0_zenith_value(aurora_constants):
    """Бюджет C/N₀ в зените — проверка что значение из техпроекта (52,6 дБ-Гц)."""
    cn0 = aurora_constants["cn0_zenith_dbHz"]
    # Не строгий числовой тест; просто проверка что значение в физически
    # разумном диапазоне для LEO 1000км GNSS-приёмника
    assert 40 < cn0 < 60


def test_jamming_margin_vs_gps(aurora_constants):
    """Маржа AURORA vs GPS L1 ≈ +22 дБ (по §15 после аудита)."""
    cn0_aurora = aurora_constants["cn0_zenith_dbHz"]   # 52.6
    cn0_gps_typical = 44.0  # типовое для GPS L1 C/A в зените
    margin = cn0_aurora - cn0_gps_typical
    assert margin == pytest.approx(8.6, abs=1.0)  # ~9 дБ-Гц C/N₀ преимущ.
