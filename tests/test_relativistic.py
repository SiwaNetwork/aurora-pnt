"""
Тесты релятивистских поправок (§33 техпроекта).
"""
import math
import pytest


GM = 3.986e14   # м³/с²
c = 299792458.0  # м/с
R_E = 6.371e6    # м (средний)


def gravitational_redshift_leo(altitude_m: float) -> float:
    """Δf/f|_grav = GM/c² × (1/R_E - 1/r) для LEO над уровнем моря."""
    r = R_E + altitude_m
    return GM / c**2 * (1.0 / R_E - 1.0 / r)


def second_order_doppler(velocity_ms: float) -> float:
    """Δf/f|_Doppler2 = -v² / (2c²)."""
    return -(velocity_ms**2) / (2 * c**2)


def test_grav_redshift_LEO_1000_positive(aurora_constants):
    """Для LEO 1000 км: Δf/f|_grav > 0 (часы идут быстрее на высоте)."""
    h = aurora_constants["altitude_km"] * 1000.0
    df_f = gravitational_redshift_leo(h)
    assert df_f > 0
    # По §33 ожидаем ≈ +0.094 ppb = 9.4e-11
    assert df_f == pytest.approx(9.44e-11, rel=0.05)


def test_grav_redshift_LEO_offset_per_day(aurora_constants):
    """Гравитационная поправка по §33: ≈ +8,16 мкс/день для 1000 км."""
    h = aurora_constants["altitude_km"] * 1000.0
    df_f = gravitational_redshift_leo(h)
    seconds_per_day = 86400.0
    offset_us_per_day = df_f * seconds_per_day * 1e6
    assert offset_us_per_day == pytest.approx(8.16, rel=0.1)


def test_doppler2_LEO_negative(aurora_constants):
    """2-й порядок Доплера для LEO: отрицательный (часы идут медленнее)."""
    v = aurora_constants["v_orbit_kmps"] * 1000.0
    df_f = second_order_doppler(v)
    assert df_f < 0


def test_doppler2_LEO_magnitude(aurora_constants):
    """|Δf/f|_Dopp2 для v=7,35 км/с ≈ 3·10⁻¹⁰."""
    v = aurora_constants["v_orbit_kmps"] * 1000.0
    df_f = second_order_doppler(v)
    # v²/(2c²) ≈ (7350)²/(2 × (3e8)²) ≈ 3·10⁻¹⁰
    expected = -3.005e-10
    assert df_f == pytest.approx(expected, rel=0.02)


def test_total_relativistic_LEO_clock_runs_slow():
    """Чистый эффект для LEO 1000 км: суммарный сдвиг ОТРИЦАТЕЛЬНЫЙ
    (часы LEO идут медленнее наземных), по §33 −17.8 мкс/сут."""
    h = 1000e3
    v = 7350.0
    df_grav = gravitational_redshift_leo(h)
    df_dopp = second_order_doppler(v)
    df_total = df_grav + df_dopp
    # Доплер 2-го порядка доминирует над grav: суммарно отрицательно для LEO
    assert df_total < 0
    # Поправка ~−2,06×10⁻¹⁰ → ~−17,8 мкс/сут
    seconds_per_day = 86400.0
    us_per_day = df_total * seconds_per_day * 1e6
    assert us_per_day == pytest.approx(-17.8, abs=1.0)


def test_sagnac_correction_sign():
    """Эффект Саньяка: δt = 2Ω·A / c² > 0 для положительной площади."""
    omega = 7.292e-5  # рад/с угловая скорость вращения Земли
    A_proj = 1.0e10   # м² проекция площади (test value, positive)
    delta_t = 2 * omega * A_proj / c**2
    assert delta_t > 0
