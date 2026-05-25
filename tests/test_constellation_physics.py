"""
Физические инварианты созвездия AURORA.

Проверяет что базовые орбитальные формулы дают значения, согласованные с
техпроектом §4 и §40.
"""
import math
import pytest


def test_semimajor_axis(aurora_constants):
    a_calc = aurora_constants["R_earth_km"] + aurora_constants["altitude_km"]
    assert a_calc == pytest.approx(aurora_constants["semimajor_km"], rel=1e-6)


def test_orbital_period(aurora_constants):
    a_m = aurora_constants["semimajor_km"] * 1000.0
    T = 2 * math.pi * math.sqrt(a_m**3 / aurora_constants["mu_earth"])
    assert T == pytest.approx(aurora_constants["T_orbit_s"], rel=0.01)
    # ~105 мин по §4
    assert 6200 < T < 6400


def test_orbital_velocity(aurora_constants):
    a_m = aurora_constants["semimajor_km"] * 1000.0
    v = math.sqrt(aurora_constants["mu_earth"] / a_m) / 1000.0  # км/с
    assert v == pytest.approx(aurora_constants["v_orbit_kmps"], rel=0.01)


def test_mean_motion(aurora_constants):
    a_m = aurora_constants["semimajor_km"] * 1000.0
    n = math.sqrt(aurora_constants["mu_earth"] / a_m**3)
    assert n == pytest.approx(aurora_constants["n_mean_motion"], rel=0.01)


def test_walker_delta_structure(aurora_constants):
    """Walker 300/15: 15 плоскостей × 20 спутников = 300."""
    n = aurora_constants["n_planes"] * aurora_constants["n_per_plane"]
    assert n == aurora_constants["n_sats"]


def test_raan_spacing_deg(aurora_constants):
    """Δ RAAN = 360° / 15 плоскостей = 24°."""
    delta_raan = 360.0 / aurora_constants["n_planes"]
    assert delta_raan == pytest.approx(24.0, abs=0.01)


def test_phase_spacing_within_plane(aurora_constants):
    """Δu внутри плоскости = 360° / 20 спутников = 18°."""
    delta_u = 360.0 / aurora_constants["n_per_plane"]
    assert delta_u == pytest.approx(18.0, abs=0.01)


def test_phase_offset_walker_f1(aurora_constants):
    """F=1 phase offset between planes: 360° × F / T = 1.2°."""
    F = 1
    T = aurora_constants["n_sats"]
    delta_phi = 360.0 * F / T
    assert delta_phi == pytest.approx(1.2, abs=0.001)


def test_doppler_max_L1(aurora_constants):
    """Δf_max = (v / c) × f_0 для L1 ≈ 38,6 кГц."""
    c = 299792458.0
    v_ms = aurora_constants["v_orbit_kmps"] * 1000.0
    fd_max = v_ms / c * aurora_constants["freq_L1_Hz"]
    assert fd_max == pytest.approx(aurora_constants["doppler_max_Hz"], rel=0.01)


def test_j2_raan_precession_sign(aurora_constants):
    """J2 RAAN-прецессия для i=75° должна быть отрицательной (retrograde)."""
    i_rad = math.radians(aurora_constants["inclination_deg"])
    J2 = 1.0826e-3
    R_E_m = aurora_constants["R_earth_km"] * 1000.0
    a_m = aurora_constants["semimajor_km"] * 1000.0
    n = aurora_constants["n_mean_motion"]
    # Ω̇ = -1.5 n J2 (Re/a)^2 cos i
    omega_dot = -1.5 * n * J2 * (R_E_m / a_m) ** 2 * math.cos(i_rad)
    # Перевод рад/с → °/сут
    omega_dot_deg_per_day = math.degrees(omega_dot) * 86400.0
    # По §40 ожидаем ≈ −1,55 °/сут
    assert omega_dot_deg_per_day == pytest.approx(-1.55, rel=0.05)
    assert omega_dot_deg_per_day < 0  # retrograde
