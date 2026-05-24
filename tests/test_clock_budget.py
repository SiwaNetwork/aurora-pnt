"""
Тесты часового бюджета (§8 техпроекта, после аудита σ_t).
"""
import math
import pytest


def sigma_t_white_fm(sigma_0: float, tau: float) -> float:
    """
    Время-ошибка для белого FM-шума: σ_t = σ_0 · √τ.

    Исправлено в аудите формул (commit 2243bff): убран двойной множитель τ.
    """
    return sigma_0 * math.sqrt(tau)


def test_sigma_t_at_100s():
    """Cs-эталон σ_0=3e-11, τ=100с → σ_t = 300 пс (не 30 пс как было)."""
    sigma = sigma_t_white_fm(3e-11, 100.0)
    # σ_t = 3·10⁻¹¹ × 10 = 3·10⁻¹⁰ с = 300 пс
    assert sigma == pytest.approx(3e-10, rel=0.01)
    sigma_ps = sigma * 1e12
    assert sigma_ps == pytest.approx(300.0, rel=0.01)


def test_sigma_t_at_3600s():
    """Cs τ=3600с → σ_t = 1,8 нс (не 6,5 нс как было)."""
    sigma = sigma_t_white_fm(3e-11, 3600.0)
    # σ_t = 3·10⁻¹¹ × 60 = 1,8 нс
    assert sigma == pytest.approx(1.8e-9, rel=0.01)


@pytest.mark.parametrize("sigma_y, tau, expected_ns", [
    (1e-11, 100, 0.1),     # Cs class
    (1e-12, 100, 0.01),    # H-maser class
    (1e-10, 100, 1.0),     # Rb class
    (1e-9, 100, 10.0),     # OCXO
])
def test_clock_classes(sigma_y, tau, expected_ns):
    """Параметризованные тесты для разных классов часов."""
    sigma_s = sigma_t_white_fm(sigma_y, tau)
    sigma_ns = sigma_s * 1e9
    assert sigma_ns == pytest.approx(expected_ns, rel=0.05)


def test_isl_chain_grows_as_sqrt_N():
    """σ_ISL для цепочки N звеньев растёт как √N."""
    sigma_hop = 1e-9  # 1 нс per hop (code-based)
    chain_lengths = [1, 4, 9, 16, 25]
    for N in chain_lengths:
        sigma_chain = math.sqrt(N) * sigma_hop
        # Цепочка N=16 даёт 4 нс, N=25 даёт 5 нс — √N закон
        assert sigma_chain == pytest.approx(math.sqrt(N) * 1e-9, rel=1e-6)


def test_cs_holdover_24h_acceptable():
    """Cs σ_0=1e-11, holdover 24 ч → σ_t < 5 нс."""
    sigma = sigma_t_white_fm(1e-11, 86400.0)
    sigma_ns = sigma * 1e9
    # √86400 ≈ 294 → σ_t ≈ 2,94 нс
    assert sigma_ns < 5.0
    assert sigma_ns == pytest.approx(2.94, rel=0.05)
