"""
Поддержание орбиты (Station Keeping) АВРОРА.

Рассчитывает возмущения орбиты и необходимый ΔV для Walker Delta 300/15/1
на 1000 км / 75°:
  - J2-прецессия RAAN: Ω̇ = -3/2 · n · J₂ · (R_E/a)² · cos(i) / (1-e²)²
  - Атмосферное торможение: da/dt = -C_D · (A/m) · ρ · v² / n
  - Солнечное давление: da/dt_srp
  - Относительный дрейф спутников одной плоскости
  - Бюджет ΔV за 7 лет (тяговые манёвры) и расчёт запаса топлива

Ссылки:
  Vallado (2013) — Fundamentals of Astrodynamics and Applications.
  Wertz & Larson (2011) — Space Mission Engineering: The New SMAD.
  ECSS-E-ST-10-04C (2008) — Space Environment.
"""

import math, os, csv
from typing import Dict, List, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Физические константы ──────────────────────────────────────────────────────
MU        = 3.986004418e14   # м³/с²
R_E       = 6_371_000.0      # м
J2        = 1.082626e-3      # безразмерный
J4        = -1.6199e-6
C_LIGHT   = 299_792_458.0
P_SOLAR   = 4.56e-6          # Па (солнечное давление на 1 а.е.)

# ── Параметры орбиты АВРОРА ───────────────────────────────────────────────────
ALT_M     = 1_000_000.0      # 1000 км
INC_DEG   = 75.0
ECC       = 0.001            # почти круговая
A_M       = R_E + ALT_M     # большая полуось

# ── Параметры спутника ────────────────────────────────────────────────────────
MASS_KG   = 120.0
A_DRAG_M2 = 0.8    # поперечное сечение аэродинамики
CD        = 2.2
A_SRP_M2  = 1.5    # площадь под давление
CR        = 1.5

# ── Плотность атмосферы (NRLMSISE-00, высота 1000 км) ────────────────────────
# Варианты: солнечный минимум, средняя, максимум
RHO_MIN_KG_M3  = 2e-15
RHO_MEAN_KG_M3 = 3e-15
RHO_MAX_KG_M3  = 1e-14

MISSION_YEARS = 7.0
SECONDS_PER_YEAR = 365.25 * 86400.0


def orbital_velocity_m_s(a: float = A_M) -> float:
    return math.sqrt(MU / a)


def mean_motion_rad_s(a: float = A_M) -> float:
    return math.sqrt(MU / a**3)


def raan_drift_rate_deg_day(a: float = A_M, inc_deg: float = INC_DEG,
                             ecc: float = ECC) -> float:
    """
    RAAN прецессия из-за J2.
    Ω̇ = -3/2 · n · J₂ · (R_E/a)² · cos(i) / (1-e²)²
    """
    n   = mean_motion_rad_s(a)
    denom = (1 - ecc**2)**2
    omega_dot_rad_s = -1.5 * n * J2 * (R_E / a)**2 * math.cos(math.radians(inc_deg)) / denom
    return math.degrees(omega_dot_rad_s) * 86400   # °/сут


def altitude_decay_rate_m_s(rho_kg_m3: float = RHO_MEAN_KG_M3,
                              a: float = A_M) -> float:
    """
    da/dt = -C_D · (A/m) · ρ · v²
    (в м/с)
    """
    v = orbital_velocity_m_s(a)
    return -CD * (A_DRAG_M2 / MASS_KG) * rho_kg_m3 * v**2


def srp_altitude_effect_m_s(a: float = A_M) -> float:
    """
    Влияние солнечного давления на большую полуось (усреднённое за орбиту).
    da/dt_srp ≈ -2 · CR · (A_srp/m) · P_solar · a / v  (в м/с, знак переменный)
    """
    v = orbital_velocity_m_s(a)
    return 2 * CR * (A_SRP_M2 / MASS_KG) * P_SOLAR * a / v


def dv_drag_m_s_yr(rho_kg_m3: float = RHO_MEAN_KG_M3) -> float:
    """ΔV/год для компенсации атмосферного торможения."""
    da_dt = altitude_decay_rate_m_s(rho_kg_m3)
    n = mean_motion_rad_s()
    v = orbital_velocity_m_s()
    # ΔV = -da/dt · n / (2·v) · T_yr ... но проще: ΔV = v_loss
    # Потеря скорости за год: Δv ≈ n · |da/dt| / 2
    return abs(da_dt) * n * SECONDS_PER_YEAR / 2


def dv_raan_correction_m_s_yr(delta_h_m: float = 500.0,
                                a: float = A_M) -> float:
    """
    ΔV/год для компенсации дифференциального дрейфа RAAN из-за разброса высот.
    В Walker Delta все плоскости дрейфуют с одной скоростью — абсолютный дрейф
    не компенсируется. Нужна только коррекция относительного фазового разброса.

    Оценка: типовое требование < 5 коррекций по ~1 м/с в год.
    """
    # dΩ̇/da = Ω̇ × (-7/2) / a  (логарифмическое дифференцирование)
    omega_dot = raan_drift_rate_deg_day(a)  # °/сут
    omega_dot_rad_s = math.radians(omega_dot) / 86400
    d_omega_da_rad_s_per_m = omega_dot_rad_s * (-3.5) / a
    # Дифференциальный дрейф RAAN за год при разбросе высот ΔH
    delta_raan_rad_yr = abs(d_omega_da_rad_s_per_m * delta_h_m) * SECONDS_PER_YEAR
    delta_raan_deg_yr = math.degrees(delta_raan_rad_yr)
    # Допустимый разброс RAAN между спутниками плоскости: 0.05°
    allowable_deg = 0.05
    if delta_raan_deg_yr < allowable_deg:
        return 0.0
    n_corr = math.ceil(delta_raan_deg_yr / allowable_deg)
    return n_corr * 1.0   # ~1 м/с на коррекцию (малые манёвры)


def dv_total_7yr() -> Dict[str, float]:
    """Суммарный бюджет ΔV за 7 лет."""
    dv_drag_min  = dv_drag_m_s_yr(RHO_MIN_KG_M3)  * MISSION_YEARS
    dv_drag_mean = dv_drag_m_s_yr(RHO_MEAN_KG_M3) * MISSION_YEARS
    dv_drag_max  = dv_drag_m_s_yr(RHO_MAX_KG_M3)  * MISSION_YEARS
    dv_raan      = dv_raan_correction_m_s_yr() * MISSION_YEARS
    dv_disposal  = orbital_velocity_m_s() * 0.001  # де-орбита: Δv ≈ 0.1% v_orb ≈ 8 м/с
    return {
        "drag_min_m_s":   dv_drag_min,
        "drag_mean_m_s":  dv_drag_mean,
        "drag_max_m_s":   dv_drag_max,
        "raan_m_s":       dv_raan,
        "disposal_m_s":   dv_disposal,
        "total_mean_m_s": dv_drag_mean + dv_raan + dv_disposal,
        "total_max_m_s":  dv_drag_max + dv_raan + dv_disposal,
    }


def propellant_mass_kg(dv_m_s: float, isp_s: float = 220.0) -> float:
    """Масса топлива по уравнению Циолковского: m_p = m_0 · (1 - e^(-ΔV/g0/Isp))."""
    g0 = 9.80665
    exp_term = math.exp(-dv_m_s / (g0 * isp_s))
    return MASS_KG * (1 - exp_term)


def relative_drift_m_per_day(da_m: float = 100.0) -> float:
    """
    Относительный дрейф спутников одной плоскости при разнице ΔA = da_m.
    Δv = -3/2 · n · Δa; интегрируем за сутки.
    """
    n = mean_motion_rad_s()
    dv_rel = 1.5 * n * da_m   # м/с
    return dv_rel * 86400      # м/сутки


def run_station_keeping_analysis(output_dir: str, label: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    years_range = np.linspace(0, MISSION_YEARS, 300)

    # Высотный дрейф при разной плотности
    alt_decay_min  = [ALT_M + altitude_decay_rate_m_s(RHO_MIN_KG_M3)  * y * SECONDS_PER_YEAR for y in years_range]
    alt_decay_mean = [ALT_M + altitude_decay_rate_m_s(RHO_MEAN_KG_M3) * y * SECONDS_PER_YEAR for y in years_range]
    alt_decay_max  = [ALT_M + altitude_decay_rate_m_s(RHO_MAX_KG_M3)  * y * SECONDS_PER_YEAR for y in years_range]

    # RAAN дрейф
    raan_rate = raan_drift_rate_deg_day()
    raan_drift = [raan_rate * y * 365.25 for y in years_range]

    dv_budget = dv_total_7yr()
    prop_mean = propellant_mass_kg(dv_budget["total_mean_m_s"])
    prop_max  = propellant_mass_kg(dv_budget["total_max_m_s"])

    _plot_altitude_decay(years_range, alt_decay_min, alt_decay_mean, alt_decay_max, output_dir, label)
    _plot_raan_drift(years_range, raan_drift, output_dir, label)
    _plot_dv_budget(dv_budget, output_dir, label)
    _plot_manoeuvre_schedule(output_dir, label)
    _save_csv(dv_budget, prop_mean, prop_max, raan_rate, output_dir, label)

    return {
        "raan_drift_deg_day":  raan_rate,
        "dv_budget":           dv_budget,
        "prop_mean_kg":        prop_mean,
        "prop_max_kg":         prop_max,
        "alt_decay_m_yr_mean": abs(altitude_decay_rate_m_s()) * SECONDS_PER_YEAR,
    }


def _plot_altitude_decay(years_range, alt_min, alt_mean, alt_max, output_dir, label):
    fig, ax = plt.subplots(figsize=(10, 6))
    alt_min_km  = np.array(alt_min)  / 1e3
    alt_mean_km = np.array(alt_mean) / 1e3
    alt_max_km  = np.array(alt_max)  / 1e3
    ax.plot(years_range, alt_mean_km, color="#0984e3", lw=2.5, label="Среднее (F10.7=150)")
    ax.fill_between(years_range, alt_min_km, alt_max_km,
                    alpha=0.2, color="#0984e3", label="Диапазон (солн. мин – макс)")
    ax.axhline(ALT_M / 1e3 - 20, ls="--", color="#e17055", lw=1.5,
               label="Порог коррекции высоты (-20 км)")
    ax.set_xlabel("Время (лет)")
    ax.set_ylabel("Высота орбиты (км)")
    ax.set_title(f"АВРОРА — Высотный дрейф без коррекции [{label}]")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"sk_alt_decay_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_raan_drift(years_range, raan_drift, output_dir, label):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(years_range, raan_drift, color="#6c5ce7", lw=2.5,
            label=f"Дрейф RAAN = {raan_drift_rate_deg_day():.4f}°/сут")

    # Для Walker Delta важно чтобы все плоскости дрейфовали одинаково
    # Отклонение от среднего (из-за разницы высот)
    da_100m = 100   # разброс высот в пределах плоскости, м
    dn = mean_motion_rad_s()
    draan_da = math.degrees(-3/2 * dn * J2 * (R_E / A_M)**2 *
                             math.cos(math.radians(INC_DEG)) / (1 - ECC**2)**2 *
                             (-2 / A_M)) * 86400
    drift_spread = [draan_da * da_100m * y * 365.25 for y in years_range]
    ax.fill_between(years_range,
                    np.array(raan_drift) - np.abs(drift_spread),
                    np.array(raan_drift) + np.abs(drift_spread),
                    alpha=0.2, color="#6c5ce7",
                    label=f"Разброс при ΔH=±{da_100m} м")

    ax.set_xlabel("Время (лет)")
    ax.set_ylabel("Дрейф RAAN (градусы)")
    ax.set_title(f"АВРОРА — Прецессия RAAN из-за J2 [{label}]")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"sk_raan_drift_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_dv_budget(dv_budget, output_dir, label):
    fig, ax = plt.subplots(figsize=(9, 5))
    items = {
        "Торможение\n(ср.)":  dv_budget["drag_mean_m_s"],
        "Торможение\n(макс.)": dv_budget["drag_max_m_s"],
        "Коррекция\nRAAN":    dv_budget["raan_m_s"],
        "Де-орбита":          dv_budget["disposal_m_s"],
        "Итого\n(среднее)":   dv_budget["total_mean_m_s"],
        "Итого\n(максимум)":  dv_budget["total_max_m_s"],
    }
    colors = ["#74b9ff", "#0984e3", "#6c5ce7", "#e17055", "#00b894", "#fdcb6e"]
    bars = ax.bar(list(items.keys()), list(items.values()),
                  color=colors, edgecolor="white", width=0.5)
    for bar, v in zip(bars, items.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("ΔV (м/с) за 7 лет")
    ax.set_title(f"АВРОРА — Бюджет ΔV поддержания орбиты [{label}]")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"sk_dv_budget_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_manoeuvre_schedule(output_dir, label):
    """График манёвров за 7 лет (примерное расписание)."""
    dv_yr = dv_drag_m_s_yr() + dv_raan_correction_m_s_yr()
    # Выполняем манёвры каждые 90 дней
    manoeuvre_interval_days = 90
    dv_per_manouevre = dv_yr / 365.25 * manoeuvre_interval_days

    days = np.arange(0, MISSION_YEARS * 365, 1)
    dv_cumulative = []
    alt_adjusted = []
    alt = ALT_M
    cum_dv = 0.0
    for d in days:
        alt += altitude_decay_rate_m_s() * 86400
        if d % manoeuvre_interval_days == 0 and d > 0:
            alt += dv_per_manouevre / mean_motion_rad_s() * 2  # приближение Хоманна
            cum_dv += dv_per_manouevre
        dv_cumulative.append(cum_dv)
        alt_adjusted.append(alt / 1e3)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1.plot(days / 365, alt_adjusted, color="#0984e3", lw=1.5)
    ax1.axhline(ALT_M / 1e3, ls="--", color="#b2bec3", lw=1, label="Номинальная высота")
    ax1.set_ylabel("Высота (км)")
    ax1.set_title(f"АВРОРА — Расписание манёвров поддержания орбиты [{label}]")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    ax2.plot(days / 365, dv_cumulative, color="#6c5ce7", lw=2)
    ax2.set_xlabel("Время (лет)")
    ax2.set_ylabel("Накопленный ΔV (м/с)")
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"sk_manoeuvre_schedule_{label}.png"), dpi=150)
    plt.close(fig)


def _save_csv(dv_budget, prop_mean, prop_max, raan_rate, output_dir, label):
    path = os.path.join(output_dir, f"station_keeping_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["parameter", "value", "unit"])
        w.writerow(["orbit_alt_km", ALT_M / 1e3, "км"])
        w.writerow(["inclination_deg", INC_DEG, "°"])
        w.writerow(["raan_drift_rate", f"{raan_rate:.4f}", "°/сут"])
        w.writerow(["alt_decay_mean", f"{abs(altitude_decay_rate_m_s())*SECONDS_PER_YEAR:.1f}", "м/год"])
        for key, val in dv_budget.items():
            w.writerow([key, f"{val:.2f}", "м/с"])
        w.writerow(["propellant_mean_kg", f"{prop_mean:.2f}", "кг"])
        w.writerow(["propellant_max_kg", f"{prop_max:.2f}", "кг"])


def print_station_keeping_summary(label: str, result: Dict) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  Station Keeping Analysis -- {label}")
    print(sep)
    dv = result["dv_budget"]
    print(f"  Прецессия RAAN (J2):        {result['raan_drift_deg_day']:.4f} °/сут")
    print(f"  Высотный дрейф (ср.):       {result['alt_decay_m_yr_mean']:.1f} м/год")
    print()
    print(f"  Бюджет ΔV за 7 лет:")
    print(f"    Торможение (среднее):  {dv['drag_mean_m_s']:>8.2f} м/с")
    print(f"    Торможение (максимум): {dv['drag_max_m_s']:>8.2f} м/с")
    print(f"    Коррекция RAAN:        {dv['raan_m_s']:>8.2f} м/с")
    print(f"    Де-орбита:             {dv['disposal_m_s']:>8.2f} м/с")
    print(f"    Итого (среднее):       {dv['total_mean_m_s']:>8.2f} м/с")
    print(f"    Итого (максимум):      {dv['total_max_m_s']:>8.2f} м/с")
    print()
    print(f"  Масса топлива (среднее): {result['prop_mean_kg']:.2f} кг")
    print(f"  Масса топлива (максимум):{result['prop_max_kg']:.2f} кг")
    print(sep)
