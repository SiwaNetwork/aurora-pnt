"""
Система управления ориентацией и стабилизацией (ADCS) АВРОРА.

Рассчитывает требования к ADCS для навигационного спутника LEO 1000 км:
  - Бюджет точности наведения антенны (pointing budget)
  - Возмущающие моменты: J2/гравитационный градиент, солнечное давление,
    аэродинамическое торможение, остаточный магнитный момент
  - Требования к маховикам (reaction wheels): момент, импульс, мощность
  - ISL point-ahead angle (Ka-диапазон, 26 ГГц)
  - Влияние ошибки ориентации на усиление антенны

Ссылки:
  Hughes (2004) — Spacecraft Attitude Dynamics.
  Wertz & Larson (2011) — Space Mission Engineering: The New SMAD.
  ECSS-E-ST-60-30C (2013) — Pointing Budget.
"""

import math, os, csv
from typing import Dict, List, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Параметры орбиты ──────────────────────────────────────────────────────────
ALT_M    = 1_000_000.0    # 1000 км
R_E      = 6_371_000.0    # радиус Земли
MU       = 3.986004418e14  # гравитационный параметр
OMEGA_E  = 7.292115e-5     # угловая скорость Земли
C        = 299_792_458.0

# ── Параметры спутника ────────────────────────────────────────────────────────
MASS_KG       = 120.0   # масса спутника
A_SOLAR_M2    = 1.5     # площадь солнечной панели (раскрытая)
A_DRAG_M2     = 0.8     # площадь лобового сечения
CD            = 2.2     # коэффициент аэродинамического торможения
CR            = 1.5     # коэффициент отражения солнечного давления
MOM_INERTIA_X = 80.0    # кг·м² (поперечная ось)
MOM_INERTIA_Y = 80.0
MOM_INERTIA_Z = 40.0    # кг·м² (продольная ось)
MAG_MOM_A_M2  = 1.0     # остаточный магнитный момент, А·м²

# ── Параметры антенны навигационного сигнала ─────────────────────────────────
ANTENNA_GAIN_dBi  = 13.0   # дБи
ANTENNA_BW3_DEG   = 20.0   # ширина ДН по -3 дБ (градусы)
ANTENNA_DIAM_M    = 0.25   # диаметр (м)

# ── Параметры ISL (Ka-диапазон) ───────────────────────────────────────────────
ISL_FREQ_GHZ   = 26.0    # ГГц
ISL_RANGE_KM   = 3000.0  # расстояние до соседнего спутника
ISL_BW3_DEG    = 2.0     # ширина ДН ISL антенны по -3 дБ

# ── Плотность атмосферы (1000 км, модель NRLMSISE-00) ────────────────────────
RHO_1000KM_KG_M3 = 3e-15   # кг/м³ (типовая для солнечного минимума)

# ── Солнечное давление ────────────────────────────────────────────────────────
P_SOLAR_PA = 4.56e-6   # Па (на 1 а.е.)


def orbital_velocity_m_s(alt_m: float = ALT_M) -> float:
    r = R_E + alt_m
    return math.sqrt(MU / r)


def orbital_rate_rad_s(alt_m: float = ALT_M) -> float:
    r = R_E + alt_m
    return math.sqrt(MU / r**3)


def gravity_gradient_torque_Nm(theta_deg: float = 1.0) -> float:
    """
    Гравитационный градиентный момент.
    T_gg = (3μ/2r³) × |Iz - Ix| × sin(2θ)
    """
    r = R_E + ALT_M
    n2 = MU / r**3
    dI = abs(MOM_INERTIA_Z - MOM_INERTIA_X)
    return 1.5 * n2 * dI * abs(math.sin(2 * math.radians(theta_deg)))


def solar_pressure_torque_Nm(moment_arm_m: float = 0.3) -> float:
    """
    T_srp = P_solar × CR × A_solar × d_arm
    """
    return P_SOLAR_PA * CR * A_SOLAR_M2 * moment_arm_m


def aerodynamic_torque_Nm(moment_arm_m: float = 0.1) -> float:
    """
    Аэродинамический момент: T_aero = 0.5 × ρ × v² × CD × A × d_arm
    """
    v = orbital_velocity_m_s()
    F = 0.5 * RHO_1000KM_KG_M3 * v**2 * CD * A_DRAG_M2
    return F * moment_arm_m


def magnetic_torque_Nm(altitude_km: float = 1000.0) -> float:
    """
    T_mag = m_res × B_earth
    B ≈ B0 × (R_E/r)³ × 2 (максимум у полюсов)
    B0 ≈ 3e-5 Тл на экваторе
    """
    B0 = 3e-5
    r = R_E + altitude_km * 1e3
    B = B0 * (R_E / r)**3 * 2.0
    return MAG_MOM_A_M2 * B


def total_disturbance_torque_Nm() -> float:
    return (gravity_gradient_torque_Nm() +
            solar_pressure_torque_Nm() +
            aerodynamic_torque_Nm() +
            magnetic_torque_Nm())


def isl_point_ahead_angle_urad() -> float:
    """
    Угол опережения для ISL (point-ahead angle).
    θ_pa = v_rel / c   (рад), затем в мкрад.
    v_rel ~ 2 × v_sat (для спутника в противоположном направлении в той же плоскости)
    """
    v = orbital_velocity_m_s()
    v_rel = 2 * v * math.sin(math.pi / 2)  # 90° разделение
    theta_rad = v_rel / C
    return theta_rad * 1e6  # мкрад


def pointing_budget() -> Dict:
    """
    Бюджет точности наведения навигационной антенны.
    Источники ошибок: star tracker, rate gyro, control error, flex, thermal.
    """
    return {
        "Звёздный датчик (3-sigma)":        0.015,   # градусов
        "Волоконный гироскоп":              0.010,
        "Ошибка управления (контроллер)":   0.010,
        "Термоупругая деформация":          0.020,
        "Гибкость конструкции":             0.015,
        "Неточность монтажа антенны":       0.010,
        "Резерв":                           0.010,
    }


def total_pointing_error_deg() -> float:
    budget = pointing_budget()
    rss = math.sqrt(sum(v**2 for v in budget.values()))
    return rss


def gain_loss_dB(pointing_error_deg: float) -> float:
    """Потеря усиления антенны при ошибке наведения δθ."""
    k = -12.0 / ANTENNA_BW3_DEG**2   # параболическое приближение
    return k * pointing_error_deg**2


def reaction_wheel_sizing() -> Dict:
    """
    Размер маховика: угловой импульс для компенсации возмущений за полуорбиту.
    H = T_dist × T_orbit/4
    """
    T_orbit = 2 * math.pi / orbital_rate_rad_s()
    T_dist  = total_disturbance_torque_Nm()
    H_req   = T_dist * T_orbit / 4   # Н·м·с
    Power_W = H_req * 0.01           # грубая оценка мощности
    return {
        "T_orbit_min":    T_orbit / 60,
        "T_disturb_Nm":   T_dist,
        "H_required_Nms": H_req,
        "wheel_torque_Nm": T_dist * 2,   # маховик обеспечивает 2× от возмущения
        "wheel_power_W":  Power_W,
    }


def run_adcs_analysis(output_dir: str, label: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    # Возмущения для разных ошибок ориентации
    theta_range = np.linspace(0.1, 10.0, 200)   # градусы
    gg_torques  = [gravity_gradient_torque_Nm(t) * 1e6 for t in theta_range]  # мкН·м
    srp_torque  = solar_pressure_torque_Nm() * 1e6
    aero_torque = aerodynamic_torque_Nm() * 1e6
    mag_torque  = magnetic_torque_Nm() * 1e6

    # Бюджет наведения
    pb = pointing_budget()
    total_err = total_pointing_error_deg()
    gain_loss  = gain_loss_dB(total_err)

    # Маховик
    rw = reaction_wheel_sizing()

    # ISL pointing
    pa_urad = isl_point_ahead_angle_urad()

    _plot_disturbance_torques(theta_range, gg_torques, srp_torque,
                               aero_torque, mag_torque, output_dir, label)
    _plot_pointing_budget(pb, total_err, gain_loss, output_dir, label)
    _plot_isl_pointing(output_dir, label)
    _plot_wheel_sizing(output_dir, label)
    _save_csv(pb, total_err, rw, pa_urad, output_dir, label)

    return {
        "pointing_budget":   pb,
        "total_error_deg":   total_err,
        "gain_loss_dB":      gain_loss,
        "rw":                rw,
        "isl_pa_urad":       pa_urad,
        "gg_torque_Nm":      gravity_gradient_torque_Nm(),
        "srp_torque_Nm":     solar_pressure_torque_Nm(),
        "aero_torque_Nm":    aerodynamic_torque_Nm(),
        "mag_torque_Nm":     magnetic_torque_Nm(),
    }


def _plot_disturbance_torques(theta_range, gg_torques, srp, aero, mag, output_dir, label):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(theta_range, gg_torques, color="#0984e3", lw=2, label="Гравитац. градиент T_gg(θ)")
    ax.axhline(srp,  ls="--", color="#e17055", lw=2, label=f"Солнечное давление = {srp:.2f} мкН·м")
    ax.axhline(aero, ls=":",  color="#00b894", lw=2, label=f"Аэродинамика = {aero:.3f} мкН·м")
    ax.axhline(mag,  ls="-.", color="#6c5ce7", lw=2, label=f"Магнитный = {mag:.2f} мкН·м")
    ax.axvline(1.0, ls="--", color="#fdcb6e", lw=1.5, label="1° ошибка наведения")
    ax.set_xlabel("Угол отклонения θ (градусы)")
    ax.set_ylabel("Возмущающий момент (мкН·м)")
    ax.set_title(f"АВРОРА — Возмущающие моменты [{label}]")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"adcs_disturbances_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_pointing_budget(pb, total_err, gain_loss, output_dir, label):
    names  = list(pb.keys())
    values = list(pb.values())
    colors = ["#0984e3", "#74b9ff", "#00b894", "#fdcb6e", "#e17055", "#6c5ce7", "#b2bec3"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Бюджет по источникам
    bars = ax1.barh(names, [v * 1000 for v in values],
                    color=colors[:len(names)], edgecolor="white", height=0.6)
    ax1.axvline(total_err * 1000, ls="--", color="#2d3436", lw=1.5,
                label=f"RSS = {total_err*1000:.1f} мДег ({total_err*3600:.1f}\")")
    for bar, v in zip(bars, values):
        ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                 f"{v*1000:.1f}", va="center", fontsize=9)
    ax1.set_xlabel("Ошибка наведения 3σ (мДег)")
    ax1.set_title(f"Бюджет ошибки наведения антенны [{label}]")
    ax1.legend(fontsize=9)
    ax1.grid(axis="x", alpha=0.3)

    # Потеря усиления vs ошибка наведения
    errs = np.linspace(0, ANTENNA_BW3_DEG / 2, 200)
    losses = [gain_loss_dB(e) for e in errs]
    ax2.plot(errs, losses, color="#6c5ce7", lw=2)
    ax2.axvline(total_err, ls="--", color="#e17055", lw=1.5,
                label=f"Суммарная ошибка = {total_err:.3f}° → {gain_loss:.4f} dB")
    ax2.axhline(-0.5, ls=":", color="#00b894", lw=1.2, label="-0.5 dB допуск")
    ax2.set_xlabel("Ошибка наведения (градусы)")
    ax2.set_ylabel("Потеря усиления (dB)")
    ax2.set_title(f"Потеря усиления антенны [{label}]")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"adcs_pointing_budget_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_isl_pointing(output_dir, label):
    """ISL point-ahead angle в зависимости от расстояния до соседа."""
    ranges_km = np.linspace(500, 6000, 200)
    v = orbital_velocity_m_s()
    angles_urad = []
    for r_km in ranges_km:
        sep_frac = (r_km / 3000.0)  # нормировано к 3000 км
        v_rel = 2 * v * min(sep_frac, 1.0)
        angles_urad.append(v_rel / C * 1e6)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ranges_km, angles_urad, color="#00b894", lw=2)
    ax.axvline(ISL_RANGE_KM, ls="--", color="#e17055", lw=1.5,
               label=f"АВРОРА ISL = {ISL_RANGE_KM:.0f} км")
    pa = isl_point_ahead_angle_urad()
    ax.axhline(pa, ls=":", color="#0984e3", lw=1.5,
               label=f"Point-ahead = {pa:.1f} мкрад")
    bw_half_urad = ISL_BW3_DEG / 2 * math.pi / 180 * 1e6
    ax.axhline(bw_half_urad, ls="-.", color="#fdcb6e", lw=1.5,
               label=f"Полуширина ДН ISL = {bw_half_urad:.0f} мкрад")
    ax.set_xlabel("Расстояние до соседнего спутника (км)")
    ax.set_ylabel("Угол опережения (мкрад)")
    ax.set_title(f"АВРОРА — ISL point-ahead angle [{label}]")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"adcs_isl_pointing_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_wheel_sizing(output_dir, label):
    """Требования к маховику для разных уровней возмущений."""
    T_orbits = np.linspace(20, 120, 100)   # мин
    T_dist   = total_disturbance_torque_Nm() * 1e6  # мкН·м

    H_vals = [T_dist * t * 60 / 4 for t in T_orbits]   # мкН·м·с

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(T_orbits, H_vals, color="#6c5ce7", lw=2)
    rw = reaction_wheel_sizing()
    ax1.axvline(rw["T_orbit_min"], ls="--", color="#e17055", lw=1.5,
                label=f"АВРОРА T = {rw['T_orbit_min']:.0f} мин")
    ax1.axhline(rw["H_required_Nms"] * 1e6, ls=":", color="#00b894", lw=1.5,
                label=f"H_req = {rw['H_required_Nms']*1e6:.0f} мкН·м·с")
    ax1.set_xlabel("Период орбиты (мин)")
    ax1.set_ylabel("Требуемый угловой импульс маховика (мкН·м·с)")
    ax1.set_title(f"Угловой импульс маховика [{label}]")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Типовые коммерческие маховики
    wheels = {
        "Rockwell Collins RWA": {"H_Nms": 4.0, "T_Nm": 0.1, "mass_kg": 2.2},
        "Bradford WSAT":        {"H_Nms": 1.0, "T_Nm": 0.05, "mass_kg": 0.8},
        "Astro ARES 4":         {"H_Nms": 4.0, "T_Nm": 0.2, "mass_kg": 2.1},
        "АВРОРА-RW (target)":   {"H_Nms": rw["H_required_Nms"], "T_Nm": rw["wheel_torque_Nm"],
                                  "mass_kg": 1.5},
    }
    w_names = list(wheels.keys())
    h_vals  = [wheels[w]["H_Nms"] for w in w_names]
    m_vals  = [wheels[w]["mass_kg"] for w in w_names]
    colors_w = ["#74b9ff", "#a29bfe", "#fdcb6e", "#00b894"]
    bars = ax2.bar(w_names, h_vals, color=colors_w, edgecolor="white", width=0.5)
    ax2b = ax2.twinx()
    ax2b.plot(range(len(w_names)), m_vals, "o-", color="#e17055", lw=2, ms=8, label="Масса (кг)")
    ax2.set_xticklabels(w_names, rotation=15, ha="right", fontsize=8)
    ax2.set_ylabel("Угловой импульс (Н·м·с)")
    ax2b.set_ylabel("Масса маховика (кг)")
    ax2.set_title(f"Сравнение маховиков [{label}]")
    ax2b.legend(loc="upper right", fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"adcs_wheel_sizing_{label}.png"), dpi=150)
    plt.close(fig)


def _save_csv(pb, total_err, rw, pa_urad, output_dir, label):
    path = os.path.join(output_dir, f"adcs_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["parameter", "value", "unit"])
        for name, val in pb.items():
            w.writerow([f"pointing_{name}", f"{val*1000:.3f}", "мДег (3sigma)"])
        w.writerow(["total_pointing_rss", f"{total_err*1000:.3f}", "мДег (3sigma)"])
        w.writerow(["gain_loss", f"{gain_loss_dB(total_err):.4f}", "dB"])
        w.writerow(["isl_point_ahead", f"{pa_urad:.1f}", "мкрад"])
        w.writerow(["rw_H_required", f"{rw['H_required_Nms']:.4f}", "Н·м·с"])
        w.writerow(["rw_torque_required", f"{rw['wheel_torque_Nm']*1e6:.3f}", "мкН·м"])
        w.writerow(["gg_torque", f"{gravity_gradient_torque_Nm()*1e9:.3f}", "нН·м"])
        w.writerow(["srp_torque", f"{solar_pressure_torque_Nm()*1e9:.3f}", "нН·м"])
        w.writerow(["aero_torque", f"{aerodynamic_torque_Nm()*1e9:.4f}", "нН·м"])
        w.writerow(["mag_torque", f"{magnetic_torque_Nm()*1e9:.3f}", "нН·м"])


def print_adcs_summary(label: str, result: Dict) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  ADCS Requirements Analysis -- {label}")
    print(sep)
    print(f"  Возмущающие моменты:")
    print(f"    Гравит. градиент:  {result['gg_torque_Nm']*1e9:.3f} нН·м")
    print(f"    Солнечное давл.:   {result['srp_torque_Nm']*1e9:.3f} нН·м")
    print(f"    Аэродинамика:      {result['aero_torque_Nm']*1e9:.4f} нН·м")
    print(f"    Магнитный:         {result['mag_torque_Nm']*1e9:.3f} нН·м")
    print()
    print(f"  Наведение антенны:")
    print(f"    Суммарная ошибка:  {result['total_error_deg']*1000:.2f} мДег (3σ)")
    print(f"    Потеря усиления:   {result['gain_loss_dB']:.4f} dB")
    print()
    print(f"  ISL point-ahead:     {result['isl_pa_urad']:.1f} мкрад")
    print(f"  Маховик H_req:       {result['rw']['H_required_Nms']*1000:.2f} мН·м·с")
    print(sep)
