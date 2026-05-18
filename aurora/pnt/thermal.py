"""
Satellite Thermal Analysis for AURORA PNT.

Models orbital temperature profile, equilibrium temperatures, and thermal
requirements for OCXO stabilization.

Reference: SMAD Ch.11.5; Gilmore, "Spacecraft Thermal Control Handbook" Vol.1.
"""

import math
import os
import csv
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R_EARTH_M   = 6_371_000.0
MU_EARTH    = 3.986004418e14
SIGMA_SB    = 5.670374419e-8   # W/m²/K⁴ (Stefan-Boltzmann)
SOLAR_CONST = 1361.0            # W/m² at 1 AU
AURORA_ALT_M = 1_000_000.0


def orbital_period_s(altitude_m: float) -> float:
    r = R_EARTH_M + altitude_m
    return 2 * math.pi * math.sqrt(r**3 / MU_EARTH)


def eclipse_fraction(altitude_m: float, beta_deg: float = 0.0) -> float:
    r = R_EARTH_M + altitude_m
    sin_rho = R_EARTH_M / r
    rho_max = math.degrees(math.asin(sin_rho))
    if abs(beta_deg) >= rho_max:
        return 0.0
    cos_frac = math.sqrt(sin_rho**2 - math.sin(math.radians(beta_deg))**2) \
               / math.cos(math.radians(beta_deg))
    cos_frac = min(1.0, max(0.0, cos_frac))
    return math.acos(1 - 2 * cos_frac) / (2 * math.pi) if cos_frac <= 1 else 0.0


# ── Satellite thermal parameters ──────────────────────────────────────────────
# Two-node simplified thermal model: sun-facing side, cold-space side

THERMAL_PARAMS = {
    # Sun-facing panel (solar array side)
    "alpha_s":        0.92,      # solar absorptivity (GaAs cell)
    "eps_s":          0.85,      # IR emissivity (solar array back)
    "area_sun_m2":    1.0,       # effective sun-facing area
    # Cold radiator (anti-nadir, deep space)
    "alpha_r":        0.10,      # radiator absorptivity (OSR tiles, low alpha)
    "eps_r":          0.85,      # radiator emissivity
    "area_rad_m2":    0.6,       # radiator area
    # Internal dissipation
    "q_internal_sunlit_w":  18.0,   # internal heat load in sunlit (electronics)
    "q_internal_eclipse_w": 12.0,   # reduced load in eclipse
    # Earth IR (albedo + emitted)
    "albedo":          0.30,     # Earth albedo factor
    "earth_ir_w_m2":  237.0,    # Earth IR emission (W/m²) at 1000 km
    "view_factor_earth": 0.15,  # satellite view factor to Earth
}

# Component temperature limits
TEMP_LIMITS = {
    "OCXO":              (-5, 60),    # operating range (°C), ideally oven at +70°C
    "Battery (Li-ion)":  (0,  40),   # charge: 0-45°C, discharge: -20-60°C
    "Solar cells":       (-150, 100), # operational
    "Electronics":       (-20, 70),   # commercial grade
    "Structure":         (-100, 150), # Al alloy
    "RF amplifiers":     (-10, 70),   # GaN PA operating
}


def equilibrium_temp_k(
    alpha: float, eps: float, area_sun_m2: float, area_rad_m2: float,
    q_internal_w: float, in_eclipse: bool = False,
    earth_ir_w: float = 0.0, albedo_w: float = 0.0,
) -> float:
    """
    Compute equilibrium temperature via energy balance:
      Q_in = Q_out
      alpha * S * A_sun + Q_int + Q_earth = eps * sigma * A_rad * T^4

    Returns temperature in Kelvin.
    """
    if in_eclipse:
        q_solar = 0.0
    else:
        q_solar = alpha * SOLAR_CONST * area_sun_m2

    q_in = q_solar + q_internal_w + earth_ir_w + albedo_w

    # T = (Q_in / (eps * sigma * A_rad))^0.25
    if q_in <= 0 or area_rad_m2 <= 0:
        return 0.0
    return (q_in / (eps * SIGMA_SB * area_rad_m2)) ** 0.25


def satellite_temp_profile(
    altitude_m: float = AURORA_ALT_M,
    beta_deg: float = 0.0,
    n_steps: int = 360,
) -> Dict:
    """
    Compute temperature over one orbit at given beta angle.

    Returns arrays of time and temperature for sun-facing and cold panels.
    """
    period_s = orbital_period_s(altitude_m)
    f_eclipse = eclipse_fraction(altitude_m, beta_deg)
    t_eclipse_start = (1 - f_eclipse) * period_s   # fraction through orbit

    p = THERMAL_PARAMS
    times_s = np.linspace(0, period_s, n_steps)
    temps_body_k = []

    for t in times_s:
        theta_norm = t / period_s   # 0..1
        in_ecl = theta_norm >= (1 - f_eclipse)

        # Earth IR heating
        earth_ir = p["earth_ir_w_m2"] * p["view_factor_earth"] * p["alpha_r"] * p["area_rad_m2"]

        # Albedo (only in sunlit)
        albedo_w = (SOLAR_CONST * p["albedo"] * p["view_factor_earth"]
                    * p["alpha_s"] * p["area_sun_m2"] if not in_ecl else 0.0)

        # Body equilibrium temperature
        q_int = p["q_internal_eclipse_w"] if in_ecl else p["q_internal_sunlit_w"]
        t_k = equilibrium_temp_k(
            p["alpha_s"], p["eps_s"], p["area_sun_m2"], p["area_rad_m2"],
            q_int, in_ecl, earth_ir, albedo_w,
        )
        temps_body_k.append(t_k)

    temps_c = np.array(temps_body_k) - 273.15
    t_min_c = float(np.min(temps_c))
    t_max_c = float(np.max(temps_c))
    delta_t  = t_max_c - t_min_c

    return {
        "times_s":   times_s,
        "temps_c":   temps_c,
        "t_min_c":   t_min_c,
        "t_max_c":   t_max_c,
        "delta_t_k": delta_t,
        "beta_deg":  beta_deg,
        "period_s":  period_s,
        "f_eclipse": f_eclipse,
    }


def ocxo_thermal_budget(delta_t_k: float, oven_power_w: float = 2.0) -> Dict:
    """
    OCXO temperature stability analysis.

    An OCXO oven maintains crystal at +80°C regardless of satellite body temperature.
    Power needed = heat loss through oven walls = lambda * A * ΔT.
    """
    # Oven thermal conductance (typical mini-OCXO oven)
    conductance_w_k = 0.1   # W/K heat leak through oven insulation
    q_heater_needed = conductance_w_k * delta_t_k
    q_available     = oven_power_w
    margin_w        = q_available - q_heater_needed

    # Residual temperature error at crystal (if oven margin < 0)
    temp_err_k = max(0.0, -margin_w / conductance_w_k)

    # Frequency error from temperature deviation
    temp_coeff = 1e-10   # /K, typical OCXO
    df_f = temp_coeff * temp_err_k
    timing_err_ns_per_10s = df_f * 10 * 1e9   # 10 s ISL sync

    return {
        "delta_t_body_k":      delta_t_k,
        "oven_power_w":        oven_power_w,
        "conductance_w_k":     conductance_w_k,
        "q_heater_needed_w":   q_heater_needed,
        "margin_w":            margin_w,
        "temp_err_k":          temp_err_k,
        "df_f":                df_f,
        "timing_err_ns_10s":   timing_err_ns_per_10s,
        "oven_adequate":       margin_w >= 0,
    }


def run_thermal_analysis(
    output_dir: str,
    label: str,
    altitude_m: float = AURORA_ALT_M,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    beta_cases = [0, 15, 25, 35, 45, 60]
    profiles   = [satellite_temp_profile(altitude_m, b) for b in beta_cases]
    ocxo_budgets = [ocxo_thermal_budget(p["delta_t_k"]) for p in profiles]

    # Worst case (beta=0)
    worst = profiles[0]
    ocxo_worst = ocxo_budgets[0]

    _plot_temp_profile(profiles, beta_cases, output_dir, label, altitude_m)
    _plot_temp_vs_beta(beta_cases, profiles, ocxo_budgets, output_dir, label)
    _save_thermal_csv(beta_cases, profiles, ocxo_budgets, output_dir, label)

    return {
        "altitude_m":     altitude_m,
        "beta_cases":     beta_cases,
        "profiles":       profiles,
        "ocxo_budgets":   ocxo_budgets,
        "worst_tmin_c":   worst["t_min_c"],
        "worst_tmax_c":   worst["t_max_c"],
        "worst_delta_t":  worst["delta_t_k"],
        "ocxo_worst":     ocxo_worst,
    }


def _plot_temp_profile(profiles, beta_cases, output_dir, label, altitude_m):
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(beta_cases)))
    for prof, beta, color in zip(profiles, beta_cases, colors):
        ax.plot(prof["times_s"] / 60, prof["temps_c"],
                color=color, lw=1.8, label=f"бета={beta}°")
    ax.axhline(TEMP_LIMITS["OCXO"][0], ls="--", color="#e17055", lw=1.2, label=f"Нижний предел OCXO {TEMP_LIMITS['OCXO'][0]}°C")
    ax.axhline(TEMP_LIMITS["OCXO"][1], ls="--", color="#0984e3", lw=1.2, label=f"Верхний предел OCXO {TEMP_LIMITS['OCXO'][1]}°C")
    ax.axhline(TEMP_LIMITS["Battery (Li-ion)"][0], ls=":", color="#fdcb6e", lw=1.0, label="Нижний предел АКБ")
    ax.set_xlabel("Время на орбите (минуты)")
    ax.set_ylabel("Температура корпуса спутника (°C)")
    ax.set_title(f"AURORA — Профиль температуры на орбите [{label}]  ({altitude_m/1000:.0f} км)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"thermal_profile_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_temp_vs_beta(beta_cases, profiles, ocxo_budgets, output_dir, label):
    t_min = [p["t_min_c"] for p in profiles]
    t_max = [p["t_max_c"] for p in profiles]
    delta = [p["delta_t_k"] for p in profiles]
    oven_margin = [o["margin_w"] for o in ocxo_budgets]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].fill_between(beta_cases, t_min, t_max, alpha=0.3, color="#6c5ce7")
    axes[0].plot(beta_cases, t_min, "o-", color="#0984e3", lw=2, label="T мин")
    axes[0].plot(beta_cases, t_max, "s-", color="#e17055", lw=2, label="T макс")
    axes[0].axhline(TEMP_LIMITS["OCXO"][0], ls="--", color="#e17055", lw=1.0, label="Нижний предел OCXO")
    axes[0].axhline(TEMP_LIMITS["Battery (Li-ion)"][0], ls=":", color="#fdcb6e", lw=1.0, label="Нижний предел АКБ")
    axes[0].set_xlabel("Бета-угол Солнца (°)")
    axes[0].set_ylabel("Температура (°C)")
    axes[0].set_title("Мин/Макс температура корпуса от бета-угла")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    bar_colors = ["#00b894" if m >= 0 else "#e17055" for m in oven_margin]
    axes[1].bar(beta_cases, oven_margin, width=6, color=bar_colors, edgecolor="white")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xlabel("Бета-угол Солнца (°)")
    axes[1].set_ylabel("Запас нагревателя OCXO (Вт)")
    axes[1].set_title("Тепловой запас OCXO от бета-угла")
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle(f"AURORA PNT — Тепловой анализ [{label}]")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"thermal_vs_beta_{label}.png"), dpi=150)
    plt.close(fig)


def _save_thermal_csv(beta_cases, profiles, ocxo_budgets, output_dir, label):
    path = os.path.join(output_dir, f"thermal_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["beta_deg", "t_min_c", "t_max_c", "delta_t_k",
                    "ocxo_heater_needed_w", "ocxo_margin_w", "ocxo_ok"])
        for beta, prof, ocxo in zip(beta_cases, profiles, ocxo_budgets):
            w.writerow([beta, f"{prof['t_min_c']:.1f}", f"{prof['t_max_c']:.1f}",
                        f"{prof['delta_t_k']:.1f}", f"{ocxo['q_heater_needed_w']:.2f}",
                        f"{ocxo['margin_w']:.2f}", ocxo["oven_adequate"]])


def print_thermal_summary(label: str, result: Dict) -> None:
    sep = "=" * 68
    worst = result["profiles"][0]
    ocxo  = result["ocxo_worst"]
    print(f"\n{sep}")
    print(f"  Satellite Thermal Analysis -- {label}")
    print(sep)
    print(f"  Altitude: {result['altitude_m']/1000:.0f} km  "
          f"(worst case: beta = {result['beta_cases'][0]}°)")
    print()
    print(f"  Body temperature range:")
    print(f"    T_min: {worst['t_min_c']:.1f} °C  (eclipse, cold face)")
    print(f"    T_max: {worst['t_max_c']:.1f} °C  (sunlit side)")
    print(f"    Delta T: {worst['delta_t_k']:.1f} K per orbit")
    print()
    print(f"  Component temperature limits:")
    for comp, (lo, hi) in TEMP_LIMITS.items():
        status = "OK" if worst["t_min_c"] >= lo and worst["t_max_c"] <= hi else "REVIEW"
        print(f"    {comp:<25}  [{lo:>4}°C, {hi:>4}°C]  body range in limits: [{status}]")
    print()
    print(f"  OCXO oven thermal budget (body ΔT = {worst['delta_t_k']:.1f} K):")
    print(f"    Oven conductance:    {ocxo['conductance_w_k']:.2f} W/K")
    print(f"    Heater needed:       {ocxo['q_heater_needed_w']:.2f} W")
    print(f"    Heater available:    {ocxo['oven_power_w']:.2f} W")
    print(f"    Margin:              {ocxo['margin_w']:+.2f} W  "
          f"{'[OK]' if ocxo['oven_adequate'] else '[INSUFFICIENT — review insulation]'}")
    print(f"    Crystal temp error:  {ocxo['temp_err_k']:.2f} K")
    print(f"    Freq error:          {ocxo['df_f']:.2e} (df/f)")
    print(f"    Timing error/sync:   {ocxo['timing_err_ns_10s']:.2f} ns  (per 10 s ISL sync)")
    print(sep)
