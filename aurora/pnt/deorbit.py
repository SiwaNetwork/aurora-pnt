"""
Orbital Lifetime and Deorbit Analysis for AURORA PNT.

Models atmospheric drag, orbital decay, debris lifetime, and deorbit
requirements for 1000 km altitude constellation.

Risk: at 1000 km, debris can remain in orbit for >100 years without deorbit.
ITU/IADC: deorbit within 25 years (LEO rule). Active deorbit required.
"""

import math
import os
import csv
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Physical constants
MU_EARTH = 3.986004418e14   # m^3/s^2
R_EARTH  = 6_371_000.0      # m
M_EARTH  = 5.972e24         # kg

# Atmospheric density model (NRLMSISE-00 representative values)
# [altitude_km, density_kg_m3]
ATM_DENSITY = [
    (200,  2.54e-10),
    (300,  1.92e-11),
    (400,  2.08e-12),
    (500,  5.22e-13),
    (600,  1.74e-13),
    (700,  6.97e-14),
    (800,  3.15e-14),
    (900,  1.54e-14),
    (1000, 7.75e-15),
    (1100, 4.05e-15),
    (1200, 2.19e-15),
    (1500, 3.77e-16),
    (2000, 2.32e-17),
]

# AURORA satellite parameters
AURORA_SAT = {
    "mass_kg":         150.0,      # estimated 150 kg satellite
    "cross_section_m2":  1.5,     # cross-sectional area
    "cd":               2.2,       # drag coefficient (sphere-like)
    "area_mass_ratio":  None,      # computed
}
AURORA_SAT["area_mass_ratio"] = (AURORA_SAT["cd"] * AURORA_SAT["cross_section_m2"]
                                  / AURORA_SAT["mass_kg"])

# IADC/ITU deorbit requirements
IADC_DEORBIT_YEARS = 25.0   # maximum orbital lifetime for LEO


def atm_density_kg_m3(alt_km: float) -> float:
    """Interpolate atmospheric density at given altitude."""
    for i in range(len(ATM_DENSITY) - 1):
        alt0, rho0 = ATM_DENSITY[i]
        alt1, rho1 = ATM_DENSITY[i+1]
        if alt0 <= alt_km <= alt1:
            t = (alt_km - alt0) / (alt1 - alt0)
            return rho0 * (rho1 / rho0) ** t
    if alt_km < ATM_DENSITY[0][0]:
        return ATM_DENSITY[0][1]
    return ATM_DENSITY[-1][1]


def orbital_velocity(alt_km: float) -> float:
    """Circular orbital velocity (m/s)."""
    return math.sqrt(MU_EARTH / (R_EARTH + alt_km * 1000))


def drag_acceleration(alt_km: float, sat: Dict) -> float:
    """Drag deceleration (m/s^2)."""
    rho = atm_density_kg_m3(alt_km)
    v   = orbital_velocity(alt_km)
    return 0.5 * rho * sat["cd"] * sat["cross_section_m2"] * v**2 / sat["mass_kg"]


def orbital_decay_rate_km_per_day(alt_km: float, sat: Dict) -> float:
    """Altitude loss rate due to atmospheric drag (km/day)."""
    a_drag = drag_acceleration(alt_km, sat)
    v      = orbital_velocity(alt_km)
    r      = R_EARTH + alt_km * 1000
    # da/dt = -2 * a_drag * a / v (vis-viva derived)
    da_dt  = -2 * a_drag * r / v   # m/s
    return da_dt * 86400 / 1000    # km/day (negative -> loss)


def lifetime_years(alt_start_km: float, alt_end_km: float,
                   sat: Dict, dt_days: float = 10.0) -> float:
    """
    Simulate orbital decay from alt_start to alt_end by numerical integration.
    Returns total time in years.
    """
    alt  = float(alt_start_km)
    t    = 0.0
    while alt > alt_end_km:
        rate = orbital_decay_rate_km_per_day(alt, sat)  # negative
        alt += rate * dt_days
        t   += dt_days
        if t > 365 * 500:   # safety cap 500 years
            break
    return t / 365.25


def deorbit_dv_ms(alt_km: float, target_alt_km: float = 200.0) -> float:
    """
    Delta-V required for Hohmann deorbit from alt_km to target_alt_km (m/s).
    Simplified: perigee lowering maneuver.
    """
    r1 = R_EARTH + alt_km * 1000
    r2 = R_EARTH + target_alt_km * 1000
    v1 = math.sqrt(MU_EARTH / r1)
    v_transfer_apo = math.sqrt(MU_EARTH * (2 / r1 - 2 / (r1 + r2)))
    dv = abs(v1 - v_transfer_apo)
    return dv


def propellant_mass_kg(dv_ms: float, sat_mass_kg: float, isp_s: float = 220.0) -> float:
    """Tsiolkovsky rocket equation: propellant mass for given dv."""
    g0 = 9.80665
    return sat_mass_kg * (math.exp(dv_ms / (isp_s * g0)) - 1)


def run_deorbit_analysis(
    output_dir: str,
    label: str,
    altitude_km: float = 1000.0,
    n_sats: int = 180,
    sat_params: Dict = None,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    sat = sat_params or AURORA_SAT

    # 1. Natural decay lifetime from current altitude
    life_natural = lifetime_years(altitude_km, 120.0, sat)

    # 2. Lifetime from various initial altitudes
    alt_range = [300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
    lifetimes  = [lifetime_years(a, 120.0, sat) for a in alt_range]

    # 3. Deorbit maneuver requirements
    dv_deorbit  = deorbit_dv_ms(altitude_km, 200.0)
    prop_mass   = propellant_mass_kg(dv_deorbit, sat["mass_kg"])
    prop_frac   = prop_mass / sat["mass_kg"]

    # 4. Station-keeping budget (orbit maintenance at 1000 km)
    drag_acc    = drag_acceleration(altitude_km, sat)
    # Annual delta-v for drag compensation (simplified)
    sk_dv_annual = drag_acc * 3600 * 24 * 365   # m/s/year (very rough)
    sk_prop_year = propellant_mass_kg(sk_dv_annual, sat["mass_kg"])

    # 5. Debris lifetime scenarios
    debris_scenarios = {
        "1 cm fragment (1g, Cd=2.5, A=0.0001m2)": {
            "mass_kg": 0.001, "cross_section_m2": 0.0001, "cd": 2.5
        },
        "10 cm fragment (100g, Cd=2.2, A=0.01m2)": {
            "mass_kg": 0.1,   "cross_section_m2": 0.01,   "cd": 2.2
        },
        "Spent satellite (150 kg, no deorbit)": {
            "mass_kg": 150.0, "cross_section_m2": 1.5,    "cd": 2.2
        },
    }
    debris_lives = {}
    for desc, params in debris_scenarios.items():
        debris_lives[desc] = lifetime_years(altitude_km, 120.0, params)

    # 6. Plots
    _plot_decay_curves(alt_range, lifetimes, output_dir, label)
    _plot_deorbit_dv_vs_altitude(output_dir, label, sat)
    _save_deorbit_csv(alt_range, lifetimes, debris_lives, output_dir, label)

    return {
        "altitude_km":       altitude_km,
        "n_sats":            n_sats,
        "life_natural_years": life_natural,
        "iadc_limit_years":  IADC_DEORBIT_YEARS,
        "compliant":         life_natural <= IADC_DEORBIT_YEARS,
        "dv_deorbit_ms":     dv_deorbit,
        "prop_mass_kg":      prop_mass,
        "prop_frac":         prop_frac,
        "sk_dv_annual_ms":   sk_dv_annual,
        "sk_prop_year_kg":   sk_prop_year,
        "debris_lives":      debris_lives,
        "fleet_prop_kg":     prop_mass * n_sats,
    }


def _plot_decay_curves(alt_range, lifetimes, output_dir, label) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(alt_range, lifetimes, "o-", color="#00b894", lw=2, ms=7)
    ax.axhline(IADC_DEORBIT_YEARS, ls="--", color="#e17055", lw=1.5,
               label=f"IADC 25-year limit")
    ax.axvline(1000, ls=":", color="#6c5ce7", lw=1.2, label="AURORA altitude 1000 km")

    # Mark where 25-year limit is crossed
    for i in range(len(lifetimes) - 1):
        if lifetimes[i] <= IADC_DEORBIT_YEARS < lifetimes[i+1]:
            ax.plot(alt_range[i], lifetimes[i], "r^", ms=12, zorder=5)
            ax.text(alt_range[i], lifetimes[i]*1.5, f"~{alt_range[i]} km\n25yr limit",
                    ha="center", fontsize=8, color="#e17055")

    ax.set_xlabel("Orbital altitude (km)")
    ax.set_ylabel("Natural decay lifetime (years)")
    ax.set_title(f"AURORA — Orbital Decay Lifetime [{label}]")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"deorbit_lifetime_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_deorbit_dv_vs_altitude(output_dir: str, label: str, sat: Dict) -> None:
    alts = list(range(400, 1501, 50))
    dvs  = [deorbit_dv_ms(a, 200.0) for a in alts]
    props = [propellant_mass_kg(dv, sat["mass_kg"]) for dv in dvs]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(alts, dvs, color="#0984e3", lw=2)
    axes[0].axvline(1000, ls=":", color="#6c5ce7", lw=1.2)
    axes[0].set_xlabel("Altitude (km)")
    axes[0].set_ylabel("Delta-V for deorbit (m/s)")
    axes[0].set_title("Deorbit Delta-V")
    axes[0].grid(alpha=0.3)

    axes[1].plot(alts, props, color="#e17055", lw=2)
    axes[1].axvline(1000, ls=":", color="#6c5ce7", lw=1.2)
    axes[1].set_xlabel("Altitude (km)")
    axes[1].set_ylabel("Propellant mass for deorbit (kg)")
    axes[1].set_title(f"Deorbit Propellant ({sat['mass_kg']:.0f} kg sat, Isp=220s)")
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"AURORA — Deorbit Requirements [{label}]", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"deorbit_dv_{label}.png"), dpi=150)
    plt.close(fig)


def _save_deorbit_csv(alt_range, lifetimes, debris_lives, output_dir, label) -> None:
    path = os.path.join(output_dir, f"deorbit_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["altitude_km", "lifetime_years", "iadc_compliant"])
        for alt, life in zip(alt_range, lifetimes):
            w.writerow([alt, f"{life:.1f}", life <= IADC_DEORBIT_YEARS])
        w.writerow(["", "", ""])
        w.writerow(["debris_type", "lifetime_years"])
        for desc, life in debris_lives.items():
            w.writerow([desc, f"{life:.1f}"])


def print_deorbit_summary(label: str, result: Dict) -> None:
    sep = "=" * 66

    print(f"\n{sep}")
    print(f"  Orbital Lifetime & Deorbit Analysis -- {label}")
    print(sep)
    print(f"  Altitude: {result['altitude_km']:.0f} km  "
          f"Constellation: {result['n_sats']} satellites")
    print()
    print(f"  Natural orbital lifetime (no deorbit):   "
          f"{result['life_natural_years']:.0f} years")
    print(f"  IADC 25-year rule:                       "
          f"{'[VIOLATION]' if not result['compliant'] else '[OK]'}")
    print(f"  => Active deorbit is MANDATORY at 1000 km")
    print()
    print(f"  Deorbit maneuver (to 200 km perigee):")
    print(f"    Delta-V required:    {result['dv_deorbit_ms']:.1f} m/s")
    print(f"    Propellant per sat:  {result['prop_mass_kg']:.2f} kg "
          f"({result['prop_frac']*100:.1f}% of mass)")
    print(f"    Fleet total (x{result['n_sats']}): {result['fleet_prop_kg']:.0f} kg "
          f"propellant reserved")
    print()
    print(f"  Station-keeping (annual drag compensation):")
    print(f"    Annual delta-V:      {result['sk_dv_annual_ms']:.3f} m/s/year")
    print(f"    Annual propellant:   {result['sk_prop_year_kg']:.4f} kg/year/sat")
    print()
    print(f"  Debris lifetime at 1000 km:")
    for desc, life in result["debris_lives"].items():
        flag = "[!!]" if life > IADC_DEORBIT_YEARS else "[OK]"
        print(f"    {flag} {desc[:55]:<55} {life:.0f} yr")
    print()
    print(f"  Mitigation strategy:")
    print(f"    - Each satellite equipped with propulsion for controlled deorbit")
    print(f"    - Deorbit within 1-3 years of EOL (target < 5 years)")
    print(f"    - Passivation: discharge batteries/pressure vessels before deorbit")
    print(f"    - Track all objects via MCC space surveillance cooperation")
    print(sep)
