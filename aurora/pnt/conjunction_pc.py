"""
Conjunction Probability and Collision Avoidance for АВРОРА.

Monte Carlo analysis of satellite conjunction events at 1000 km orbit.
Models:
  - Debris population at 1000 km (Iridium breakup, ASAT test residuals)
  - Collision cross-section for 150 kg class satellites
  - Probability of collision per year per satellite
  - Fleet-wide risk and manoeuvre budget

References:
  Klinkrad, "Space Debris", 2006, Springer.
  MASTER-8 orbital debris model (ESA).
  NASA Debris Assessment Software (DAS) methodology.
  Kessler & Cour-Palais (1978) — collision rate model.
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
R_EARTH_M = 6_371_000.0

# ── Orbit parameters ──────────────────────────────────────────────────────────
AURORA_ALT_M   = 1_000_000.0    # 1000 km
AURORA_INC_DEG = 75.0
N_SATS_PHASE4  = 300

# ── Debris environment at 1000 km ─────────────────────────────────────────────
# Approximate flux from MASTER-8 / ESA:
#   Trackable (>10 cm): ~0.8 obj/km shell at 1000 km
#   Lethal (1-10 cm):   ~10x more
#   Fragmenting (<1 cm): ~100x more
DEBRIS = {
    "> 10 см (отслеж.)": {
        "flux_per_km2_yr": 2.5e-5,   # #/(m², year) at 1000 km for trackable
        "can_track":       True,
        "lethal":          True,
        "color":           "#e17055",
    },
    "1–10 см (неотслеж.)": {
        "flux_per_km2_yr": 2.5e-4,
        "can_track":       False,
        "lethal":          True,
        "color":           "#fdcb6e",
    },
    "< 1 см (повреждение)": {
        "flux_per_km2_yr": 2.5e-3,
        "can_track":       False,
        "lethal":          False,
        "color":           "#6c5ce7",
    },
}

# ── Satellite geometry ────────────────────────────────────────────────────────
SAT = {
    "mass_kg":        140.0,
    "cross_section_m2": 1.5,   # presented area (m²) — 150 kg class satellite
    "Bc_m2_kg":       0.025,   # ballistic coefficient (m²/kg) for drag
    "collision_dia_m": 0.5,    # effective collision diameter (combined hard body)
}

# ── Avoidance manoeuvre parameters ────────────────────────────────────────────
MANOEUVRE = {
    "pc_threshold":      1e-4,     # probability of collision threshold (0.01%)
    "miss_dist_req_m":   1000.0,   # required minimum miss distance (1 km)
    "dv_per_event_ms":   0.5,      # ΔV per avoidance manoeuvre (m/s)
    "dv_budget_ms":      21.0,     # total ΔV budget for station-keeping (7yr)
    "manoeuvre_rate_per_yr": 0.3,  # expected CAM rate per satellite per year
}


def collision_probability_per_year(
    flux_per_m2_yr: float,
    cross_section_m2: float,
) -> float:
    """Poisson probability of at least one collision per year."""
    rate = flux_per_m2_yr * cross_section_m2
    return 1 - math.exp(-rate)


def monte_carlo_conjunction(
    n_trials: int = 100_000,
    n_sats: int = N_SATS_PHASE4,
    mission_years: float = 7.0,
    seed: int = 42,
) -> Dict:
    """
    Monte Carlo simulation of conjunction events over mission life.

    Each satellite, each year: draw from Poisson distribution to determine
    number of conjunctions with trackable objects; compute miss distance
    distribution and P(c) using Foster's equation (simplified).
    """
    rng = np.random.default_rng(seed)

    # Annual collision rate for trackable debris per satellite
    rate_trackable = DEBRIS["> 10 см (отслеж.)"]["flux_per_km2_yr"] * SAT["cross_section_m2"]
    rate_lethal    = DEBRIS["1–10 см (неотслеж.)"]["flux_per_km2_yr"] * SAT["cross_section_m2"]

    # Simulate conjunctions
    n_conjunctions = rng.poisson(rate_trackable * 100,  # scale up for trackable warnings
                                  size=(n_trials,))
    # Expected miss distances (km, lognormal)
    miss_dists_km = rng.lognormal(mean=math.log(20), sigma=1.5,
                                   size=(n_trials,))   # median ~20 km

    # P(c) for each conjunction: simplified Foster model
    # P(c) ≈ (1/(2π σ_r σ_t)) * Ac * exp(−0.5*(d/σ)²)
    # Simplified: P(c) = Ac / (2π σ_r σ_t) for worst case miss geometry
    sigma_pos_km = 0.1   # typical RSO position uncertainty (km)
    ac_km2       = (SAT["collision_dia_m"] / 2000)**2 * math.pi   # km²

    # Fraction of conjunctions with P(c) > threshold
    pc_per_miss = ac_km2 / (2 * math.pi * sigma_pos_km**2) * np.exp(
        -0.5 * (miss_dists_km / sigma_pos_km)**2
    )
    n_high_pc = np.sum(pc_per_miss > MANOEUVRE["pc_threshold"])

    # Annual events
    cam_per_year = MANOEUVRE["manoeuvre_rate_per_yr"]
    total_cams   = cam_per_year * n_sats * mission_years
    dv_cams      = total_cams * MANOEUVRE["dv_per_event_ms"] / n_sats   # per satellite

    # P(zero collisions) over fleet life
    fleet_rate = rate_lethal * n_sats * mission_years
    p_zero_collision = math.exp(-fleet_rate)
    p_one_plus = 1 - p_zero_collision

    return {
        "n_trials":          n_trials,
        "n_sats":            n_sats,
        "mission_years":     mission_years,
        "rate_trackable_yr": rate_trackable,
        "rate_lethal_yr":    rate_lethal,
        "miss_dists_km":     miss_dists_km,
        "pc_per_miss":       pc_per_miss,
        "n_high_pc_frac":    n_high_pc / n_trials,
        "cam_per_year_fleet": cam_per_year * n_sats,
        "total_cams":        total_cams,
        "dv_cams_ms":        dv_cams,
        "fleet_rate":        fleet_rate,
        "p_zero_collision":  p_zero_collision,
        "p_one_plus":        p_one_plus,
    }


def run_conjunction_analysis(
    output_dir: str,
    label: str,
    n_trials: int = 50_000,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    mc = monte_carlo_conjunction(n_trials=n_trials)

    # ── Debris flux vs altitude ───────────────────────────────────────────────
    alt_km = list(range(600, 1400, 10))
    # Rough altitude dependence: flux peaks near 800-1000 km (Iridium debris band)
    flux_peak_alt = 1000.0   # km
    flux_profile  = [DEBRIS["> 10 см (отслеж.)"]["flux_per_km2_yr"] *
                     math.exp(-0.5 * ((a - flux_peak_alt) / 200)**2)
                     for a in alt_km]

    # ── P(c) vs altitude ─────────────────────────────────────────────────────
    pc_by_alt = [collision_probability_per_year(f, SAT["cross_section_m2"])
                 for f in flux_profile]

    # ── CAM ΔV budget ─────────────────────────────────────────────────────────
    n_sats_range  = [3, 18, 60, 180, 300]
    total_cams_by_phase = [MANOEUVRE["manoeuvre_rate_per_yr"] * n * 7
                           for n in n_sats_range]

    _plot_debris_flux(alt_km, flux_profile, output_dir, label)
    _plot_miss_distance_hist(mc, output_dir, label)
    _plot_cam_budget(n_sats_range, total_cams_by_phase, output_dir, label)
    _save_conjunction_csv(mc, output_dir, label)

    return {
        "mc":                mc,
        "alt_km":            alt_km,
        "flux_profile":      flux_profile,
        "pc_by_alt":         pc_by_alt,
        "n_sats_range":      n_sats_range,
        "total_cams_phases": total_cams_by_phase,
    }


def _plot_debris_flux(alt_km, flux_profile, output_dir, label):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(alt_km, flux_profile, color="#e17055", lw=2, label="> 10 см (мусор отслеж.)")
    ax.semilogy(alt_km, [f * 10 for f in flux_profile], color="#fdcb6e",
                lw=2, ls="--", label="1–10 см (≈×10)")
    ax.axvline(AURORA_ALT_M / 1000, ls=":", color="#0984e3", lw=1.5,
               label=f"АВРОРА {AURORA_ALT_M/1000:.0f} км")
    ax.set_xlabel("Высота орбиты (км)")
    ax.set_ylabel("Поток мусора (#/м²/год)")
    ax.set_title(f"АВРОРА — Поток орбитального мусора от высоты [{label}]")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"conjunction_flux_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_miss_distance_hist(mc, output_dir, label):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(mc["miss_dists_km"], bins=np.linspace(0, 100, 51), color="#0984e3",
                 alpha=0.75, edgecolor="white", linewidth=0.4, density=True)
    axes[0].axvline(1.0, ls="--", color="#e17055", lw=1.5, label="1 км (манёвр порог)")
    axes[0].axvline(5.0, ls=":", color="#fdcb6e", lw=1.2, label="5 км (предупреждение)")
    axes[0].set_xlabel("Расстояние сближения (км)")
    axes[0].set_ylabel("Плотность вероятности")
    axes[0].set_title("Распределение расстояний сближения (Монте-Карло)")
    axes[0].legend(fontsize=9)
    axes[0].set_xlim(0, 100)
    axes[0].grid(alpha=0.3)

    axes[1].hist(np.log10(mc["pc_per_miss"] + 1e-15), bins=80,
                 color="#6c5ce7", alpha=0.7, edgecolor="white", density=True)
    axes[1].axvline(math.log10(MANOEUVRE["pc_threshold"]),
                    ls="--", color="#e17055", lw=1.5,
                    label=f"Порог P(c) = {MANOEUVRE['pc_threshold']:.0e}")
    axes[1].set_xlabel("log₁₀[P(collision)]")
    axes[1].set_ylabel("Плотность вероятности")
    axes[1].set_title("Распределение вероятности столкновения")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"АВРОРА — Монте-Карло сближений [{label}]  "
                 f"({mc['n_trials']:,} испытаний)", fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"conjunction_mc_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_cam_budget(n_sats_range, total_cams, output_dir, label):
    phase_names = ["Фаза 0\n(3)", "Фаза 1\n(18)", "Фаза 2\n(60)",
                   "Фаза 3\n(180)", "Фаза 4\n(300)"]
    colors = ["#fdcb6e", "#74b9ff", "#00b894", "#6c5ce7", "#e17055"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(phase_names, total_cams, color=colors, edgecolor="white", alpha=0.85)
    for i, (lbl, v) in enumerate(zip(phase_names, total_cams)):
        ax.text(i, v + 1, f"{v:.0f} манёвров", ha="center", fontsize=8)
    ax.set_ylabel("Всего манёвров предотвращения за 7 лет")
    ax.set_title(f"АВРОРА — Бюджет манёвров CAM по фазам [{label}]")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"conjunction_cam_{label}.png"), dpi=150)
    plt.close(fig)


def _save_conjunction_csv(mc, output_dir, label):
    path = os.path.join(output_dir, f"conjunction_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        rows = [
            ("n_trials",             mc["n_trials"]),
            ("n_sats",               mc["n_sats"]),
            ("mission_years",        mc["mission_years"]),
            ("rate_lethal_yr",       f"{mc['rate_lethal_yr']:.3e}"),
            ("cam_per_yr_fleet",     f"{mc['cam_per_year_fleet']:.1f}"),
            ("total_cams_7yr",       f"{mc['total_cams']:.0f}"),
            ("dv_cams_ms_per_sat",   f"{mc['dv_cams_ms']:.2f}"),
            ("p_zero_collision",     f"{mc['p_zero_collision']:.4f}"),
            ("p_one_plus_collision", f"{mc['p_one_plus']:.4e}"),
        ]
        for r in rows:
            w.writerow(r)


def print_conjunction_summary(label: str, result: Dict) -> None:
    sep = "=" * 68
    mc = result["mc"]
    print(f"\n{sep}")
    print(f"  Conjunction / Collision Avoidance -- {label}")
    print(sep)
    print(f"  Орбита: {AURORA_ALT_M/1000:.0f} км, {AURORA_INC_DEG:.0f}°  |  "
          f"Флот: {mc['n_sats']} спутников × {mc['mission_years']:.0f} лет")
    print()
    print(f"  Дебрисная среда:")
    for dk, d in DEBRIS.items():
        pc = collision_probability_per_year(d["flux_per_km2_yr"], SAT["cross_section_m2"])
        print(f"    {dk:<25}  поток: {d['flux_per_km2_yr']:.1e} /м²/год  "
              f"P(столкн./год): {pc:.2e}")
    print()
    print(f"  Монте-Карло ({mc['n_trials']:,} испытаний):")
    print(f"    Тест. летальных/год/спутник: {mc['rate_lethal_yr']:.2e}")
    print(f"    CAM/год (флот):              {mc['cam_per_year_fleet']:.1f} манёвров")
    print(f"    ΔV на CAM/спутник (7 лет):   {mc['dv_cams_ms']:.2f} м/с")
    print(f"    P(0 столкновений, флот, 7л): {mc['p_zero_collision']:.4f}")
    print(f"    P(≥1 столкновение, флот, 7л):{mc['p_one_plus']:.2e}")
    print(f"    Порог CAM:                   P(c) > {MANOEUVRE['pc_threshold']:.0e}")
    print(sep)
