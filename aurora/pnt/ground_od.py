"""
Ground-Only Orbit Determination (OD) for АВРОРА.

Models orbit determination accuracy using only ground station
network observations, without ISL measurements. Compares:
  - Ground-only OD (АВРОРА 21 MCS stations)
  - ISL-augmented OD
  - GPS-like ground network (reference)

Key outputs:
  - Radial/along-track/cross-track (RAC) uncertainty
  - Time-averaged along-track error (= clock/ranging error contribution)
  - UERE contribution from orbit determination
  - Ephemeris prediction accuracy vs. time since last ground contact

References:
  Montenbruck et al. (2015) GPS Solutions (LEO orbit determination).
  Zhu et al. (2022) GPS Solutions (LEO PNT ground network).
  IS-GPS-200 (GPS broadcast ephemeris accuracy spec).
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
GM     = 3.986004418e14   # m³/s² (Earth gravitational parameter)
R_E    = 6_371_000.0      # m
C      = 299_792_458.0    # m/s

AURORA_ALT_M = 1_000_000.0   # 1000 km orbit
AURORA_INC   = 75.0           # inclination (°)

# ── Ground station networks ───────────────────────────────────────────────────
NETWORKS = {
    "АВРОРА МКС (21 ст.)": {
        "n_stations":   21,
        "lat_coverage": "Россия + партнёры (15°–70° с.ш.)",
        "arc_h":        2.0,    # tracking arc (hours) before contact gap
        "gap_h":        0.5,    # typical contact gap (hours)
        "ranging_m":    0.30,   # pseudorange RMS
        "phase_mm":     2.0,    # carrier phase RMS (mm)
        "color": "#0984e3",
    },
    "GPS (глобал., 18 ст.)": {
        "n_stations":   18,
        "lat_coverage": "Глобальное (ВВС США)",
        "arc_h":        6.0,
        "gap_h":        0.1,
        "ranging_m":    0.30,
        "phase_mm":     2.0,
        "color": "#e17055",
    },
    "АВРОРА МКС + ИСЛ": {
        "n_stations":   21,
        "lat_coverage": "МКС + ISL дополнение",
        "arc_h":        24.0,   # ISL provides continuous coverage
        "gap_h":        0.0,
        "ranging_m":    0.10,   # ISL code: 0.10 m
        "phase_mm":     2.0,
        "color": "#00b894",
    },
}

# ── OD accuracy model ─────────────────────────────────────────────────────────
# Radial/Along/Cross-track uncertainties at various prediction intervals
# Based on Montenbruck et al. empirical data for LEO

def orbit_accuracy_m(
    arc_h: float,
    gap_h: float,
    ranging_m: float,
    n_stations: int,
    t_predict_h: float = 0.0,
) -> Dict:
    """
    Estimate orbit determination accuracy in RAC frame.

    Args:
        arc_h:      Continuous tracking arc length (hours)
        gap_h:      Contact gap (hours, no observations)
        ranging_m:  Pseudorange measurement noise (1σ, metres)
        n_stations: Number of ground stations in view
        t_predict_h: Prediction interval beyond last obs (hours)

    Returns:
        dict with radial, along-track, cross-track errors and 3D position error.
    """
    # Observability: more stations → better OD
    station_factor = math.sqrt(1.0 / max(n_stations, 1))

    # Arc length factor: longer arc → better gravity model fit
    arc_factor = math.exp(-arc_h / 4.0) + 0.02   # diminishing returns

    # Base OD accuracy (radial is tightest for LEO)
    sigma_r_base  = 0.05   # m  (radial, 1σ, best case)
    sigma_a_base  = 0.50   # m  (along-track)
    sigma_c_base  = 0.30   # m  (cross-track)

    # Scale by measurement noise, station count, arc
    scale = ranging_m * station_factor * (1 + arc_factor)

    sigma_r = sigma_r_base * scale * 5
    sigma_a = sigma_a_base * scale * 20   # along-track grows fastest
    sigma_c = sigma_c_base * scale * 10

    # Prediction degradation: drag uncertainty dominates along-track
    if t_predict_h > 0:
        drag_err_m_per_h = 0.15   # m/h along-track growth (1000 km LEO)
        gap_err_m        = gap_h * drag_err_m_per_h * 5
        sigma_a += gap_err_m + drag_err_m_per_h * t_predict_h
        sigma_r += t_predict_h * 0.005   # small radial growth
        sigma_c += t_predict_h * 0.010

    sigma_3d = math.sqrt(sigma_r**2 + sigma_a**2 + sigma_c**2)

    # UERE contribution from orbit (along-track → range error via cos(el))
    uere_m = math.sqrt(sigma_r**2 + (sigma_a * 0.3)**2 + (sigma_c * 0.3)**2)

    return {
        "sigma_r_m":    sigma_r,
        "sigma_a_m":    sigma_a,
        "sigma_c_m":    sigma_c,
        "sigma_3d_m":   sigma_3d,
        "uere_orbit_m": uere_m,
        "t_predict_h":  t_predict_h,
    }


def run_ground_od_analysis(output_dir: str, label: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    t_predict_range = np.linspace(0, 12, 200)   # hours

    results = {}
    for nk, n in NETWORKS.items():
        results[nk] = {
            "base":       orbit_accuracy_m(n["arc_h"], n["gap_h"],
                                           n["ranging_m"], n["n_stations"], 0.0),
            "curves_r":   [orbit_accuracy_m(n["arc_h"], n["gap_h"],
                                             n["ranging_m"], n["n_stations"],
                                             t)["sigma_r_m"] for t in t_predict_range],
            "curves_a":   [orbit_accuracy_m(n["arc_h"], n["gap_h"],
                                             n["ranging_m"], n["n_stations"],
                                             t)["sigma_a_m"] for t in t_predict_range],
            "curves_3d":  [orbit_accuracy_m(n["arc_h"], n["gap_h"],
                                             n["ranging_m"], n["n_stations"],
                                             t)["sigma_3d_m"] for t in t_predict_range],
        }

    _plot_rac_accuracy(results, output_dir, label)
    _plot_prediction_degradation(t_predict_range, results, output_dir, label)
    _plot_network_comparison(results, output_dir, label)
    _save_od_csv(results, output_dir, label)

    return {
        "results":   results,
        "networks":  list(NETWORKS.keys()),
        "t_range_h": t_predict_range.tolist(),
    }


def _plot_rac_accuracy(results, output_dir, label):
    networks = list(results.keys())
    metrics  = ["sigma_r_m", "sigma_a_m", "sigma_c_m", "sigma_3d_m"]
    met_labels = ["Радиальная", "Вдоль трека", "Поперёк трека", "3D суммарная"]

    fig, ax = plt.subplots(figsize=(11, 6))
    x  = np.arange(len(networks))
    w  = 0.18
    colors = ["#0984e3", "#e17055", "#00b894", "#6c5ce7"]

    for i, (met, lbl) in enumerate(zip(metrics, met_labels)):
        vals = [results[nk]["base"][met] for nk in networks]
        ax.bar(x + i * w, vals, w, label=lbl, color=colors[i], edgecolor="white")

    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(networks, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Погрешность (м, 1σ)")
    ax.set_title(f"АВРОРА — Точность орбитального определения: RAC сравнение [{label}]")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"od_rac_comparison_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_prediction_degradation(t_range, results, output_dir, label):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for nk, n in NETWORKS.items():
        color = NETWORKS[nk]["color"]
        axes[0].plot(t_range, results[nk]["curves_a"], color=color, lw=2, label=nk)
        axes[1].plot(t_range, results[nk]["curves_3d"], color=color, lw=2, label=nk)

    axes[0].axvline(2.0, ls=":", color="gray", lw=0.8, label="2 ч (типовой пропуск)")
    axes[0].axhline(2.0, ls="--", color="#e17055", lw=1.2, label="2 м UERE цель")
    axes[0].set_xlabel("Время с последнего наблюдения (ч)")
    axes[0].set_ylabel("Ошибка вдоль трека (м, 1σ)")
    axes[0].set_title("Деградация прогноза вдоль трека")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].axvline(2.0, ls=":", color="gray", lw=0.8)
    axes[1].axhline(5.0, ls="--", color="#e17055", lw=1.2, label="5 м цель")
    axes[1].set_xlabel("Время с последнего наблюдения (ч)")
    axes[1].set_ylabel("3D погрешность орбиты (м, 1σ)")
    axes[1].set_title("Деградация 3D прогноза")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"АВРОРА — Деградация прогноза эфемерид [{label}]")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"od_prediction_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_network_comparison(results, output_dir, label):
    fig, ax = plt.subplots(figsize=(9, 5))
    networks = list(results.keys())
    uere     = [results[nk]["base"]["uere_orbit_m"] for nk in networks]
    sigma3d  = [results[nk]["base"]["sigma_3d_m"]   for nk in networks]
    x = np.arange(len(networks))
    colors = [NETWORKS[nk]["color"] for nk in networks]

    ax.bar(x - 0.2, uere,   0.35, label="UERE орбита (м)", color=colors, edgecolor="white", alpha=0.85)
    ax.bar(x + 0.2, sigma3d, 0.35, label="3D орбита (м)",  color=colors, edgecolor="white", alpha=0.50)
    ax.axhline(0.80, ls="--", color="#e17055", lw=1.2, label="0.8 м UERE цель")
    ax.set_xticks(x)
    ax.set_xticklabels(networks, rotation=10, ha="right", fontsize=9)
    ax.set_ylabel("Погрешность (м, 1σ)")
    ax.set_title(f"АВРОРА — Сравнение сетей НС для орбитального определения [{label}]")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"od_network_compare_{label}.png"), dpi=150)
    plt.close(fig)


def _save_od_csv(results, output_dir, label):
    path = os.path.join(output_dir, f"ground_od_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["network", "sigma_r_m", "sigma_a_m", "sigma_c_m",
                    "sigma_3d_m", "uere_orbit_m"])
        for nk, r in results.items():
            b = r["base"]
            w.writerow([nk, f"{b['sigma_r_m']:.3f}", f"{b['sigma_a_m']:.3f}",
                        f"{b['sigma_c_m']:.3f}", f"{b['sigma_3d_m']:.3f}",
                        f"{b['uere_orbit_m']:.3f}"])


def print_ground_od_summary(label: str, result: Dict) -> None:
    sep = "=" * 68
    print(f"\n{sep}")
    print(f"  Ground-Only Orbit Determination Analysis -- {label}")
    print(sep)
    print(f"  Высота орбиты: {AURORA_ALT_M/1000:.0f} км  |  Наклонение: {AURORA_INC:.0f}°")
    print()
    print(f"  {'Сеть НС':<30} {'R (м)':>7} {'A (м)':>7} {'C (м)':>7} "
          f"{'3D (м)':>8} {'UERE (м)':>9}")
    print(f"  {'':─<65}")
    for nk in result["networks"]:
        b = result["results"][nk]["base"]
        print(f"  {nk:<30} {b['sigma_r_m']:>6.2f}  {b['sigma_a_m']:>6.2f}  "
              f"{b['sigma_c_m']:>6.2f}  {b['sigma_3d_m']:>7.2f}  {b['uere_orbit_m']:>8.2f}")
    print()
    print(f"  Наихудший случай: 12 ч без контакта (только МКС):")
    n = NETWORKS["АВРОРА МКС (21 ст.)"]
    b12 = orbit_accuracy_m(n["arc_h"], n["gap_h"], n["ranging_m"], n["n_stations"], 12.0)
    print(f"    Вдоль трека: {b12['sigma_a_m']:.1f} м  |  3D: {b12['sigma_3d_m']:.1f} м")
    print(sep)
