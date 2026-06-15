"""
Multipath Environment Model for АВРОРА.

Models multipath-induced ranging error by environment type.
LEO satellites provide significantly better multipath geometry than GPS
due to high elevation angles and faster sky motion.
"""

import math
import os
import csv
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Environment definitions with multipath 1-sigma error
ENVIRONMENTS = {
    "Open Sky":    {
        "mp_l1_m": 0.30, "mp_l5_m": 0.15,
        "leo_factor": 1.0,   "gps_factor": 1.0,
        "color": "#00b894", "desc": "Open field, no obstructions",
    },
    "Suburban":    {
        "mp_l1_m": 0.80, "mp_l5_m": 0.40,
        "leo_factor": 0.7,   "gps_factor": 1.0,
        "color": "#fdcb6e", "desc": "Suburban road with scattered buildings",
    },
    "Urban":       {
        "mp_l1_m": 2.00, "mp_l5_m": 1.00,
        "leo_factor": 0.5,   "gps_factor": 1.0,
        "color": "#e17055", "desc": "Urban canyon, buildings >10 m",
    },
    "Deep Urban":  {
        "mp_l1_m": 5.00, "mp_l5_m": 2.50,
        "leo_factor": 0.4,   "gps_factor": 1.0,
        "color": "#d63031", "desc": "Dense city center, narrow streets",
    },
    "Indoor":      {
        "mp_l1_m": 8.00, "mp_l5_m": 4.00,
        "leo_factor": 0.6,   "gps_factor": None,
        "color": "#a855f7", "desc": "Indoor, concrete building (GPS unavailable)",
    },
    "Maritime":    {
        "mp_l1_m": 0.50, "mp_l5_m": 0.25,
        "leo_factor": 0.8,   "gps_factor": 1.0,
        "color": "#0984e3", "desc": "Open sea, sea surface reflection",
    },
}

# LEO advantage on multipath:
# Higher elevation angles -> less multipath (sin(el) factor)
# Faster satellite motion -> multipath averages out faster
# Stronger signal -> less prone to tracking errors from multipath
LEO_AVG_ELEVATION_DEG = 42.0   # Phase 3, mean visible elevation
GPS_AVG_ELEVATION_DEG = 28.0   # GPS MEO, mean visible elevation

# UERE budget (baseline, open sky, L1+L5)
UERE_NO_MP = {
    "clock_m": 1.50, "eph_m": 0.10, "iono_m": 0.05, "tropo_m": 0.50,
}


def mp_factor_from_elevation(elevation_deg: float) -> float:
    """Multipath error scales roughly as 1/sin(elevation)."""
    return 1.0 / math.sin(math.radians(max(elevation_deg, 5.0)))


def leo_mp_advantage_db(leo_el: float, gps_el: float) -> float:
    """dB advantage of LEO over GPS from elevation alone."""
    factor = mp_factor_from_elevation(gps_el) / mp_factor_from_elevation(leo_el)
    return 20 * math.log10(factor)


def compute_environment_uere(
    env_name: str,
    freq: str = "L1+L5",
    system: str = "АВРОРА",
) -> Dict:
    env = ENVIRONMENTS[env_name]
    base = dict(UERE_NO_MP)

    if freq == "L1+L5":
        mp_open = env["mp_l5_m"]   # dual-freq suppresses some multipath
    else:
        mp_open = env["mp_l1_m"]

    # Apply LEO factor (elevated angles reduce multipath)
    if system == "АВРОРА":
        mp = mp_open * env["leo_factor"]
        system_mp_factor = env["leo_factor"]
    else:  # GPS
        mp = mp_open * (env["gps_factor"] or 10.0)
        system_mp_factor = env["gps_factor"]

    base["multipath_m"] = mp
    uere = math.sqrt(sum(v**2 for v in base.values()))
    return {
        "env": env_name,
        "system": system,
        "freq": freq,
        "mp_error_m": mp,
        "uere_total_m": uere,
        "system_mp_factor": system_mp_factor,
    }


def run_multipath_analysis(
    output_dir: str,
    label: str,
    pdop_leo: float = 5.15,
    pdop_gps: float = 3.50,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    results = []
    for env_name in ENVIRONMENTS:
        for system, pdop in [("АВРОРА", pdop_leo), ("GPS", pdop_gps)]:
            for freq in ["L1+L5", "L1"]:
                r = compute_environment_uere(env_name, freq, system)
                r["pdop"]  = pdop
                r["cep_m"] = 0.59 * pdop * r["uere_total_m"]
                results.append(r)

    # LEO elevation advantage
    el_advantage = leo_mp_advantage_db(LEO_AVG_ELEVATION_DEG, GPS_AVG_ELEVATION_DEG)

    # Summary: АВРОРА L1+L5 vs GPS L1 CEP per environment
    summary = []
    for env_name in ENVIRONMENTS:
        aurora = next(r for r in results if r["env"] == env_name
                      and r["system"] == "АВРОРА" and r["freq"] == "L1+L5")
        gps    = next(r for r in results if r["env"] == env_name
                      and r["system"] == "GPS" and r["freq"] == "L1")
        summary.append({
            "env":              env_name,
            "mp_aurora_m":      aurora["mp_error_m"],
            "mp_gps_m":         gps["mp_error_m"],
            "cep_aurora_m":     aurora["cep_m"],
            "cep_gps_m":        gps["cep_m"] if gps["mp_error_m"] < 50 else None,
            "improvement":      (gps["cep_m"] / aurora["cep_m"])
                                if aurora["cep_m"] > 0 and gps["mp_error_m"] < 50 else None,
        })

    _plot_mp_by_environment(summary, output_dir, label)
    _plot_uere_breakdown_env(results, output_dir, label)
    _save_mp_csv(results, summary, output_dir, label)

    return {
        "results":          results,
        "summary":          summary,
        "elevation_advantage_db": el_advantage,
        "leo_avg_el_deg":   LEO_AVG_ELEVATION_DEG,
        "gps_avg_el_deg":   GPS_AVG_ELEVATION_DEG,
    }


def _plot_mp_by_environment(summary: List[Dict], output_dir: str, label: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    envs   = [s["env"] for s in summary]
    colors = [ENVIRONMENTS[e]["color"] for e in envs]

    # Left: multipath error
    ax = axes[0]
    x = np.arange(len(envs))
    w = 0.35
    mp_aurora = [s["mp_aurora_m"] for s in summary]
    mp_gps    = [s["mp_gps_m"]    for s in summary]
    b1 = ax.bar(x - w/2, mp_aurora, w, label="АВРОРА L1+L5", color="#00b894", edgecolor="white")
    b2 = ax.bar(x + w/2, mp_gps,    w, label="GPS L1",       color="#0984e3", edgecolor="white")
    ax.bar_label(b1, fmt="%.2f м", padding=2, fontsize=7)
    ax.bar_label(b2, fmt="%.2f м", padding=2, fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(envs, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Ошибка многолучёвости 1σ (м)")
    ax.set_title("Ошибка многолучёвости по среде")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Right: CEP comparison
    ax2 = axes[1]
    cep_aurora = [s["cep_aurora_m"] for s in summary]
    cep_gps    = [s["cep_gps_m"]    if s["cep_gps_m"] else s["cep_aurora_m"] * 3
                  for s in summary]
    b3 = ax2.bar(x - w/2, cep_aurora, w, label="АВРОРА L1+L5", color="#00b894", edgecolor="white")
    b4 = ax2.bar(x + w/2, cep_gps,    w, label="GPS L1",       color="#0984e3", edgecolor="white")
    ax2.bar_label(b3, fmt="%.1f м", padding=2, fontsize=7)
    ax2.bar_label(b4, fmt="%.1f м", padding=2, fontsize=7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(envs, rotation=20, ha="right", fontsize=9)
    ax2.set_ylabel("CEP 50% (м)")
    ax2.set_title("CEP по среде (АВРОРА L1+L5 vs GPS L1)")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(f"АВРОРА — Анализ многолучёвости [{label}]", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"multipath_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_uere_breakdown_env(results: List[Dict], output_dir: str, label: str) -> None:
    aurora_l1l5 = [r for r in results if r["system"] == "АВРОРА" and r["freq"] == "L1+L5"]
    envs   = [r["env"] for r in aurora_l1l5]
    ueres  = [r["uere_total_m"] for r in aurora_l1l5]
    mps    = [r["mp_error_m"] for r in aurora_l1l5]
    non_mp = [math.sqrt(max(u**2 - m**2, 0)) for u, m in zip(ueres, mps)]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(envs))
    ax.bar(x, non_mp, label="UERE без многолучёвости", color="#74b9ff", edgecolor="white")
    ax.bar(x, mps, bottom=non_mp, label="Многолучёвость", color="#e17055", edgecolor="white")
    for xi, u in zip(x, ueres):
        ax.text(xi, u + 0.05, f"{u:.2f} м", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(envs, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("UERE 1σ (м)")
    ax.set_title(f"Состав UERE АВРОРА L1+L5 по среде [{label}]")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"multipath_uere_{label}.png"), dpi=150)
    plt.close(fig)


def _save_mp_csv(results, summary, output_dir, label) -> None:
    path = os.path.join(output_dir, f"multipath_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["env","system","freq","mp_error_m",
                                           "uere_total_m","pdop","cep_m"])
        w.writeheader()
        for r in results:
            w.writerow({k: round(v, 4) if isinstance(v, float) else v
                        for k, v in r.items() if k in w.fieldnames})


def print_multipath_summary(label: str, result: Dict) -> None:
    sep = "=" * 68

    print(f"\n{sep}")
    print(f"  Multipath Environment Analysis -- {label}")
    print(sep)
    print(f"  LEO avg elevation: {result['leo_avg_el_deg']:.0f} deg  "
          f"GPS avg elevation: {result['gps_avg_el_deg']:.0f} deg")
    print(f"  LEO elevation advantage over GPS: "
          f"{result['elevation_advantage_db']:.1f} dB")
    print()
    print(f"  CEP comparison (АВРОРА L1+L5 vs GPS L1):")
    print(f"  {'Environment':<14} {'MP АВРОРА':>10} {'MP GPS':>8} "
          f"{'CEP АВРОРА':>12} {'CEP GPS':>10} {'Improv.':>9}")
    print("  " + "-" * 67)
    for s in result["summary"]:
        gps_cep = f"{s['cep_gps_m']:.1f}m" if s["cep_gps_m"] else "N/A"
        improv  = f"{s['improvement']:.1f}x" if s["improvement"] else ">"
        print(f"  {s['env']:<14} {s['mp_aurora_m']:>9.2f}m {s['mp_gps_m']:>7.2f}m "
              f"{s['cep_aurora_m']:>11.2f}m {gps_cep:>10} {improv:>9}")
    print(sep)
