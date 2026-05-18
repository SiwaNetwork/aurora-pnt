"""
Constellation Deployment Timeline for AURORA PNT.

Models the launch sequence, orbital injection, phasing, and
service availability ramp-up across all 4 phases.

Phase 0: 3 sats  — proof of concept (technology demonstration)
Phase 1: 18 sats — regional (Russia + adjacent), LPT time scale
Phase 2: 60 sats — extended regional (SDCM + neighbours)
Phase 3: 180 sats — near-global, autonomous navigation
Phase 4: 300 sats — full global, redundant constellation

Launch vehicle: Soyuz-2.1b / Fregat  or  Angara-1.2 (6.5 t to 1000 km SSO)
Satellite mass: ~140 kg wet → 46 sats per rideshare
Deployment cadence: ~4 launches/year at peak

References:
  AURORA PNT System Design Document (SDD) v1.0 (2026).
  Wertz, "Space Mission Analysis and Design", 3rd ed., Ch. 20.
"""

import math
import os
import csv
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Constellation phases ──────────────────────────────────────────────────────
PHASES = {
    0: {
        "name":        "Фаза 0 — Демонстратор",
        "n_sats":      3,
        "n_planes":    1,
        "sats_plane":  3,
        "n_launches":  1,
        "launch_year": 2026.5,
        "duration_yr": 0.5,
        "color":       "#fdcb6e",
        "services": {
            "LPT (AURORA time)":     True,
            "ISL ranging":           True,
            "Navigation (L1/L5)":    False,
            "Coverage Russia":       False,
            "Coverage global":       False,
            "SDCM augmentation":     False,
        },
        "coverage_russia_pct":  0.0,
        "pdop_p95_russia":      None,
    },
    1: {
        "name":        "Фаза 1 — Региональная",
        "n_sats":      18,
        "n_planes":    6,
        "sats_plane":  3,
        "n_launches":  1,
        "launch_year": 2027.5,
        "duration_yr": 1.0,
        "color":       "#74b9ff",
        "services": {
            "LPT (AURORA time)":     True,
            "ISL ranging":           True,
            "Navigation (L1/L5)":    True,
            "Coverage Russia":       True,
            "Coverage global":       False,
            "SDCM augmentation":     True,
        },
        "coverage_russia_pct":  75.0,
        "pdop_p95_russia":      8.0,
    },
    2: {
        "name":        "Фаза 2 — Расширенная",
        "n_sats":      60,
        "n_planes":    10,
        "sats_plane":  6,
        "n_launches":  4,
        "launch_year": 2029.0,
        "duration_yr": 1.5,
        "color":       "#00b894",
        "services": {
            "LPT (AURORA time)":     True,
            "ISL ranging":           True,
            "Navigation (L1/L5)":    True,
            "Coverage Russia":       True,
            "Coverage global":       False,
            "SDCM augmentation":     True,
        },
        "coverage_russia_pct":  99.0,
        "pdop_p95_russia":      4.5,
    },
    3: {
        "name":        "Фаза 3 — Полуглобальная",
        "n_sats":      180,
        "n_planes":    12,
        "sats_plane":  15,
        "n_launches":  10,
        "launch_year": 2031.0,
        "duration_yr": 2.0,
        "color":       "#6c5ce7",
        "services": {
            "LPT (AURORA time)":     True,
            "ISL ranging":           True,
            "Navigation (L1/L5)":    True,
            "Coverage Russia":       True,
            "Coverage global":       True,
            "SDCM augmentation":     True,
        },
        "coverage_russia_pct":  100.0,
        "pdop_p95_russia":      2.5,
    },
    4: {
        "name":        "Фаза 4 — Полная группировка",
        "n_sats":      300,
        "n_planes":    15,
        "sats_plane":  20,
        "n_launches":  8,
        "launch_year": 2033.5,
        "duration_yr": 2.0,
        "color":       "#e17055",
        "services": {
            "LPT (AURORA time)":     True,
            "ISL ranging":           True,
            "Navigation (L1/L5)":    True,
            "Coverage Russia":       True,
            "Coverage global":       True,
            "SDCM augmentation":     True,
        },
        "coverage_russia_pct":  100.0,
        "pdop_p95_russia":      1.8,
    },
}

SAT_MASS_WET_KG  = 140.0     # per satellite wet mass
LAUNCH_CAP_KG    = 6_500.0   # Soyuz-2.1b / Fregat to 1000 km
COST_PER_SAT_MUSD = 3.0      # $3M per satellite manufacturing (rough estimate)
COST_PER_LAUNCH_MUSD = 50.0  # $50M per launch (Soyuz-2.1b, rough)

ORBITAL_PERIOD_MIN = 105.0   # minutes at 1000 km
PHASING_DAYS_PER_PLANE = 15.0   # days to drift to correct RAAN spacing


def sats_per_launch() -> int:
    return max(1, int(LAUNCH_CAP_KG / SAT_MASS_WET_KG))


def total_launches(n_sats: int) -> int:
    return math.ceil(n_sats / sats_per_launch())


def phase_cost_musd(phase_num: int) -> Dict:
    p = PHASES[phase_num]
    prev_sats = sum(PHASES[i]["n_sats"] for i in range(phase_num))
    new_sats  = p["n_sats"] - prev_sats
    n_launches = p["n_launches"]

    sat_cost     = new_sats * COST_PER_SAT_MUSD
    launch_cost  = n_launches * COST_PER_LAUNCH_MUSD
    total_cost   = sat_cost + launch_cost

    return {
        "new_sats":       new_sats,
        "n_launches":     n_launches,
        "sat_cost_musd":  sat_cost,
        "launch_cost_musd": launch_cost,
        "total_musd":     total_cost,
    }


def run_deployment_analysis(output_dir: str, label: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    # ── Timeline events ────────────────────────────────────────────────────────
    timeline = []
    cumulative_sats = 0
    total_cost = 0.0
    for ph_num in PHASES:
        ph = PHASES[ph_num]
        cost = phase_cost_musd(ph_num)
        prev_sats = sum(PHASES[i]["n_sats"] for i in range(ph_num))
        new_sats  = ph["n_sats"] - prev_sats

        timeline.append({
            "phase":          ph_num,
            "name":           ph["name"],
            "year":           ph["launch_year"],
            "n_total_sats":   ph["n_sats"],
            "new_sats":       new_sats,
            "n_launches":     ph["n_launches"],
            "coverage_pct":   ph["coverage_russia_pct"],
            "pdop":           ph["pdop_p95_russia"],
            "total_musd":     cost["total_musd"],
            "color":          ph["color"],
        })
        total_cost += cost["total_musd"]

    n_total = PHASES[4]["n_sats"]
    sats_per_lv = sats_per_launch()
    n_launches_total = sum(ph["n_launches"] for ph in PHASES.values())

    _plot_deployment_timeline(timeline, output_dir, label)
    _plot_sats_vs_time(timeline, output_dir, label)
    _plot_coverage_ramp(timeline, output_dir, label)
    _plot_cost_breakdown(timeline, total_cost, output_dir, label)
    _save_deployment_csv(timeline, output_dir, label)

    return {
        "timeline":       timeline,
        "total_sats":     n_total,
        "sats_per_lv":    sats_per_lv,
        "n_launches_total": n_launches_total,
        "total_cost_musd": total_cost,
        "phases":         PHASES,
    }


def _plot_deployment_timeline(timeline, output_dir, label):
    fig, ax = plt.subplots(figsize=(13, 6))

    for t in timeline:
        ax.barh(t["phase"], t["new_sats"],
                left=t["year"], height=0.5, color=t["color"],
                edgecolor="white", alpha=0.85)
        ax.text(t["year"] + 0.1, t["phase"],
                f"  +{t['new_sats']} спутн.\n  {t['n_launches']} запусков",
                va="center", fontsize=8)

    ax.set_yticks(list(PHASES.keys()))
    ax.set_yticklabels([PHASES[i]["name"] for i in PHASES], fontsize=9)
    ax.set_xlabel("Год")
    ax.set_title(f"AURORA PNT — График развёртывания группировки [{label}]")
    ax.set_xlim(2025.5, 2037)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"deployment_timeline_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_sats_vs_time(timeline, output_dir, label):
    years = [t["year"] for t in timeline]
    n_sats = [t["n_total_sats"] for t in timeline]
    colors = [t["color"] for t in timeline]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.step([2026] + years, [0] + n_sats, where="post", color="#6c5ce7", lw=2)
    ax.scatter(years, n_sats, s=100, c=colors, zorder=5)
    for t in timeline:
        ax.text(t["year"] + 0.1, t["n_total_sats"] + 5,
                f"Ф{t['phase']}: {t['n_total_sats']} сп.", fontsize=8)
    ax.set_xlabel("Год")
    ax.set_ylabel("Количество спутников на орбите")
    ax.set_title(f"AURORA PNT — Наращивание группировки [{label}]")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"deployment_sats_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_coverage_ramp(timeline, output_dir, label):
    years = [t["year"] for t in timeline]
    covs  = [t["coverage_pct"] for t in timeline]
    pdops = [t["pdop"] if t["pdop"] else 0.0 for t in timeline]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].step([2026] + years, [0] + covs, where="post", color="#00b894", lw=2)
    axes[0].scatter(years, covs, s=80, color=[t["color"] for t in timeline], zorder=5)
    axes[0].axhline(95, ls="--", color="#e17055", lw=1.2, label="95% цель")
    axes[0].set_xlabel("Год")
    axes[0].set_ylabel("Покрытие России (% времени, 4+ спутника)")
    axes[0].set_title("Покрытие по России от года")
    axes[0].set_ylim(0, 105)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    valid_phases = [(t["year"], t["pdop"]) for t in timeline if t["pdop"]]
    if valid_phases:
        vy, vp = zip(*valid_phases)
        axes[1].step(vy, vp, where="post", color="#0984e3", lw=2)
        axes[1].scatter(list(vy), list(vp),
                        s=80, color=[t["color"] for t in timeline if t["pdop"]], zorder=5)
    axes[1].axhline(5.0, ls="--", color="#e17055", lw=1.2, label="PDOP < 5 цель")
    axes[1].axhline(2.0, ls=":",  color="#00b894", lw=1.2, label="PDOP < 2 цель")
    axes[1].set_xlabel("Год")
    axes[1].set_ylabel("PDOP p95 (Россия)")
    axes[1].set_title("PDOP p95 от года")
    axes[1].set_ylim(0, 12)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"AURORA PNT — Рост покрытия и точности [{label}]")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"deployment_coverage_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_cost_breakdown(timeline, total_cost, output_dir, label):
    phase_names   = [t["name"].split("—")[1].strip() for t in timeline]
    phase_costs   = [t["total_musd"] for t in timeline]
    phase_colors  = [t["color"] for t in timeline]

    fig, ax = plt.subplots(figsize=(10, 6))
    wedges, texts, autotexts = ax.pie(
        phase_costs, colors=phase_colors, autopct="%1.0f%%",
        startangle=90, pctdistance=0.75,
    )
    ax.legend(wedges, [f"Ф{t['phase']}: {t['name'].split('—')[1].strip()} "
                       f"(${c:.0f}M)" for t, c in zip(timeline, phase_costs)],
              loc="lower left", fontsize=8, bbox_to_anchor=(-0.1, -0.1))
    ax.set_title(f"AURORA PNT — Распределение затрат по фазам [{label}]\n"
                 f"Итого: ${total_cost:.0f}M")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"deployment_cost_{label}.png"), dpi=150)
    plt.close(fig)


def _save_deployment_csv(timeline, output_dir, label):
    path = os.path.join(output_dir, f"deployment_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["phase", "name", "launch_year", "n_total_sats", "new_sats",
                    "n_launches", "coverage_russia_pct", "pdop_p95", "cost_musd"])
        for t in timeline:
            w.writerow([t["phase"], t["name"], t["year"], t["n_total_sats"],
                        t["new_sats"], t["n_launches"], t["coverage_pct"],
                        t["pdop"] or "", f"{t['total_musd']:.1f}"])


def print_deployment_summary(label: str, result: Dict) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  Constellation Deployment Plan -- {label}")
    print(sep)
    print(f"  Ракета: Союз-2.1б / Ангара-1.2  |  "
          f"Полезная нагрузка: {LAUNCH_CAP_KG/1000:.1f} т  |  "
          f"Спутников на запуск: ~{result['sats_per_lv']}")
    print(f"  Всего запусков: {result['n_launches_total']}  |  "
          f"Полный состав: {result['total_sats']} спутников  |  "
          f"Стоимость: ${result['total_cost_musd']:.0f}M")
    print()
    print(f"  {'Фаза':<5} {'Год':>6} {'Сп. орб.':>9} {'Запусков':>9} "
          f"{'Покрытие РФ':>12} {'PDOP p95':>9} {'Стоимость':>10}")
    print(f"  {'':─<65}")
    for t in result["timeline"]:
        pdop_str = f"{t['pdop']:.1f}" if t["pdop"] else "  —"
        print(f"  Ф{t['phase']}    {t['year']:>6.1f}  {t['n_total_sats']:>8}  "
              f"{t['n_launches']:>8}  {t['coverage_pct']:>10.0f}%  {pdop_str:>9}  "
              f"${t['total_musd']:>7.0f}M")
    print(sep)
