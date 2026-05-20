"""
SDCM (Sistema Differentsial'noy Korrektsii i Monitoringa) module for AURORA PNT.

Russian SBAS/DGNSS augmentation system. Models differential correction accuracy,
coverage zone, and UERE improvement for AURORA Mode C (Combined + SDCM).

Reference: SDCM ICD v2.0, GOST R 56231-2014.
"""

import math
import os
import csv
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Physical constants
R_EARTH = 6_371_000.0   # m

# SDCM reference stations (Phase 4, 7 stations in Russia)
SDCM_STATIONS = [
    {"name": "Moscow",       "lat":  55.75, "lon":  37.62},
    {"name": "Korolev",      "lat":  55.92, "lon":  37.80},
    {"name": "Novosibirsk",  "lat":  54.98, "lon":  82.90},
    {"name": "Khabarovsk",   "lat":  48.48, "lon": 135.07},
    {"name": "Magadan",      "lat":  59.57, "lon": 150.79},
    {"name": "Petropavlovsk","lat":  53.01, "lon": 158.65},
    {"name": "Anadyr",       "lat":  64.73, "lon": 177.52},
]

# SDCM coverage: effective within this radius of any reference station
SDCM_COVERAGE_RADIUS_KM = 3000.0   # 3000 km service area per station
SDCM_BORDER_FADE_KM     =  500.0   # accuracy degrades in last 500 km

# UERE budget (meters, 1-sigma, L1+L5 dual-freq)
UERE_BUDGET = {
    "autonomous": {
        "clock_m": 3.00, "eph_m": 0.50, "iono_m": 0.05,
        "tropo_m": 0.50, "multipath_m": 0.30, "isb_m": 0.00,
    },
    "combined": {
        "clock_m": 1.50, "eph_m": 0.10, "iono_m": 0.05,
        "tropo_m": 0.50, "multipath_m": 0.30, "isb_m": 0.50,
    },
    "sdcm_core": {       # within 1500 km of a reference station
        "clock_m": 0.80, "eph_m": 0.05, "iono_m": 0.04,
        "tropo_m": 0.25, "multipath_m": 0.30, "isb_m": 0.30,
    },
    "sdcm_edge": {       # 1500-3000 km, accuracy starts to degrade
        "clock_m": 1.20, "eph_m": 0.08, "iono_m": 0.05,
        "tropo_m": 0.35, "multipath_m": 0.30, "isb_m": 0.40,
    },
}


def uere_rss(budget: Dict) -> float:
    return math.sqrt(sum(v**2 for k, v in budget.items() if k.endswith("_m")))


def cep_50(uere_m: float, pdop: float) -> float:
    return 0.59 * pdop * uere_m


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    r = R_EARTH / 1000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * r * math.asin(math.sqrt(a))


def sdcm_coverage_factor(lat: float, lon: float) -> float:
    """
    Returns 0..1 factor indicating SDCM correction quality at a grid point.
    1.0 = full SDCM coverage (< SDCM_COVERAGE_RADIUS_KM/2 from nearest station)
    0.0 = outside SDCM coverage
    Linear interpolation in the outer half.
    """
    min_dist = min(haversine_km(lat, lon, s["lat"], s["lon"]) for s in SDCM_STATIONS)
    core_r   = SDCM_COVERAGE_RADIUS_KM - SDCM_BORDER_FADE_KM
    if min_dist <= core_r:
        return 1.0
    elif min_dist <= SDCM_COVERAGE_RADIUS_KM:
        return 1.0 - (min_dist - core_r) / SDCM_BORDER_FADE_KM
    return 0.0


def uere_at_point(lat: float, lon: float, mode: str = "combined") -> Tuple[float, str]:
    """
    Compute UERE (m) and effective mode at a grid point.
    In SDCM coverage zone, corrects to sdcm_core/sdcm_edge.
    """
    factor = sdcm_coverage_factor(lat, lon)
    if mode in ("combined", "sdcm") and factor > 0:
        if factor >= 0.5:
            budget = UERE_BUDGET["sdcm_core"]
            eff_mode = "sdcm_core"
        else:
            # Interpolate between sdcm_edge and combined
            b_edge = UERE_BUDGET["sdcm_edge"]
            b_comb = UERE_BUDGET["combined"]
            budget = {k: b_edge[k] * factor * 2 + b_comb[k] * (1 - factor * 2)
                      for k in b_edge}
            eff_mode = "sdcm_edge"
        return uere_rss(budget), eff_mode
    else:
        budget = UERE_BUDGET.get(mode, UERE_BUDGET["combined"])
        return uere_rss(budget), mode


# ---------------------------------------------------------------------------
# Coverage grid analysis
# ---------------------------------------------------------------------------

def compute_sdcm_grid(
    lat_min: float = 40.0, lat_max: float = 82.0,
    lon_min: float = 20.0, lon_max: float = 180.0,
    step_deg: float = 2.0,
    pdop_autonomous: float = 5.15,
    pdop_combined:   float = 1.67,
) -> List[Dict]:
    """Compute UERE and CEP for a grid of points under all three modes."""
    rows = []
    lat = lat_min
    while lat <= lat_max:
        lon = lon_min
        while lon <= lon_max:
            factor = sdcm_coverage_factor(lat, lon)

            uere_auto = uere_rss(UERE_BUDGET["autonomous"])
            uere_comb = uere_rss(UERE_BUDGET["combined"])

            uere_sdcm, eff = uere_at_point(lat, lon, mode="combined")

            rows.append({
                "lat":                lat,
                "lon":                lon,
                "sdcm_factor":        round(factor, 3),
                "sdcm_zone":          eff if factor > 0 else "none",
                "uere_autonomous_m":  round(uere_auto, 3),
                "uere_combined_m":    round(uere_comb, 3),
                "uere_sdcm_m":        round(uere_sdcm, 3),
                "cep_autonomous_m":   round(cep_50(uere_auto, pdop_autonomous), 2),
                "cep_combined_m":     round(cep_50(uere_comb, pdop_combined), 2),
                "cep_sdcm_m":         round(cep_50(uere_sdcm, pdop_combined), 2),
            })
            lon += step_deg
        lat += step_deg
    return rows


def summarize_grid(rows: List[Dict]) -> Dict:
    factors    = [r["sdcm_factor"]    for r in rows]
    cep_auto   = [r["cep_autonomous_m"] for r in rows]
    cep_comb   = [r["cep_combined_m"]   for r in rows]
    cep_sdcm   = [r["cep_sdcm_m"]       for r in rows]
    in_zone    = [r for r in rows if r["sdcm_factor"] > 0]

    pct_covered = 100.0 * len(in_zone) / len(rows) if rows else 0.0

    return {
        "n_points":            len(rows),
        "pct_covered":         pct_covered,
        "uere_autonomous_m":   uere_rss(UERE_BUDGET["autonomous"]),
        "uere_combined_m":     uere_rss(UERE_BUDGET["combined"]),
        "uere_sdcm_core_m":    uere_rss(UERE_BUDGET["sdcm_core"]),
        "cep_autonomous_p50":  float(np.percentile(cep_auto, 50)),
        "cep_autonomous_p95":  float(np.percentile(cep_auto, 95)),
        "cep_combined_p50":    float(np.percentile(cep_comb, 50)),
        "cep_combined_p95":    float(np.percentile(cep_comb, 95)),
        "cep_sdcm_p50":        float(np.percentile(cep_sdcm, 50)),
        "cep_sdcm_p95":        float(np.percentile(cep_sdcm, 95)),
        "cep_sdcm_in_zone_p50": float(np.percentile([r["cep_sdcm_m"] for r in in_zone], 50)) if in_zone else 0,
        "cep_sdcm_in_zone_p95": float(np.percentile([r["cep_sdcm_m"] for r in in_zone], 95)) if in_zone else 0,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_sdcm_coverage(rows: List[Dict], output_dir: str, label: str) -> None:
    lats = [r["lat"] for r in rows]
    lons = [r["lon"] for r in rows]
    cep  = [r["cep_sdcm_m"] for r in rows]
    fac  = [r["sdcm_factor"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: SDCM coverage factor map
    ax = axes[0]
    sc = ax.scatter(lons, lats, c=fac, cmap="RdYlGn", s=18, vmin=0, vmax=1)
    for st in SDCM_STATIONS:
        ax.plot(st["lon"], st["lat"], "^", color="#0984e3", ms=8, zorder=5)
        ax.text(st["lon"], st["lat"] + 1.2, st["name"], fontsize=6.5,
                ha="center", color="#0984e3")
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Качество коррекции SDCM (0=нет, 1=полная)")
    ax.set_xlabel("Долгота")
    ax.set_ylabel("Широта")
    ax.set_title(f"Зона покрытия SDCM [{label}]")
    ax.grid(alpha=0.3)

    # Right: CEP comparison bar chart
    ax2 = axes[1]
    modes = ["Автономный\n(Режим A)", "Комбинированный\n(Режим B)", "Зона SDCM\n(Режим C)"]
    ueres = [
        uere_rss(UERE_BUDGET["autonomous"]),
        uere_rss(UERE_BUDGET["combined"]),
        uere_rss(UERE_BUDGET["sdcm_core"]),
    ]
    pdops = [5.15, 1.67, 1.67]
    ceps  = [cep_50(u, p) for u, p in zip(ueres, pdops)]
    colors = ["#e17055", "#0984e3", "#00b894"]
    bars = ax2.bar(modes, ceps, color=colors, edgecolor="white", width=0.5)
    ax2.bar_label(bars, fmt="%.2f м", padding=3, fontsize=10, fontweight="bold")
    ax2.set_ylabel("CEP 50% (м)")
    ax2.set_title(f"CEP по режимам [{label}]")
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(0, max(ceps) * 1.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"sdcm_coverage_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_uere_breakdown(output_dir: str, label: str) -> None:
    modes   = ["Автономный", "Комбинированный", "Ядро SDCM", "Край SDCM"]
    keys    = ["clock_m", "eph_m", "iono_m", "tropo_m", "multipath_m", "isb_m"]
    labels  = ["Часы", "Эфемериды", "Ионосфера", "Тропосфера", "Многолучёвость", "ISB"]
    colors  = ["#e17055", "#fdcb6e", "#74b9ff", "#55efc4", "#a29bfe", "#fd79a8"]

    data = [[UERE_BUDGET[m.lower().replace(" ", "_")].get(k, 0.0) for m in
             ["autonomous","combined","sdcm_core","sdcm_edge"]] for k in keys]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(modes))
    bottom = np.zeros(len(modes))
    for i, (vals, lbl, col) in enumerate(zip(data, labels, colors)):
        ax.bar(x, vals, bottom=bottom, label=lbl, color=col, edgecolor="white", width=0.55)
        bottom += np.array(vals)

    totals = [uere_rss(UERE_BUDGET[m]) for m in ["autonomous","combined","sdcm_core","sdcm_edge"]]
    for xi, tot in zip(x, totals):
        ax.text(xi, tot + 0.05, f"{tot:.2f} м", ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel("Компонента UERE (м, 1σ)")
    ax.set_title(f"Состав бюджета UERE по режимам [{label}]")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"sdcm_uere_{label}.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_sdcm_analysis(
    output_dir: str,
    label: str,
    pdop_autonomous: float = 5.15,
    pdop_combined:   float = 1.67,
    grid_step_deg:   float = 2.0,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    rows    = compute_sdcm_grid(pdop_autonomous=pdop_autonomous,
                                pdop_combined=pdop_combined,
                                step_deg=grid_step_deg)
    summary = summarize_grid(rows)

    # Save grid CSV
    path = os.path.join(output_dir, f"sdcm_grid_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # UERE table CSV
    uere_path = os.path.join(output_dir, f"sdcm_uere_{label}.csv")
    with open(uere_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mode", "clock_m", "eph_m", "iono_m", "tropo_m",
                    "multipath_m", "isb_m", "uere_total_m", "pdop", "cep_m"])
        for mode_key, pdop in [("autonomous", pdop_autonomous), ("combined", pdop_combined),
                                ("sdcm_core", pdop_combined), ("sdcm_edge", pdop_combined)]:
            b = UERE_BUDGET[mode_key]
            tot = uere_rss(b)
            w.writerow([mode_key, b["clock_m"], b["eph_m"], b["iono_m"],
                        b["tropo_m"], b["multipath_m"], b.get("isb_m", 0),
                        f"{tot:.3f}", pdop, f"{cep_50(tot, pdop):.3f}"])

    _plot_sdcm_coverage(rows, output_dir, label)
    _plot_uere_breakdown(output_dir, label)

    return {"summary": summary, "rows": rows, "pdop_combined": pdop_combined}


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def print_sdcm_summary(label: str, result: Dict) -> None:
    s   = result["summary"]
    sep = "=" * 66

    print(f"\n{sep}")
    print(f"  SDCM Differential Corrections -- {label}")
    print(sep)
    print(f"  Reference stations: {len(SDCM_STATIONS)}  "
          f"Coverage radius per station: {SDCM_COVERAGE_RADIUS_KM:.0f} km")
    print(f"  Grid points analyzed: {s['n_points']}  "
          f"In SDCM zone: {s['pct_covered']:.1f}%")
    print()
    print(f"  UERE Budget (L1+L5 dual-freq, 1-sigma):")
    print(f"  {'Mode':<18} {'UERE':>8}  {'PDOP':>6}  {'CEP p50':>9}  {'CEP p95':>9}")
    print("  " + "-" * 56)
    print(f"  {'Autonomous (A)':<18} {s['uere_autonomous_m']:>7.2f}m  {5.15:>6.2f}  "
          f"{s['cep_autonomous_p50']:>8.2f}m  {s['cep_autonomous_p95']:>8.2f}m")
    print(f"  {'Combined (B)':<18} {s['uere_combined_m']:>7.2f}m  "
          f"{result['pdop_combined']:>6.2f}  "
          f"{s['cep_combined_p50']:>8.2f}m  {s['cep_combined_p95']:>8.2f}m")
    print(f"  {'SDCM core (C)':<18} {s['uere_sdcm_core_m']:>7.2f}m  "
          f"{result['pdop_combined']:>6.2f}  "
          f"{s['cep_sdcm_in_zone_p50']:>8.2f}m  {s['cep_sdcm_in_zone_p95']:>8.2f}m")
    print()
    print(f"  UERE breakdown (mode C / SDCM core):")
    b = UERE_BUDGET["sdcm_core"]
    for comp, val in b.items():
        print(f"    {comp:<14} {val:.2f} m")
    print(f"    {'RSS total':<14} {uere_rss(b):.2f} m")
    print()
    print(f"  CEP improvement over autonomous: "
          f"{s['cep_autonomous_p50'] / s['cep_sdcm_in_zone_p50']:.1f}x")
    print(f"  CEP improvement over combined:   "
          f"{s['cep_combined_p50'] / s['cep_sdcm_in_zone_p50']:.1f}x")
    print(sep)
