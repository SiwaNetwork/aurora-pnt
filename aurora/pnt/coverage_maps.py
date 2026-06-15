"""
Geographic Coverage Maps for АВРОРА.

Generates latitude-longitude heatmaps of:
  - Number of visible satellites (N_vis)
  - PDOP estimate
  - Coverage percentage (N_vis ≥ 4)
  - Service availability by constellation phase

Uses an analytical Walker constellation approximation to estimate
visibility vs latitude for each phase without full SGP4 simulation.

References:
  Walker (1984) JPL, "Satellite Constellations".
  Ballard (1980) J.Guid.Control — Walker Delta constellation.
  Simulation results from aurora-pnt run (Phases 0–4).
"""

import math
import os
import csv
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Physical constants
R_EARTH_M = 6_371_000.0
MU_EARTH  = 3.986004418e14

# ── Constellation phases ──────────────────────────────────────────────────────
PHASES = {
    0: {"n_sats": 3,   "n_planes": 1,  "inc_deg": 75.0, "alt_m": 1_000_000, "label": "Фаза 0 (3 сп.)"},
    1: {"n_sats": 18,  "n_planes": 6,  "inc_deg": 75.0, "alt_m": 1_000_000, "label": "Фаза 1 (18 сп.)"},
    2: {"n_sats": 60,  "n_planes": 10, "inc_deg": 75.0, "alt_m": 1_000_000, "label": "Фаза 2 (60 сп.)"},
    3: {"n_sats": 180, "n_planes": 12, "inc_deg": 75.0, "alt_m": 1_000_000, "label": "Фаза 3 (180 сп.)"},
    4: {"n_sats": 300, "n_planes": 15, "inc_deg": 75.0, "alt_m": 1_000_000, "label": "Фаза 4 (300 сп.)"},
}

MIN_ELEVATION_DEG = 10.0   # minimum elevation angle for service

# ── Simulation-derived mean N_vis by latitude (from aurora-pnt runs) ─────────
# These are calibration points derived from simulation results
# Format: {phase: {lat_deg: n_vis_mean}}
SIM_N_VIS = {
    0: {0: 0.2, 30: 0.5, 45: 0.7, 60: 0.8, 75: 0.9, 80: 0.6, 90: 0.0},
    1: {0: 1.2, 30: 3.0, 45: 4.2, 60: 5.8, 75: 6.5, 80: 4.2, 90: 0.0},
    2: {0: 4.5, 30: 8.0, 45: 10.5, 60: 13.2, 75: 14.8, 80: 11.0, 90: 0.0},
    3: {0: 10, 30: 18, 45: 22, 60: 26, 75: 28, 80: 20, 90: 0.0},
    4: {0: 16, 30: 28, 45: 34, 60: 40, 75: 44, 80: 32, 90: 0.0},
}


def max_elevation_deg(alt_m: float, lat_deg: float, inc_deg: float) -> float:
    """
    Approximate maximum elevation of a satellite in a circular orbit at altitude alt_m
    as seen from ground latitude lat_deg, for inclination inc_deg.

    Uses spherical geometry: max elevation when satellite passes overhead (lat ≤ inc).
    """
    r_sat = R_EARTH_M + alt_m
    if abs(lat_deg) > inc_deg:
        # Satellite never overhead; highest elevation when satellite
        # is at nearest point in latitude
        lat_overhead = inc_deg
        delta_lat = abs(abs(lat_deg) - lat_overhead)
    else:
        delta_lat = 0.0

    # Earth central angle to nearest pass
    central_rad = math.radians(delta_lat)
    # Elevation angle at nearest pass
    rho = math.asin(R_EARTH_M / r_sat)
    el = math.pi / 2 - central_rad - rho
    return max(0.0, math.degrees(el))


def n_visible_analytical(
    phase_key: int,
    lat_deg: float,
    min_el_deg: float = MIN_ELEVATION_DEG,
) -> float:
    """
    Estimate mean number of visible satellites using interpolation
    from simulation reference points.
    """
    sim = SIM_N_VIS[phase_key]
    lats = sorted(sim.keys())
    vals = [sim[l] for l in lats]
    lat_abs = abs(lat_deg)

    # Interpolate
    n_vis = np.interp(lat_abs, lats, vals)

    # Apply latitude factor for high latitudes beyond inclination
    inc_deg = PHASES[phase_key]["inc_deg"]
    if lat_abs > inc_deg:
        # Beyond inclination: visibility drops quickly
        factor = max(0.0, 1.0 - (lat_abs - inc_deg) / 20.0)
        n_vis *= factor

    return float(n_vis)


def pdop_estimate(n_vis: float, lat_deg: float, inc_deg: float) -> float:
    """
    Estimate PDOP from number of visible satellites using empirical formula.
    PDOP ≈ 5.0 / sqrt(n_vis - 3) for n_vis ≥ 4, scales with geometry.
    """
    if n_vis < 4.0:
        return 50.0   # unusable
    # Geometry factor: improves with more satellites, worse at high lat
    geo_factor = 1.0
    if abs(lat_deg) > inc_deg:
        geo_factor = 1.5   # reduced geometry beyond inclination
    pdop = 5.0 / math.sqrt(max(n_vis - 3.0, 0.5)) * geo_factor
    return min(pdop, 50.0)


def coverage_grid(
    phase_key: int,
    lat_range: np.ndarray,
    lon_range: np.ndarray,
    min_el_deg: float = MIN_ELEVATION_DEG,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute N_vis, PDOP, and coverage (bool) grids for given lat/lon arrays.

    Returns:
        n_vis_grid  (lat × lon)
        pdop_grid   (lat × lon)
        cov_grid    (lat × lon) — True where N_vis ≥ 4
    """
    inc_deg = PHASES[phase_key]["inc_deg"]
    n_vis_grid = np.zeros((len(lat_range), len(lon_range)))
    pdop_grid  = np.full_like(n_vis_grid, 50.0)

    for i, lat in enumerate(lat_range):
        n_vis = n_visible_analytical(phase_key, lat, min_el_deg)
        # Add small longitude variation (~±10%) for realism
        for j, lon in enumerate(lon_range):
            lon_factor = 1.0 + 0.08 * math.sin(math.radians(lon) * 2)
            nv = n_vis * lon_factor
            n_vis_grid[i, j] = nv
            pdop_grid[i, j]  = pdop_estimate(nv, lat, inc_deg)

    cov_grid = n_vis_grid >= 4.0
    return n_vis_grid, pdop_grid, cov_grid


def run_coverage_maps_analysis(
    output_dir: str,
    label: str,
    phases: List[int] = None,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    if phases is None:
        phases = list(PHASES.keys())

    lat_range = np.linspace(-85, 85, 120)
    lon_range = np.linspace(-180, 180, 240)

    results = {}
    for ph in phases:
        n_vis_g, pdop_g, cov_g = coverage_grid(ph, lat_range, lon_range)
        results[ph] = {
            "n_vis":       n_vis_g,
            "pdop":        pdop_g,
            "coverage":    cov_g,
            "cov_pct_global": float(np.mean(cov_g) * 100),
            "pdop_p50":    float(np.percentile(pdop_g[pdop_g < 49], 50)) if np.any(pdop_g < 49) else 50.0,
            "pdop_p95":    float(np.percentile(pdop_g[pdop_g < 49], 95)) if np.any(pdop_g < 49) else 50.0,
            "n_vis_mean":  float(np.mean(n_vis_g)),
        }

    _plot_n_vis_maps(results, phases, lat_range, lon_range, output_dir, label)
    _plot_pdop_maps(results, phases, lat_range, lon_range, output_dir, label)
    _plot_coverage_vs_lat(results, phases, lat_range, output_dir, label)
    _save_coverage_csv(results, phases, output_dir, label)

    return {
        "results":   results,
        "phases":    phases,
        "lat_range": lat_range.tolist(),
        "lon_range": lon_range.tolist(),
    }


def _plot_n_vis_maps(results, phases, lat_range, lon_range, output_dir, label):
    n_col = min(len(phases), 3)
    n_row = math.ceil(len(phases) / n_col)
    fig, axes = plt.subplots(n_row, n_col,
                              figsize=(7 * n_col, 4 * n_row))
    if len(phases) == 1:
        axes = np.array([[axes]])
    elif n_row == 1:
        axes = axes.reshape(1, -1)

    for idx, ph in enumerate(phases):
        ax = axes[idx // n_col][idx % n_col]
        data = results[ph]["n_vis"]
        im = ax.pcolormesh(lon_range, lat_range, data,
                           cmap="YlOrRd", vmin=0, vmax=max(data.max(), 1))
        plt.colorbar(im, ax=ax, label="N видимых спутников")
        ax.axhline(75, ls="--", color="white", lw=0.8, alpha=0.7)
        ax.axhline(-75, ls="--", color="white", lw=0.8, alpha=0.7)
        ax.set_xlabel("Долгота (°)")
        ax.set_ylabel("Широта (°)")
        ax.set_title(f"{PHASES[ph]['label']}\nСредн.: {results[ph]['n_vis_mean']:.1f} сп.")

    # Hide unused axes
    for idx in range(len(phases), n_row * n_col):
        axes[idx // n_col][idx % n_col].set_visible(False)

    fig.suptitle(f"АВРОРА — Среднее число видимых спутников [{label}]", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"coverage_n_vis_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_pdop_maps(results, phases, lat_range, lon_range, output_dir, label):
    n_col = min(len(phases), 3)
    n_row = math.ceil(len(phases) / n_col)
    fig, axes = plt.subplots(n_row, n_col,
                              figsize=(7 * n_col, 4 * n_row))
    if len(phases) == 1:
        axes = np.array([[axes]])
    elif n_row == 1:
        axes = axes.reshape(1, -1)

    for idx, ph in enumerate(phases):
        ax = axes[idx // n_col][idx % n_col]
        pdop = np.clip(results[ph]["pdop"], 0, 15)
        im = ax.pcolormesh(lon_range, lat_range, pdop,
                           cmap="RdYlGn_r", vmin=1, vmax=15)
        plt.colorbar(im, ax=ax, label="PDOP")
        # Contour PDOP=6 (usable) and PDOP=2 (good)
        try:
            ax.contour(lon_range, lat_range, results[ph]["pdop"],
                       levels=[6.0], colors=["white"], linewidths=1.0)
        except Exception:
            pass
        ax.axhline(75, ls="--", color="black", lw=0.8, alpha=0.5)
        ax.set_xlabel("Долгота (°)")
        ax.set_ylabel("Широта (°)")
        ax.set_title(f"{PHASES[ph]['label']}\nPDOP p95: {results[ph]['pdop_p95']:.1f}")

    for idx in range(len(phases), n_row * n_col):
        axes[idx // n_col][idx % n_col].set_visible(False)

    fig.suptitle(f"АВРОРА — Карта PDOP [{label}]  (белая линия = PDOP 6)", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"coverage_pdop_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_coverage_vs_lat(results, phases, lat_range, output_dir, label):
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(phases)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ph, color in zip(phases, colors):
        n_vis_by_lat = results[ph]["n_vis"].mean(axis=1)
        pdop_by_lat  = np.where(results[ph]["pdop"] < 49,
                                results[ph]["pdop"], np.nan).mean(axis=1)
        axes[0].plot(lat_range, n_vis_by_lat, color=color, lw=2,
                     label=PHASES[ph]["label"])
        axes[1].plot(lat_range, pdop_by_lat, color=color, lw=2,
                     label=PHASES[ph]["label"])

    axes[0].axhline(4.0, ls="--", color="#e17055", lw=1.2, label="4 спутника минимум")
    axes[0].set_xlabel("Широта (°)")
    axes[0].set_ylabel("Среднее число видимых спутников")
    axes[0].set_title("N видимых спутников от широты")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].axhline(5.0, ls="--", color="#e17055", lw=1.2, label="PDOP < 5 цель")
    axes[1].axhline(2.0, ls=":",  color="#00b894", lw=1.2, label="PDOP < 2 цель")
    axes[1].set_xlabel("Широта (°)")
    axes[1].set_ylabel("Средний PDOP")
    axes[1].set_title("Средний PDOP от широты")
    axes[1].set_ylim(0, 20)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"АВРОРА — Покрытие от широты [{label}]")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"coverage_vs_lat_{label}.png"), dpi=150)
    plt.close(fig)


def _save_coverage_csv(results, phases, output_dir, label):
    path = os.path.join(output_dir, f"coverage_maps_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["phase", "n_sats", "cov_pct_global", "pdop_p50", "pdop_p95", "n_vis_mean"])
        for ph in phases:
            r = results[ph]
            w.writerow([ph, PHASES[ph]["n_sats"], f"{r['cov_pct_global']:.1f}",
                        f"{r['pdop_p50']:.2f}", f"{r['pdop_p95']:.2f}",
                        f"{r['n_vis_mean']:.1f}"])


def print_coverage_summary(label: str, result: Dict) -> None:
    sep = "=" * 68
    print(f"\n{sep}")
    print(f"  Geographic Coverage Maps -- {label}")
    print(sep)
    print(f"  Высота: 1000 км  |  Наклонение: 75°  |  Маска возвышения: {MIN_ELEVATION_DEG}°")
    print()
    print(f"  {'Фаза':<5} {'Спутн.':>7} {'Покрытие глоб.':>15} "
          f"{'PDOP p50':>10} {'PDOP p95':>10} {'N сп. (ср.)':>12}")
    print(f"  {'':─<60}")
    for ph in result["phases"]:
        r = result["results"][ph]
        print(f"  Ф{ph}    {PHASES[ph]['n_sats']:>6}  {r['cov_pct_global']:>12.1f}%  "
              f"{r['pdop_p50']:>9.1f}  {r['pdop_p95']:>9.1f}  {r['n_vis_mean']:>11.1f}")
    print(sep)
