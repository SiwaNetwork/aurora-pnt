"""
Combined LEO + GLONASS PNT simulation.

Architecture:
  - LEO PNT satellites: relay/retransmit GNSS + encrypted auth overlay
  - GLONASS: MEO primary reference (24 sats, 3 planes, 64.8 deg, 19136 km)
  - Ground receiver: combined geometry with separate clock bias per system (ISB)
  - Expected improvement: PDOP ~5.0 (LEO only) -> ~1.5-2.0 (LEO+GLONASS)

Multi-constellation DOP model (IS-GPS-705 / ICD-GLONASS):
  - Each system adds one clock bias column to the H matrix
  - ISB (Inter-System Bias) treated as additional unknown -> solved simultaneously
  - UERE for combined solution reduced by better geometry and ISB calibration

References:
  - GLONASS ICD Ed. 5.1, 2008
  - Montenbruck et al., "Multi-GNSS receiver clock characterization", GPS Solutions 2014
  - Walter & Enge, "Weighted RAIM for Precision Approach", ION GPS 1995
"""

import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from astropy.time import Time
from sgp4.api import SatrecArray
from tqdm import tqdm

from aurora import logger
from aurora.pnt.coverage import (
    _ecef_to_enu,
    _gmst_rad,
    _teme_to_ecef,
    _enu_to_az_el,
    altitude_to_mean_motion,
    geodetic_to_ecef_km,
    load_satrec_from_tle_file,
)
from aurora.pnt.dop import compute_dop, compute_dop_multiconstellation
from aurora.pnt.glonass import (
    GLONASS_PARAMS,
    generate_glonass_satrec_array,
)
from aurora.pnt.grid import get_grid
from aurora.pnt.metrics import aggregate_simulation, aggregate_timestep
from aurora.pnt.report import (
    plot_coverage_map,
    plot_coverage_over_time,
    save_summary_json,
    save_timestep_metrics_csv,
)
from aurora.tles.generate_tles_from_scratch import generate_tles_from_scratch_with_sgp

log = logger.get_logger(__name__)

_SECONDS_PER_DAY = 86400.0


def _epoch_to_jd(epoch: Time) -> tuple[float, float]:
    jd_full = epoch.jd
    return float(math.floor(jd_full)), float(jd_full - math.floor(jd_full))


def compute_combined_visibility_at_time(
    leo_array: SatrecArray,
    glonass_array: SatrecArray,
    grid_points: list[tuple],
    jd: float,
    fr: float,
    leo_min_el_deg: float = 10.0,
    glo_min_el_deg: float = 5.0,
    mode: str = "combined",
) -> list[dict]:
    """
    Compute LEO+GLONASS visibility and DOP at a single time step.

    mode="combined"   — 5-column H: (dx,dy,dz,dt_lpt,dt_glonass); ISB as 5th unknown.
    mode="autonomous" — 4-column H: (dx,dy,dz,dt_lpt); only LEO sats used, LPT time scale.
    mode="glonass"    — 4-column H: (dx,dy,dz,dt_glo); only GLONASS sats (for comparison).

    GLONASS uses a lower elevation mask (5 deg) because MEO geometry provides
    better signal quality at shallow elevation compared to LEO.
    """
    def _propagate(sat_array):
        e, r_teme, _ = sat_array.sgp4(
            np.array([jd], dtype=np.float64),
            np.array([fr], dtype=np.float64),
        )
        return np.asarray(e)[:, 0], np.asarray(r_teme)[:, 0, :]

    e_leo, r_teme_leo = _propagate(leo_array)
    e_glo, r_teme_glo = _propagate(glonass_array)

    gmst = _gmst_rad(jd, fr)
    r_ecef_leo = _teme_to_ecef(r_teme_leo, gmst)
    r_ecef_glo = _teme_to_ecef(r_teme_glo, gmst)

    valid_leo = (e_leo == 0)
    valid_glo = (e_glo == 0)
    leo_min_rad = math.radians(leo_min_el_deg)
    glo_min_rad = math.radians(glo_min_el_deg)

    results = []
    for lat, lon, alt_m in grid_points:
        lat_r = math.radians(lat)
        lon_r = math.radians(lon)
        obs_ecef = geodetic_to_ecef_km(lat, lon, alt_m)

        enu_leo = _ecef_to_enu(r_ecef_leo, lat_r, lon_r, obs_ecef)
        az_leo, el_leo = _enu_to_az_el(enu_leo)
        vis_leo = valid_leo & (el_leo >= leo_min_rad)

        enu_glo = _ecef_to_enu(r_ecef_glo, lat_r, lon_r, obs_ecef)
        az_glo, el_glo = _enu_to_az_el(enu_glo)
        vis_glo = valid_glo & (el_glo >= glo_min_rad)

        n_leo = int(vis_leo.sum())
        n_glo = int(vis_glo.sum())

        if mode == "autonomous":
            # LPT time scale: use only LEO satellites, 4-parameter solution
            dop = compute_dop(az_leo[vis_leo], el_leo[vis_leo]) if n_leo >= 4 else None
            n_used = n_leo
            n_glo_used = 0
        elif mode == "glonass":
            dop = compute_dop(az_glo[vis_glo], el_glo[vis_glo]) if n_glo >= 4 else None
            n_used = n_glo
            n_glo_used = n_glo
        else:
            # Combined: 5-parameter solution (ISB as additional unknown)
            dop = compute_dop_multiconstellation(
                az_leo[vis_leo], el_leo[vis_leo],
                az_glo[vis_glo], el_glo[vis_glo],
            )
            n_used = n_leo + n_glo
            n_glo_used = n_glo

        results.append({
            "lat": lat,
            "lon": lon,
            "n_sats":    n_used,
            "n_leo":     n_leo,
            "n_glonass": n_glo_used,
            "pdop": dop["pdop"] if dop else None,
            "hdop": dop["hdop"] if dop else None,
            "vdop": dop["vdop"] if dop else None,
            "gdop": dop["gdop"] if dop else None,
            "tdop": dop["tdop"] if dop else None,
        })

    return results


def _aggregate_combined_timestep(point_results: list[dict]) -> dict:
    """Extend aggregate_timestep with per-system satellite counts."""
    base = aggregate_timestep(point_results)
    n_leo_vals = [r["n_leo"] for r in point_results]
    n_glo_vals = [r["n_glonass"] for r in point_results]
    base["n_leo_mean"]     = round(float(np.mean(n_leo_vals)), 2)
    base["n_glonass_mean"] = round(float(np.mean(n_glo_vals)), 2)
    return base


def run_combined_simulation(
    config: dict,
    output_dir: str,
    label: str = "combined",
    leo_min_el_deg: float = 10.0,
    glo_min_el_deg: float = 5.0,
    mode: str = "combined",
) -> dict:
    """
    Run combined LEO+GLONASS PNT simulation.

    mode="combined"   — combined geometry, ISB as 5th unknown (best PDOP)
    mode="autonomous" — LEO-only, LPT time scale (sovereign, no GLONASS dependency)
    mode="glonass"    — GLONASS-only (comparison baseline)

    Generates LEO TLEs from config, generates GLONASS Walker-Delta TLEs,
    then for every time step computes DOP over the grid according to mode.

    Returns overall summary dict (same structure as run_pnt_simulation).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── 1. LEO TLEs ──────────────────────────────────────────────────────────
    cc = config["constellation"]
    tle_path = os.path.join(output_dir, f"tles_{label}.txt")
    log.info(f"Generating LEO TLEs: {cc['num_orbits']}x{cc['num_sats_per_orbit']}")
    generate_tles_from_scratch_with_sgp(
        tle_path,
        cc["name"],
        cc["num_orbits"],
        cc["num_sats_per_orbit"],
        cc.get("phase_diff", True),
        cc["inclination_degree"],
        cc.get("eccentricity", 1e-7),
        cc.get("arg_of_perigee_degree", 0.0),
        cc["mean_motion_rev_per_day"],
    )
    leo_satrecs = load_satrec_from_tle_file(tle_path)
    leo_array = SatrecArray(leo_satrecs)
    n_leo = len(leo_satrecs)
    log.info(f"LEO: {n_leo} satellites")

    # ── 2. GLONASS TLEs ──────────────────────────────────────────────────────
    log.info(f"Generating GLONASS TLEs: {GLONASS_PARAMS['num_orbits']}x"
             f"{GLONASS_PARAMS['num_sats_per_orbit']}")
    glo_array, glo_tle_path = generate_glonass_satrec_array(output_dir)
    n_glo = GLONASS_PARAMS["num_orbits"] * GLONASS_PARAMS["num_sats_per_orbit"]
    log.info(f"GLONASS: {n_glo} satellites")

    # ── 3. Grid ───────────────────────────────────────────────────────────────
    pnt_cfg = config.get("pnt", {})
    grid_name = pnt_cfg.get("grid", "global")
    grid_step = pnt_cfg.get("grid_step_deg", 5.0)
    min_el = pnt_cfg.get("min_elevation_deg", leo_min_el_deg)
    grid_points = get_grid(grid_name, grid_step)
    log.info(f"Grid '{grid_name}' step={grid_step}deg: {len(grid_points)} points")

    # ── 4. Epoch ──────────────────────────────────────────────────────────────
    epoch = Time("2000-01-01 00:00:00", scale="tdb")
    jd0, fr0 = _epoch_to_jd(epoch)

    # ── 5. Time steps ─────────────────────────────────────────────────────────
    sim_cfg = config["simulation"]
    end_time_s = sim_cfg["end_time_hours"] * 3600.0
    step_s = sim_cfg["time_step_minutes"] * 60.0
    offset_s = sim_cfg.get("offset_ns", 0) / 1e9
    times_s = [
        t for t in (offset_s + i * step_s for i in range(int((end_time_s - offset_s) / step_s)))
        if t < end_time_s
    ]
    log.info(f"Simulation: {sim_cfg['end_time_hours']}h, step={sim_cfg['time_step_minutes']}min, "
             f"{len(times_s)} steps")

    # ── 6. Main loop ──────────────────────────────────────────────────────────
    timestep_metrics: list[dict] = []
    kept_steps: list[list[dict]] = []

    for step_idx, t_s in enumerate(tqdm(times_s, desc=f"Combined [{label}]")):
        t_days = t_s / _SECONDS_PER_DAY
        fr = fr0 + t_days
        jd = jd0 + math.floor(fr)
        fr = fr - math.floor(fr)

        point_results = compute_combined_visibility_at_time(
            leo_array, glo_array, grid_points, jd, fr,
            leo_min_el_deg=min_el, glo_min_el_deg=glo_min_el_deg,
            mode=mode,
        )

        step_agg = _aggregate_combined_timestep(point_results)
        step_agg["time_s"] = t_s
        step_agg["time_h"] = round(t_s / 3600.0, 4)
        timestep_metrics.append(step_agg)

        if len(kept_steps) < 50:
            kept_steps.append(point_results)

    # ── 7. Aggregate + save ───────────────────────────────────────────────────
    summary = aggregate_simulation(timestep_metrics)
    summary["label"]           = label
    summary["n_satellites"]    = n_leo + n_glo
    summary["n_leo"]           = n_leo
    summary["n_glonass"]       = n_glo
    summary["grid"]            = grid_name
    summary["grid_step_deg"]   = grid_step
    summary["n_orbits"]        = cc["num_orbits"]
    summary["n_sats_per_orbit"]= cc["num_sats_per_orbit"]
    summary["altitude_km"]     = config["satellite"]["altitude_m"] / 1000
    summary["inclination_deg"] = cc["inclination_degree"]
    summary["n_leo_mean"]      = round(
        float(np.mean([m["n_leo_mean"] for m in timestep_metrics])), 2
    )
    summary["n_glonass_mean"]  = round(
        float(np.mean([m["n_glonass_mean"] for m in timestep_metrics])), 2
    )
    summary["mode"]            = mode
    summary["multi_constellation"] = (mode == "combined")
    summary["glo_min_el_deg"]  = glo_min_el_deg
    summary["leo_min_el_deg"]  = min_el

    save_timestep_metrics_csv(timestep_metrics, output_dir)
    save_summary_json(summary, output_dir, label)

    # ── 8. Plots ──────────────────────────────────────────────────────────────
    plot_coverage_over_time(timestep_metrics, output_dir, label, sim_cfg["time_step_minutes"])
    if kept_steps:
        plot_coverage_map(kept_steps, output_dir, label, "n_sats")
        plot_coverage_map(kept_steps, output_dir, label, "pdop")

    _plot_combined_comparison(timestep_metrics, output_dir, label, n_leo, n_glo)
    _save_combined_csv(timestep_metrics, output_dir, label)

    _print_combined_summary(label, summary)
    return summary


def _plot_combined_comparison(
    timestep_metrics: list[dict],
    output_dir: str,
    label: str,
    n_leo: int,
    n_glo: int,
) -> None:
    """Plot combined vs LEO-only PDOP degradation and satellite counts."""
    times  = [m["time_h"] for m in timestep_metrics]
    pdop   = [m.get("pdop_p95") for m in timestep_metrics]
    n_leo_ts  = [m.get("n_leo_mean", 0) for m in timestep_metrics]
    n_glo_ts  = [m.get("n_glonass_mean", 0) for m in timestep_metrics]
    n_tot  = [m.get("n_sats_mean", 0) for m in timestep_metrics]
    cov    = [m.get("coverage_4sats_pct", 0) for m in timestep_metrics]

    fig, axes = plt.subplots(3, 1, figsize=(13, 11))
    fig.suptitle(f"Combined LEO+GLONASS — {label}\n"
                 f"LEO: {n_leo} sats @ 1000 km  |  GLONASS: {n_glo} sats @ 19136 km",
                 fontweight="bold")

    # PDOP over time
    ax = axes[0]
    pdop_clean = [v if v is not None else float("nan") for v in pdop]
    ax.plot(times, pdop_clean, color="#1565C0", linewidth=1.2, label="PDOP p95 (combined)")
    ax.axhline(6.0, color="red",    linestyle="--", linewidth=1.5, label="PDOP=6 limit")
    ax.axhline(2.0, color="green",  linestyle=":",  linewidth=1.0, label="PDOP=2 excellent")
    ax.set_ylabel("PDOP p95")
    ax.set_title("PDOP Dilution of Precision (multi-constellation)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 8)

    # Satellite counts stacked area
    ax = axes[1]
    n_leo_arr = np.array(n_leo_ts)
    n_glo_arr = np.array(n_glo_ts)
    ax.fill_between(times, n_glo_arr, alpha=0.7, color="#E53935", label="GLONASS visible (mean)")
    ax.fill_between(times, n_glo_arr + n_leo_arr, n_glo_arr, alpha=0.7,
                    color="#1E88E5", label="LEO visible (mean)")
    ax.axhline(5, color="grey",   linestyle="--", linewidth=1, label="5 sat min (multi-GNSS)")
    ax.set_ylabel("Satellites visible (mean)")
    ax.set_title("Satellite Visibility by Constellation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Coverage
    ax = axes[2]
    ax.plot(times, cov, color="#2E7D32", linewidth=1.2, label="5-sat coverage %")
    ax.axhline(99.0, color="orange", linestyle="--", linewidth=1, label="99% target")
    ax.axhline(95.0, color="red",    linestyle="--", linewidth=1.5, label="95% requirement")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Coverage (%)")
    ax.set_title("5-Satellite Coverage (combined geometry)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(80, 102)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"combined_{label}.png"),
        dpi=120, bbox_inches="tight"
    )
    plt.close()


def _save_combined_csv(timestep_metrics: list[dict], output_dir: str, label: str) -> None:
    fields = ["time_h", "n_sats_mean", "n_leo_mean", "n_glonass_mean",
              "pdop_mean", "pdop_p95", "coverage_4sats_pct", "coverage_good_pdop_pct"]
    with open(os.path.join(output_dir, f"combined_{label}.csv"), "w", encoding="utf-8") as f:
        f.write(",".join(fields) + "\n")
        for m in timestep_metrics:
            f.write(",".join(str(m.get(k, "")) for k in fields) + "\n")


def _print_combined_summary(label: str, summary: dict) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  Combined LEO+GLONASS PNT — {label}")
    print(sep)
    print(f"  LEO satellites:           {summary.get('n_leo', '?'):>6}  "
          f"(mean visible: {summary.get('n_leo_mean', '?'):.1f})")
    print(f"  GLONASS satellites:       {summary.get('n_glonass', '?'):>6}  "
          f"(mean visible: {summary.get('n_glonass_mean', '?'):.1f})")
    print(f"  Total visible (mean):     {summary.get('n_sats_mean_overall', '?')!r:>6}")
    print()
    print(f"  PDOP p95 (mean):          {summary.get('pdop_p95_mean', 'N/A')!r:>8}")
    print(f"  PDOP p95 (worst step):    {summary.get('pdop_p95_worst', 'N/A')!r:>8}")
    print(f"  Coverage 4+ sats:         {summary.get('coverage_4sats_mean_pct', 'N/A'):>7.2f}%")
    print(f"  Coverage PDOP<6:          {summary.get('coverage_good_pdop_mean_pct', 'N/A'):>7.2f}%")
    print(f"  Time steps:               {summary.get('n_timesteps', 0):>7d}")
    print(f"{sep}\n")
