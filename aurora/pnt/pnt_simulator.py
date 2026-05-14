"""
Main PNT simulation orchestrator.

Standalone pipeline (does NOT require the routing simulation):
  1. Generate TLEs from config
  2. Load Satrec objects for fast batch SGP4
  3. For each time step → batch compute visibility + DOP over grid
  4. Aggregate metrics and save results
"""

import math
import os
from pathlib import Path

import yaml
from astropy.time import Time
from sgp4.api import SatrecArray
from tqdm import tqdm

from aurora import logger
from aurora.pnt.coverage import (
    altitude_to_mean_motion,
    compute_grid_visibility_at_time,
    load_satrec_from_tle_file,
)
from aurora.pnt.grid import get_grid, STATIONS
from aurora.pnt.metrics import aggregate_simulation, aggregate_timestep, print_summary
from aurora.pnt.passes import (
    extract_pass_windows,
    print_pass_summary,
    save_pass_windows_csv,
    save_station_visibility_csv,
)
from aurora.pnt.report import (
    plot_coverage_map,
    plot_coverage_over_time,
    plot_experiment_comparison,
    save_summary_json,
    save_timestep_metrics_csv,
)
from aurora.tles.generate_tles_from_scratch import generate_tles_from_scratch_with_sgp

log = logger.get_logger(__name__)

# SGP4 epoch reference (J2000)
_JD_J2000 = 2451545.0
_SECONDS_PER_DAY = 86400.0


def _epoch_to_jd(epoch: Time) -> tuple[float, float]:
    """Convert astropy Time to (jd, fr) Julian date parts."""
    jd_full = epoch.jd
    jd_int = math.floor(jd_full)
    fr = jd_full - jd_int
    return float(jd_int), float(fr)


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(config: dict, overrides: dict) -> dict:
    """
    Apply dot-notation overrides to a nested config dict.
    E.g. {"satellite.altitude_m": 700000} sets config["satellite"]["altitude_m"] = 700000
    """
    import copy
    cfg = copy.deepcopy(config)
    for key, value in overrides.items():
        parts = key.split(".")
        node = cfg
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    # Auto-recalculate mean_motion if altitude changed but mean_motion not explicitly overridden
    if "satellite.altitude_m" in overrides and "constellation.mean_motion_rev_per_day" not in overrides:
        alt_m = cfg["satellite"]["altitude_m"]
        cfg["constellation"]["mean_motion_rev_per_day"] = altitude_to_mean_motion(alt_m)
        log.info(
            f"Auto-computed mean_motion={cfg['constellation']['mean_motion_rev_per_day']:.4f} "
            f"rev/day for altitude={alt_m/1000:.0f} km"
        )
    return cfg


def run_pnt_simulation(
    config: dict,
    output_dir: str,
    label: str = "run",
    save_per_step: bool = False,
) -> dict:
    """
    Run a single PNT simulation from a config dict.

    Args:
        config:        Full simulation config dict.
        output_dir:    Directory to write results.
        label:         Human-readable run label (used in filenames/plots).
        save_per_step: If True, save per-point CSVs for each time step (can be large).

    Returns:
        Overall summary dict.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── 1. Generate TLEs ─────────────────────────────────────────────────────
    cc = config["constellation"]
    tle_path = os.path.join(output_dir, f"tles_{label}.txt")
    log.info(f"Генерирую TLE: {cc['num_orbits']} орбит × {cc['num_sats_per_orbit']} спутников")
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

    # ── 2. Load Satrec objects ────────────────────────────────────────────────
    satrecs = load_satrec_from_tle_file(tle_path)
    sat_array = SatrecArray(satrecs)
    n_sats = len(satrecs)
    log.info(f"Загружено {n_sats} спутников.")

    # ── 3. Grid ───────────────────────────────────────────────────────────────
    pnt_cfg = config.get("pnt", {})
    grid_name = pnt_cfg.get("grid", "russia")
    grid_step = pnt_cfg.get("grid_step_deg", 5.0)
    min_el = pnt_cfg.get("min_elevation_deg", 10.0)
    grid_points = get_grid(grid_name, grid_step)
    log.info(f"Сетка '{grid_name}' шаг={grid_step}°: {len(grid_points)} точек, маска={min_el}°")

    # ── 4. Epoch (J2000 = TLE epoch) ─────────────────────────────────────────
    epoch = Time("2000-01-01 00:00:00", scale="tdb")
    jd0, fr0 = _epoch_to_jd(epoch)

    # ── 5. Time steps ─────────────────────────────────────────────────────────
    sim_cfg = config["simulation"]
    end_time_s = sim_cfg["end_time_hours"] * 3600.0
    step_s = sim_cfg["time_step_minutes"] * 60.0
    offset_s = sim_cfg.get("offset_ns", 0) / 1e9
    times_s = list(
        t for t in (offset_s + i * step_s for i in range(int((end_time_s - offset_s) / step_s)))
        if t < end_time_s
    )

    log.info(
        f"Симуляция: {sim_cfg['end_time_hours']}ч, шаг={sim_cfg['time_step_minutes']}мин, "
        f"{len(times_s)} шагов"
    )

    # ── 6. Main loop ──────────────────────────────────────────────────────────
    # Ground station pass-window tracking
    gs_list = config.get("ground_stations", [])
    station_points = [
        (s["latitude"], s["longitude"], s.get("elevation_m", 0.0)) for s in gs_list
    ]
    station_ts: dict[str, list] = {s["name"]: [] for s in gs_list}

    timestep_metrics: list[dict] = []
    all_point_results: list[list[dict]] = []

    for step_idx, t_s in enumerate(tqdm(times_s, desc=f"PNT [{label}]")):
        t_days = t_s / _SECONDS_PER_DAY
        fr = fr0 + t_days
        jd = jd0 + math.floor(fr)
        fr = fr - math.floor(fr)

        point_results = compute_grid_visibility_at_time(
            sat_array, grid_points, jd, fr, min_el
        )

        step_agg = aggregate_timestep(point_results)
        step_agg["time_s"] = t_s
        step_agg["time_h"] = round(t_s / 3600.0, 4)
        timestep_metrics.append(step_agg)

        # Station visibility (cheap — only a few points)
        if gs_list:
            st_results = compute_grid_visibility_at_time(
                sat_array, station_points, jd, fr, min_el
            )
            for i, s in enumerate(gs_list):
                station_ts[s["name"]].append((t_s, st_results[i]["n_sats"]))

        if save_per_step:
            all_point_results.append(point_results)
        elif len(all_point_results) < 50:
            # Always keep first 50 steps for coverage map
            all_point_results.append(point_results)

    # ── 7. Aggregate + save ───────────────────────────────────────────────────
    summary = aggregate_simulation(timestep_metrics)
    summary["label"] = label
    summary["n_satellites"] = n_sats
    summary["grid"] = grid_name
    summary["grid_step_deg"] = grid_step
    summary["n_orbits"] = cc["num_orbits"]
    summary["n_sats_per_orbit"] = cc["num_sats_per_orbit"]
    summary["altitude_km"] = config["satellite"]["altitude_m"] / 1000
    summary["inclination_deg"] = cc["inclination_degree"]

    save_timestep_metrics_csv(timestep_metrics, output_dir)
    save_summary_json(summary, output_dir, label)

    # ── 8. Pass windows ───────────────────────────────────────────────────────
    if gs_list:
        windows = extract_pass_windows(station_ts, threshold=1)
        save_station_visibility_csv(station_ts, output_dir, label)
        save_pass_windows_csv(windows, output_dir, label)
        print_pass_summary(label, windows)
        summary["pass_windows_per_station"] = {
            name: len(wins) for name, wins in windows.items()
        }

    # ── 10. Plots ─────────────────────────────────────────────────────────────
    plot_coverage_over_time(
        timestep_metrics, output_dir, label, sim_cfg["time_step_minutes"]
    )
    if all_point_results:
        plot_coverage_map(all_point_results, output_dir, label, "n_sats")
        plot_coverage_map(all_point_results, output_dir, label, "pdop")

    print_summary(label, summary)
    return summary


def run_experiment(experiment_config_path: str, output_base_dir: str | None = None) -> list[dict]:
    """
    Run a parameter sweep experiment defined in a YAML file.

    Returns list of {label, summary} dicts for comparison.
    """
    with open(experiment_config_path, encoding="utf-8") as f:
        exp = yaml.safe_load(f)

    exp_name = exp["experiment"]["name"]
    base_cfg_path = exp["experiment"]["base_config"]
    out_dir = output_base_dir or exp["experiment"].get("output_dir", f"results/{exp_name}")
    sweep_param = exp["experiment"].get("sweep_parameter", "parameter")
    runs = exp["experiment"]["runs"]

    base_config = load_config(base_cfg_path)
    all_results = []

    print(f"\n{'═'*55}")
    print(f"  Эксперимент: {exp_name}")
    print(f"  Базовый конфиг: {base_cfg_path}")
    print(f"  Конфигурации: {len(runs)}")
    print(f"{'═'*55}")

    for run in runs:
        lbl = run["label"]
        overrides = run.get("overrides", {})
        run_cfg = apply_overrides(base_config, overrides)
        run_dir = os.path.join(out_dir, lbl)

        summary = run_pnt_simulation(run_cfg, run_dir, label=lbl)
        all_results.append({"label": lbl, "summary": summary})

    # Comparison plot
    plot_experiment_comparison(all_results, out_dir, sweep_param)

    # Save combined summary table
    _save_comparison_table(all_results, out_dir, sweep_param)

    return all_results


def _save_comparison_table(results: list[dict], output_dir: str, sweep_param: str) -> None:
    """Save comparison CSV with key metrics across all runs."""
    import csv
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(output_dir, f"comparison_{sweep_param}.csv")
    keys = [
        "label", "n_satellites", "altitude_km", "n_orbits", "n_sats_per_orbit",
        "inclination_deg", "coverage_4sats_mean_pct", "coverage_4sats_min_pct",
        "pdop_p95_mean", "pdop_p95_worst", "coverage_good_pdop_mean_pct",
        "n_sats_mean_overall",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow({**r["summary"], "label": r["label"]})
    print(f"  Таблица сравнения: {path}")
