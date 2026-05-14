"""
CLI entry point for the LEO PNT simulator.

Commands:
  leopath-pnt run            -c CONFIG [-o OUTPUT] [-l LABEL]
  leopath-pnt experiment     -c EXPERIMENT_CONFIG [-o OUTPUT]
  leopath-pnt info           -c CONFIG
  leopath-pnt viz            -c CONFIG [-o OUTPUT] [-l LABEL] [--tle TLE]
  leopath-pnt cesium         -c CONFIG [-o OUTPUT] [-l LABEL] [--tle TLE]
  leopath-pnt network-metrics -c CONFIG [-o OUTPUT] [-l LABEL] [--tle TLE]
"""

import argparse
import sys

from aurora.pnt.pnt_simulator import load_config, run_experiment, run_pnt_simulation
from aurora.pnt.visualize import generate_all_visuals


def cmd_run(args):
    config = load_config(args.config)
    label = args.label or "run"
    output_dir = args.output or f"results/{label}"
    run_pnt_simulation(config, output_dir, label=label, save_per_step=args.save_steps)


def cmd_experiment(args):
    out = args.output or None
    run_experiment(args.config, output_base_dir=out)


def cmd_info(args):
    """Print simulation plan without running."""
    config = load_config(args.config)
    cc = config["constellation"]
    sc = config["satellite"]
    sim = config["simulation"]
    pnt = config.get("pnt", {})

    n_sats = cc["num_orbits"] * cc["num_sats_per_orbit"]
    n_steps = int(sim["end_time_hours"] * 60 / sim["time_step_minutes"])
    grid_name = pnt.get("grid", "russia")
    grid_step = pnt.get("grid_step_deg", 5.0)

    from aurora.pnt.grid import get_grid
    grid_pts = get_grid(grid_name, grid_step)

    print("\n" + "═" * 52)
    print("  LEO PNT — план симуляции")
    print("═" * 52)
    print(f"  Созвездие:       {cc['name']}")
    print(f"  Орбит:           {cc['num_orbits']}")
    print(f"  Спутн./орбиту:   {cc['num_sats_per_orbit']}")
    print(f"  Всего спутников: {n_sats}")
    print(f"  Высота:          {sc['altitude_m']/1000:.0f} км")
    print(f"  Наклонение:      {cc['inclination_degree']}°")
    print(f"  Продолжит.:      {sim['end_time_hours']} ч")
    print(f"  Шаг:             {sim['time_step_minutes']} мин  ({n_steps} итераций)")
    print(f"  Сетка:           {grid_name} @ {grid_step}° ({len(grid_pts)} точек)")
    print(f"  Маска возвышения:{pnt.get('min_elevation_deg', 10.0)}°")
    print("─" * 52)
    total_calcs = n_sats * len(grid_pts) * n_steps
    print(f"  Расчётов SGP4:   ~{total_calcs:,}")
    print("═" * 52 + "\n")


def cmd_viz(args):
    """Generate 3D globe and 2D ground track map from existing TLE file."""
    config = load_config(args.config)
    label = args.label or "viz"
    output_dir = args.output or f"results/{label}"
    tle_path = args.tle

    if not tle_path:
        # Try to find TLE in output dir
        import glob as _glob
        candidates = _glob.glob(f"{output_dir}/tles_*.txt")
        if not candidates:
            print(f"ERROR: no TLE file found. Use --tle to specify path.")
            sys.exit(1)
        tle_path = candidates[0]
        print(f"  Using TLE file: {tle_path}")

    gs_list = config.get("ground_stations", [])
    alt_m = config.get("satellite", {}).get("altitude_m", 1_000_000)
    min_el = config.get("pnt", {}).get("min_elevation_deg", 10.0)

    generate_all_visuals(tle_path, gs_list, output_dir, label, alt_m, min_el)


def cmd_cesium(args):
    """Generate interactive CesiumJS animated globe visualization."""
    from aurora.pnt.cesium_pnt import generate_cesium_visualization

    config = load_config(args.config)
    label = args.label or "cesium"
    output_dir = args.output or f"results/{label}"
    tle_path = args.tle

    if not tle_path:
        import glob as _glob
        candidates = _glob.glob(f"{output_dir}/tles_*.txt")
        if not candidates:
            print(f"ERROR: no TLE file found. Use --tle to specify path.")
            sys.exit(1)
        tle_path = candidates[0]
        print(f"  Using TLE file: {tle_path}")

    gs_list = config.get("ground_stations", [])
    alt_m   = config.get("satellite", {}).get("altitude_m", 1_000_000)
    min_el  = config.get("pnt", {}).get("min_elevation_deg", 10.0)

    generate_cesium_visualization(
        tle_path, gs_list, output_dir, label,
        duration_h=args.duration, step_s=args.step, speed_multiplier=args.speed,
        ion_token=args.token or "",
        altitude_m=alt_m,
        min_elevation_deg=min_el,
    )


def cmd_ranging(args):
    """Ranging accuracy: UERE budget, position error, timing error."""
    import glob as _glob
    import json
    from aurora.ranging.runner import run_ranging_analysis, compute_ranging_summary
    from aurora.ranging.report import save_ranging_report, print_summary as print_ranging

    config = load_config(args.config)
    label = args.label or "ranging"
    output_dir = args.output or f"results/{label}"

    # Load pre-computed link budget snapshots
    lb_csv = args.link_budget_csv
    if not lb_csv:
        candidates = _glob.glob(f"{output_dir}/link_budget_*.csv")
        if not candidates:
            print("ERROR: no link_budget CSV found. Run link-budget first or use --link-budget-csv.")
            sys.exit(1)
        lb_csv = candidates[0]
        print(f"  Using link budget: {lb_csv}")

    import csv
    with open(lb_csv, encoding="utf-8") as f:
        lb_snaps = [
            {k: (float(v) if v.replace(".", "").replace("-", "").isdigit() else v)
             for k, v in row.items()}
            for row in csv.DictReader(f)
        ]

    # Load pre-computed network metrics snapshots (optional — runner has fallback)
    nm_csv = args.network_csv
    if not nm_csv:
        candidates = _glob.glob(f"{output_dir}/isl_stats_*.csv")
        nm_csv = candidates[0] if candidates else None
        if nm_csv:
            print(f"  Using network stats: {nm_csv}")
        else:
            print("  No isl_stats CSV; using default ISL distance (3000 km)")

    nm_snaps = []
    if nm_csv:
        with open(nm_csv, encoding="utf-8") as f:
            nm_snaps = [
                {"time_h": float(r["time_h"]),
                 "isl_mean_dist_m": float(r["isl_mean_dist_km"]) * 1000}
                for r in csv.DictReader(f)
            ]

    # Load PNT summary for PDOP
    pnt_json = args.pnt_summary
    if not pnt_json:
        candidates = _glob.glob(f"{output_dir}/summary_*.json")
        if candidates:
            pnt_json = candidates[0]
    pnt_summary = {}
    if pnt_json:
        with open(pnt_json) as f:
            pnt_summary = json.load(f)

    gs_names = list({s["gs_name"] for s in lb_snaps})

    dual = getattr(args, "dual_freq", False)
    freq_mode = "L1+L5 dual-freq" if dual else "L1 single-freq"
    print(f"\n  Running ranging analysis: {label} "
          f"(clock {args.clock_ppb} ppb, sync {args.sync_interval}s, {freq_mode})")
    results = run_ranging_analysis(
        lb_snaps, nm_snaps, pnt_summary, config,
        oscillator_stability_ppb=args.clock_ppb,
        sync_interval_s=args.sync_interval,
        dual_frequency=dual,
    )
    summary = compute_ranging_summary(results, gs_names)
    print_ranging(label, summary)
    save_ranging_report(results, summary, gs_names, output_dir, label)
    print(f"  Results saved to: {output_dir}/")


def cmd_link_budget(args):
    """Compute link budget (FSPL, Doppler, C/N0) for all GS-satellite pairs."""
    import glob as _glob
    from aurora.link_budget.runner import run_link_budget_sim
    from aurora.link_budget.report import save_link_budget_report, print_summary

    config = load_config(args.config)
    label = args.label or "link_budget"
    output_dir = args.output or f"results/{label}"

    tle_path = args.tle
    if not tle_path:
        candidates = _glob.glob(f"{output_dir}/tles_*.txt")
        if not candidates:
            print(f"ERROR: no TLE file found. Use --tle to specify path.")
            sys.exit(1)
        tle_path = candidates[0]
        print(f"  Using TLE: {tle_path}")

    print(f"\n  Running link budget simulation: {label} ({args.band} band)")
    snapshots, gs_names = run_link_budget_sim(
        config, tle_path,
        freq_band=args.band,
        tx_power_dbw=args.tx_power,
        tx_gain_dbi=args.tx_gain,
        rx_gain_dbi=args.rx_gain,
    )
    from aurora.link_budget.report import _compute_summary
    summary = _compute_summary(snapshots, gs_names)
    print_summary(label, summary, args.band)
    save_link_budget_report(snapshots, gs_names, output_dir, label, freq_band=args.band)
    print(f"  Results saved to: {output_dir}/")


def _print_path_stability(label: str, path_m: dict) -> None:
    """Print Hypatia-style path stability summary for synchronization analysis."""
    if not path_m:
        return
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  Path Stability (Hypatia) — {label}")
    print(sep)
    print(f"  {'Pair':<30} {'Life(steps)':>11} {'Jitter(ms)':>11} {'Hops':>6} {'T-uncert(ns)':>13}")
    print("  " + "-" * 67)
    for (a, b), m in list(path_m.items())[:15]:
        lt = f"{m['path_lifetime_mean_steps']:.1f}" if m['path_lifetime_mean_steps'] else "N/A"
        jt = f"{m['latency_jitter_ms']:.3f}" if m['latency_jitter_ms'] else "N/A"
        hp = f"{m['hop_count_mean']:.1f}" if m['hop_count_mean'] else "N/A"
        tu = f"{m['timing_path_uncertainty_ns']:.1f}" if m['timing_path_uncertainty_ns'] else "N/A"
        print(f"  {a+'_to_'+b:<30} {lt:>11} {jt:>11} {hp:>6} {tu:>13}")
    print(f"{sep}\n")


def cmd_network_metrics(args):
    """Run ISL/GSL topology simulation and compute network metrics."""
    import glob as _glob
    from aurora.network_metrics.runner import run_network_sim
    from aurora.network_metrics.metrics import (
        compute_gsl_metrics, compute_isl_metrics,
        compute_latency_metrics, compute_routing_stability,
        compute_path_stability,
    )
    from aurora.network_metrics.report import save_network_metrics, print_summary

    config = load_config(args.config)
    label = args.label or "network"
    output_dir = args.output or f"results/{label}"

    tle_path = args.tle
    if not tle_path:
        candidates = _glob.glob(f"{output_dir}/tles_*.txt")
        if not candidates:
            print(f"ERROR: no TLE file found. Use --tle to specify path.")
            sys.exit(1)
        tle_path = candidates[0]
        print(f"  Using TLE: {tle_path}")

    print(f"\n  Running network metrics simulation: {label}")
    snapshots, gs_names = run_network_sim(config, tle_path)

    gsl_m    = compute_gsl_metrics(snapshots, gs_names)
    isl_m    = compute_isl_metrics(snapshots)
    lat_m    = compute_latency_metrics(snapshots, gs_names)
    stab_m   = compute_routing_stability(snapshots, gs_names)
    path_m   = compute_path_stability(snapshots, gs_names)

    print_summary(label, gsl_m, isl_m, stab_m)
    _print_path_stability(label, path_m)
    save_network_metrics(snapshots, gsl_m, isl_m, lat_m, stab_m, gs_names, output_dir, label,
                         path_stability=path_m)
    print(f"  Results saved to: {output_dir}/")


def cmd_clock_analysis(args):
    """Clock type comparison and ISL sync chain analysis."""
    from aurora.ranging.clock_analysis import run_clock_analysis, print_clock_summary
    label = args.label or "global"
    output_dir = args.output or "results/clock_analysis"
    print(f"\n  Running clock analysis: {label} "
          f"(sync={args.sync_interval:.0f}s, target={args.target_ns:.0f}ns, "
          f"max_hops={args.max_hops})")
    summary = run_clock_analysis(
        sync_interval_s=args.sync_interval,
        target_ns=args.target_ns,
        max_hops=args.max_hops,
        output_dir=output_dir,
        label=label,
    )
    print_clock_summary(label, summary, target_ns=args.target_ns)
    print(f"  Results saved to: {output_dir}/")


def cmd_raim(args):
    """RAIM integrity analysis from PNT timestep data."""
    import csv, glob as _glob, json
    from aurora.pnt.raim import run_raim_analysis, save_raim_report, print_raim_summary

    config = load_config(args.config)
    label = args.label or "raim"
    output_dir = args.output or f"results/{label}"

    pnt_csv = args.pnt_csv
    if not pnt_csv:
        # Try to find timestep_metrics.csv in output dir
        out_search = args.output or f"results/{args.label or 'phase4'}"
        candidates = _glob.glob(f"{out_search}/timestep_metrics.csv")
        if not candidates:
            print("ERROR: --pnt-csv required (path to timestep_metrics.csv)")
            sys.exit(1)
        pnt_csv = candidates[0]
        print(f"  Using PNT data: {pnt_csv}")

    with open(pnt_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"\n  Running RAIM analysis: {label} "
          f"(UERE={args.uere_m}m, HPL<{args.hpl_req}m, VPL<{args.vpl_req}m)")
    analysis = run_raim_analysis(rows, uere_m=args.uere_m,
                                 hpl_req_m=args.hpl_req, vpl_req_m=args.vpl_req)
    print_raim_summary(label, analysis)
    save_raim_report(analysis, output_dir, label)
    print(f"  Results saved to: {output_dir}/")


def cmd_combined(args):
    """Combined LEO+GLONASS multi-constellation PNT simulation."""
    from aurora.pnt.combined_sim import run_combined_simulation

    config = load_config(args.config)
    label = args.label or "combined"
    output_dir = args.output or f"results/{label}"
    mode = args.mode

    mode_desc = {
        "combined":   "LEO+GLONASS, ISB as 5th unknown (best PDOP)",
        "autonomous": "LEO-only, LPT time scale (sovereign, no GLONASS)",
        "glonass":    "GLONASS-only (comparison baseline)",
    }.get(mode, mode)

    print(f"\n  Running LEO+GLONASS simulation: {label}  [{mode}]")
    print(f"  Mode: {mode_desc}")
    print(f"  LEO: {config['constellation']['num_orbits']}x"
          f"{config['constellation']['num_sats_per_orbit']} sats @ "
          f"{config['satellite']['altitude_m']/1000:.0f} km, "
          f"{config['constellation']['inclination_degree']} deg")
    print(f"  GLONASS: 3x8=24 sats @ 19136 km, 64.8 deg")
    print(f"  Elevation masks: LEO {args.leo_min_el} deg, GLONASS {args.glo_min_el} deg")

    summary = run_combined_simulation(
        config, output_dir, label=label,
        leo_min_el_deg=args.leo_min_el,
        glo_min_el_deg=args.glo_min_el,
        mode=mode,
    )
    print(f"  Results saved to: {output_dir}/")


def cmd_time_scale(args):
    """LEO-PNT time scale and accuracy analysis (autonomous vs combined vs SDCM)."""
    import json, glob as _glob
    from aurora.pnt.time_scale import run_time_scale_analysis, print_time_scale_summary

    label = args.label or "phase4"
    output_dir = args.output or "results/time_scale"

    # Try to load PDOP values from previous simulation results
    pdop_leo = args.pdop_leo
    pdop_combined = args.pdop_combined

    # Auto-detect from combined simulation summary if not provided
    if pdop_combined is None:
        candidates = _glob.glob(f"results/*/summary_combined*.json") + \
                     _glob.glob(f"results/phase4_combined/summary_*.json")
        if candidates:
            with open(candidates[0]) as f:
                s = json.load(f)
            pdop_combined = s.get("pdop_p95_mean") or 1.67
            print(f"  PDOP combined from {candidates[0]}: {pdop_combined}")
        else:
            pdop_combined = 1.67
            print(f"  PDOP combined not found, using default: {pdop_combined}")

    if pdop_leo is None:
        # Try to find LEO-only Phase 4 summary
        candidates = _glob.glob("results/phase4/summary_*.json") + \
                     _glob.glob("results/phase3*/summary_*.json")
        if candidates:
            with open(candidates[0]) as f:
                s = json.load(f)
            pdop_leo = s.get("pdop_p95_mean") or 5.0
            print(f"  PDOP LEO-only from {candidates[0]}: {pdop_leo}")
        else:
            pdop_leo = 5.0
            print(f"  PDOP LEO-only not found, using default: {pdop_leo}")

    print(f"\n  Running time scale analysis: {label}")
    print(f"  PDOP LEO-only={pdop_leo:.2f}, PDOP combined={pdop_combined:.2f}")
    analysis = run_time_scale_analysis(
        pdop_leo=pdop_leo,
        pdop_combined=pdop_combined,
        output_dir=output_dir,
        label=label,
    )
    print_time_scale_summary(label, analysis)
    print(f"  Results saved to: {output_dir}/")


def cmd_resilience(args):
    """Satellite failure resilience analysis."""
    import json, glob as _glob
    from aurora.pnt.resilience import run_resilience_sweep, print_resilience_summary

    config = load_config(args.config)
    label = args.label or "resilience"
    output_dir = args.output or f"results/{label}"

    # Load base PNT summary for baseline metrics
    summary_candidates = _glob.glob(f"{args.output or 'results/' + (args.label or 'phase4')}/summary_*.json")
    base_summary = {}
    if summary_candidates:
        with open(summary_candidates[0]) as f:
            base_summary = json.load(f)
        print(f"  Using baseline: {summary_candidates[0]}")
    else:
        # Fallback defaults from config
        import math
        n_orb = config["constellation"]["num_orbits"]
        n_spo = config["constellation"]["num_sats_per_orbit"]
        base_summary = {
            "n_satellites": n_orb * n_spo,
            "pdop_p95_mean": 5.0,
            "coverage_4sats_mean_pct": 100.0,
            "n_sats_mean_overall": 12.0,
        }

    failure_pcts = [0, 5, 10, 15, 20, 25, 30, 40, 50]
    print(f"\n  Running resilience analysis: {label} "
          f"(base: {base_summary.get('n_satellites','?')} satellites)")
    analysis = run_resilience_sweep(base_summary, failure_pcts, output_dir, label)
    print_resilience_summary(label, analysis)
    print(f"  Results saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        prog="aurora-pnt",
        description="AURORA PNT Simulation Framework — Shiwa Network",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = sub.add_parser("run", help="Запустить одну симуляцию PNT")
    p_run.add_argument("-c", "--config", required=True, help="Путь к YAML конфигу")
    p_run.add_argument("-o", "--output", help="Директория для результатов")
    p_run.add_argument("-l", "--label", help="Метка запуска (имя в файлах)")
    p_run.add_argument("--save-steps", action="store_true",
                       help="Сохранять CSV для каждого временного шага")

    # experiment
    p_exp = sub.add_parser("experiment", help="Запустить параметрический эксперимент")
    p_exp.add_argument("-c", "--config", required=True, help="Путь к YAML эксперимента")
    p_exp.add_argument("-o", "--output", help="Базовая директория для результатов")

    # info
    p_info = sub.add_parser("info", help="Показать план симуляции без запуска")
    p_info.add_argument("-c", "--config", required=True, help="Путь к YAML конфигу")

    # viz
    p_viz = sub.add_parser("viz", help="Визуализация группировки (3D глобус + наземные треки)")
    p_viz.add_argument("-c", "--config", required=True, help="Путь к YAML конфигу")
    p_viz.add_argument("-o", "--output", help="Директория для результатов")
    p_viz.add_argument("-l", "--label", help="Метка (имя в файлах)")
    p_viz.add_argument("--tle", help="Путь к TLE файлу (если не указан — ищется в output dir)")

    # cesium
    p_ces = sub.add_parser("cesium", help="Интерактивный 3D глобус CesiumJS с анимацией спутников")
    p_ces.add_argument("-c", "--config", required=True, help="Путь к YAML конфигу")
    p_ces.add_argument("-o", "--output", help="Директория для результатов")
    p_ces.add_argument("-l", "--label", help="Метка (имя в файлах)")
    p_ces.add_argument("--tle", help="Путь к TLE файлу (если не указан — ищется в output dir)")
    p_ces.add_argument("--duration", type=float, default=24.0, help="Продолжительность (часы, по умолч. 24)")
    p_ces.add_argument("--step", type=float, default=60.0, help="Шаг позиций (секунды, по умолч. 60)")
    p_ces.add_argument("--speed", type=int, default=60, help="Множитель скорости анимации (по умолч. 60x)")
    p_ces.add_argument("--token", default="", help="Cesium Ion access token для текстур Земли")

    # ranging
    p_rng = sub.add_parser("ranging", help="Ranging accuracy: UERE, position error, timing")
    p_rng.add_argument("-c", "--config", required=True)
    p_rng.add_argument("-o", "--output")
    p_rng.add_argument("-l", "--label")
    p_rng.add_argument("--link-budget-csv", help="Path to link_budget_*.csv (auto-detect if omitted)")
    p_rng.add_argument("--network-csv",     help="Path to isl_stats_*.csv (auto-detect if omitted)")
    p_rng.add_argument("--pnt-summary",     help="Path to summary_*.json for PDOP values")
    p_rng.add_argument("--clock-ppb", type=float, default=0.1,
                       help="Oscillator stability ppb (default 0.1 = rubidium; 10 = TCXO)")
    p_rng.add_argument("--sync-interval", type=float, default=60.0,
                       help="ISL sync interval seconds (default 60)")
    p_rng.add_argument("--dual-freq", action="store_true",
                       help="Dual-frequency receiver (L1+L5): ionosphere cancelled to ~0.05m")

    # link-budget
    p_lb = sub.add_parser("link-budget", help="Link budget: FSPL, Doppler, C/N0 per GS-satellite pair")
    p_lb.add_argument("-c", "--config", required=True)
    p_lb.add_argument("-o", "--output")
    p_lb.add_argument("-l", "--label")
    p_lb.add_argument("--tle")
    p_lb.add_argument("--band", default="L1", choices=["L1","L2","L5","G1","G2","S","Ka","Ku"],
                      help="Frequency band (default: L1)")
    p_lb.add_argument("--tx-power", type=float, default=16.0, help="TX power dBW (default: 16)")
    p_lb.add_argument("--tx-gain",  type=float, default=14.0, help="TX antenna gain dBi (default: 14)")
    p_lb.add_argument("--rx-gain",  type=float, default=3.0,  help="RX antenna gain dBi (default: 3)")

    # network-metrics
    p_nm = sub.add_parser("network-metrics", help="ISL/GSL topology simulation + handover analysis")
    p_nm.add_argument("-c", "--config", required=True, help="Путь к YAML конфигу")
    p_nm.add_argument("-o", "--output", help="Директория для результатов")
    p_nm.add_argument("-l", "--label", help="Метка (имя в файлах)")
    p_nm.add_argument("--tle", help="Путь к TLE файлу (если не указан — ищется в output dir)")

    # clock-analysis
    p_ca = sub.add_parser("clock-analysis", help="Clock type comparison and ISL sync chain analysis")
    p_ca.add_argument("-o", "--output", default="results/clock_analysis")
    p_ca.add_argument("-l", "--label", default="global")
    p_ca.add_argument("--sync-interval", type=float, default=60.0, help="ISL sync interval (s)")
    p_ca.add_argument("--target-ns", type=float, default=10.0, help="Timing error target (ns)")
    p_ca.add_argument("--max-hops", type=int, default=20, help="Max hops to analyze")

    # raim
    p_raim = sub.add_parser("raim", help="RAIM integrity: HPL, VPL, availability analysis")
    p_raim.add_argument("-c", "--config", required=True)
    p_raim.add_argument("-o", "--output")
    p_raim.add_argument("-l", "--label")
    p_raim.add_argument("--pnt-csv", help="Path to timestep_metrics.csv")
    p_raim.add_argument("--hpl-req", type=float, default=40.0, help="HPL requirement (m, default 40)")
    p_raim.add_argument("--vpl-req", type=float, default=50.0, help="VPL requirement (m, default 50)")
    p_raim.add_argument("--uere-m",  type=float, default=4.3,  help="UERE (m, default 4.3 = dual-freq)")

    # combined
    p_com = sub.add_parser("combined",
                           help="LEO+GLONASS multi-constellation PNT: combined/autonomous/glonass modes")
    p_com.add_argument("-c", "--config", required=True, help="LEO constellation YAML config")
    p_com.add_argument("-o", "--output", help="Output directory")
    p_com.add_argument("-l", "--label",  help="Run label")
    p_com.add_argument("--mode", default="combined",
                       choices=["combined", "autonomous", "glonass"],
                       help="Operating mode: combined=LEO+GLONASS ISB, "
                            "autonomous=LEO-only LPT, glonass=GLONASS-only (default: combined)")
    p_com.add_argument("--leo-min-el",  type=float, default=10.0,
                       help="LEO elevation mask in degrees (default: 10)")
    p_com.add_argument("--glo-min-el",  type=float, default=5.0,
                       help="GLONASS elevation mask in degrees (default: 5)")

    # time-scale
    p_ts = sub.add_parser("time-scale",
                          help="LEO-PNT time scale analysis: LPT stability, UERE, accuracy by mode")
    p_ts.add_argument("-o", "--output", default="results/time_scale", help="Output directory")
    p_ts.add_argument("-l", "--label",  default="phase4", help="Run label")
    p_ts.add_argument("--pdop-leo",      type=float, default=None,
                      help="PDOP p95 for autonomous LEO-only mode (auto-detect if omitted)")
    p_ts.add_argument("--pdop-combined", type=float, default=None,
                      help="PDOP p95 for combined LEO+GLONASS mode (auto-detect if omitted)")

    # resilience
    p_res = sub.add_parser("resilience", help="Constellation resilience: satellite and ISL failure scenarios")
    p_res.add_argument("-c", "--config", required=True)
    p_res.add_argument("-o", "--output")
    p_res.add_argument("-l", "--label")
    p_res.add_argument("--tle")
    p_res.add_argument("--failure-pct", type=float, default=10.0,
                       help="Satellite failure percentage (default: 10)")

    args = parser.parse_args()

    dispatch = {
        "run": cmd_run,
        "experiment": cmd_experiment,
        "info": cmd_info,
        "viz": cmd_viz,
        "cesium": cmd_cesium,
        "ranging": cmd_ranging,
        "link-budget": cmd_link_budget,
        "network-metrics": cmd_network_metrics,
        "clock-analysis": cmd_clock_analysis,
        "raim": cmd_raim,
        "resilience": cmd_resilience,
        "combined":   cmd_combined,
        "time-scale": cmd_time_scale,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
