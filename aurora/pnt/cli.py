"""
CLI entry point for the LEO PNT simulator.

Commands:
  aurora-pnt run            -c CONFIG [-o OUTPUT] [-l LABEL]
  aurora-pnt experiment     -c EXPERIMENT_CONFIG [-o OUTPUT]
  aurora-pnt info           -c CONFIG
  aurora-pnt viz            -c CONFIG [-o OUTPUT] [-l LABEL] [--tle TLE]
  aurora-pnt cesium         -c CONFIG [-o OUTPUT] [-l LABEL] [--tle TLE]
  aurora-pnt network-metrics -c CONFIG [-o OUTPUT] [-l LABEL] [--tle TLE]
"""

import argparse
import sys

# Ensure UTF-8 output on Windows (cp1251 console can't handle Unicode symbols)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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


def cmd_serve(args):
    """Start a local HTTP server to view Cesium HTML without file:// restrictions."""
    import http.server
    import webbrowser
    import threading

    port = args.port
    directory = args.directory or "."

    os.chdir(directory)

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, fmt, *a):
            pass  # suppress per-request noise

    with http.server.HTTPServer(("", port), QuietHandler) as httpd:
        url = f"http://localhost:{port}/"
        print(f"  Serving '{os.path.abspath(directory)}' at {url}")
        print(f"  Open in browser: {url}")
        print(f"  Example:  {url}results/phase4/cesium_phase4_global.html")
        print(f"  Press Ctrl+C to stop.")
        if args.open:
            threading.Timer(0.5, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  Server stopped.")


def cmd_download_cesium(args):
    """Download CesiumJS once for offline globe visualization."""
    from aurora.pnt.cesium_pnt import download_cesium, is_cesium_local, _CESIUM_LOCAL_DIR, _CESIUM_VERSION

    if is_cesium_local() and not args.force:
        print(f"  CesiumJS {_CESIUM_VERSION} already installed at:")
        print(f"  {_CESIUM_LOCAL_DIR}")
        print(f"  Use --force to re-download.")
        return

    ok = download_cesium(target_dir=args.target or None, version=_CESIUM_VERSION)
    if ok:
        print(f"\n  Done. All future 'cesium' commands will use local files.")
        print(f"  No internet connection needed for globe visualization.")
    else:
        print(f"\n  Download failed. Check your internet connection and try again.")
        sys.exit(1)


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


def cmd_clock_arch(args):
    """Mixed-clock constellation analysis: OCXO/Rb/Cs per satellite tier."""
    from aurora.pnt.timing_service import run_mixed_clock_analysis

    label      = args.label  or "phase4"
    output_dir = args.output or "results/clock_arch"

    print(f"\n  Running mixed-clock constellation analysis: {label}")
    print(f"  {args.n_planes} planes x {args.n_sats} sats/plane = "
          f"{args.n_planes * args.n_sats} total")
    print(f"  Cs/plane: {args.cs_per_plane}  Rb/plane: {args.rb_per_plane}  "
          f"OCXO/plane: {args.n_sats - args.cs_per_plane - args.rb_per_plane}")

    ocxo_sync = getattr(args, "ocxo_sync_interval", 10.0)
    run_mixed_clock_analysis(
        output_dir=output_dir,
        label=label,
        n_planes=args.n_planes,
        n_sats_per_plane=args.n_sats,
        cs_per_plane=args.cs_per_plane,
        rb_per_plane=args.rb_per_plane,
        sync_interval_s=args.sync_interval,
        ocxo_sync_interval_s=ocxo_sync,
    )
    print(f"  Results saved to: {output_dir}/")


def cmd_timing_service(args):
    """AURORA-T timing service: PTP/NTP grandmaster accuracy by clock type."""
    from aurora.pnt.timing_service import run_timing_service_analysis

    label = args.label or "phase4"
    output_dir = args.output or "results/timing_service"

    print(f"\n  Running AURORA-T timing service analysis: {label}")
    print(f"  ISL chain: {args.isl_hops} hops, sync interval {args.sync_interval} s")
    print(f"  UERE: autonomous={args.uere_autonomous} m, combined={args.uere_combined} m")

    run_timing_service_analysis(
        output_dir=output_dir,
        label=label,
        n_isl_hops=args.isl_hops,
        sync_interval_s=args.sync_interval,
        uere_autonomous_m=args.uere_autonomous,
        uere_combined_m=args.uere_combined,
    )
    print(f"  Results saved to: {output_dir}/")


def cmd_sdcm(args):
    """SDCM differential corrections: Mode C UERE, coverage, CEP improvement."""
    from aurora.pnt.sdcm import run_sdcm_analysis, print_sdcm_summary
    label      = args.label  or "phase3"
    output_dir = args.output or "results/sdcm"
    print(f"\n  Running SDCM differential corrections analysis: {label}")
    result = run_sdcm_analysis(output_dir=output_dir, label=label,
                               pdop_autonomous=args.pdop_auto,
                               pdop_combined=args.pdop_combined,
                               grid_step_deg=args.grid_step)
    print_sdcm_summary(label, result)
    print(f"  Results saved to: {output_dir}/")


def cmd_isl_link(args):
    """ISL RF link budget: Ka/V/Optical bands, margin vs range."""
    from aurora.pnt.isl_link_budget import run_isl_link_budget_analysis, print_isl_link_summary
    label      = args.label  or "phase3"
    output_dir = args.output or "results/isl_link"
    print(f"\n  Running ISL link budget analysis: {label}")
    result = run_isl_link_budget_analysis(output_dir=output_dir, label=label,
                                          n_planes=args.n_planes,
                                          n_sats_per_plane=args.n_sats)
    print_isl_link_summary(label, result)
    print(f"  Results saved to: {output_dir}/")


def cmd_anti_jam(args):
    """Anti-jamming analysis: J/S ratio, effective jamming radius vs GPS."""
    from aurora.pnt.anti_jam import run_anti_jam_analysis, print_anti_jam_summary
    label      = args.label  or "phase3"
    output_dir = args.output or "results/anti_jam"
    print(f"\n  Running anti-jamming analysis: {label}  "
          f"(elevation {args.elevation} deg)")
    result = run_anti_jam_analysis(output_dir=output_dir, label=label,
                                   elevation_deg=args.elevation)
    print_anti_jam_summary(label, result)
    print(f"  Results saved to: {output_dir}/")


def cmd_tesla_mac(args):
    """TESLA MAC anti-spoofing: key chain, vulnerability window, attack mitigation."""
    from aurora.pnt.tesla_mac import run_tesla_mac_analysis, print_tesla_summary
    label      = args.label  or "phase3"
    output_dir = args.output or "results/tesla_mac"
    print(f"\n  Running TESLA MAC anti-spoofing analysis: {label}")
    result = run_tesla_mac_analysis(output_dir=output_dir, label=label)
    print_tesla_summary(label, result)
    print(f"  Results saved to: {output_dir}/")


def cmd_multipath(args):
    """Multipath environment model: CEP by scenario vs GPS."""
    from aurora.pnt.multipath import run_multipath_analysis, print_multipath_summary
    label      = args.label  or "phase3"
    output_dir = args.output or "results/multipath"
    print(f"\n  Running multipath environment analysis: {label}")
    result = run_multipath_analysis(output_dir=output_dir, label=label,
                                    pdop_leo=args.pdop_leo,
                                    pdop_gps=args.pdop_gps)
    print_multipath_summary(label, result)
    print(f"  Results saved to: {output_dir}/")


def cmd_acquisition(args):
    """Signal acquisition and TTFF analysis: Doppler search, cold/warm/hot start."""
    from aurora.pnt.acquisition import run_acquisition_analysis, print_acquisition_summary
    label      = args.label  or "phase3"
    output_dir = args.output or "results/acquisition"
    print(f"\n  Running signal acquisition & TTFF analysis: {label}  "
          f"(clock {args.clock_ppm} ppm, {args.channels} channels)")
    result = run_acquisition_analysis(output_dir=output_dir, label=label,
                                      rx_clock_ppm=args.clock_ppm,
                                      parallel_ch=args.channels)
    print_acquisition_summary(label, result)
    print(f"  Results saved to: {output_dir}/")


def cmd_deorbit(args):
    """Orbital lifetime and deorbit analysis: decay, IADC compliance, delta-V."""
    from aurora.pnt.deorbit import run_deorbit_analysis, print_deorbit_summary
    label      = args.label  or "phase3"
    output_dir = args.output or "results/deorbit"
    print(f"\n  Running orbital lifetime & deorbit analysis: {label}  "
          f"({args.altitude_km:.0f} km, {args.n_sats} sats)")
    result = run_deorbit_analysis(output_dir=output_dir, label=label,
                                  altitude_km=args.altitude_km,
                                  n_sats=args.n_sats)
    print_deorbit_summary(label, result)
    print(f"  Results saved to: {output_dir}/")


def cmd_monte_carlo(args):
    """Monte Carlo position accuracy: empirical CEP histograms vs analytical estimates."""
    from aurora.pnt.monte_carlo import run_monte_carlo, print_mc_summary

    label      = args.label  or "phase3"
    output_dir = args.output or "results/monte_carlo"

    phase_configs = [
        {"mode": "autonomous",    "n_sats": args.n_sats_auto,
         "pdop_target": args.pdop_auto,    "name": "Autonomous (LEO-only)"},
        {"mode": "combined",      "n_sats": args.n_sats_comb,
         "pdop_target": args.pdop_combined, "name": "Combined (LEO+GLONASS)"},
        {"mode": "combined_sdcm", "n_sats": args.n_sats_comb,
         "pdop_target": args.pdop_combined, "name": "Combined+SDCM"},
    ]

    print(f"\n  Running Monte Carlo position accuracy simulation: {label}")
    print(f"  Trials: {args.n_trials:,}  |  Seed: {args.seed}")
    print(f"  Autonomous: {args.n_sats_auto} sats, PDOP={args.pdop_auto}")
    print(f"  Combined:   {args.n_sats_comb} sats, PDOP={args.pdop_combined}")

    results = run_monte_carlo(
        output_dir=output_dir,
        label=label,
        n_trials=args.n_trials,
        seed=args.seed,
        phase_configs=phase_configs,
    )
    print_mc_summary(label, results)
    print(f"  Results saved to: {output_dir}/")


def cmd_freq_plan(args):
    """Frequency plan: ITU allocation, interference analysis, Doppler profile."""
    from aurora.pnt.frequency_plan import run_frequency_plan_analysis, print_frequency_plan_summary

    label      = args.label  or "phase3"
    output_dir = args.output or "results/freq_plan"

    print(f"\n  Running frequency plan analysis: {label}")
    print(f"  Constellation: {args.n_sats} sats @ {args.altitude_km:.0f} km")
    print(f"  TX: {args.tx_power:.1f} dBW, TX ant: {args.tx_gain:.1f} dBi")

    result = run_frequency_plan_analysis(
        output_dir=output_dir,
        label=label,
        n_sats=args.n_sats,
        altitude_m=args.altitude_km * 1000,
        tx_power_dbw=args.tx_power,
        tx_gain_dbi=args.tx_gain,
        rx_gain_dbi=args.rx_gain,
    )
    print_frequency_plan_summary(label, result)
    print(f"  Results saved to: {output_dir}/")


def cmd_isl_ranging(args):
    """ISL ranging: autonomous orbit determination via crosslink distance measurements."""
    from aurora.pnt.isl_ranging import run_isl_ranging_analysis, print_isl_summary

    label      = args.label  or "phase3"
    output_dir = args.output or "results/isl_ranging"

    print(f"\n  Running ISL ranging & autonomous OD analysis: {label}")
    print(f"  Constellation: {args.n_planes}x{args.n_sats}/plane @ {args.altitude_km:.0f} km, "
          f"{args.inclination_deg:.0f} deg")
    print(f"  Ranging mode: {args.mode}  |  MCS stations: {args.n_stations}")

    result = run_isl_ranging_analysis(
        output_dir=output_dir,
        label=label,
        n_planes=args.n_planes,
        n_sats_per_plane=args.n_sats,
        altitude_m=args.altitude_km * 1000,
        inclination_deg=args.inclination_deg,
        n_mcs_stations=args.n_stations,
        ranging_mode=args.mode,
        base_uere_m=args.uere_m,
        pdop=args.pdop,
    )
    print_isl_summary(label, result)
    print(f"  Results saved to: {output_dir}/")


def cmd_user_link_budget(args):
    """User terminal RF link budget: C/N0, pseudorange noise, margins vs elevation."""
    from aurora.pnt.link_budget import run_link_budget_analysis, print_link_budget_summary
    label      = args.label  or "phase4"
    output_dir = args.output or "results/user_link_budget"
    print(f"\n  Running user terminal link budget: {label}  "
          f"(altitude {args.altitude_km:.0f} km)")
    result = run_link_budget_analysis(output_dir=output_dir, label=label,
                                      altitude_m=args.altitude_km * 1000)
    print_link_budget_summary(label, result)
    print(f"  Results saved to: {output_dir}/")


def cmd_eclipse(args):
    """Eclipse / Earth shadow analysis: fraction, duration, battery sizing, OCXO thermal."""
    from aurora.pnt.eclipse import run_eclipse_analysis, print_eclipse_summary
    label      = args.label  or "phase4"
    output_dir = args.output or "results/eclipse"
    print(f"\n  Running eclipse analysis: {label}  "
          f"(altitude {args.altitude_km:.0f} km, {args.n_sats} sats)")
    result = run_eclipse_analysis(output_dir=output_dir, label=label,
                                  altitude_m=args.altitude_km * 1000,
                                  n_sats=args.n_sats)
    print_eclipse_summary(label, result)
    print(f"  Results saved to: {output_dir}/")


def cmd_nav_message(args):
    """Navigation message structure: bit budget, frame, TTFF, authentication."""
    from aurora.pnt.nav_message import run_nav_message_analysis, print_nav_message_summary
    label      = args.label  or "phase4"
    output_dir = args.output or "results/nav_message"
    print(f"\n  Running navigation message analysis: {label}  "
          f"({args.data_rate} bps, {args.n_sats} sats)")
    result = run_nav_message_analysis(output_dir=output_dir, label=label,
                                      data_rate_bps=args.data_rate,
                                      n_sats=args.n_sats)
    print_nav_message_summary(label, result)
    print(f"  Results saved to: {output_dir}/")


def cmd_iono_correction(args):
    """Ionospheric correction: Klobuchar vs NeQuick-G vs dual-freq residuals."""
    from aurora.pnt.iono_correction import run_iono_analysis, print_iono_summary
    label      = args.label  or "phase4"
    output_dir = args.output or "results/iono_correction"
    print(f"\n  Running ionospheric correction analysis: {label}")
    result = run_iono_analysis(output_dir=output_dir, label=label)
    print_iono_summary(label, result)
    print(f"  Results saved to: {output_dir}/")


def cmd_user_dynamics(args):
    """User dynamics: moving platform accuracy (aviation/maritime/land vehicle)."""
    from aurora.pnt.user_dynamics import run_user_dynamics_analysis, print_user_dynamics_summary
    label      = args.label  or "phase4"
    output_dir = args.output or "results/user_dynamics"
    print(f"\n  Running user dynamics analysis: {label}")
    result = run_user_dynamics_analysis(output_dir=output_dir, label=label)
    print_user_dynamics_summary(label, result)
    print(f"  Results saved to: {output_dir}/")


def cmd_ground_od(args):
    """Ground-only orbit determination: RAC accuracy, prediction degradation."""
    from aurora.pnt.ground_od import run_ground_od_analysis, print_ground_od_summary
    label      = args.label  or "phase4"
    output_dir = args.output or "results/ground_od"
    print(f"\n  Running ground orbit determination analysis: {label}")
    result = run_ground_od_analysis(output_dir=output_dir, label=label)
    print_ground_od_summary(label, result)
    print(f"  Results saved to: {output_dir}/")


def cmd_deployment(args):
    """Deployment timeline: launch sequence, coverage ramp-up, cost breakdown."""
    from aurora.pnt.deployment import run_deployment_analysis, print_deployment_summary
    label      = args.label  or "full"
    output_dir = args.output or "results/deployment"
    print(f"\n  Running deployment timeline analysis: {label}")
    result = run_deployment_analysis(output_dir=output_dir, label=label)
    print_deployment_summary(label, result)
    print(f"  Results saved to: {output_dir}/")


def cmd_conjunction(args):
    """Conjunction probability: Monte Carlo debris collision risk analysis."""
    from aurora.pnt.conjunction_pc import run_conjunction_analysis, print_conjunction_summary
    label      = args.label  or "phase4"
    output_dir = args.output or "results/conjunction"
    print(f"\n  Running conjunction probability analysis: {label}  "
          f"({args.n_trials:,} trials)")
    result = run_conjunction_analysis(output_dir=output_dir, label=label,
                                      n_trials=args.n_trials)
    print_conjunction_summary(label, result)
    print(f"  Results saved to: {output_dir}/")


def cmd_coverage_maps(args):
    """Geographic coverage maps: N_vis and PDOP heatmaps by phase."""
    from aurora.pnt.coverage_maps import run_coverage_maps_analysis, print_coverage_summary
    label      = args.label  or "all_phases"
    output_dir = args.output or "results/coverage_maps"
    phases = list(range(5)) if args.all_phases else [args.phase]
    print(f"\n  Running coverage maps analysis: {label}  "
          f"(phases: {phases})")
    result = run_coverage_maps_analysis(output_dir=output_dir, label=label,
                                        phases=phases)
    print_coverage_summary(label, result)
    print(f"  Results saved to: {output_dir}/")


def cmd_power_budget(args):
    """Satellite power budget: solar array, battery, subsystem loads, margins."""
    from aurora.pnt.power_budget import run_power_budget_analysis, print_power_budget_summary
    label      = args.label  or "phase4"
    output_dir = args.output or "results/power_budget"
    print(f"\n  Running power budget analysis: {label}  "
          f"(altitude {args.altitude_km:.0f} km, {args.n_sats} sats)")
    result = run_power_budget_analysis(output_dir=output_dir, label=label,
                                       altitude_m=args.altitude_km * 1000,
                                       n_sats=args.n_sats)
    print_power_budget_summary(label, result)
    print(f"  Results saved to: {output_dir}/")


def cmd_mass_budget(args):
    """Satellite mass and volume budget: subsystems, propellant, wet mass, fleet totals."""
    from aurora.pnt.mass_budget import run_mass_budget_analysis, print_mass_budget_summary
    label      = args.label  or "phase4"
    output_dir = args.output or "results/mass_budget"
    print(f"\n  Running mass budget analysis: {label}  "
          f"({args.n_sats} sats, {args.mission_years:.0f} yr mission)")
    result = run_mass_budget_analysis(output_dir=output_dir, label=label,
                                      n_sats=args.n_sats,
                                      mission_years=args.mission_years)
    print_mass_budget_summary(label, result)
    print(f"  Results saved to: {output_dir}/")


def cmd_thermal(args):
    """Satellite thermal analysis: orbital temperature profile, OCXO oven budget."""
    from aurora.pnt.thermal import run_thermal_analysis, print_thermal_summary
    label      = args.label  or "phase4"
    output_dir = args.output or "results/thermal"
    print(f"\n  Running thermal analysis: {label}  "
          f"(altitude {args.altitude_km:.0f} km)")
    result = run_thermal_analysis(output_dir=output_dir, label=label,
                                  altitude_m=args.altitude_km * 1000)
    print_thermal_summary(label, result)
    print(f"  Results saved to: {output_dir}/")


def cmd_timing_chain(args):
    """Timing chain: Cs→Rb→OCXO→user 1PPS ADEV, ISL transfer, holdover."""
    from aurora.pnt.timing_chain import run_timing_chain_analysis, print_timing_chain_summary
    label      = args.label  or "phase4"
    output_dir = args.output or "results/timing_chain"
    print(f"\n  Running timing chain analysis: {label}  "
          f"(max {args.max_hops} ISL hops)")
    result = run_timing_chain_analysis(output_dir=output_dir, label=label,
                                       n_hops_max=args.max_hops)
    print_timing_chain_summary(label, result)
    print(f"  Results saved to: {output_dir}/")


def cmd_radiation(args):
    """Radiation environment: TID, SEU rates, OCXO drift vs Al shielding."""
    from aurora.pnt.radiation import run_radiation_analysis, print_radiation_summary
    label      = args.label  or "phase4"
    output_dir = args.output or "results/radiation"
    print(f"\n  Running radiation analysis: {label}")
    result = run_radiation_analysis(output_dir=output_dir, label=label)
    print_radiation_summary(label, result)
    print(f"  Results saved to: {output_dir}/")


def cmd_relativistic(args):
    """Relativistic corrections: Sagnac, gravitational redshift, clock bias."""
    from aurora.pnt.relativistic import run_relativistic_analysis, print_relativistic_summary
    label      = args.label  or "phase4"
    output_dir = args.output or "results/relativistic"
    print(f"\n  Running relativistic corrections analysis: {label}")
    result = run_relativistic_analysis(output_dir=output_dir, label=label)
    print_relativistic_summary(label, result)
    print(f"  Results saved to: {output_dir}/")


def cmd_troposphere(args):
    """Tropospheric delay: ZHD/ZWD Saastamoinen, NMF, seasonal variation."""
    from aurora.pnt.troposphere import run_troposphere_analysis, print_troposphere_summary
    label      = args.label  or "phase4"
    output_dir = args.output or "results/troposphere"
    print(f"\n  Running troposphere analysis: {label}")
    result = run_troposphere_analysis(output_dir=output_dir, label=label)
    print_troposphere_summary(label, result)
    print(f"  Results saved to: {output_dir}/")


def cmd_reliability(args):
    """Reliability & MTBF: subsystem R(t), fleet degradation, spare satellites."""
    from aurora.pnt.reliability import run_reliability_analysis, print_reliability_summary
    label      = args.label  or "phase4"
    output_dir = args.output or "results/reliability"
    print(f"\n  Running reliability analysis: {label}")
    result = run_reliability_analysis(output_dir=output_dir, label=label)
    print_reliability_summary(label, result)
    print(f"  Results saved to: {output_dir}/")


def cmd_ppp_convergence(args):
    """PPP convergence: Kalman filter, LEO vs MEO, dual vs single freq."""
    from aurora.pnt.ppp_convergence import run_ppp_convergence_analysis, print_ppp_summary
    label      = args.label  or "phase4"
    output_dir = args.output or "results/ppp_convergence"
    print(f"\n  Running PPP convergence analysis: {label}")
    result = run_ppp_convergence_analysis(output_dir=output_dir, label=label)
    print_ppp_summary(label, result)
    print(f"  Results saved to: {output_dir}/")


def cmd_signal_quality(args):
    """Signal quality: SISA, SISRA, URE, UERE budget vs GPS/Galileo."""
    from aurora.pnt.signal_quality import run_signal_quality_analysis, print_signal_quality_summary
    label      = args.label  or "phase4"
    output_dir = args.output or "results/signal_quality"
    print(f"\n  Running signal quality analysis: {label}")
    result = run_signal_quality_analysis(output_dir=output_dir, label=label)
    print_signal_quality_summary(label, result)
    print(f"  Results saved to: {output_dir}/")


def cmd_adcs(args):
    """ADCS requirements: pointing budget, disturbance torques, reaction wheels, ISL."""
    from aurora.pnt.adcs_requirements import run_adcs_analysis, print_adcs_summary
    label      = args.label  or "phase4"
    output_dir = args.output or "results/adcs"
    print(f"\n  Running ADCS requirements analysis: {label}")
    result = run_adcs_analysis(output_dir=output_dir, label=label)
    print_adcs_summary(label, result)
    print(f"  Results saved to: {output_dir}/")


def cmd_ground_network(args):
    """Ground network optimization: MCS geometry, OD quality, coverage analysis."""
    from aurora.pnt.ground_network_opt import run_ground_network_analysis, print_ground_network_summary
    label      = args.label  or "phase4"
    output_dir = args.output or "results/ground_network"
    print(f"\n  Running ground network analysis: {label}")
    result = run_ground_network_analysis(output_dir=output_dir, label=label)
    print_ground_network_summary(label, result)
    print(f"  Results saved to: {output_dir}/")


def cmd_signal_design(args):
    """Signal design: modulation (BOC/BPSK/TMBOC), code sequences (Gold/Weil/Memory), nav message."""
    from aurora.pnt.signal_design import run_signal_design_analysis, print_signal_design_summary
    label      = args.label  or "phase4"
    output_dir = args.output or "results/signal_design"
    print(f"\n  Running signal design analysis: {label}")
    result = run_signal_design_analysis(output_dir=output_dir, label=label)
    print_signal_design_summary(label, result)
    print(f"  Results saved to: {output_dir}/")


def cmd_competitor_analysis(args):
    """Competitor analysis: AURORA vs GLONASS, GPS, Galileo, LEO PNT systems."""
    from aurora.pnt.competitor_analysis import run_competitor_analysis, print_competitor_summary
    label      = args.label  or "phase4"
    output_dir = args.output or "results/competitor_analysis"
    print(f"\n  Running competitor analysis: {label}")
    result = run_competitor_analysis(output_dir=output_dir, label=label)
    print_competitor_summary(label, result)
    print(f"  Results saved to: {output_dir}/")


def cmd_user_segment(args):
    """User segment: Doppler, PLL bandwidth, channel count, receiver classes."""
    from aurora.pnt.user_segment import run_user_segment_analysis, print_user_segment_summary
    label = args.label or "phase4"
    output_dir = args.output or f"results/user_segment"
    print(f"\n  Running user segment analysis: {label}")
    results = run_user_segment_analysis(output_dir, label)
    print_user_segment_summary(label, results)
    print(f"  Results saved to: {output_dir}/")


def cmd_itu_coordination(args):
    """ITU/МСЭ coordination: PSD, SSC matrix, band plan, OOB emission mask."""
    from aurora.pnt.itu_coordination import run_itu_coordination_analysis, print_itu_summary
    label = args.label or "phase4"
    output_dir = args.output or f"results/itu_coordination"
    print(f"\n  Running ITU coordination analysis: {label}")
    results = run_itu_coordination_analysis(output_dir, label)
    print_itu_summary(label, results)
    print(f"  Results saved to: {output_dir}/")


def cmd_rtk_ppp(args):
    """PPP-RTK architecture: accuracy vs baseline, convergence, latency, RSN map."""
    from aurora.pnt.rtk_ppp import run_rtk_ppp_analysis, print_rtk_ppp_summary
    label = args.label or "phase4"
    output_dir = args.output or f"results/rtk_ppp"
    print(f"\n  Running PPP-RTK analysis: {label}")
    results = run_rtk_ppp_analysis(output_dir, label)
    print_rtk_ppp_summary(label, results)
    print(f"  Results saved to: {output_dir}/")


def cmd_real_data(args):
    """Интеграция с реальными источниками данных (IGS, VMF3, SLR)."""
    from aurora.pnt.real_data import run_real_data_analysis, print_real_data_summary
    label = args.label or "phase5"
    output_dir = args.output or "results/real_data"
    print(f"\n  Running real data integration: {label}")
    r = run_real_data_analysis(output_dir, label); print_real_data_summary(label, r)
    print(f"  Results saved to: {output_dir}/")


def cmd_validate(args):
    """Валидация моделей (Klobuchar §11, Saastamoinen §34, POD §47) на реальных данных."""
    from aurora.pnt.validate_models import run_validate_analysis, print_validate_summary
    label = args.label or "phase5"
    output_dir = args.output or "results/validate"
    print(f"\n  Running model validation: {label}")
    r = run_validate_analysis(output_dir, label); print_validate_summary(label, r)
    print(f"  Results saved to: {output_dir}/")


def cmd_code_gen(args):
    """Генератор Weil/Gold/Extended Memory кодов для L1/L5."""
    from aurora.pnt.code_gen import run_code_gen_analysis, print_code_gen_summary
    label = args.label or "phase5"
    output_dir = args.output or "results/code_gen"
    print(f"\n  Running code generator analysis: {label}")
    r = run_code_gen_analysis(output_dir, label); print_code_gen_summary(label, r)
    print(f"  Results saved to: {output_dir}/")


def cmd_sdr_receiver(args):
    """Software-defined receiver: захват, слежение, TTFF."""
    from aurora.pnt.sdr_receiver import run_sdr_receiver_analysis, print_sdr_receiver_summary
    label = args.label or "phase5"
    output_dir = args.output or "results/sdr"
    print(f"\n  Running SDR receiver analysis: {label}")
    r = run_sdr_receiver_analysis(output_dir, label); print_sdr_receiver_summary(label, r)
    print(f"  Results saved to: {output_dir}/")


def cmd_risks(args):
    """Реестр рисков AURORA PNT (P×S матрица)."""
    from aurora.pnt.risks import run_risks_analysis, print_risks_summary
    label = args.label or "phase5"
    output_dir = args.output or "results/risks"
    print(f"\n  Running risk register analysis: {label}")
    r = run_risks_analysis(output_dir, label); print_risks_summary(label, r)
    print(f"  Results saved to: {output_dir}/")


def cmd_schedule(args):
    """График работ AURORA PNT (Gantt + критический путь)."""
    from aurora.pnt.schedule import run_schedule_analysis, print_schedule_summary
    label = args.label or "phase5"
    output_dir = args.output or "results/schedule"
    print(f"\n  Running schedule analysis: {label}")
    r = run_schedule_analysis(output_dir, label); print_schedule_summary(label, r)
    print(f"  Results saved to: {output_dir}/")


def cmd_cybersec(args):
    """Модель угроз (STRIDE/PASTA): угрозы, риск-матрица, меры."""
    from aurora.pnt.cybersec_threat import run_cybersec_analysis, print_cybersec_summary
    label = args.label or "phase5"
    output_dir = args.output or "results/cybersec"
    print(f"\n  Running cybersec threat analysis: {label}")
    r = run_cybersec_analysis(output_dir, label); print_cybersec_summary(label, r)
    print(f"  Results saved to: {output_dir}/")


def cmd_e2e_pipeline(args):
    """Сквозная end-to-end PVT симуляция 24 ч для 4 пользователей."""
    from aurora.pnt.e2e_pipeline import run_e2e_pipeline_analysis, print_e2e_pipeline_summary
    label = args.label or "phase5"
    output_dir = args.output or "results/e2e"
    print(f"\n  Running end-to-end PVT pipeline: {label}")
    r = run_e2e_pipeline_analysis(output_dir, label); print_e2e_pipeline_summary(label, r)
    print(f"  Results saved to: {output_dir}/")


def cmd_pvt_montecarlo(args):
    """End-to-end Monte-Carlo PVT error budget."""
    from aurora.pnt.pvt_montecarlo import run_pvt_montecarlo_analysis, print_pvt_montecarlo_summary
    label = args.label or "phase5"
    output_dir = args.output or "results/pvt_montecarlo"
    print(f"\n  Running PVT Monte-Carlo analysis: {label}")
    results = run_pvt_montecarlo_analysis(output_dir, label)
    print_pvt_montecarlo_summary(label, results)
    print(f"  Results saved to: {output_dir}/")


def cmd_dop_temporal(args):
    """Temporal DOP / availability maps over the constellation."""
    from aurora.pnt.dop_temporal import run_dop_temporal_analysis, print_dop_temporal_summary
    label = args.label or "phase5"
    output_dir = args.output or "results/dop_temporal"
    print(f"\n  Running temporal DOP analysis: {label}")
    results = run_dop_temporal_analysis(output_dir, label)
    print_dop_temporal_summary(label, results)
    print(f"  Results saved to: {output_dir}/")


def cmd_pod(args):
    """Precise orbit determination: force budget, R/A/C accuracy, SLR, observability."""
    from aurora.pnt.pod_filter import run_pod_analysis, print_pod_summary
    label = args.label or "phase5"
    output_dir = args.output or "results/pod"
    print(f"\n  Running POD analysis: {label}")
    results = run_pod_analysis(output_dir, label)
    print_pod_summary(label, results)
    print(f"  Results saved to: {output_dir}/")


def cmd_autonav(args):
    """Autonomous navigation via ISL ranging: ephemeris growth, rank deficiency."""
    from aurora.pnt.autonav_isl import run_autonav_analysis, print_autonav_summary
    label = args.label or "phase5"
    output_dir = args.output or "results/autonav"
    print(f"\n  Running AutoNav ISL analysis: {label}")
    results = run_autonav_analysis(output_dir, label)
    print_autonav_summary(label, results)
    print(f"  Results saved to: {output_dir}/")


def cmd_araim(args):
    """ARAIM: solution separation, VPL/HPL, P_HMI risk tree."""
    from aurora.pnt.araim import run_araim_analysis, print_araim_summary
    label = args.label or "phase5"
    output_dir = args.output or "results/araim"
    print(f"\n  Running ARAIM analysis: {label}")
    results = run_araim_analysis(output_dir, label)
    print_araim_summary(label, results)
    print(f"  Results saved to: {output_dir}/")


def cmd_integrity(args):
    """Integrity budget: Stanford diagram, LPV-200/CAT-I availability, ISM."""
    from aurora.pnt.integrity_budget import run_integrity_analysis, print_integrity_summary
    label = args.label or "phase5"
    output_dir = args.output or "results/integrity"
    print(f"\n  Running integrity budget analysis: {label}")
    results = run_integrity_analysis(output_dir, label)
    print_integrity_summary(label, results)
    print(f"  Results saved to: {output_dir}/")


def cmd_cost(args):
    """Life-cycle cost model: CAPEX/OPEX, learning curve, LCC, sensitivity."""
    from aurora.pnt.cost_model import run_cost_analysis, print_cost_summary
    label = args.label or "phase5"
    output_dir = args.output or "results/cost"
    print(f"\n  Running cost model analysis: {label}")
    results = run_cost_analysis(output_dir, label)
    print_cost_summary(label, results)
    print(f"  Results saved to: {output_dir}/")


def cmd_production(args):
    """Production & AIT for 300 sats: flow, rate, throughput, deployment."""
    from aurora.pnt.production_ait import run_production_analysis, print_production_summary
    label = args.label or "phase5"
    output_dir = args.output or "results/production"
    print(f"\n  Running production/AIT analysis: {label}")
    results = run_production_analysis(output_dir, label)
    print_production_summary(label, results)
    print(f"  Results saved to: {output_dir}/")


def cmd_launch_campaign(args):
    """Launch & deployment: manifest, RAAN phasing, timeline, dV raise budget."""
    from aurora.pnt.launch_campaign import run_launch_campaign_analysis, print_launch_campaign_summary
    label = args.label or "phase5"
    output_dir = args.output or "results/launch"
    print(f"\n  Running launch campaign analysis: {label}")
    results = run_launch_campaign_analysis(output_dir, label)
    print_launch_campaign_summary(label, results)
    print(f"  Results saved to: {output_dir}/")


def cmd_ground_segment(args):
    """Ground segment (MCS/TT&C): architecture, latency, redundancy."""
    from aurora.pnt.ground_segment import run_ground_segment_analysis, print_ground_segment_summary
    label = args.label or "phase5"
    output_dir = args.output or "results/ground_segment"
    print(f"\n  Running ground segment analysis: {label}")
    results = run_ground_segment_analysis(output_dir, label)
    print_ground_segment_summary(label, results)
    print(f"  Results saved to: {output_dir}/")


def cmd_system_concept(args):
    """Concept illustrations: system overview, service scenarios, LEO vs MEO, signal flow."""
    from aurora.pnt.system_concept import run_system_concept
    label      = args.label  or "phase4"
    output_dir = args.output or "results/system_concept"
    results = run_system_concept(output_dir, label)
    print(f"  Generated {len(results['figures'])} concept figures in {output_dir}/")


def cmd_station_keeping(args):
    """Station keeping: J2 RAAN drift, atmospheric drag, delta-V budget."""
    from aurora.pnt.station_keeping import run_station_keeping_analysis, print_station_keeping_summary
    label      = args.label  or "phase4"
    output_dir = args.output or "results/station_keeping"
    print(f"\n  Running station keeping analysis: {label}")
    result = run_station_keeping_analysis(output_dir=output_dir, label=label)
    print_station_keeping_summary(label, result)
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

    # serve
    p_srv = sub.add_parser(
        "serve",
        help="Запустить HTTP-сервер для просмотра Cesium HTML без ограничений file://"
    )
    p_srv.add_argument("--port", type=int, default=8765, help="Порт (по умолчанию 8765)")
    p_srv.add_argument("--directory", default=None, help="Корневая директория (по умолчанию текущая)")
    p_srv.add_argument("--open", action="store_true", help="Открыть браузер автоматически")

    # download-cesium
    p_dlc = sub.add_parser(
        "download-cesium",
        help="Скачать CesiumJS один раз для работы глобуса без интернета"
    )
    p_dlc.add_argument(
        "--target", default=None,
        help="Папка назначения (по умолчанию assets/cesium/ в корне проекта)"
    )
    p_dlc.add_argument(
        "--force", action="store_true",
        help="Перезагрузить, даже если уже установлен"
    )

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

    # clock-arch
    p_ca2 = sub.add_parser("clock-arch",
                           help="Mixed-clock architecture: OCXO/Rb/Cs per tier, ISL chain, holdover")
    p_ca2.add_argument("-o", "--output", default="results/clock_arch")
    p_ca2.add_argument("-l", "--label",  default="phase4")
    p_ca2.add_argument("--n-planes",     type=int,   default=15,
                       help="Number of orbital planes (default: 15)")
    p_ca2.add_argument("--n-sats",       type=int,   default=20,
                       help="Satellites per plane (default: 20)")
    p_ca2.add_argument("--cs-per-plane", type=int,   default=1,
                       help="Cs timing anchors per plane (default: 1)")
    p_ca2.add_argument("--rb-per-plane", type=int,   default=3,
                       help="Rb timing relays per plane (default: 3)")
    p_ca2.add_argument("--sync-interval", type=float, default=60.0,
                       help="ISL sync interval for Cs/Rb tiers in seconds (default: 60)")
    p_ca2.add_argument("--ocxo-sync-interval", type=float, default=10.0,
                       help="ISL sync interval for OCXO tier in seconds (default: 10, must be <=24 for PTP Class 25)")

    # timing-service
    p_tms = sub.add_parser("timing-service",
                           help="AURORA-T: PTP/NTP grandmaster accuracy, timing protocol stack")
    p_tms.add_argument("-o", "--output", default="results/timing_service", help="Output directory")
    p_tms.add_argument("-l", "--label",  default="phase4", help="Run label")
    p_tms.add_argument("--isl-hops",       type=int,   default=8,
                       help="Number of ISL hops in clock distribution chain (default: 8)")
    p_tms.add_argument("--sync-interval",  type=float, default=60.0,
                       help="ISL synchronization interval in seconds (default: 60)")
    p_tms.add_argument("--uere-autonomous", type=float, default=3.10,
                       help="UERE for autonomous mode in meters (default: 3.10)")
    p_tms.add_argument("--uere-combined",   type=float, default=1.69,
                       help="UERE for combined LEO+GLONASS mode in meters (default: 1.69)")

    # sdcm
    p_sdcm = sub.add_parser("sdcm", help="SDCM differential corrections: Mode C UERE/CEP analysis")
    p_sdcm.add_argument("-o","--output", default="results/sdcm")
    p_sdcm.add_argument("-l","--label",  default="phase3")
    p_sdcm.add_argument("--pdop-auto",     type=float, default=5.15, help="Autonomous PDOP p95")
    p_sdcm.add_argument("--pdop-combined", type=float, default=1.67, help="Combined PDOP p95")
    p_sdcm.add_argument("--grid-step",    type=float, default=2.0,  help="Grid step degrees")

    # isl-link
    p_isl_lnk = sub.add_parser("isl-link", help="ISL RF link budget: Ka/V/Optical bands")
    p_isl_lnk.add_argument("-o","--output",  default="results/isl_link")
    p_isl_lnk.add_argument("-l","--label",   default="phase3")
    p_isl_lnk.add_argument("--n-planes", type=int, default=12)
    p_isl_lnk.add_argument("--n-sats",   type=int, default=15)

    # anti-jam
    p_aj = sub.add_parser("anti-jam", help="Anti-jamming: J/S ratio, jamming radius vs GPS")
    p_aj.add_argument("-o","--output",    default="results/anti_jam")
    p_aj.add_argument("-l","--label",     default="phase3")
    p_aj.add_argument("--elevation", type=float, default=10.0,
                      help="Satellite elevation angle in degrees (default: 10)")

    # tesla-mac
    p_tsl = sub.add_parser("tesla-mac", help="TESLA MAC anti-spoofing analysis")
    p_tsl.add_argument("-o","--output", default="results/tesla_mac")
    p_tsl.add_argument("-l","--label",  default="phase3")

    # multipath
    p_mp = sub.add_parser("multipath", help="Multipath model: CEP by environment vs GPS")
    p_mp.add_argument("-o","--output",   default="results/multipath")
    p_mp.add_argument("-l","--label",    default="phase3")
    p_mp.add_argument("--pdop-leo", type=float, default=5.15)
    p_mp.add_argument("--pdop-gps", type=float, default=3.50)

    # acquisition
    p_acq = sub.add_parser("acquisition", help="Signal acquisition & TTFF: Doppler, search, TTFF")
    p_acq.add_argument("-o","--output",   default="results/acquisition")
    p_acq.add_argument("-l","--label",    default="phase3")
    p_acq.add_argument("--clock-ppm",  type=float, default=1.0,
                       help="RX oscillator uncertainty in ppm (default: 1.0 = TCXO)")
    p_acq.add_argument("--channels",   type=int,   default=32,
                       help="Number of parallel acquisition channels (default: 32)")

    # deorbit
    p_deo = sub.add_parser("deorbit", help="Orbital lifetime & deorbit: IADC compliance, delta-V")
    p_deo.add_argument("-o","--output",       default="results/deorbit")
    p_deo.add_argument("-l","--label",        default="phase3")
    p_deo.add_argument("--altitude-km",   type=float, default=1000.0)
    p_deo.add_argument("--n-sats",        type=int,   default=180)

    # freq-plan
    p_fp = sub.add_parser("freq-plan",
                          help="Frequency plan: ITU allocation, GNSS interference, Doppler profile")
    p_fp.add_argument("-o", "--output",       default="results/freq_plan", help="Output directory")
    p_fp.add_argument("-l", "--label",        default="phase3",            help="Run label")
    p_fp.add_argument("--n-sats",             type=int,   default=180,
                      help="Total satellites in constellation (default: 180 = Phase 3)")
    p_fp.add_argument("--altitude-km",        type=float, default=1000.0,  help="Orbit altitude km")
    p_fp.add_argument("--tx-power",           type=float, default=16.0,
                      help="TX power dBW (default: 16 = 40 W)")
    p_fp.add_argument("--tx-gain",            type=float, default=14.0,
                      help="TX antenna gain dBi (default: 14)")
    p_fp.add_argument("--rx-gain",            type=float, default=3.0,
                      help="RX antenna gain dBi (default: 3)")

    # isl-ranging
    p_isl = sub.add_parser("isl-ranging",
                           help="ISL ranging: autonomous orbit determination, ephemeris holdover analysis")
    p_isl.add_argument("-o", "--output",       default="results/isl_ranging", help="Output directory")
    p_isl.add_argument("-l", "--label",        default="phase3",              help="Run label")
    p_isl.add_argument("--n-planes",           type=int,   default=12,
                       help="Number of orbital planes (default: 12 = Phase 3)")
    p_isl.add_argument("--n-sats",             type=int,   default=15,
                       help="Satellites per plane (default: 15 = Phase 3)")
    p_isl.add_argument("--altitude-km",        type=float, default=1000.0,   help="Orbit altitude km")
    p_isl.add_argument("--inclination-deg",    type=float, default=75.0,     help="Inclination degrees")
    p_isl.add_argument("--n-stations",         type=int,   default=21,
                       help="Number of MCS ground stations (default: 21)")
    p_isl.add_argument("--mode",               default="code",
                       choices=["code", "phase"],
                       help="ISL ranging mode: code (0.3m) or phase (0.01m) (default: code)")
    p_isl.add_argument("--uere-m",             type=float, default=3.10,
                       help="Baseline autonomous UERE in meters (default: 3.10)")
    p_isl.add_argument("--pdop",               type=float, default=5.15,
                       help="PDOP p95 value for CEP computation (default: 5.15 = Phase 3)")

    # monte-carlo
    p_mc = sub.add_parser("monte-carlo",
                          help="Monte Carlo accuracy: empirical CEP histograms vs analytical estimates")
    p_mc.add_argument("-o", "--output",     default="results/monte_carlo", help="Output directory")
    p_mc.add_argument("-l", "--label",      default="phase3",              help="Run label")
    p_mc.add_argument("--n-trials",         type=int,   default=10_000,
                      help="Number of Monte Carlo trials (default: 10000)")
    p_mc.add_argument("--seed",             type=int,   default=42,
                      help="Random seed for reproducibility (default: 42)")
    p_mc.add_argument("--pdop-auto",        type=float, default=5.15,
                      help="PDOP p95 for autonomous mode (default: 5.15 = Phase 3)")
    p_mc.add_argument("--pdop-combined",    type=float, default=1.67,
                      help="PDOP p95 for combined mode (default: 1.67 = Phase 4)")
    p_mc.add_argument("--n-sats-auto",      type=int,   default=10,
                      help="Visible satellites for autonomous mode (default: 10)")
    p_mc.add_argument("--n-sats-comb",      type=int,   default=22,
                      help="Visible satellites for combined mode (default: 22)")

    # user-link-budget
    p_ulb = sub.add_parser("user-link-budget",
                           help="User terminal link budget: L1/L5 C/N0, pseudorange noise, margins")
    p_ulb.add_argument("-o", "--output",     default="results/user_link_budget")
    p_ulb.add_argument("-l", "--label",      default="phase4")
    p_ulb.add_argument("--altitude-km",  type=float, default=1000.0,
                       help="Satellite altitude km (default: 1000)")

    # eclipse
    p_ecl = sub.add_parser("eclipse",
                           help="Eclipse / Earth shadow: duration, battery sizing, OCXO thermal effect")
    p_ecl.add_argument("-o", "--output",     default="results/eclipse")
    p_ecl.add_argument("-l", "--label",      default="phase4")
    p_ecl.add_argument("--altitude-km",  type=float, default=1000.0)
    p_ecl.add_argument("--n-sats",       type=int,   default=180)

    # nav-message
    p_nav = sub.add_parser("nav-message",
                           help="Navigation message: bit budget, frame structure, TTFF, OSNMA auth")
    p_nav.add_argument("-o", "--output",     default="results/nav_message")
    p_nav.add_argument("-l", "--label",      default="phase4")
    p_nav.add_argument("--data-rate",    type=int,   default=500,
                       help="Data rate bps (default: 500)")
    p_nav.add_argument("--n-sats",       type=int,   default=180,
                       help="Constellation size for almanac budget (default: 180)")

    # iono-correction
    p_ion = sub.add_parser("iono-correction",
                           help="Ionospheric correction: Klobuchar vs NeQuick-G vs dual-freq")
    p_ion.add_argument("-o", "--output",     default="results/iono_correction")
    p_ion.add_argument("-l", "--label",      default="phase4")

    # user-dynamics
    p_ud = sub.add_parser("user-dynamics",
                          help="User dynamics: moving platform accuracy (aviation/maritime/land)")
    p_ud.add_argument("-o", "--output",     default="results/user_dynamics")
    p_ud.add_argument("-l", "--label",      default="phase4")

    # ground-od
    p_god = sub.add_parser("ground-od",
                           help="Ground orbit determination: RAC accuracy, prediction degradation")
    p_god.add_argument("-o", "--output",     default="results/ground_od")
    p_god.add_argument("-l", "--label",      default="phase4")

    # deployment
    p_dep = sub.add_parser("deployment",
                           help="Deployment timeline: launch sequence, coverage ramp-up, costs")
    p_dep.add_argument("-o", "--output",     default="results/deployment")
    p_dep.add_argument("-l", "--label",      default="full")

    # conjunction
    p_conj = sub.add_parser("conjunction",
                            help="Conjunction probability: Monte Carlo debris collision analysis")
    p_conj.add_argument("-o", "--output",     default="results/conjunction")
    p_conj.add_argument("-l", "--label",      default="phase4")
    p_conj.add_argument("--n-trials", type=int, default=50_000,
                        help="Monte Carlo trials (default: 50000)")

    # coverage-maps
    p_cov = sub.add_parser("coverage-maps",
                           help="Geographic coverage maps: N_vis and PDOP heatmaps by phase")
    p_cov.add_argument("-o", "--output",     default="results/coverage_maps")
    p_cov.add_argument("-l", "--label",      default="all_phases")
    p_cov.add_argument("--all-phases", action="store_true", default=True,
                       help="Generate maps for all phases (default: True)")
    p_cov.add_argument("--phase", type=int, default=4,
                       help="Single phase to analyze (ignored if --all-phases)")

    # power-budget
    p_pb = sub.add_parser("power-budget",
                          help="Satellite power budget: solar array BOL/EOL, battery, subsystem loads")
    p_pb.add_argument("-o", "--output",     default="results/power_budget")
    p_pb.add_argument("-l", "--label",      default="phase4")
    p_pb.add_argument("--altitude-km",  type=float, default=1000.0)
    p_pb.add_argument("--n-sats",       type=int,   default=180)

    # mass-budget
    p_mb = sub.add_parser("mass-budget",
                          help="Satellite mass & volume budget: subsystems, propellant, fleet totals")
    p_mb.add_argument("-o", "--output",     default="results/mass_budget")
    p_mb.add_argument("-l", "--label",      default="phase4")
    p_mb.add_argument("--n-sats",       type=int,   default=180)
    p_mb.add_argument("--mission-years", type=float, default=7.0)

    # thermal
    p_th = sub.add_parser("thermal",
                          help="Satellite thermal analysis: orbital temp profile, OCXO oven budget")
    p_th.add_argument("-o", "--output",     default="results/thermal")
    p_th.add_argument("-l", "--label",      default="phase4")
    p_th.add_argument("--altitude-km",  type=float, default=1000.0)

    # timing-chain
    p_tc = sub.add_parser("timing-chain",
                          help="Timing chain: Cs→Rb→OCXO→user 1PPS, ADEV, ISL transfer, holdover")
    p_tc.add_argument("-o", "--output",     default="results/timing_chain")
    p_tc.add_argument("-l", "--label",      default="phase4")
    p_tc.add_argument("--max-hops",     type=int,   default=12,
                      help="Maximum ISL hops to analyze (default: 12)")

    # radiation
    p_rad = sub.add_parser("radiation", help="Radiation environment: TID, SEU, OCXO drift vs Al shielding")
    p_rad.add_argument("-o", "--output", default="results/radiation")
    p_rad.add_argument("-l", "--label",  default="phase4")

    # relativistic
    p_rel = sub.add_parser("relativistic", help="Relativistic corrections: Sagnac, redshift, clock bias LEO vs MEO")
    p_rel.add_argument("-o", "--output", default="results/relativistic")
    p_rel.add_argument("-l", "--label",  default="phase4")

    # troposphere
    p_tro = sub.add_parser("troposphere", help="Tropospheric delay: ZHD/ZWD Saastamoinen, NMF, seasonal")
    p_tro.add_argument("-o", "--output", default="results/troposphere")
    p_tro.add_argument("-l", "--label",  default="phase4")

    # reliability
    p_rel2 = sub.add_parser("reliability", help="Reliability & MTBF: R(t), fleet degradation, spare satellites")
    p_rel2.add_argument("-o", "--output", default="results/reliability")
    p_rel2.add_argument("-l", "--label",  default="phase4")

    # ppp-convergence
    p_ppp = sub.add_parser("ppp-convergence", help="PPP convergence: Kalman, LEO vs MEO, dual/single freq")
    p_ppp.add_argument("-o", "--output", default="results/ppp_convergence")
    p_ppp.add_argument("-l", "--label",  default="phase4")

    # signal-quality
    p_sq = sub.add_parser("signal-quality", help="Signal quality: SISA, URE, UERE budget vs GPS/Galileo")
    p_sq.add_argument("-o", "--output", default="results/signal_quality")
    p_sq.add_argument("-l", "--label",  default="phase4")

    # adcs
    p_adcs = sub.add_parser("adcs", help="ADCS: pointing budget, disturbance torques, reaction wheels, ISL")
    p_adcs.add_argument("-o", "--output", default="results/adcs")
    p_adcs.add_argument("-l", "--label",  default="phase4")

    # ground-network
    p_gn = sub.add_parser("ground-network", help="Ground network: MCS geometry, OD quality, coverage optimization")
    p_gn.add_argument("-o", "--output", default="results/ground_network")
    p_gn.add_argument("-l", "--label",  default="phase4")

    # signal-design
    p_sd = sub.add_parser("signal-design",
                          help="Signal design: modulation BOC/BPSK/TMBOC, codes Gold/Weil/Memory, nav-msg ANAV")
    p_sd.add_argument("-o", "--output", default="results/signal_design")
    p_sd.add_argument("-l", "--label",  default="phase4")

    # competitor-analysis
    p_comp = sub.add_parser("competitor-analysis",
                            help="Competitor analysis: AURORA vs GLONASS, GPS, Galileo, LEO PNT systems")
    p_comp.add_argument("-o", "--output", default="results/competitor_analysis")
    p_comp.add_argument("-l", "--label",  default="phase4")

    # user-segment
    p_us = sub.add_parser("user-segment",
                          help="User segment: Doppler dynamics, PLL bandwidth, channel count, receiver classes")
    p_us.add_argument("-o", "--output", default="results/user_segment")
    p_us.add_argument("-l", "--label",  default="phase4")

    # itu-coordination
    p_itu = sub.add_parser("itu-coordination",
                           help="ITU/МСЭ coordination: PSD, SSC matrix, RNSS band plan, OOB mask")
    p_itu.add_argument("-o", "--output", default="results/itu_coordination")
    p_itu.add_argument("-l", "--label",  default="phase4")

    # rtk-ppp
    p_rtk = sub.add_parser("rtk-ppp",
                           help="PPP-RTK architecture: accuracy vs baseline, convergence, latency, RSN map")
    p_rtk.add_argument("-o", "--output", default="results/rtk_ppp")
    p_rtk.add_argument("-l", "--label",  default="phase4")

    # station-keeping
    p_sk = sub.add_parser("station-keeping", help="Station keeping: J2 RAAN drift, drag, delta-V 7yr budget")
    p_sk.add_argument("-o", "--output", default="results/station_keeping")
    p_sk.add_argument("-l", "--label",  default="phase4")

    # ── Phase 5: Реальные данные и валидация ───────────────────────────────
    for _name, _help, _odir in [
        ("real-data", "Интеграция реальных данных (IGS, VMF3, SLR)", "results/real_data"),
        ("validate",  "Валидация моделей на реальных/embedded данных", "results/validate"),
    ]:
        _p = sub.add_parser(_name, help=_help)
        _p.add_argument("-o", "--output", default=_odir)
        _p.add_argument("-l", "--label",  default="phase5")

    # ── Phase 4: Прототипирование сигнала ──────────────────────────────────
    for _name, _help, _odir in [
        ("code-gen",     "Генератор Weil/Gold/Extended Memory кодов",  "results/code_gen"),
        ("sdr-receiver", "SDR-приёмник: захват, слежение, TTFF",       "results/sdr"),
    ]:
        _p = sub.add_parser(_name, help=_help)
        _p.add_argument("-o", "--output", default=_odir)
        _p.add_argument("-l", "--label",  default="phase5")

    # ── Phase 3 expansion modules (risk/schedule/cyber/e2e) ────────────────
    for _name, _help, _odir in [
        ("risks",      "Реестр рисков и матрица P×S",              "results/risks"),
        ("schedule",   "График работ (Gantt) и критический путь",  "results/schedule"),
        ("cybersec",   "Модель угроз и меры (STRIDE/PASTA)",       "results/cybersec"),
        ("e2e",        "Сквозная PVT-симуляция 24 ч",              "results/e2e"),
    ]:
        _p = sub.add_parser(_name, help=_help)
        _p.add_argument("-o", "--output", default=_odir)
        _p.add_argument("-l", "--label",  default="phase5")

    # ── Phase 2 expansion modules ──────────────────────────────────────────
    for _name, _help, _odir in [
        ("pvt-montecarlo", "End-to-end Monte-Carlo PVT error budget", "results/pvt_montecarlo"),
        ("dop-temporal",   "Temporal DOP / availability maps",        "results/dop_temporal"),
        ("pod",            "Precise orbit determination (force budget, R/A/C, SLR)", "results/pod"),
        ("autonav",        "Autonomous navigation via ISL ranging",   "results/autonav"),
        ("araim",          "ARAIM: solution separation, VPL/HPL, P_HMI", "results/araim"),
        ("integrity",      "Integrity budget: Stanford, LPV-200/CAT-I, ISM", "results/integrity"),
        ("cost",           "Life-cycle cost model (CAPEX/OPEX, LCC)", "results/cost"),
        ("production",     "Production & AIT for 300 sats",           "results/production"),
        ("launch-campaign","Launch & deployment campaign",            "results/launch"),
        ("ground-segment", "Ground segment (MCS/TT&C) architecture",  "results/ground_segment"),
    ]:
        _p = sub.add_parser(_name, help=_help)
        _p.add_argument("-o", "--output", default=_odir)
        _p.add_argument("-l", "--label",  default="phase5")

    # system-concept
    p_sc = sub.add_parser("system-concept",
                          help="Concept illustrations: system overview, service scenarios, LEO vs MEO, signal flow")
    p_sc.add_argument("-o", "--output", default="results/system_concept")
    p_sc.add_argument("-l", "--label",  default="phase4")

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
        "sdcm":           cmd_sdcm,
        "isl-link":       cmd_isl_link,
        "anti-jam":       cmd_anti_jam,
        "tesla-mac":      cmd_tesla_mac,
        "multipath":      cmd_multipath,
        "acquisition":    cmd_acquisition,
        "deorbit":        cmd_deorbit,
        "freq-plan":      cmd_freq_plan,
        "isl-ranging":    cmd_isl_ranging,
        "monte-carlo":    cmd_monte_carlo,
        "resilience":     cmd_resilience,
        "user-link-budget": cmd_user_link_budget,
        "eclipse":          cmd_eclipse,
        "nav-message":      cmd_nav_message,
        "power-budget":     cmd_power_budget,
        "mass-budget":      cmd_mass_budget,
        "thermal":          cmd_thermal,
        "timing-chain":     cmd_timing_chain,
        "iono-correction":  cmd_iono_correction,
        "user-dynamics":    cmd_user_dynamics,
        "ground-od":        cmd_ground_od,
        "deployment":       cmd_deployment,
        "conjunction":      cmd_conjunction,
        "coverage-maps":    cmd_coverage_maps,
        "radiation":        cmd_radiation,
        "relativistic":     cmd_relativistic,
        "troposphere":      cmd_troposphere,
        "reliability":      cmd_reliability,
        "ppp-convergence":  cmd_ppp_convergence,
        "signal-quality":   cmd_signal_quality,
        "adcs":             cmd_adcs,
        "ground-network":   cmd_ground_network,
        "station-keeping":    cmd_station_keeping,
        "system-concept":     cmd_system_concept,
        "user-segment":       cmd_user_segment,
        "itu-coordination":   cmd_itu_coordination,
        "rtk-ppp":            cmd_rtk_ppp,
        "pvt-montecarlo":     cmd_pvt_montecarlo,
        "dop-temporal":       cmd_dop_temporal,
        "pod":                cmd_pod,
        "autonav":            cmd_autonav,
        "araim":              cmd_araim,
        "integrity":          cmd_integrity,
        "cost":               cmd_cost,
        "production":         cmd_production,
        "launch-campaign":    cmd_launch_campaign,
        "ground-segment":     cmd_ground_segment,
        "real-data":          cmd_real_data,
        "validate":           cmd_validate,
        "code-gen":           cmd_code_gen,
        "sdr-receiver":       cmd_sdr_receiver,
        "risks":              cmd_risks,
        "schedule":           cmd_schedule,
        "cybersec":           cmd_cybersec,
        "e2e":                cmd_e2e_pipeline,
        "signal-design":       cmd_signal_design,
        "competitor-analysis": cmd_competitor_analysis,
        "combined":         cmd_combined,
        "time-scale":       cmd_time_scale,
        "timing-service":   cmd_timing_service,
        "clock-arch":       cmd_clock_arch,
        "download-cesium":  cmd_download_cesium,
        "serve":            cmd_serve,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
