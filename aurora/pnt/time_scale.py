"""
LEO-PNT Time Scale (LPT) model and accuracy comparison by operating mode.

The LEO-PNT system maintains its own time scale (LPT) independently of any
external GNSS. Two primary operating modes:

  Mode A — Autonomous (LEO-PNT standalone):
    - Receiver uses only LEO-PNT signals referenced to LPT
    - No GLONASS dependency; fully sovereign time and position solution
    - H matrix: 4 columns (dx, dy, dz, dt_lpt)
    - PDOP: from LEO geometry (~5.0 Phase 4)
    - Clock bias: LPT stability = master_clock + ISL_chain_noise

  Mode B — Combined (LEO-PNT + GLONASS):
    - Receiver uses both LEO and GLONASS signals
    - ISB (GLONASST - LPT) solved simultaneously as 5th unknown
    - H matrix: 5 columns (dx, dy, dz, dt_lpt, dt_glonass)
    - PDOP: combined geometry (~1.67 Phase 4)
    - Clock bias: reduced by ISB estimation; ephemeris improved by GLONASS

  Mode C — Combined + SDCM (GLONASS differential corrections):
    - Same as Mode B but receiver applies SDCM corrections
    - UERE: clock 0.8 m, ephemeris 0.05 m, iono 0.5 m (single-freq)
    - Only available in SDCM coverage (Russia + surrounding regions)

LPT time scale realization:
  - Physical master: H-maser or Cs at control center (Zheleznogorsk)
  - Distribution: ISL sync chain to all LEO satellites (sqrt(N)*ppb*T_sync noise)
  - Broadcast: each satellite transmits LPT polynomial coefficients (like GLONASS Kp)
  - Ground: receiver computes position in LPT frame, no external reference needed

Transition between modes (seamless):
  - System always maintains LPT (no reinitialization required)
  - If GLONASS available: receiver may enter Mode B for improved geometry
  - If GLONASS signal lost: automatic fallback to Mode A at same LPT epoch
  - ISB re-calibration time after GLONASS re-acquisition: ~30-60 s

References:
  - GLONASS ICD Ed. 5.1, 2008 (ISB treatment)
  - Satelles STL Architecture: independent LEO time scale, Iridium constellation
  - Xona PULSAR: autonomous LEO PNT time scale, sub-ns ISL sync
  - Kaplan & Hegarty, "GPS: Principles and Applications", Ch. 7 (DOP)
  - Montenbruck et al., "Multi-GNSS receiver clock characterization", GPS Solutions 2014
"""

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_C = 299_792_458.0       # m/s
_NS_PER_S = 1e9

# LPT design target: timing error < 10 ns -> ranging error < 3 m
LPT_TARGET_NS = 10.0


# ── UERE budget by operating mode ──────────────────────────────────────────
# Sources:
#   clock_bias:    LPT master clock + ISL chain noise (or ISB residual in combined)
#   ephemeris:     LEO orbit determination accuracy (GNSS-aided onboard receiver)
#   iono_sf:       L1 single-freq ionosphere  = 2.0/sin(el) m, typical 2.0 m @ 30 deg
#   iono_df:       L1+L5 dual-freq residual  ~ 0.05 m (dispersive cancellation)
#   tropo:         Saastamoinen model residual
#   multipath:     Ground receiver, open-sky environment
#   thermal:       Receiver noise floor
#   isb_residual:  ISB estimation error after calibration (combined modes only)
#   relativistic:  LEO relativistic correction residual (~0.02 ns, much smaller than GPS)

UERE_BUDGET = {
    "autonomous": {
        "clock_bias_m":   3.0,    # LPT: Cs master + ISL chain (8 hops, 60s sync)
        "ephemeris_m":    0.5,    # GNSS-aided onboard orbit determination
        "iono_sf_m":      2.0,    # L1 single-freq (elevation-dependent, typical value)
        "iono_df_m":      0.05,   # L1+L5 dual-freq residual
        "tropo_m":        0.5,
        "multipath_m":    0.3,
        "thermal_m":      0.1,
        "isb_residual_m": 0.0,    # No ISB in autonomous mode
        "relativistic_m": 0.02,
    },
    "combined": {
        "clock_bias_m":   1.5,    # ISB calibration eliminates most GLONASST offset
        "ephemeris_m":    0.1,    # Better: GLONASS measurements constrain LEO orbit
        "iono_sf_m":      2.0,
        "iono_df_m":      0.05,
        "tropo_m":        0.5,
        "multipath_m":    0.3,
        "thermal_m":      0.1,
        "isb_residual_m": 0.5,    # ~1.5 ns ISB estimation residual -> 0.45 m
        "relativistic_m": 0.02,
    },
    "combined_sdcm": {
        # SDCM provides satellite clock + ephemeris corrections for GLONASS
        "clock_bias_m":   0.8,    # SDCM clock corrections applied
        "ephemeris_m":    0.05,   # SDCM ephemeris corrections applied
        "iono_sf_m":      0.5,    # SDCM GIVE (Grid Ionosphere Vertical Error)
        "iono_df_m":      0.05,
        "tropo_m":        0.3,    # Improved troposphere model with SDCM
        "multipath_m":    0.3,
        "thermal_m":      0.1,
        "isb_residual_m": 0.3,    # Reduced ISB residual with SDCM
        "relativistic_m": 0.02,
    },
}


def compute_uere(mode: str, dual_frequency: bool = False) -> dict:
    """
    Compute User Equivalent Range Error for a given operating mode.

    Returns per-component and total RSS UERE.
    """
    budget = UERE_BUDGET[mode]
    iono_m = budget["iono_df_m"] if dual_frequency else budget["iono_sf_m"]

    components = {
        "clock_bias_m":   budget["clock_bias_m"],
        "ephemeris_m":    budget["ephemeris_m"],
        "ionosphere_m":   iono_m,
        "troposphere_m":  budget["tropo_m"],
        "multipath_m":    budget["multipath_m"],
        "thermal_m":      budget["thermal_m"],
        "isb_residual_m": budget["isb_residual_m"],
        "relativistic_m": budget["relativistic_m"],
    }
    uere_rss = math.sqrt(sum(v ** 2 for v in components.values()))

    return {
        "mode": mode,
        "dual_frequency": dual_frequency,
        "components": components,
        "uere_rss_m": round(uere_rss, 3),
    }


def compute_position_accuracy(pdop: float, uere_m: float) -> dict:
    """
    Derive position accuracy figures from PDOP and UERE.

    Assumes isotropic Gaussian errors (standard navigation approximation):
      sigma_3d = PDOP * UERE
      HDOP ~ PDOP / sqrt(2)  (for typical LEO/MEO geometry)
      CEP 50% ~ 0.59 * HDOP * UERE  (Groves, eq 7.24)
    """
    hdop = pdop / math.sqrt(2)
    vdop = pdop / math.sqrt(2)

    return {
        "pdop":              round(pdop, 3),
        "uere_m":            round(uere_m, 3),
        "sigma_3d_m":        round(pdop * uere_m, 2),
        "sigma_h_m":         round(hdop * uere_m, 2),
        "sigma_v_m":         round(vdop * uere_m, 2),
        "cep_50_m":          round(0.59 * hdop * uere_m, 2),
        "sep_50_m":          round(0.51 * pdop * uere_m, 2),
        "timing_1sigma_ns":  round(uere_m / _C * _NS_PER_S, 2),
    }


def compute_lpt_stability(
    master_clock_ppb: float = 0.01,
    n_isl_hops: int = 8,
    sync_interval_s: float = 60.0,
    duration_h: float = 24.0,
) -> dict:
    """
    LPT (LEO PNT Time) stability — two separate metrics:

    1. Inter-satellite sync error (navigation-relevant):
       sigma_isl = sqrt(N_hops) * ppb * T_sync  [ns]
       This determines the UERE clock-bias term for position accuracy.
       It resets every sync_interval_s, so it does NOT grow over time.

    2. LPT-UTC offset (timing service / external traceability):
       sigma_utc = ppb * duration_s  [ns]
       This grows linearly with time and matters for applications needing
       absolute UTC reference (e.g., time dissemination, synchronization service).
       It does NOT affect position accuracy — all satellites share the same LPT,
       so systematic offset cancels in the position solution.

    In practice, sigma_utc is managed by periodic comparison of the LPT master
    clock to UTC via GLONASS signals (when available) or ground reference.
    In fully autonomous mode, sigma_utc grows until the next external correction.
    """
    # Navigation-relevant: inter-satellite ISL sync error
    sigma_isl_ns   = math.sqrt(n_isl_hops) * master_clock_ppb * sync_interval_s
    sigma_isl_m    = sigma_isl_ns * 1e-9 * _C

    # Timing-service-relevant: LPT-UTC offset (master clock free-run)
    sigma_utc_ns   = master_clock_ppb * duration_h * 3600.0
    sigma_utc_m    = sigma_utc_ns * 1e-9 * _C

    return {
        "master_clock_ppb":  master_clock_ppb,
        "n_isl_hops":        n_isl_hops,
        "sync_interval_s":   sync_interval_s,
        "duration_h":        duration_h,
        # Navigation accuracy (ISL chain, constant):
        "sigma_isl_ns":      round(sigma_isl_ns, 5),
        "sigma_isl_m":       round(sigma_isl_m, 5),
        # UTC traceability (master clock drift, grows over time):
        "sigma_utc_ns":      round(sigma_utc_ns, 5),
        "sigma_utc_m":       round(sigma_utc_m, 5),
        # Alias for backward compat (ISL = navigation-relevant):
        "sigma_lpt_ns":      round(sigma_isl_ns, 5),
        "sigma_lpt_m":       round(sigma_isl_m, 5),
    }


def run_time_scale_analysis(
    pdop_leo: float = 5.0,
    pdop_combined: float = 1.67,
    output_dir: str = "results/time_scale",
    label: str = "phase4",
) -> dict:
    """
    Compare accuracy across all operating modes: autonomous, combined, combined+SDCM.

    Runs both single-frequency and dual-frequency variants.
    Also computes LPT stability for all relevant master clock types.

    Args:
        pdop_leo:      PDOP p95 for autonomous LEO-only mode
        pdop_combined: PDOP p95 for combined LEO+GLONASS mode
        output_dir:    Where to write plots/CSV/JSON
        label:         Run label

    Returns:
        Summary dict with accuracy metrics for all modes and LPT stability table.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Define all mode configurations
    mode_configs = [
        # (mode_key, budget_key, label, pdop, dual_freq, color)
        ("autonomous_sf",    "autonomous",     "Autonomous LPT\n(L1 single-freq)",        pdop_leo,      False, "#E53935"),
        ("autonomous_df",    "autonomous",     "Autonomous LPT\n(L1+L5 dual-freq)",       pdop_leo,      True,  "#FB8C00"),
        ("combined_sf",      "combined",       "Combined LEO+GLONASS\n(L1 single-freq)",  pdop_combined, False, "#43A047"),
        ("combined_df",      "combined",       "Combined LEO+GLONASS\n(L1+L5 dual-freq)", pdop_combined, True,  "#1E88E5"),
        ("combined_sdcm_df", "combined_sdcm",  "Combined + SDCM\n(L1+L5 dual-freq)",      pdop_combined, True,  "#8E24AA"),
    ]

    mode_results = {}
    for mk, budget_k, ml, pdop, df, color in mode_configs:
        uere = compute_uere(budget_k, dual_frequency=df)
        acc  = compute_position_accuracy(pdop, uere["uere_rss_m"])
        mode_results[mk] = {
            "label": ml, "color": color,
            "pdop": pdop, "dual_frequency": df,
            "uere": uere, "accuracy": acc,
        }

    # LPT stability for master clock types
    clock_types = {
        "OCXO":  1.0,
        "Rb":    0.1,
        "Cs":    0.01,
        "Maser": 0.001,
    }
    lpt_stability = {
        name: compute_lpt_stability(ppb, n_isl_hops=8, sync_interval_s=60.0, duration_h=24.0)
        for name, ppb in clock_types.items()
    }

    _plot_accuracy_comparison(mode_results, output_dir, label)
    _plot_lpt_stability(lpt_stability, output_dir, label)
    _plot_uere_breakdown(mode_results, output_dir, label)
    _save_summary_json(mode_results, lpt_stability, output_dir, label)

    return {
        "label": label,
        "pdop_leo": pdop_leo,
        "pdop_combined": pdop_combined,
        "modes": mode_results,
        "lpt_stability": lpt_stability,
    }


def _plot_accuracy_comparison(results: dict, output_dir: str, label: str) -> None:
    keys   = list(results.keys())
    labels = [results[k]["label"] for k in keys]
    cep    = [results[k]["accuracy"]["cep_50_m"] for k in keys]
    sv     = [results[k]["accuracy"]["sigma_v_m"] for k in keys]
    pdops  = [results[k]["pdop"] for k in keys]
    ueres  = [results[k]["uere"]["uere_rss_m"] for k in keys]
    colors = [results[k]["color"] for k in keys]
    x = np.arange(len(keys))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Position Accuracy by Mode — {label}", fontweight="bold", fontsize=13)

    for ax, vals, ylabel, title, limits in [
        (axes[0, 0], cep,   "CEP 50% (m)",          "Horizontal Accuracy (CEP)",  [(10.0,"red","10m req"), (3.0,"green","3m target")]),
        (axes[0, 1], sv,    "Vertical 1-sigma (m)", "Vertical Accuracy",           [(15.0,"red","15m req"), (5.0,"green","5m target")]),
        (axes[1, 0], pdops, "PDOP",                  "Geometry (PDOP)",             [(6.0,"red","PDOP=6"), (2.0,"green","PDOP=2 excellent")]),
        (axes[1, 1], ueres, "UERE 1-sigma (m)",      "Range Error Budget (UERE)",   []),
    ]:
        bars = ax.bar(x, vals, color=colors, alpha=0.85, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")
        for lval, lcolor, llabel in limits:
            ax.axhline(lval, color=lcolor, linestyle="--", linewidth=1.5, label=llabel)
        if limits:
            ax.legend(fontsize=8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + max(vals) * 0.01,
                    f"{v:.1f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"accuracy_modes_{label}.png"),
                dpi=120, bbox_inches="tight")
    plt.close()


def _plot_uere_breakdown(results: dict, output_dir: str, label: str) -> None:
    keys     = list(results.keys())
    labels   = [results[k]["label"].replace("\n", " ") for k in keys]
    comp_keys = ["clock_bias_m", "ephemeris_m", "ionosphere_m",
                 "troposphere_m", "multipath_m", "isb_residual_m"]
    comp_labels = ["Clock bias", "Ephemeris", "Ionosphere",
                   "Troposphere", "Multipath", "ISB residual"]
    comp_colors = ["#E53935", "#FB8C00", "#FDD835", "#43A047", "#1E88E5", "#8E24AA"]

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.suptitle(f"UERE Budget Breakdown by Mode — {label}", fontweight="bold")

    x = np.arange(len(keys))
    bottom = np.zeros(len(keys))
    for ck, cl, cc in zip(comp_keys, comp_labels, comp_colors):
        vals = np.array([
            results[k]["uere"]["components"].get(ck, 0.0) for k in keys
        ])
        ax.bar(x, vals, bottom=bottom, label=cl, color=cc, alpha=0.85, width=0.6)
        bottom += vals

    # Plot RSS total as dots
    total = [results[k]["uere"]["uere_rss_m"] for k in keys]
    ax.scatter(x, total, color="black", zorder=5, s=60, marker="D", label="UERE RSS (total)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Error contribution (m, 1-sigma)")
    ax.set_title("UERE Breakdown — stacked linear, RSS diamond")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"uere_breakdown_{label}.png"),
                dpi=120, bbox_inches="tight")
    plt.close()


def _plot_lpt_stability(lpt_stability: dict, output_dir: str, label: str) -> None:
    """Two panels: (left) ISL inter-sat sync error = navigation accuracy;
    (right) LPT-UTC free-run drift = timing-service traceability."""
    durations = [0.5, 1, 2, 4, 8, 12, 24, 48, 72]
    colors_clk = {
        "OCXO": "#FB8C00", "Rb": "#43A047",
        "Cs": "#1E88E5", "Maser": "#8E24AA",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(
        f"LPT (LEO-PNT Time) Stability — {label}\n"
        f"8 ISL hops | sync interval 60 s | master clock types", fontweight="bold"
    )

    for name, data in lpt_stability.items():
        ppb = data["master_clock_ppb"]
        # Navigation: ISL sync error (constant vs duration, shown as horizontal lines)
        isl_ns = data["sigma_isl_ns"]
        # Timing: UTC offset (grows linearly)
        utc_ns = [compute_lpt_stability(ppb, 8, 60.0, dh)["sigma_utc_ns"] for dh in durations]
        c = colors_clk[name]
        ax1.axhline(isl_ns, color=c, lw=2, label=f"{name} ({ppb} ppb)  {isl_ns:.2f} ns")
        ax2.semilogy(durations, utc_ns, "o-", color=c, lw=2, label=f"{name} ({ppb} ppb)")

    ax1.axhline(LPT_TARGET_NS, color="black", ls=":", lw=1.5, label=f"{LPT_TARGET_NS:.0f} ns target")
    ax1.axhline(1.0, color="grey", ls=":", lw=1, label="1 ns precision")
    ax1.set_xlabel("(Constant — independent of duration)")
    ax1.set_ylabel("ISL sync error 1-sigma (ns)")
    ax1.set_title("Navigation Clock Bias\n(ISL chain, 8 hops @ 60 s sync)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(LPT_TARGET_NS * 1.5, 20))

    ax2.axhline(1000, color="orange", ls="--", lw=1.5, label="1 us (UTC correction limit)")
    ax2.axhline(10,   color="black",  ls=":",  lw=1.0, label="10 ns precision target")
    ax2.set_xlabel("Autonomous operation duration (hours)")
    ax2.set_ylabel("LPT-UTC offset 1-sigma (ns, log)")
    ax2.set_title("Timing Service Traceability\n(LPT-UTC drift, free-running master)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"lpt_stability_{label}.png"),
                dpi=120, bbox_inches="tight")
    plt.close()


def _save_summary_json(results: dict, lpt_stability: dict, output_dir: str, label: str) -> None:
    import json
    out = {}
    for k, v in results.items():
        out[k] = {
            "mode_label":       v["label"].replace("\n", " "),
            "pdop":             v["pdop"],
            "dual_frequency":   v["dual_frequency"],
            "uere_rss_m":       v["uere"]["uere_rss_m"],
            "uere_components":  v["uere"]["components"],
            "cep_50_m":         v["accuracy"]["cep_50_m"],
            "sep_50_m":         v["accuracy"]["sep_50_m"],
            "sigma_3d_m":       v["accuracy"]["sigma_3d_m"],
            "sigma_v_m":        v["accuracy"]["sigma_v_m"],
            "timing_1sigma_ns": v["accuracy"]["timing_1sigma_ns"],
        }
    out["lpt_stability_24h"] = {
        name: {
            "ppb":           d["master_clock_ppb"],
            "sigma_lpt_ns":  d["sigma_lpt_ns"],
            "sigma_lpt_m":   d["sigma_lpt_m"],
        }
        for name, d in lpt_stability.items()
    }
    with open(os.path.join(output_dir, f"time_scale_{label}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def print_time_scale_summary(label: str, analysis: dict) -> None:
    sep = "=" * 76
    print(f"\n{sep}")
    print(f"  LEO-PNT Time Scale & Accuracy — {label}")
    print(sep)
    print(f"\n  {'Mode':<38} {'PDOP':>5} {'UERE':>7} {'CEP':>6} {'Vert':>6} {'t-sig(ns)':>10}")
    print("  " + "-" * 72)
    for k, v in analysis["modes"].items():
        acc = v["accuracy"]
        lbl = v["label"].replace("\n", " ")
        print(f"  {lbl:<38} {acc['pdop']:>5.2f} {v['uere']['uere_rss_m']:>7.3f} "
              f"{acc['cep_50_m']:>6.2f} {acc['sigma_v_m']:>6.2f} "
              f"{acc['timing_1sigma_ns']:>10.2f}")

    print(f"\n  LPT inter-satellite sync error  (8 ISL hops, sync=60 s):")
    print(f"  [Navigation-relevant: determines clock bias in UERE; constant, not time-dependent]")
    print(f"  {'Clock':>8}  {'ppb':>10}  {'sigma_ISL (ns)':>16}  {'range bias (m)':>16}")
    print("  " + "-" * 58)
    for name, d in analysis["lpt_stability"].items():
        print(f"  {name:>8}  {d['master_clock_ppb']:>10.4f}  "
              f"{d['sigma_isl_ns']:>16.5f}  {d['sigma_isl_m']:>16.5f}")

    print(f"\n  LPT-UTC offset (master clock free-run, autonomous 24 h):")
    print(f"  [Timing-service-relevant: does NOT affect position accuracy]")
    print(f"  {'Clock':>8}  {'ppb':>10}  {'sigma_UTC (ns)':>16}  {'time bias (m)':>16}")
    print("  " + "-" * 58)
    for name, d in analysis["lpt_stability"].items():
        print(f"  {name:>8}  {d['master_clock_ppb']:>10.4f}  "
              f"{d['sigma_utc_ns']:>16.3f}  {d['sigma_utc_m']:>16.3f}")

    print(f"\n  Key comparison (dual-freq, PDOP autonomous={analysis['pdop_leo']:.2f}, "
          f"combined={analysis['pdop_combined']:.2f}):")
    for k in ("autonomous_df", "combined_df", "combined_sdcm_df"):
        if k in analysis["modes"]:
            v = analysis["modes"][k]
            acc = v["accuracy"]
            print(f"    {v['label'].replace(chr(10),' '):<42}  "
                  f"CEP={acc['cep_50_m']:.2f} m  Vert={acc['sigma_v_m']:.2f} m")
    print(f"{sep}\n")
