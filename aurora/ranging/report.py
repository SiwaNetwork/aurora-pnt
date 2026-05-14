"""
Ranging report: CSV, JSON, and plots.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def save_ranging_report(
    results: list[dict],
    summary: dict,
    gs_names: list[str],
    output_dir: str,
    label: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    _save_csv(results, output_dir, label)
    _save_json(summary, output_dir, label)
    _plot_uere_breakdown(results, gs_names, output_dir, label)
    _plot_position_error_cdf(results, gs_names, output_dir, label)
    _plot_timing_error_timeseries(results, gs_names, output_dir, label)
    _plot_accuracy_vs_elevation(results, output_dir, label)


def _save_csv(results, output_dir, label):
    path = os.path.join(output_dir, f"ranging_{label}.csv")
    fields = ["time_h", "gs_name", "sat_id", "elevation_deg", "distance_km",
              "cn0_dbhz", "pseudorange_noise_m", "clock_sync_error_m",
              "tropo_error_m", "iono_error_m", "uere_m",
              "pos_error_3d_m", "time_error_ns", "propagation_delay_ms"]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(fields) + "\n")
        for r in results:
            f.write(",".join(str(r.get(k, "")) for k in fields) + "\n")


def _save_json(summary, output_dir, label):
    path = os.path.join(output_dir, f"ranging_summary_{label}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def _plot_uere_breakdown(results, gs_names, output_dir, label):
    """Stacked bar showing UERE components per GS (mean values)."""
    components = ["pseudorange_noise_m", "clock_sync_error_m", "tropo_error_m",
                  "iono_error_m"]
    labels_nice = ["Thermal noise", "Clock sync (ISL)", "Troposphere", "Ionosphere"]
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]

    means = {c: [] for c in components}
    gs_valid = []
    for gs in gs_names:
        gs_r = [r for r in results if r["gs_name"] == gs]
        if not gs_r:
            continue
        gs_valid.append(gs)
        for c in components:
            means[c].append(float(np.mean([r[c] for r in gs_r])))

    x = np.arange(len(gs_valid))
    fig, ax = plt.subplots(figsize=(12, 5))
    bottom = np.zeros(len(gs_valid))
    for c, lab, col in zip(components, labels_nice, colors):
        vals = np.array(means[c])
        ax.bar(x, vals, bottom=bottom, label=lab, color=col, alpha=0.85)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(gs_valid, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Range error contribution (m, 1-sigma)")
    ax.set_title(f"UERE Error Budget by Component — {label}", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"uere_breakdown_{label}.png"), dpi=120, bbox_inches="tight")
    plt.close()


def _plot_position_error_cdf(results, gs_names, output_dir, label):
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.tab10

    for i, gs in enumerate(gs_names[:10]):
        errs = sorted([r["pos_error_3d_m"] for r in results if r["gs_name"] == gs])
        if not errs:
            continue
        p = np.arange(1, len(errs) + 1) / len(errs) * 100
        ax.plot(errs, p, color=cmap(i), linewidth=1.5, label=gs)

    ax.axvline(10.0, color="green", linestyle="--", linewidth=1, label="10 m threshold")
    ax.axvline(5.0, color="orange", linestyle="--", linewidth=1, label="5 m threshold")
    ax.axhline(95, color="grey", linestyle=":", linewidth=1, alpha=0.6, label="95th percentile")
    ax.set_xlabel("3D Position Error (m, 1-sigma)")
    ax.set_ylabel("CDF (%)")
    ax.set_title(f"Position Error CDF — {label}", fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"pos_error_cdf_{label}.png"), dpi=120, bbox_inches="tight")
    plt.close()


def _plot_timing_error_timeseries(results, gs_names, output_dir, label):
    fig, ax = plt.subplots(figsize=(13, 4))
    cmap = plt.cm.tab10

    for i, gs in enumerate(gs_names[:10]):
        gs_r = sorted([r for r in results if r["gs_name"] == gs], key=lambda x: x["time_h"])
        if not gs_r:
            continue
        t = [r["time_h"] for r in gs_r]
        te = [r["time_error_ns"] for r in gs_r]
        ax.plot(t, te, color=cmap(i), linewidth=1.2, label=gs, alpha=0.85)

    ax.axhline(10.0, color="red", linestyle="--", linewidth=1, label="10 ns target")
    ax.axhline(5.0, color="green", linestyle="--", linewidth=1, label="5 ns target")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Timing error (ns, 1-sigma)")
    ax.set_title(f"Timing Accuracy over 24h — {label}", fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"timing_error_{label}.png"), dpi=120, bbox_inches="tight")
    plt.close()


def _plot_accuracy_vs_elevation(results, output_dir, label):
    els = np.array([r["elevation_deg"] for r in results])
    pos = np.array([r["pos_error_3d_m"] for r in results])
    t_err = np.array([r["time_error_ns"] for r in results])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Accuracy vs Elevation Angle — {label}", fontweight="bold")

    ax1.scatter(els, pos, s=4, alpha=0.3, c="#E91E63")
    ax1.set_xlabel("Elevation (deg)")
    ax1.set_ylabel("3D Position Error (m)")
    ax1.set_title("Position Error")
    ax1.grid(True, alpha=0.3)

    ax2.scatter(els, t_err, s=4, alpha=0.3, c="#009688")
    ax2.set_xlabel("Elevation (deg)")
    ax2.set_ylabel("Timing Error (ns)")
    ax2.set_title("Timing Error")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"accuracy_vs_elevation_{label}.png"), dpi=120, bbox_inches="tight")
    plt.close()


def print_summary(label: str, summary: dict) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  Ranging & Timing Accuracy — {label}")
    print(sep)
    print(f"  {'Station':<22} {'UERE p95':>9} {'Pos 3D p95':>11} {'Time p95':>10} {'Delay':>8}")
    print("  " + "-" * 65)
    for gs, m in summary.items():
        print(f"  {gs:<22}"
              f"  {m['uere_p95_m']:>6.2f} m"
              f"  {m['pos_error_p95_m']:>8.2f} m"
              f"  {m['time_error_p95_ns']:>7.2f} ns"
              f"  {m['prop_delay_mean_ms']:>5.1f} ms")
    print(f"{sep}\n")
