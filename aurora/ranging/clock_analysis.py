"""
Clock type comparison and ISL synchronization chain analysis.

Clock types modelled:
  TCXO    — 10 ppb   (cheap oscillator, commercial smallsat)
  MEMS    —  5 ppb   (MEMS oscillator)
  OCXO    —  1 ppb   (oven-controlled, mid-range)
  Rb      —  0.1 ppb (rubidium atomic, navigation satellite standard)
  Cs      —  0.01 ppb (cesium atomic, high-stability)
  Maser   —  0.001 ppb (hydrogen maser, ground master clock)

ISL sync chain: each hop adds clock uncertainty. For N-hop chain:
  sigma_total(N) = sqrt(N) * sigma_single_hop

References:
  - IS-GPS-200: satellite clock specs
  - Lombardi et al., "Time & Frequency" (NIST), 2002
  - Capuano et al., "LEO Clock Sync via ISL", ION GNSS+ 2022
"""

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_C = 299_792_458.0   # m/s
_NS_PER_S = 1e9

CLOCK_TYPES = {
    "TCXO":   {"ppb": 10.0,   "color": "#E53935", "linestyle": "-"},
    "MEMS":   {"ppb":  5.0,   "color": "#FB8C00", "linestyle": "--"},
    "OCXO":   {"ppb":  1.0,   "color": "#FDD835", "linestyle": "-."},
    "Rb":     {"ppb":  0.1,   "color": "#43A047", "linestyle": "-"},
    "Cs":     {"ppb":  0.01,  "color": "#1E88E5", "linestyle": "--"},
    "Maser":  {"ppb":  0.001, "color": "#8E24AA", "linestyle": "-."},
}


def clock_sync_error_ns(ppb: float, sync_interval_s: float) -> float:
    """
    One-sigma clock drift between ISL synchronization events (nanoseconds).
    sigma = stability_ppb * 1e-9 * sync_interval_s * 1e9
          = stability_ppb * sync_interval_s  [ns]
    """
    return ppb * sync_interval_s  # ns


def isl_chain_error_ns(ppb: float, sync_interval_s: float, n_hops: int) -> float:
    """Clock sync error after N ISL hops (random walk, root-sum-square)."""
    single_hop_ns = clock_sync_error_ns(ppb, sync_interval_s)
    return math.sqrt(n_hops) * single_hop_ns


def max_hops_for_target(ppb: float, sync_interval_s: float, target_ns: float) -> float:
    """Maximum ISL hops before timing error exceeds target (nanoseconds)."""
    single_hop_ns = clock_sync_error_ns(ppb, sync_interval_s)
    if single_hop_ns <= 0:
        return float("inf")
    return (target_ns / single_hop_ns) ** 2


def max_sync_interval_for_target(ppb: float, n_hops: int, target_ns: float) -> float:
    """Maximum sync interval (seconds) to stay within target timing error."""
    if n_hops <= 0 or ppb <= 0:
        return float("inf")
    # target_ns = sqrt(n_hops) * ppb * T  =>  T = target_ns / (sqrt(n_hops) * ppb)
    return target_ns / (math.sqrt(n_hops) * ppb)


def run_clock_analysis(
    sync_interval_s: float = 60.0,
    target_ns: float = 10.0,
    max_hops: int = 20,
    output_dir: str = "results/clock_analysis",
    label: str = "phase4_global",
) -> dict:
    """
    Full clock analysis: per-type sync error vs hop count, max hops for 10ns target.
    Returns summary dict.
    """
    os.makedirs(output_dir, exist_ok=True)
    hops_range = list(range(1, max_hops + 1))
    summary = {}

    for name, info in CLOCK_TYPES.items():
        ppb = info["ppb"]
        errors_ns = [isl_chain_error_ns(ppb, sync_interval_s, n) for n in hops_range]
        max_h = max_hops_for_target(ppb, sync_interval_s, target_ns)
        max_T = max_sync_interval_for_target(ppb, max_hops, target_ns)
        summary[name] = {
            "ppb": ppb,
            "single_hop_error_ns": round(clock_sync_error_ns(ppb, sync_interval_s), 4),
            "max_hops_at_10ns": round(max_h, 1),
            "max_sync_interval_s_at_10ns_20hops": round(max_T, 2),
            "errors_ns_by_hop": [round(e, 4) for e in errors_ns],
        }

    _plot_clock_analysis(summary, hops_range, target_ns, sync_interval_s, output_dir, label)
    _plot_sync_interval_sweep(sync_interval_s, max_hops, target_ns, output_dir, label)
    _save_clock_csv(summary, hops_range, output_dir, label)
    return summary


def _plot_clock_analysis(summary, hops_range, target_ns, sync_interval_s, output_dir, label):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"ISL Clock Synchronization Analysis — {label}  "
                 f"(sync interval={sync_interval_s:.0f}s)", fontweight="bold")

    for name, data in summary.items():
        info = CLOCK_TYPES[name]
        ax1.semilogy(hops_range, data["errors_ns_by_hop"],
                     color=info["color"], linestyle=info["linestyle"],
                     linewidth=2, label=f"{name} ({data['ppb']} ppb)")

    ax1.axhline(target_ns, color="black", linestyle=":", linewidth=1.5,
                label=f"{target_ns:.0f} ns target")
    ax1.axhline(1.0, color="grey", linestyle=":", linewidth=1, label="1 ns (precision)")
    ax1.set_xlabel("Number of ISL hops from master clock")
    ax1.set_ylabel("Timing error 1-sigma (ns, log scale)")
    ax1.set_title("Sync Error vs Hop Count")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, which="both")
    ax1.set_xlim(1, max(hops_range))

    # Bar chart: max hops before exceeding target
    names = list(summary.keys())
    max_hops_vals = [min(summary[n]["max_hops_at_10ns"], max(hops_range)) for n in names]
    colors = [CLOCK_TYPES[n]["color"] for n in names]
    bars = ax2.barh(names, max_hops_vals, color=colors, alpha=0.85)
    ax2.axvline(max(hops_range), color="grey", linestyle="--", linewidth=1)
    for bar, val in zip(bars, [summary[n]["max_hops_at_10ns"] for n in names]):
        label_txt = f"{val:.1f}" if val <= max(hops_range) else f">{max(hops_range)}"
        ax2.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                 label_txt, va="center", fontsize=9)
    ax2.set_xlabel(f"Max ISL hops before >{target_ns:.0f} ns timing error")
    ax2.set_title(f"Max Hops at {target_ns:.0f} ns threshold")
    ax2.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"clock_analysis_{label}.png"), dpi=120, bbox_inches="tight")
    plt.close()


def _plot_sync_interval_sweep(sync_interval_s, max_hops, target_ns, output_dir, label):
    """Show how sync interval affects timing error at fixed hop count."""
    intervals = [1, 5, 10, 30, 60, 120, 300, 600]
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f"Effect of Sync Interval on Timing Error — {label}  "
                 f"({max_hops} hops)", fontweight="bold")

    for name, info in CLOCK_TYPES.items():
        errors = [isl_chain_error_ns(info["ppb"], T, max_hops) for T in intervals]
        ax.loglog(intervals, errors, color=info["color"], linestyle=info["linestyle"],
                  linewidth=2, marker="o", markersize=5, label=f"{name} ({info['ppb']} ppb)")

    ax.axhline(target_ns, color="black", linestyle=":", linewidth=1.5,
               label=f"{target_ns:.0f} ns target")
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=1, label="1 ns precision")
    ax.set_xlabel("ISL sync interval (seconds)")
    ax.set_ylabel("Timing error 1-sigma (ns, log scale)")
    ax.set_title(f"Timing Error vs Sync Interval  ({max_hops} hops)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"clock_sync_interval_{label}.png"),
                dpi=120, bbox_inches="tight")
    plt.close()


def _save_clock_csv(summary, hops_range, output_dir, label):
    path = os.path.join(output_dir, f"clock_analysis_{label}.csv")
    with open(path, "w", encoding="utf-8") as f:
        header = ["clock_type", "ppb", "single_hop_ns", "max_hops_10ns"] + \
                 [f"hop_{n}_ns" for n in hops_range]
        f.write(",".join(header) + "\n")
        for name, data in summary.items():
            row = [name, str(data["ppb"]),
                   str(data["single_hop_error_ns"]),
                   str(data["max_hops_at_10ns"])]
            row += [str(e) for e in data["errors_ns_by_hop"]]
            f.write(",".join(row) + "\n")


def print_clock_summary(label: str, summary: dict, target_ns: float = 10.0) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  ISL Clock Synchronization Analysis — {label}")
    print(sep)
    print(f"  {'Clock':>8} {'Stability':>12} {'1-hop err':>12} {'Max hops @{:.0f}ns'.format(target_ns):>18} {'Max T @20hops':>14}")
    print("  " + "-" * 68)
    for name, data in summary.items():
        mh = data["max_hops_at_10ns"]
        mh_str = f"{mh:.1f}" if mh < 1e6 else ">1M"
        print(f"  {name:>8}  {data['ppb']:>8.3f} ppb  {data['single_hop_error_ns']:>9.4f} ns  "
              f"{mh_str:>18}  {data['max_sync_interval_s_at_10ns_20hops']:>10.2f} s")
    print(f"{sep}\n")
