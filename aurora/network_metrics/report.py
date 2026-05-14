"""
Save network metrics results to CSV, JSON, and PNG plots.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def save_network_metrics(
    snapshots: list[dict],
    gsl_metrics: dict,
    isl_metrics: dict,
    latency_metrics: dict,
    stability: dict,
    gs_names: list[str],
    output_dir: str,
    label: str,
    path_stability: dict | None = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    _save_summary_json(gsl_metrics, isl_metrics, latency_metrics, stability, output_dir, label,
                       path_stability=path_stability)
    _save_gsl_csv(snapshots, gs_names, output_dir, label)
    _save_isl_csv(snapshots, output_dir, label)
    _save_latency_csv(latency_metrics, output_dir, label)
    if path_stability:
        _save_path_stability_csv(path_stability, output_dir, label)
    _plot_gsl_handovers(gsl_metrics, gs_names, output_dir, label)
    _plot_isl_over_time(isl_metrics, output_dir, label)
    _plot_latency_heatmap(latency_metrics, gs_names, output_dir, label)
    _plot_gsl_attachment_timeline(snapshots, gs_names, output_dir, label)
    if path_stability:
        _plot_path_stability(path_stability, output_dir, label)


# ── JSON summary ──────────────────────────────────────────────────────────────

def _save_summary_json(gsl_metrics, isl_metrics, latency_metrics, stability, output_dir, label,
                       path_stability=None):
    summary = {
        "label": label,
        "gsl": gsl_metrics,
        "isl": {k: v for k, v in isl_metrics.items() if not k.endswith("_series")},
        "latency": {
            f"{a}_to_{b}": {k: v for k, v in m.items() if k != "series_ms"}
            for (a, b), m in latency_metrics.items()
        },
        "routing_stability": stability,
    }
    if path_stability:
        summary["path_stability"] = {
            f"{a}_to_{b}": m for (a, b), m in path_stability.items()
        }
    path = os.path.join(output_dir, f"network_metrics_{label}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)


# ── CSV exports ───────────────────────────────────────────────────────────────

def _save_gsl_csv(snapshots, gs_names, output_dir, label):
    path = os.path.join(output_dir, f"gsl_attachment_{label}.csv")
    header = ["time_h"] + [f"sat_{g}" for g in gs_names] + [f"dist_km_{g}" for g in gs_names]
    rows = []
    for s in snapshots:
        row = [f"{s['time_h']:.4f}"]
        for g in gs_names:
            sat = s["gsl_attachment"].get(g)
            row.append(str(sat) if sat is not None else "")
        for g in gs_names:
            d = s["gsl_distance_m"].get(g)
            row.append(f"{d/1000:.2f}" if d is not None else "")
        rows.append(row)
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(r) + "\n")


def _save_isl_csv(snapshots, output_dir, label):
    path = os.path.join(output_dir, f"isl_stats_{label}.csv")
    with open(path, "w", encoding="utf-8") as f:
        f.write("time_h,n_active_isls,isl_mean_dist_km\n")
        for s in snapshots:
            f.write(f"{s['time_h']:.4f},{s['n_active_isls']},{s['isl_mean_dist_m']/1000:.2f}\n")


def _save_latency_csv(latency_metrics, output_dir, label):
    path = os.path.join(output_dir, f"latency_summary_{label}.csv")
    with open(path, "w", encoding="utf-8") as f:
        f.write("from,to,latency_mean_ms,latency_min_ms,latency_max_ms,availability_pct\n")
        for (a, b), m in latency_metrics.items():
            f.write(
                f"{a},{b},{m['latency_mean_ms']},{m['latency_min_ms']},"
                f"{m['latency_max_ms']},{m['availability_pct']}\n"
            )


# ── Plots ─────────────────────────────────────────────────────────────────────

def _plot_gsl_handovers(gsl_metrics, gs_names, output_dir, label):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"GSL Handover Analysis — {label}", fontsize=13, fontweight="bold")

    names = list(gs_names)
    counts = [gsl_metrics[g]["handover_count"] for g in names]
    rates = [gsl_metrics[g]["handover_rate_per_h"] for g in names]
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(names)))

    ax = axes[0]
    bars = ax.barh(names, counts, color=colors)
    ax.set_xlabel("Handovers per 24 h")
    ax.set_title("Total GSL Handover Count")
    ax.bar_label(bars, fmt="%d", padding=3)
    ax.set_xlim(0, max(counts) * 1.2 if counts else 1)

    ax = axes[1]
    bars = ax.barh(names, rates, color=colors)
    ax.set_xlabel("Handovers / hour")
    ax.set_title("Handover Rate")
    ax.bar_label(bars, fmt="%.1f", padding=3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"gsl_handovers_{label}.png"), dpi=120, bbox_inches="tight")
    plt.close()


def _plot_isl_over_time(isl_metrics, output_dir, label):
    t = isl_metrics["time_h_series"]
    counts = isl_metrics["active_counts_series"]
    dists = isl_metrics["mean_dists_km_series"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(f"ISL Topology Over Time — {label}", fontsize=13, fontweight="bold")

    ax1.plot(t, counts, color="#2196F3", linewidth=1.5)
    ax1.set_ylabel("Active ISLs")
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(np.mean(counts), color="#F44336", linestyle="--", linewidth=1, label=f"Mean={np.mean(counts):.0f}")
    ax1.legend(fontsize=9)

    ax2.plot(t, dists, color="#4CAF50", linewidth=1.5)
    ax2.set_ylabel("Mean ISL Distance (km)")
    ax2.set_xlabel("Time (hours)")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(np.mean(dists), color="#F44336", linestyle="--", linewidth=1, label=f"Mean={np.mean(dists):.0f} km")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"isl_over_time_{label}.png"), dpi=120, bbox_inches="tight")
    plt.close()


def _plot_latency_heatmap(latency_metrics, gs_names, output_dir, label):
    n = len(gs_names)
    matrix = np.full((n, n), np.nan)
    for (a, b), m in latency_metrics.items():
        if m["latency_mean_ms"] is not None:
            i = gs_names.index(a)
            j = gs_names.index(b)
            matrix[i, j] = m["latency_mean_ms"]
            matrix[j, i] = m["latency_mean_ms"]

    fig, ax = plt.subplots(figsize=(max(8, n * 0.7), max(6, n * 0.6)))
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad("lightgrey")
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("One-way latency (ms)", fontsize=10)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(gs_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(gs_names, fontsize=8)
    ax.set_title(f"Mean One-Way Propagation Latency (ms) — {label}", fontsize=11, fontweight="bold")

    for i in range(n):
        for j in range(n):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i,j]:.0f}", ha="center", va="center",
                        fontsize=6, color="black" if matrix[i, j] < np.nanmax(matrix) * 0.7 else "white")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"latency_heatmap_{label}.png"), dpi=120, bbox_inches="tight")
    plt.close()


def _plot_gsl_attachment_timeline(snapshots, gs_names, output_dir, label):
    """Timeline showing which satellite each GS is attached to over 24h."""
    times = [s["time_h"] for s in snapshots]
    fig, axes = plt.subplots(len(gs_names), 1, figsize=(14, max(6, len(gs_names) * 0.8)),
                             sharex=True)
    if len(gs_names) == 1:
        axes = [axes]

    fig.suptitle(f"GSL Attachment Timeline — {label}", fontsize=12, fontweight="bold")

    for ax, gs in zip(axes, gs_names):
        sats = [s["gsl_attachment"].get(gs) for s in snapshots]
        # Assign color per unique satellite
        unique_sats = sorted({s for s in sats if s is not None})
        cmap = plt.cm.tab20
        sat_color = {s: cmap(i % 20) for i, s in enumerate(unique_sats)}

        prev_t = times[0]
        prev_sat = sats[0]
        for i in range(1, len(times)):
            if sats[i] != prev_sat or i == len(times) - 1:
                color = sat_color.get(prev_sat, "lightgrey") if prev_sat is not None else "lightgrey"
                ax.barh(0, times[i] - prev_t, left=prev_t, height=0.6, color=color, edgecolor="none")
                prev_t = times[i]
                prev_sat = sats[i]

        ax.set_yticks([0])
        ax.set_yticklabels([gs], fontsize=8)
        ax.set_xlim(0, 24)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[-1].set_xlabel("Time (hours)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"gsl_timeline_{label}.png"), dpi=120, bbox_inches="tight")
    plt.close()


def print_summary(label: str, gsl_metrics: dict, isl_metrics: dict, stability: dict) -> None:
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  Network Metrics — {label}")
    print(sep)

    print("\n  ISL Topology")
    print(f"    Active ISLs (mean/min/max): "
          f"{isl_metrics['isl_count_mean']:.0f} / "
          f"{isl_metrics['isl_count_min']} / "
          f"{isl_metrics['isl_count_max']}")
    print(f"    ISL distance (mean/min/max km): "
          f"{isl_metrics['isl_dist_mean_km']:.0f} / "
          f"{isl_metrics['isl_dist_min_km']:.0f} / "
          f"{isl_metrics['isl_dist_max_km']:.0f}")

    print("\n  GSL Handovers (per 24 h)")
    for gs, m in gsl_metrics.items():
        print(f"    {gs:<20} {m['handover_count']:>3} handovers  "
              f"({m['handover_rate_per_h']:.1f}/h)  "
              f"avg attach {m['mean_attachment_duration_min']:.0f} min  "
              f"coverage {m['coverage_pct']:.0f}%")

    print(f"\n  Routing Stability: {stability['topology_change_rate_pct']:.0f}% of steps "
          f"have at least one GSL change")
    print(f"{sep}\n")


def _save_path_stability_csv(path_stability: dict, output_dir: str, label: str) -> None:
    path = os.path.join(output_dir, f"path_stability_{label}.csv")
    fields = ["pair", "path_lifetime_mean_steps", "path_change_count",
              "path_change_rate_pct", "latency_jitter_ms", "hop_count_mean",
              "timing_path_uncertainty_ns"]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(fields) + "\n")
        for (a, b), m in path_stability.items():
            row = [f"{a}_to_{b}"] + [str(m.get(k, "")) for k in fields[1:]]
            f.write(",".join(row) + "\n")


def _plot_path_stability(path_stability: dict, output_dir: str, label: str) -> None:
    """Scatter: latency jitter vs hop count; bubble size = path change rate."""
    pairs = list(path_stability.items())
    jitters = [m["latency_jitter_ms"] for _, m in pairs if m["latency_jitter_ms"] is not None]
    hops    = [m["hop_count_mean"]    for _, m in pairs if m["latency_jitter_ms"] is not None]
    changes = [m["path_change_rate_pct"] for _, m in pairs if m["latency_jitter_ms"] is not None]
    names   = [f"{a[:3]}-{b[:3]}" for (a, b), m in pairs if m["latency_jitter_ms"] is not None]

    if not jitters:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Path Stability (Hypatia) — {label}", fontweight="bold")

    sc = ax1.scatter(hops, jitters, c=changes, cmap="RdYlGn_r",
                     s=60, alpha=0.8, edgecolors="grey", linewidths=0.4)
    plt.colorbar(sc, ax=ax1, label="Path change rate (%)")
    ax1.set_xlabel("Mean hop count (ISL hops)")
    ax1.set_ylabel("Latency jitter (ms, 1-sigma)")
    ax1.set_title("Jitter vs Hops")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(1.0, color="orange", linestyle="--", linewidth=1, label="1 ms jitter")
    ax1.axhline(5.0, color="red",    linestyle="--", linewidth=1, label="5 ms jitter")
    ax1.legend(fontsize=8)

    # Timing uncertainty histogram
    uncerts = [m["timing_path_uncertainty_ns"] for _, m in pairs
               if m["timing_path_uncertainty_ns"] is not None]
    ax2.hist(uncerts, bins=20, color="#1565C0", alpha=0.8, edgecolor="white")
    ax2.axvline(10.0, color="orange", linestyle="--", linewidth=1.5, label="10 ns target")
    ax2.axvline(100.0, color="red",   linestyle="--", linewidth=1.5, label="100 ns limit")
    ax2.set_xlabel("Timing path uncertainty (ns, 1-sigma)")
    ax2.set_ylabel("GS pair count")
    ax2.set_title("Sync Uncertainty Distribution")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"path_stability_{label}.png"), dpi=120, bbox_inches="tight")
    plt.close()
