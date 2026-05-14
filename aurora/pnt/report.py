"""
Save simulation results to CSV/JSON and generate matplotlib plots.
"""

import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


# ── Raw data ─────────────────────────────────────────────────────────────────

def save_timestep_csv(
    results: list[dict],
    time_ns: int,
    output_dir: str,
    step_idx: int,
) -> None:
    """Save per-point results for one time step to CSV."""
    _ensure_dir(output_dir)
    fname = os.path.join(output_dir, f"step_{step_idx:05d}.csv")
    if not results:
        return
    fieldnames = list(results[0].keys()) + ["time_ns"]
    with open(fname, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({**row, "time_ns": time_ns})


def save_timestep_metrics_csv(
    timestep_metrics: list[dict],
    output_dir: str,
    filename: str = "timestep_metrics.csv",
) -> None:
    """Save per-timestep aggregate metrics to a single CSV."""
    _ensure_dir(output_dir)
    if not timestep_metrics:
        return
    path = os.path.join(output_dir, filename)
    fieldnames = list(timestep_metrics[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(timestep_metrics)


def save_summary_json(summary: dict, output_dir: str, label: str = "run") -> None:
    """Save overall simulation summary to JSON."""
    _ensure_dir(output_dir)
    path = os.path.join(output_dir, f"summary_{label}.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_coverage_over_time(
    timestep_metrics: list[dict],
    output_dir: str,
    label: str = "run",
    time_step_minutes: float = 10.0,
) -> None:
    """Line plot: coverage and PDOP p95 over simulation time."""
    _ensure_dir(output_dir)
    times_h = [i * time_step_minutes / 60.0 for i in range(len(timestep_metrics))]
    cov = [m["coverage_4sats_pct"] for m in timestep_metrics]
    pdop_p95 = [m.get("pdop_p95") or float("nan") for m in timestep_metrics]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(f"LEO PNT — {label}", fontsize=13, fontweight="bold")

    ax1.plot(times_h, cov, color="#2196F3", linewidth=1.5)
    ax1.axhline(95, color="#FF5722", linestyle="--", linewidth=1, label="95% цель")
    ax1.set_ylabel("Покрытие 4+ спутников (%)")
    ax1.set_ylim(0, 105)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(times_h, pdop_p95, color="#4CAF50", linewidth=1.5)
    ax2.axhline(6.0, color="#FF5722", linestyle="--", linewidth=1, label="PDOP<6 цель")
    ax2.set_ylabel("PDOP p95")
    ax2.set_xlabel("Время (часы)")
    ax2.set_ylim(0, None)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f"coverage_time_{label}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_coverage_map(
    point_results_list: list[list[dict]],
    output_dir: str,
    label: str = "run",
    metric: str = "pdop",
) -> None:
    """
    Heatmap of average PDOP (or n_sats) over all time steps at each grid point.

    Args:
        point_results_list: List of per-timestep per-point result lists.
        metric: "pdop" or "n_sats"
    """
    _ensure_dir(output_dir)
    if not point_results_list:
        return

    # Accumulate values per (lat, lon)
    acc: dict[tuple, list] = {}
    for step_results in point_results_list:
        for row in step_results:
            key = (row["lat"], row["lon"])
            val = row[metric] if metric == "n_sats" else row.get("pdop")
            if val is not None:
                acc.setdefault(key, []).append(val)

    if not acc:
        return

    lats = sorted(set(k[0] for k in acc))
    lons = sorted(set(k[1] for k in acc))
    grid = np.full((len(lats), len(lons)), np.nan)

    lat_idx = {v: i for i, v in enumerate(lats)}
    lon_idx = {v: i for i, v in enumerate(lons)}
    for (lat, lon), vals in acc.items():
        grid[lat_idx[lat], lon_idx[lon]] = np.mean(vals)

    fig, ax = plt.subplots(figsize=(14, 6))
    if metric == "n_sats":
        cmap, vmin, vmax = "YlGn", 0, None
        label_cb = "Среднее кол-во спутников"
    else:
        cmap, vmin, vmax = "RdYlGn_r", 1, 10
        label_cb = "Средний PDOP"

    im = ax.pcolormesh(
        lons, lats, grid,
        cmap=cmap, vmin=vmin, vmax=vmax,
        shading="auto",
    )
    plt.colorbar(im, ax=ax, label=label_cb, pad=0.02)
    ax.set_xlabel("Долгота (°)")
    ax.set_ylabel("Широта (°)")
    ax.set_title(f"LEO PNT — {label} — {label_cb}", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.2, color="white")

    path = os.path.join(output_dir, f"map_{metric}_{label}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_experiment_comparison(
    results: list[dict],    # [{"label": str, "summary": dict}, ...]
    output_dir: str,
    sweep_param: str = "parameter",
) -> None:
    """
    Bar chart comparing key metrics across experiment runs.
    """
    _ensure_dir(output_dir)
    if not results:
        return

    labels = [r["label"] for r in results]
    cov = [r["summary"].get("coverage_4sats_mean_pct", 0) for r in results]
    pdop = [r["summary"].get("pdop_p95_mean") or 0 for r in results]

    x = np.arange(len(labels))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Сравнение конфигураций — {sweep_param}", fontsize=13, fontweight="bold")

    bars1 = ax1.bar(x, cov, width, color="#2196F3", alpha=0.85)
    ax1.axhline(95, color="#FF5722", linestyle="--", linewidth=1.2, label="Цель 95%")
    ax1.set_ylabel("Покрытие 4+ спутников (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.set_ylim(0, 105)
    ax1.legend(fontsize=9)
    ax1.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars1, cov):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    bars2 = ax2.bar(x, pdop, width, color="#4CAF50", alpha=0.85)
    ax2.axhline(6.0, color="#FF5722", linestyle="--", linewidth=1.2, label="PDOP<6 цель")
    ax2.set_ylabel("PDOP p95 (среднее по времени)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right")
    ax2.legend(fontsize=9)
    ax2.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars2, pdop):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, f"comparison_{sweep_param}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  График сохранён: {path}")
