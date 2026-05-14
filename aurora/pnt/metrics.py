"""
Aggregate PNT metrics over time steps and grid points.
"""

import math
from typing import Any

import numpy as np

# Minimum satellites for PNT fix
MIN_SATS_FOR_FIX = 4

# PDOP threshold for "good" navigation
PDOP_GOOD_THRESHOLD = 6.0


def aggregate_timestep(point_results: list[dict]) -> dict:
    """
    Aggregate coverage metrics across all grid points at a single time step.

    Args:
        point_results: List of per-point dicts from compute_grid_visibility_at_time.

    Returns:
        Dict with aggregate statistics for this time step.
    """
    n_points = len(point_results)
    if n_points == 0:
        return {}

    n_sats_list = [r["n_sats"] for r in point_results]
    pdop_valid = [r["pdop"] for r in point_results if r["pdop"] is not None]
    gdop_valid = [r["gdop"] for r in point_results if r["gdop"] is not None]

    coverage_4plus = sum(1 for n in n_sats_list if n >= MIN_SATS_FOR_FIX) / n_points
    coverage_good_pdop = (
        sum(1 for p in pdop_valid if p <= PDOP_GOOD_THRESHOLD) / n_points
        if pdop_valid else 0.0
    )

    return {
        "n_points": n_points,
        "coverage_4sats_pct": round(coverage_4plus * 100, 2),
        "coverage_good_pdop_pct": round(coverage_good_pdop * 100, 2),
        "n_sats_min": int(min(n_sats_list)),
        "n_sats_max": int(max(n_sats_list)),
        "n_sats_mean": round(float(np.mean(n_sats_list)), 2),
        "pdop_min": round(min(pdop_valid), 3) if pdop_valid else None,
        "pdop_max": round(max(pdop_valid), 3) if pdop_valid else None,
        "pdop_mean": round(float(np.mean(pdop_valid)), 3) if pdop_valid else None,
        "pdop_p95": round(float(np.percentile(pdop_valid, 95)), 3) if pdop_valid else None,
        "gdop_mean": round(float(np.mean(gdop_valid)), 3) if gdop_valid else None,
        "gdop_p95": round(float(np.percentile(gdop_valid, 95)), 3) if gdop_valid else None,
    }


def aggregate_simulation(timestep_metrics: list[dict]) -> dict:
    """
    Aggregate metrics across all time steps of a simulation run.

    Args:
        timestep_metrics: List of per-timestep aggregate dicts.

    Returns:
        Summary dict with overall simulation statistics.
    """
    if not timestep_metrics:
        return {}

    cov = [m["coverage_4sats_pct"] for m in timestep_metrics]
    good_pdop = [m["coverage_good_pdop_pct"] for m in timestep_metrics]
    pdop_p95 = [m["pdop_p95"] for m in timestep_metrics if m.get("pdop_p95") is not None]
    pdop_mean = [m["pdop_mean"] for m in timestep_metrics if m.get("pdop_mean") is not None]
    n_sats_mean = [m["n_sats_mean"] for m in timestep_metrics]

    return {
        # Coverage availability
        "coverage_4sats_mean_pct": round(float(np.mean(cov)), 2),
        "coverage_4sats_min_pct": round(float(np.min(cov)), 2),
        "coverage_4sats_p05_pct": round(float(np.percentile(cov, 5)), 2),

        # PDOP quality
        "pdop_p95_mean": round(float(np.mean(pdop_p95)), 3) if pdop_p95 else None,
        "pdop_p95_worst": round(float(np.max(pdop_p95)), 3) if pdop_p95 else None,
        "pdop_mean_overall": round(float(np.mean(pdop_mean)), 3) if pdop_mean else None,

        # Coverage with good PDOP
        "coverage_good_pdop_mean_pct": round(float(np.mean(good_pdop)), 2),

        # Satellite count
        "n_sats_mean_overall": round(float(np.mean(n_sats_mean)), 2),

        # Number of time steps
        "n_timesteps": len(timestep_metrics),
    }


def print_summary(run_label: str, summary: dict) -> None:
    """Print a human-readable summary of simulation results."""
    sep = "─" * 55
    print(f"\n{sep}")
    print(f"  {run_label}")
    print(sep)
    print(f"  Покрытие 4+ спутников:  {summary.get('coverage_4sats_mean_pct', 'N/A'):>7.2f}% (среднее)")
    print(f"                          {summary.get('coverage_4sats_min_pct', 'N/A'):>7.2f}% (минимум)")
    print(f"  PDOP p95 (среднее):     {summary.get('pdop_p95_mean', 'N/A')!r:>10}")
    print(f"  PDOP p95 (худший шаг):  {summary.get('pdop_p95_worst', 'N/A')!r:>10}")
    print(f"  Покрытие PDOP<6:        {summary.get('coverage_good_pdop_mean_pct', 'N/A'):>7.2f}%")
    print(f"  Среднее кол-во спутн.:  {summary.get('n_sats_mean_overall', 'N/A'):>7.2f}")
    print(f"  Шагов симуляции:        {summary.get('n_timesteps', 0):>7d}")
    print(sep)
