"""
Ranging simulation runner.

For each timestep, per GS computes:
  - pseudorange noise from C/N0
  - ISL clock synchronization error
  - full UERE budget
  - position accuracy from PDOP × UERE
  - timing accuracy
"""

import math
import os

import numpy as np
from tqdm import tqdm

from aurora.ranging.models import (
    RangingErrorBudget,
    pseudorange_noise_m,
    isl_sync_error_m,
    propagation_delay_ns,
    relativistic_correction_m,
    position_error_m,
    time_error_ns,
)


_C = 299_792_458.0


def run_ranging_analysis(
    link_budget_snapshots: list[dict],
    network_snapshots: list[dict],
    pnt_summary: dict,
    config: dict,
    bandwidth_hz: float = 10.23e6,
    oscillator_stability_ppb: float = 0.1,   # rubidium clock default
    sync_interval_s: float = 60.0,
    dual_frequency: bool = False,             # L1+L5: cancels ionosphere to ~0.05m
) -> list[dict]:
    """
    Cross-reference link budget and network snapshots to compute per-timestep
    ranging accuracy metrics.

    Returns list of dicts with ranging analysis per GS per timestep.
    """
    altitude_m = config["satellite"]["altitude_m"]
    relativ_m = relativistic_correction_m(altitude_m)

    # Build lookup: time_h -> isl_mean_dist_km
    isl_by_time = {}
    for snap in network_snapshots:
        isl_by_time[round(snap["time_h"], 4)] = snap["isl_mean_dist_m"] / 1000.0

    results = []
    for snap in tqdm(link_budget_snapshots, desc="Ranging analysis"):
        t_key = round(snap["time_h"], 4)
        isl_dist_km = isl_by_time.get(t_key, 3000.0)  # fallback to typical

        # Pseudorange thermal noise
        cn0 = snap["cn0_dbhz"]
        noise_m = pseudorange_noise_m(cn0, bandwidth_hz)

        # Clock sync error via ISL
        clock_via_isl_m = isl_sync_error_m(isl_dist_km, oscillator_stability_ppb, sync_interval_s)

        # Atmospheric losses already in link budget; convert atm_loss to ranging error
        # Rough: troposphere ~2.5m at zenith, mapping as 1/sin(el)
        el = max(snap["elevation_deg"], 5.0)
        tropo_m = 2.5 / math.sin(math.radians(el))
        # Dual-frequency (L1+L5) cancels ionosphere to residual ~0.05 m
        iono_m = 0.05 if dual_frequency else 2.0 / math.sin(math.radians(el))

        budget = RangingErrorBudget(
            clock_bias_m=clock_via_isl_m,
            ephemeris_m=0.5,
            iono_m=iono_m,
            tropo_m=tropo_m,
            multipath_m=0.3,
            noise_m=noise_m,
            relativistic_m=relativ_m,
        )
        uere = budget.uere()

        # Position error using mean PDOP from PNT summary (None for under-coverage configs)
        pdop = pnt_summary.get("pdop_p95_mean") or 10.0
        pos_err_m = position_error_m(uere, pdop)

        # Time error (using TDOP ≈ PDOP × 0.3 approximation)
        tdop = pdop * 0.3
        t_err_ns = time_error_ns(uere, tdop)

        # One-way propagation delay
        prop_ns = propagation_delay_ns(snap["distance_km"] * 1000)

        results.append({
            "time_h": snap["time_h"],
            "gs_name": snap["gs_name"],
            "sat_id": snap["sat_id"],
            "elevation_deg": snap["elevation_deg"],
            "distance_km": snap["distance_km"],
            "cn0_dbhz": cn0,
            "pseudorange_noise_m": round(noise_m, 4),
            "clock_sync_error_m": round(clock_via_isl_m, 4),
            "tropo_error_m": round(tropo_m, 3),
            "iono_error_m": round(iono_m, 3),
            "uere_m": round(uere, 3),
            "pdop": pdop,
            "pos_error_3d_m": round(pos_err_m, 3),
            "time_error_ns": round(t_err_ns, 3),
            "propagation_delay_ms": round(prop_ns / 1e6, 3),
            "relativistic_m_per_s": round(relativ_m, 6),
        })

    return results


def compute_ranging_summary(results: list[dict], gs_names: list[str]) -> dict:
    """Per-GS aggregate ranging statistics."""
    summary = {}
    for gs in gs_names:
        gs_r = [r for r in results if r["gs_name"] == gs]
        if not gs_r:
            continue
        summary[gs] = {
            "uere_mean_m": round(float(np.mean([r["uere_m"] for r in gs_r])), 3),
            "uere_p95_m": round(float(np.percentile([r["uere_m"] for r in gs_r], 95)), 3),
            "pos_error_mean_m": round(float(np.mean([r["pos_error_3d_m"] for r in gs_r])), 2),
            "pos_error_p95_m": round(float(np.percentile([r["pos_error_3d_m"] for r in gs_r], 95)), 2),
            "time_error_mean_ns": round(float(np.mean([r["time_error_ns"] for r in gs_r])), 2),
            "time_error_p95_ns": round(float(np.percentile([r["time_error_ns"] for r in gs_r], 95)), 2),
            "prop_delay_mean_ms": round(float(np.mean([r["propagation_delay_ms"] for r in gs_r])), 3),
        }
    return summary
