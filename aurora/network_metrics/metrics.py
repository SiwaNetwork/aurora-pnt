"""
Aggregate metric computations over per-step snapshots.
"""

import numpy as np


def compute_gsl_metrics(snapshots: list[dict], gs_names: list[str]) -> dict:
    """
    Per-GS: handover count, handover rate (events/hour), mean attachment duration (min).
    """
    results = {}
    total_hours = snapshots[-1]["time_h"] - snapshots[0]["time_h"] if len(snapshots) > 1 else 24.0
    step_min = (snapshots[1]["time_h"] - snapshots[0]["time_h"]) * 60 if len(snapshots) > 1 else 10.0

    for gs in gs_names:
        attachments = [s["gsl_attachment"].get(gs) for s in snapshots]
        handovers = sum(
            1 for i in range(1, len(attachments))
            if attachments[i] is not None
            and attachments[i - 1] is not None
            and attachments[i] != attachments[i - 1]
        )
        no_coverage_steps = sum(1 for a in attachments if a is None)
        coverage_pct = 100.0 * (1 - no_coverage_steps / len(attachments)) if attachments else 0.0

        # Mean attachment duration = total covered steps / (handovers + 1) * step_min
        covered_steps = len(attachments) - no_coverage_steps
        mean_attach_min = (covered_steps / max(handovers + 1, 1)) * step_min

        results[gs] = {
            "handover_count": handovers,
            "handover_rate_per_h": handovers / total_hours if total_hours > 0 else 0.0,
            "mean_attachment_duration_min": round(mean_attach_min, 1),
            "coverage_pct": round(coverage_pct, 1),
        }
    return results


def compute_isl_metrics(snapshots: list[dict]) -> dict:
    """
    Time-series stats for ISL topology: active link count, mean/max/min distance.
    """
    active_counts = [s["n_active_isls"] for s in snapshots]
    mean_dists = [s["isl_mean_dist_m"] / 1000.0 for s in snapshots]  # km

    all_edge_dists_km = []
    for s in snapshots:
        for _, _, d in s["isl_edges"]:
            all_edge_dists_km.append(d / 1000.0)

    return {
        "isl_count_mean": float(np.mean(active_counts)),
        "isl_count_min": int(np.min(active_counts)),
        "isl_count_max": int(np.max(active_counts)),
        "isl_dist_mean_km": float(np.mean(all_edge_dists_km)) if all_edge_dists_km else 0.0,
        "isl_dist_min_km": float(np.min(all_edge_dists_km)) if all_edge_dists_km else 0.0,
        "isl_dist_max_km": float(np.max(all_edge_dists_km)) if all_edge_dists_km else 0.0,
        "active_counts_series": active_counts,
        "mean_dists_km_series": mean_dists,
        "time_h_series": [s["time_h"] for s in snapshots],
    }


def compute_latency_metrics(snapshots: list[dict], gs_names: list[str]) -> dict:
    """
    Per GS-pair: mean/min/max one-way propagation latency (ms) and path length (km).
    Returns dict keyed by (gs_a, gs_b) tuples.
    """
    results = {}
    pairs = []
    for i, a in enumerate(gs_names):
        for j, b in enumerate(gs_names):
            if j > i:
                pairs.append((a, b))

    for pair in pairs:
        latencies = [s["latency_ms"].get(pair) for s in snapshots]
        valid = [x for x in latencies if x is not None]
        if valid:
            results[pair] = {
                "latency_mean_ms": round(float(np.mean(valid)), 2),
                "latency_min_ms": round(float(np.min(valid)), 2),
                "latency_max_ms": round(float(np.max(valid)), 2),
                "latency_std_ms": round(float(np.std(valid)), 3),
                "availability_pct": round(100.0 * len(valid) / len(latencies), 1),
                "series_ms": latencies,
            }
        else:
            results[pair] = {
                "latency_mean_ms": None,
                "latency_min_ms": None,
                "latency_max_ms": None,
                "latency_std_ms": None,
                "availability_pct": 0.0,
                "series_ms": latencies,
            }
    return results


def compute_path_stability(snapshots: list[dict], gs_names: list[str]) -> dict:
    """
    Hypatia-inspired: per GS-pair path lifetime and latency jitter for timing analysis.

    path_lifetime_steps  — mean consecutive steps a route stays unchanged
    latency_jitter_ms    — std of one-way latency (timing uncertainty from routing)
    hop_count_mean       — mean number of ISL hops (1 hop = direct sat link)
    path_change_rate_pct — % of steps where the route changed
    """
    results = {}
    pairs = [(a, b) for i, a in enumerate(gs_names) for j, b in enumerate(gs_names) if j > i]

    for pair in pairs:
        latencies = [s["latency_ms"].get(pair) for s in snapshots]
        paths = [s["path_km"].get(pair) for s in snapshots]

        # Path change detection: latency changes by > 1 ms means routing path changed
        changes = 0
        run_len = 1
        run_lens = []
        for i in range(1, len(latencies)):
            l_prev, l_curr = latencies[i - 1], latencies[i]
            if l_prev is None or l_curr is None:
                run_lens.append(run_len)
                run_len = 1
                continue
            if abs(l_curr - l_prev) > 1.0:
                changes += 1
                run_lens.append(run_len)
                run_len = 1
            else:
                run_len += 1
        run_lens.append(run_len)

        valid_lat = [x for x in latencies if x is not None]
        valid_km  = [x for x in paths if x is not None]

        # Estimate hop count: path_km / mean_ISL_dist_km (rough)
        isl_mean_km = float(np.mean([
            s["isl_mean_dist_m"] / 1000.0 for s in snapshots
            if s["isl_mean_dist_m"] > 0
        ])) if snapshots else 3000.0
        hop_mean = float(np.mean(valid_km)) / isl_mean_km if valid_km and isl_mean_km > 0 else None

        results[pair] = {
            "path_lifetime_mean_steps": round(float(np.mean(run_lens)), 1) if run_lens else None,
            "path_change_count": changes,
            "path_change_rate_pct": round(100.0 * changes / max(len(latencies) - 1, 1), 1),
            "latency_jitter_ms": round(float(np.std(valid_lat)), 3) if valid_lat else None,
            "hop_count_mean": round(hop_mean, 1) if hop_mean is not None else None,
            # timing sync error from jitter: jitter_ms -> ns (1-sigma path delay uncertainty)
            "timing_path_uncertainty_ns": round(float(np.std(valid_lat)) * 1e6, 1) if valid_lat else None,
        }
    return results


def compute_routing_stability(snapshots: list[dict], gs_names: list[str]) -> dict:
    """
    How often does GSL attachment change: fraction of steps with at least one handover.
    """
    steps_with_handover = 0
    for i in range(1, len(snapshots)):
        prev = snapshots[i - 1]["gsl_attachment"]
        curr = snapshots[i]["gsl_attachment"]
        changed = any(
            curr.get(g) != prev.get(g)
            for g in gs_names
            if curr.get(g) is not None and prev.get(g) is not None
        )
        if changed:
            steps_with_handover += 1
    return {
        "steps_with_any_handover": steps_with_handover,
        "steps_total": len(snapshots),
        "topology_change_rate_pct": round(
            100.0 * steps_with_handover / max(len(snapshots) - 1, 1), 1
        ),
    }
