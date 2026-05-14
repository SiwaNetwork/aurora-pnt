"""
Resilience analysis: satellite failure and ISL outage scenarios.

Simulates what happens when N% of satellites fail and recomputes
coverage/PDOP metrics to find the system's degradation curve.
"""

import math
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def run_resilience_sweep(
    base_summary: dict,
    failure_pcts: list[float] = None,
    output_dir: str = "results/resilience",
    label: str = "phase4",
) -> dict:
    """
    Estimate degradation of key PNT metrics vs satellite failure percentage.

    Uses an analytic approximation:
    - Visible sats scale linearly with (1 - failure_pct/100)
    - PDOP scales as ~ 1 / sqrt(n_visible)  (DOP geometry improvement rule)
    - Coverage scales non-linearly (drops sharply below 4-sat threshold)

    Returns per-failure-level metrics dict.
    """
    if failure_pcts is None:
        failure_pcts = [0, 5, 10, 15, 20, 25, 30, 40, 50]

    n_total   = base_summary.get("n_satellites", 300)
    pdop_base = base_summary.get("pdop_p95_mean") or 5.0
    cov_base  = base_summary.get("coverage_4sats_mean_pct", 100.0)
    n_sat_mean_base = base_summary.get("n_sats_mean_overall", 12.0)

    # Normalization: compute logistic at full capacity to scale correctly
    _k = 3.0
    _mid = 4.0 / max(n_sat_mean_base, 1.0)
    _cov_at_full = 1.0 / (1.0 + math.exp(_k * (_mid - 1.0 + 0.05)))

    results = []
    for pct in failure_pcts:
        fraction_surviving = 1.0 - pct / 100.0
        n_surviving = n_total * fraction_surviving
        n_vis = n_sat_mean_base * fraction_surviving

        # PDOP scales as 1/sqrt(n_vis) relative to baseline
        if n_vis >= 4 and n_sat_mean_base > 0:
            pdop_est = pdop_base * math.sqrt(n_sat_mean_base / n_vis)
        else:
            pdop_est = 999.0

        # Coverage: sigmoid normalized so 0% failure = cov_base
        cov_factor = 1.0 / (1.0 + math.exp(_k * (_mid - fraction_surviving + 0.05)))
        cov_norm = (cov_factor / _cov_at_full) * cov_base
        cov_est = min(cov_norm, cov_base)

        results.append({
            "failure_pct": pct,
            "n_surviving": round(n_surviving),
            "n_vis_mean": round(n_vis, 2),
            "pdop_p95_est": round(pdop_est, 2),
            "coverage_est_pct": round(cov_est, 2),
            "raim_detect_ok": n_vis >= 5,
            "raim_isolate_ok": n_vis >= 6,
            "operational": pdop_est <= 6.0 and cov_est >= 95.0,
        })

    os.makedirs(output_dir, exist_ok=True)
    _plot_resilience(results, n_total, output_dir, label)
    _save_resilience_csv(results, output_dir, label)

    # Find critical thresholds
    op_limit = next((r for r in results if not r["operational"]), None)
    raim_limit = next((r for r in results if not r["raim_detect_ok"]), None)

    return {
        "label": label,
        "n_satellites": n_total,
        "failure_levels": results,
        "operational_until_pct": op_limit["failure_pct"] if op_limit else max(failure_pcts),
        "raim_detect_until_pct": raim_limit["failure_pct"] if raim_limit else max(failure_pcts),
    }


def _plot_resilience(results: list[dict], n_total: int, output_dir: str, label: str) -> None:
    pcts    = [r["failure_pct"] for r in results]
    pdops   = [r["pdop_p95_est"] for r in results]
    covs    = [r["coverage_est_pct"] for r in results]
    n_vis   = [r["n_vis_mean"] for r in results]
    ops     = [r["operational"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Resilience Analysis — {label}  ({n_total} satellites baseline)",
                 fontweight="bold")

    # PDOP
    ax = axes[0]
    ax.plot(pcts, pdops, "o-", color="#1565C0", linewidth=2)
    ax.axhline(6.0, color="red", linestyle="--", linewidth=1.5, label="PDOP=6 limit")
    ax.fill_between(pcts, pdops, 6.0,
                    where=[p > 6 for p in pdops], alpha=0.2, color="red")
    ax.set_xlabel("Satellite failure (%)")
    ax.set_ylabel("PDOP p95 (estimated)")
    ax.set_title("PDOP Degradation")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Coverage
    ax = axes[1]
    ax.plot(pcts, covs, "o-", color="#2E7D32", linewidth=2)
    ax.axhline(95.0, color="red",    linestyle="--", linewidth=1.5, label="95% req")
    ax.axhline(99.0, color="orange", linestyle="--", linewidth=1,   label="99% target")
    ax.set_xlabel("Satellite failure (%)")
    ax.set_ylabel("4-sat coverage (%)")
    ax.set_title("Coverage Degradation")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    # Operational / RAIM status
    ax = axes[2]
    ax.bar(pcts, [r["n_vis_mean"] for r in results],
           color=["#43A047" if r["operational"] else "#E53935" for r in results],
           alpha=0.8, width=3.5)
    ax.axhline(4, color="grey",   linestyle="--", linewidth=1, label="4 sat (nav)")
    ax.axhline(5, color="orange", linestyle="--", linewidth=1, label="5 sat (RAIM detect)")
    ax.axhline(6, color="green",  linestyle="--", linewidth=1, label="6 sat (RAIM isolate)")
    ax.set_xlabel("Satellite failure (%)")
    ax.set_ylabel("Mean visible satellites")
    ax.set_title("Visibility & Operational Status\n(green=operational, red=degraded)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"resilience_{label}.png"), dpi=120, bbox_inches="tight")
    plt.close()


def _save_resilience_csv(results: list[dict], output_dir: str, label: str) -> None:
    fields = ["failure_pct", "n_surviving", "n_vis_mean", "pdop_p95_est",
              "coverage_est_pct", "raim_detect_ok", "raim_isolate_ok", "operational"]
    with open(os.path.join(output_dir, f"resilience_{label}.csv"), "w", encoding="utf-8") as f:
        f.write(",".join(fields) + "\n")
        for r in results:
            f.write(",".join(str(r.get(k, "")) for k in fields) + "\n")


def print_resilience_summary(label: str, analysis: dict) -> None:
    sep = "=" * 68
    print(f"\n{sep}")
    print(f"  Resilience Analysis — {label}  ({analysis['n_satellites']} satellites)")
    print(sep)
    print(f"  {'Failure':>8} {'N_surviv':>9} {'N_vis':>7} {'PDOP p95':>9} "
          f"{'Cov%':>7} {'RAIM':>6} {'Status':>10}")
    print("  " + "-" * 62)
    for r in analysis["failure_levels"]:
        raim = "detect" if r["raim_detect_ok"] else "NONE"
        if r["raim_isolate_ok"]:
            raim = "isolate"
        status = "OK" if r["operational"] else "DEGRADED"
        print(f"  {r['failure_pct']:>7.0f}%  {r['n_surviving']:>8}  {r['n_vis_mean']:>6.1f}  "
              f"{r['pdop_p95_est']:>8.2f}  {r['coverage_est_pct']:>6.1f}%  "
              f"{raim:>6}  {status:>10}")
    print(f"\n  System operational up to:  {analysis['operational_until_pct']:.0f}% failures")
    print(f"  RAIM detect up to:         {analysis['raim_detect_until_pct']:.0f}% failures")
    print(f"{sep}\n")
