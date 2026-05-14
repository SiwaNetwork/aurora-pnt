"""
RAIM (Receiver Autonomous Integrity Monitoring) analysis.

Computes HPL/VPL protection levels and RAIM availability from
per-timestep satellite visibility data.

Reference:
  - RTCA DO-229: WAAS MOPS
  - Parkinson & Spilker: GPS Theory and Applications, Vol.2, Ch.5
  - Walter & Enge: "Weighted RAIM for Precision Approach"
"""

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# P_FA (probability of false alert) = 1e-5 per hour for aviation PA
# k_FA for normal distribution: P(|z|>k) = P_FA -> k ~ 5.33
_K_FA = 5.33
# P_MD (probability of missed detection) for RAIM
_K_MD = 5.33


def _dop_from_n_sats(n_sats: int, pdop: float) -> dict:
    """Approximate HDOP and VDOP from PDOP and satellite count."""
    if n_sats < 4 or pdop is None or pdop <= 0:
        return {"hdop": None, "vdop": None, "tdop": None}
    # Walker constellation: typical HDOP/VDOP split
    vdop = pdop / math.sqrt(2)
    hdop = pdop / math.sqrt(2)
    tdop = pdop * 0.3
    return {"hdop": round(hdop, 3), "vdop": round(vdop, 3), "tdop": round(tdop, 3)}


def compute_protection_levels(
    n_sats: int,
    pdop: float,
    uere_m: float,
) -> dict:
    """
    Compute HPL and VPL using the RAIM slope method (simplified).

    HPL = k_FA * HDOP * UERE  (horizontal protection level)
    VPL = k_FA * VDOP * UERE  (vertical protection level)

    For RAIM fault detection to work: need n_sats >= 5 (detect) or >= 6 (isolate).
    """
    dops = _dop_from_n_sats(n_sats, pdop)
    if dops["hdop"] is None:
        return {
            "hpl_m": None, "vpl_m": None,
            "raim_detect": False, "raim_isolate": False,
            "n_sats": n_sats,
        }

    hpl = _K_FA * dops["hdop"] * uere_m
    vpl = _K_FA * dops["vdop"] * uere_m

    return {
        "hpl_m": round(hpl, 2),
        "vpl_m": round(vpl, 2),
        "hdop": dops["hdop"],
        "vdop": dops["vdop"],
        "raim_detect":  n_sats >= 5,
        "raim_isolate": n_sats >= 6,
        "n_sats": n_sats,
    }


def run_raim_analysis(
    timestep_data: list[dict],
    uere_m: float = 4.3,
    hpl_req_m: float = 40.0,
    vpl_req_m: float = 50.0,
) -> dict:
    """
    Compute per-timestep RAIM metrics from PNT simulation output.

    timestep_data: list of rows from timestep_metrics.csv
      Keys needed: time_h, n_sats_mean, pdop_p95 (or pdop_mean)

    Returns summary with availability, HPL/VPL distributions.
    """
    results = []
    for row in timestep_data:
        n_sats = float(row.get("n_sats_mean", 0))
        pdop   = float(row.get("pdop_p95", 0) or row.get("pdop_mean", 0) or 0)
        pl = compute_protection_levels(int(round(n_sats)), pdop, uere_m)
        pl["time_h"] = float(row.get("time_h", 0))
        pl["n_sats_raw"] = n_sats
        results.append(pl)

    # Availability metrics
    total = len(results)
    detect_steps   = sum(1 for r in results if r["raim_detect"])
    isolate_steps  = sum(1 for r in results if r["raim_isolate"])
    hpl_ok = [r for r in results if r["hpl_m"] is not None and r["hpl_m"] <= hpl_req_m]
    vpl_ok = [r for r in results if r["vpl_m"] is not None and r["vpl_m"] <= vpl_req_m]
    both_ok = [r for r in results
               if r["hpl_m"] is not None and r["vpl_m"] is not None
               and r["hpl_m"] <= hpl_req_m and r["vpl_m"] <= vpl_req_m]

    hpls = [r["hpl_m"] for r in results if r["hpl_m"] is not None]
    vpls = [r["vpl_m"] for r in results if r["vpl_m"] is not None]

    return {
        "n_timesteps": total,
        "uere_m": uere_m,
        "hpl_req_m": hpl_req_m,
        "vpl_req_m": vpl_req_m,
        "raim_detect_availability_pct":  round(100.0 * detect_steps  / max(total, 1), 2),
        "raim_isolate_availability_pct": round(100.0 * isolate_steps / max(total, 1), 2),
        "hpl_ok_pct":   round(100.0 * len(hpl_ok)  / max(total, 1), 2),
        "vpl_ok_pct":   round(100.0 * len(vpl_ok)  / max(total, 1), 2),
        "both_ok_pct":  round(100.0 * len(both_ok) / max(total, 1), 2),
        "hpl_mean_m":   round(float(np.mean(hpls)), 2) if hpls else None,
        "hpl_p95_m":    round(float(np.percentile(hpls, 95)), 2) if hpls else None,
        "vpl_mean_m":   round(float(np.mean(vpls)), 2) if vpls else None,
        "vpl_p95_m":    round(float(np.percentile(vpls, 95)), 2) if vpls else None,
        "series": results,
    }


def save_raim_report(analysis: dict, output_dir: str, label: str) -> None:
    import json
    os.makedirs(output_dir, exist_ok=True)

    # JSON summary (without series)
    summary = {k: v for k, v in analysis.items() if k != "series"}
    with open(os.path.join(output_dir, f"raim_summary_{label}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # CSV series
    series = analysis["series"]
    if series:
        fields = ["time_h", "n_sats_raw", "hpl_m", "vpl_m", "hdop", "vdop",
                  "raim_detect", "raim_isolate"]
        with open(os.path.join(output_dir, f"raim_series_{label}.csv"), "w", encoding="utf-8") as f:
            f.write(",".join(fields) + "\n")
            for r in series:
                f.write(",".join(str(r.get(k, "")) for k in fields) + "\n")

    _plot_raim(analysis, output_dir, label)


def _plot_raim(analysis: dict, output_dir: str, label: str) -> None:
    series = analysis["series"]
    times  = [r["time_h"] for r in series]
    hpls   = [r["hpl_m"]  for r in series]
    vpls   = [r["vpl_m"]  for r in series]
    n_sats = [r["n_sats_raw"] for r in series]

    fig, axes = plt.subplots(3, 1, figsize=(13, 10))
    fig.suptitle(f"RAIM Integrity Analysis — {label}  "
                 f"(UERE={analysis['uere_m']:.1f}m)", fontweight="bold")

    # HPL over time
    ax = axes[0]
    ax.plot(times, hpls, color="#1565C0", linewidth=1.2, label="HPL")
    ax.axhline(analysis["hpl_req_m"], color="red", linestyle="--",
               linewidth=1.5, label=f"HPL req {analysis['hpl_req_m']:.0f}m")
    ax.fill_between(times, hpls, analysis["hpl_req_m"],
                    where=[h > analysis["hpl_req_m"] for h in (hpls or [])],
                    alpha=0.2, color="red", label="HPL violation")
    ax.set_ylabel("HPL (m)")
    ax.set_title(f"Horizontal Protection Level — ok {analysis['hpl_ok_pct']:.1f}% of time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # VPL over time
    ax = axes[1]
    ax.plot(times, vpls, color="#2E7D32", linewidth=1.2, label="VPL")
    ax.axhline(analysis["vpl_req_m"], color="red", linestyle="--",
               linewidth=1.5, label=f"VPL req {analysis['vpl_req_m']:.0f}m")
    ax.set_ylabel("VPL (m)")
    ax.set_title(f"Vertical Protection Level — ok {analysis['vpl_ok_pct']:.1f}% of time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Satellites visible
    ax = axes[2]
    ax.fill_between(times, n_sats, alpha=0.6, color="#6A1B9A")
    ax.axhline(5, color="orange", linestyle="--", linewidth=1, label="5 sat (RAIM detect)")
    ax.axhline(6, color="green",  linestyle="--", linewidth=1, label="6 sat (RAIM isolate)")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Satellites visible (mean)")
    ax.set_title(f"Satellite Visibility — RAIM detect {analysis['raim_detect_availability_pct']:.1f}%, "
                 f"isolate {analysis['raim_isolate_availability_pct']:.1f}%")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"raim_{label}.png"), dpi=120, bbox_inches="tight")
    plt.close()


def print_raim_summary(label: str, analysis: dict) -> None:
    sep = "=" * 68
    print(f"\n{sep}")
    print(f"  RAIM Integrity Analysis — {label}")
    print(sep)
    print(f"  UERE assumed: {analysis['uere_m']:.1f} m  "
          f"(HPL req: {analysis['hpl_req_m']:.0f}m,  VPL req: {analysis['vpl_req_m']:.0f}m)")
    print()
    print(f"  RAIM detect availability:   {analysis['raim_detect_availability_pct']:>6.2f}%  (n>=5 sats)")
    print(f"  RAIM isolate availability:  {analysis['raim_isolate_availability_pct']:>6.2f}%  (n>=6 sats)")
    print()
    print(f"  HPL within {analysis['hpl_req_m']:.0f}m:   {analysis['hpl_ok_pct']:>6.2f}%   "
          f"(mean {analysis['hpl_mean_m']:.1f}m, p95 {analysis['hpl_p95_m']:.1f}m)")
    print(f"  VPL within {analysis['vpl_req_m']:.0f}m:   {analysis['vpl_ok_pct']:>6.2f}%   "
          f"(mean {analysis['vpl_mean_m']:.1f}m, p95 {analysis['vpl_p95_m']:.1f}m)")
    print(f"  Both HPL+VPL ok:            {analysis['both_ok_pct']:>6.2f}%")
    print(f"{sep}\n")
