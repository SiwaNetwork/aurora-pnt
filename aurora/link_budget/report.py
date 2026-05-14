"""
Link budget report: CSV, JSON summary, and plots.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_link_budget_report(
    snapshots: list[dict],
    gs_names: list[str],
    output_dir: str,
    label: str,
    freq_band: str = "L1",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    _save_csv(snapshots, output_dir, label)
    summary = _compute_summary(snapshots, gs_names)
    _save_json(summary, output_dir, label)
    _plot_cn0_by_elevation(snapshots, output_dir, label, freq_band)
    _plot_cn0_timeseries(snapshots, gs_names, output_dir, label)
    _plot_doppler_distribution(snapshots, output_dir, label, freq_band)
    _plot_link_margin_cdf(snapshots, gs_names, output_dir, label)


def _save_csv(snapshots, output_dir, label):
    path = os.path.join(output_dir, f"link_budget_{label}.csv")
    fields = ["time_h", "gs_name", "sat_id", "distance_km", "elevation_deg",
              "fspl_db", "atm_loss_db", "rx_power_dbw", "cn0_dbhz",
              "snr_db", "doppler_hz", "link_margin_db"]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(fields) + "\n")
        for s in snapshots:
            row = [str(s.get(k, "")) for k in fields]
            f.write(",".join(row) + "\n")


def _compute_summary(snapshots, gs_names):
    summary = {}
    for gs in gs_names:
        gs_snaps = [s for s in snapshots if s["gs_name"] == gs]
        if not gs_snaps:
            continue
        cn0s = [s["cn0_dbhz"] for s in gs_snaps]
        margins = [s["link_margin_db"] for s in gs_snaps]
        dops = [abs(s["doppler_hz"]) for s in gs_snaps]
        els = [s["elevation_deg"] for s in gs_snaps]
        summary[gs] = {
            "cn0_mean_dbhz": round(float(np.mean(cn0s)), 2),
            "cn0_min_dbhz": round(float(np.min(cn0s)), 2),
            "cn0_max_dbhz": round(float(np.max(cn0s)), 2),
            "link_margin_mean_db": round(float(np.mean(margins)), 2),
            "link_margin_min_db": round(float(np.min(margins)), 2),
            "pct_margin_positive": round(100.0 * sum(m > 0 for m in margins) / len(margins), 1),
            "doppler_max_hz": round(float(np.max(dops)), 1),
            "elevation_mean_deg": round(float(np.mean(els)), 1),
        }
    return summary


def _save_json(summary, output_dir, label):
    path = os.path.join(output_dir, f"link_budget_summary_{label}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def _plot_cn0_by_elevation(snapshots, output_dir, label, freq_band):
    els = np.array([s["elevation_deg"] for s in snapshots])
    cn0s = np.array([s["cn0_dbhz"] for s in snapshots])

    fig, ax = plt.subplots(figsize=(9, 5))
    sc = ax.scatter(els, cn0s, c=cn0s, cmap="RdYlGn", s=4, alpha=0.4, vmin=30, vmax=55)
    plt.colorbar(sc, ax=ax, label="C/N0 (dB·Hz)")
    ax.axhline(35.0, color="red", linestyle="--", linewidth=1.2, label="Min acquisition (35 dBHz)")
    ax.axhline(45.0, color="green", linestyle="--", linewidth=1.2, label="Tracking threshold (45 dBHz)")
    ax.set_xlabel("Elevation angle (deg)")
    ax.set_ylabel("C/N0 (dB·Hz)")
    ax.set_title(f"C/N0 vs Elevation — {label} ({freq_band} band)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"cn0_vs_elevation_{label}.png"), dpi=120, bbox_inches="tight")
    plt.close()


def _plot_cn0_timeseries(snapshots, gs_names, output_dir, label):
    fig, ax = plt.subplots(figsize=(13, 5))
    cmap = plt.cm.tab10

    for i, gs in enumerate(gs_names[:10]):
        gs_snaps = sorted([s for s in snapshots if s["gs_name"] == gs], key=lambda x: x["time_h"])
        if not gs_snaps:
            continue
        t = [s["time_h"] for s in gs_snaps]
        cn0 = [s["cn0_dbhz"] for s in gs_snaps]
        ax.plot(t, cn0, color=cmap(i), linewidth=1.2, label=gs, alpha=0.85)

    ax.axhline(35.0, color="red", linestyle="--", linewidth=1, alpha=0.6, label="Min 35 dBHz")
    ax.axhline(45.0, color="green", linestyle="--", linewidth=1, alpha=0.6, label="Track 45 dBHz")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("C/N0 (dB·Hz)")
    ax.set_title(f"C/N0 over 24h — {label}", fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"cn0_timeseries_{label}.png"), dpi=120, bbox_inches="tight")
    plt.close()


def _plot_doppler_distribution(snapshots, output_dir, label, freq_band):
    dops_khz = np.array([s["doppler_hz"] / 1000.0 for s in snapshots])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(dops_khz, bins=60, color="#2196F3", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Doppler shift (kHz)")
    ax.set_ylabel("Count")
    ax.set_title(f"Doppler Shift Distribution — {label} ({freq_band})", fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

    # Annotate max
    vmax = np.max(np.abs(dops_khz))
    ax.axvline(vmax, color="red", linestyle=":", linewidth=1, label=f"+{vmax:.1f} kHz max")
    ax.axvline(-vmax, color="red", linestyle=":", linewidth=1, label=f"-{vmax:.1f} kHz max")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"doppler_dist_{label}.png"), dpi=120, bbox_inches="tight")
    plt.close()


def _plot_link_margin_cdf(snapshots, gs_names, output_dir, label):
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.tab10

    for i, gs in enumerate(gs_names[:10]):
        margins = sorted([s["link_margin_db"] for s in snapshots if s["gs_name"] == gs])
        if not margins:
            continue
        p = np.arange(1, len(margins) + 1) / len(margins) * 100
        ax.plot(margins, p, color=cmap(i), linewidth=1.5, label=gs)

    ax.axvline(0, color="red", linestyle="--", linewidth=1.2, label="0 dB margin")
    ax.axhline(95, color="grey", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_xlabel("Link margin (dB)")
    ax.set_ylabel("CDF (%)")
    ax.set_title(f"Link Margin CDF — {label}", fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"link_margin_cdf_{label}.png"), dpi=120, bbox_inches="tight")
    plt.close()


def print_summary(label: str, summary: dict, freq_band: str) -> None:
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  Link Budget — {label} ({freq_band} band)")
    print(sep)
    print(f"  {'Station':<20} {'C/N0 mean':>10} {'C/N0 min':>9} {'Margin min':>11} {'Doppler max':>12}")
    print("  " + "-" * 60)
    for gs, m in summary.items():
        print(f"  {gs:<20} {m['cn0_mean_dbhz']:>8.1f} dBHz"
              f"  {m['cn0_min_dbhz']:>6.1f}"
              f"  {m['link_margin_min_db']:>8.1f} dB"
              f"  {m['doppler_max_hz']/1000:>8.1f} kHz")
    print(f"{sep}\n")
