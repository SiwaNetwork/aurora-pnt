"""
ISL Link Budget for АВРОРА.

RF link budget for inter-satellite communication links.
АВРОРА uses Ka-band (26 GHz) ISL for ranging and data exchange.
Models TX power, antenna gain, FSPL, and required margin for reliable operation.
"""

import math
import os
import csv
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_LIGHT  = 299_792_458.0   # m/s
K_BOLTZ  = 1.380649e-23    # J/K

# ISL frequency options
ISL_BANDS = {
    "Ka": {
        "freq_ghz":       26.0,
        "atm_loss_db":     0.1,   # near-vacuum, very small
        "rain_fade_db":    0.0,   # no rain in space
        "pointing_loss_db": 0.5,
        "desc": "Ka-band 26 GHz (primary, all-weather)",
    },
    "V": {
        "freq_ghz":       60.0,
        "atm_loss_db":     0.1,
        "rain_fade_db":    0.0,
        "pointing_loss_db": 0.8,
        "desc": "V-band 60 GHz (oxygen absorption in atmosphere only)",
    },
    "Optical": {
        "freq_ghz":  193_400.0,   # 1550 nm
        "atm_loss_db":     0.0,
        "rain_fade_db":    0.0,
        "pointing_loss_db": 1.5,  # tighter beam, harder pointing
        "desc": "Optical 1550 nm (highest throughput, clear sky only)",
    },
}

# АВРОРА ISL hardware parameters (per satellite)
ISL_TX = {
    "tx_power_dbw":  10.0,    # 10 W = 10 dBW
    "tx_gain_dbi":   35.0,    # 30 cm aperture antenna at Ka-band
    "rx_gain_dbi":   35.0,
    "noise_temp_k": 290.0,
    "data_rate_mbps": 100.0,
    "req_eb_n0_db":   8.0,    # QPSK, BER 1e-6
    "req_margin_db":  3.0,    # link margin requirement
}

# АВРОРА Phase 3 ISL distances (from isl_ranging module)
AURORA_ISL_RANGES = {
    "in_plane_km":     3068.0,
    "cross_plane_km":  3820.0,
    "horizon_km":      3709.0,
}


def fspl_db(range_m: float, freq_hz: float) -> float:
    return 20 * math.log10(4 * math.pi * range_m * freq_hz / C_LIGHT)


def noise_power_dbw(noise_temp_k: float, bw_hz: float) -> float:
    return 10 * math.log10(K_BOLTZ * noise_temp_k * bw_hz)


def link_budget(
    range_m:         float,
    freq_ghz:        float,
    tx_power_dbw:    float,
    tx_gain_dbi:     float,
    rx_gain_dbi:     float,
    noise_temp_k:    float,
    data_rate_bps:   float,
    atm_loss_db:     float = 0.1,
    pointing_loss_db: float = 0.5,
) -> Dict:
    freq_hz = freq_ghz * 1e9
    pl      = fspl_db(range_m, freq_hz)
    eirp    = tx_power_dbw + tx_gain_dbi
    rx_pwr  = eirp + rx_gain_dbi - pl - atm_loss_db - pointing_loss_db
    n0      = 10 * math.log10(K_BOLTZ * noise_temp_k)          # dBW/Hz
    eb_n0   = rx_pwr - n0 - 10 * math.log10(data_rate_bps)
    cn0     = rx_pwr - n0                                       # dBHz
    req_eb  = ISL_TX["req_eb_n0_db"]
    margin  = eb_n0 - req_eb

    return {
        "range_km":       range_m / 1000,
        "freq_ghz":       freq_ghz,
        "eirp_dbw":       eirp,
        "fspl_db":        pl,
        "rx_power_dbw":   rx_pwr,
        "cn0_dbhz":       cn0,
        "eb_n0_db":       eb_n0,
        "link_margin_db": margin,
        "link_ok":        margin >= ISL_TX["req_margin_db"],
    }


def max_range_km(
    freq_ghz:        float,
    tx_power_dbw:    float,
    tx_gain_dbi:     float,
    rx_gain_dbi:     float,
    noise_temp_k:    float,
    data_rate_bps:   float,
    atm_loss_db:     float = 0.1,
    pointing_loss_db: float = 0.5,
    margin_req_db:   float = 3.0,
    req_eb_n0_db:    float = 8.0,
) -> float:
    """Binary search for maximum range with required margin."""
    freq_hz = freq_ghz * 1e9
    eirp = tx_power_dbw + tx_gain_dbi
    n0   = 10 * math.log10(K_BOLTZ * noise_temp_k)
    req_total = req_eb_n0_db + margin_req_db + 10 * math.log10(data_rate_bps) + n0
    max_rx_pwr = eirp + rx_gain_dbi - atm_loss_db - pointing_loss_db
    max_pl = max_rx_pwr - req_total
    # pl = 20*log10(4*pi*r*f/c) => r = c/(4*pi*f) * 10^(pl/20)
    r_m = (C_LIGHT / (4 * math.pi * freq_hz)) * 10**(max_pl / 20)
    return r_m / 1000


def run_isl_link_budget_analysis(
    output_dir: str,
    label: str,
    n_planes: int = 12,
    n_sats_per_plane: int = 15,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    range_sweep_km = list(range(500, 6001, 100))

    for band_name, band in ISL_BANDS.items():
        freq  = band["freq_ghz"]
        atm   = band["atm_loss_db"]
        ptl   = band["pointing_loss_db"]
        dr    = ISL_TX["data_rate_mbps"] * 1e6

        # Link budgets at key АВРОРА ranges
        budgets = {}
        for link_name, dist_km in AURORA_ISL_RANGES.items():
            budgets[link_name] = link_budget(
                dist_km * 1000, freq,
                ISL_TX["tx_power_dbw"], ISL_TX["tx_gain_dbi"],
                ISL_TX["rx_gain_dbi"],  ISL_TX["noise_temp_k"],
                dr, atm, ptl,
            )

        # Max range
        r_max = max_range_km(
            freq, ISL_TX["tx_power_dbw"], ISL_TX["tx_gain_dbi"],
            ISL_TX["rx_gain_dbi"], ISL_TX["noise_temp_k"], dr, atm, ptl,
        )

        # Margin vs range sweep
        margins = []
        for r_km in range_sweep_km:
            b = link_budget(r_km * 1000, freq,
                            ISL_TX["tx_power_dbw"], ISL_TX["tx_gain_dbi"],
                            ISL_TX["rx_gain_dbi"], ISL_TX["noise_temp_k"],
                            dr, atm, ptl)
            margins.append(b["link_margin_db"])

        results[band_name] = {
            "band":         band,
            "budgets":      budgets,
            "max_range_km": r_max,
            "range_sweep_km": range_sweep_km,
            "margins":      margins,
        }

    _plot_margin_vs_range(results, output_dir, label)
    _plot_budget_summary(results, output_dir, label)
    _save_budget_csv(results, output_dir, label)

    return {
        "results":       results,
        "isl_ranges_km": AURORA_ISL_RANGES,
        "tx_params":     ISL_TX,
        "n_planes":      n_planes,
        "n_sats":        n_planes * n_sats_per_plane,
    }


def _plot_margin_vs_range(results: Dict, output_dir: str, label: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = {"Ka": "#00b894", "V": "#0984e3", "Optical": "#a855f7"}

    for band_name, r in results.items():
        ax.plot(r["range_sweep_km"], r["margins"],
                color=colors.get(band_name, "gray"),
                lw=2, label=f"{band_name} ({r['band']['freq_ghz']:.0f} GHz)")

    ax.axhline(ISL_TX["req_margin_db"], ls="--", color="#e17055", lw=1.2,
               label=f"Required margin ({ISL_TX['req_margin_db']} dB)")
    ax.axhline(0, ls="-", color="black", lw=0.5)

    # Mark АВРОРА ISL ranges
    for rng_name, rng_km in AURORA_ISL_RANGES.items():
        ax.axvline(rng_km, ls=":", color="#636e72", lw=0.8, alpha=0.7)
        ax.text(rng_km, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 10,
                rng_name.replace("_", " "), fontsize=7, color="#636e72", ha="center")

    ax.set_xlabel("ISL range (km)")
    ax.set_ylabel("Link margin (dB)")
    ax.set_title(f"АВРОРА ISL Link Margin vs Range [{label}]")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(min(r["range_sweep_km"]), max(r["range_sweep_km"]))
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"isl_margin_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_budget_summary(results: Dict, output_dir: str, label: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"Ka": "#00b894", "V": "#0984e3", "Optical": "#a855f7"}

    # Left: margin at key АВРОРА ranges
    ax = axes[0]
    x     = np.arange(len(AURORA_ISL_RANGES))
    width = 0.25
    for i, (band_name, r) in enumerate(results.items()):
        margins = [r["budgets"][rn]["link_margin_db"] for rn in AURORA_ISL_RANGES]
        bars = ax.bar(x + i * width, margins, width,
                      label=band_name, color=colors.get(band_name, "gray"),
                      edgecolor="white")
        ax.bar_label(bars, fmt="%.1f dB", padding=2, fontsize=8)

    ax.axhline(ISL_TX["req_margin_db"], ls="--", color="#e17055", lw=1.2)
    ax.set_xticks(x + width)
    ax.set_xticklabels([k.replace("_km", "").replace("_", " ")
                        for k in AURORA_ISL_RANGES], fontsize=9)
    ax.set_ylabel("Link margin (dB)")
    ax.set_title("Margin at АВРОРА ISL ranges")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Right: max range comparison
    ax2 = axes[1]
    bands = list(results.keys())
    maxr  = [results[b]["max_range_km"] for b in bands]
    bar_colors = [colors.get(b, "gray") for b in bands]
    bars2 = ax2.bar(bands, maxr, color=bar_colors, edgecolor="white", width=0.5)
    ax2.bar_label(bars2, fmt="%.0f km", padding=3, fontsize=10)
    ax2.axhline(AURORA_ISL_RANGES["cross_plane_km"], ls="--",
                color="#e17055", lw=1.2, label="Max ISL range needed")
    ax2.set_ylabel("Max range with 3 dB margin (km)")
    ax2.set_title("Max ISL Range by Band")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(f"АВРОРА ISL Link Budget [{label}]", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"isl_budget_{label}.png"), dpi=150)
    plt.close(fig)


def _save_budget_csv(results: Dict, output_dir: str, label: str) -> None:
    path = os.path.join(output_dir, f"isl_budget_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["band", "link_type", "range_km", "fspl_db",
                    "rx_power_dbw", "cn0_dbhz", "eb_n0_db", "margin_db", "link_ok"])
        for band_name, r in results.items():
            for link_name, b in r["budgets"].items():
                w.writerow([band_name, link_name,
                            f"{b['range_km']:.1f}", f"{b['fspl_db']:.1f}",
                            f"{b['rx_power_dbw']:.2f}", f"{b['cn0_dbhz']:.1f}",
                            f"{b['eb_n0_db']:.1f}", f"{b['link_margin_db']:.1f}",
                            b["link_ok"]])


def print_isl_link_summary(label: str, result: Dict) -> None:
    sep = "=" * 72

    print(f"\n{sep}")
    print(f"  ISL Link Budget -- {label}")
    print(sep)
    tx = result["tx_params"]
    print(f"  TX power: {tx['tx_power_dbw']:.0f} dBW  "
          f"TX gain: {tx['tx_gain_dbi']:.0f} dBi  "
          f"RX gain: {tx['rx_gain_dbi']:.0f} dBi")
    print(f"  Data rate: {tx['data_rate_mbps']:.0f} Mbps  "
          f"Required Eb/N0: {tx['req_eb_n0_db']:.0f} dB  "
          f"Required margin: {tx['req_margin_db']:.0f} dB")
    print()

    for band_name, r in result["results"].items():
        print(f"  [{band_name}] {r['band']['desc']}")
        print(f"  {'Link type':<20} {'Range km':>10} {'FSPL dB':>9} "
              f"{'Eb/N0 dB':>10} {'Margin dB':>10} {'Status':>8}")
        print("  " + "-" * 70)
        for link_name, b in r["budgets"].items():
            status = "[OK]" if b["link_ok"] else "[FAIL]"
            print(f"  {link_name:<20} {b['range_km']:>10.0f} {b['fspl_db']:>9.1f} "
                  f"{b['eb_n0_db']:>10.1f} {b['link_margin_db']:>10.1f} {status:>8}")
        print(f"  Max range (3 dB margin): {r['max_range_km']:.0f} km")
        need = AURORA_ISL_RANGES["cross_plane_km"]
        ok = r["max_range_km"] >= need
        print(f"  Cross-plane {need:.0f} km: {'[FEASIBLE]' if ok else '[INSUFFICIENT]'}")
        print()
    print(sep)
