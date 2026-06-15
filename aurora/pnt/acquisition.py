"""
Signal Acquisition and TTFF (Time-To-First-Fix) Analysis for АВРОРА.

LEO satellites produce large Doppler shifts (~41 kHz at L1) — 20x larger than GPS.
This requires wider frequency search, but LEO's stronger signal compensates.

Models: acquisition search space, dwell time, TTFF for cold/warm/hot start.
"""

import math
import os
import csv
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_LIGHT  = 299_792_458.0
MU_EARTH = 3.986004418e14
R_EARTH  = 6_371_000.0

# АВРОРА signal parameters
AURORA_L1 = {
    "freq_mhz":       1575.42,
    "chip_rate_mcps":  1.023,
    "code_length":     1023,     # chips per code period
    "code_period_ms":  1.0,      # 1 ms
    "cn0_min_dbhz":   25.0,      # minimum C/N0 for acquisition
    "cn0_open_dbhz":  42.7,      # typical open sky C/N0 (from Phase 3)
    "altitude_km":   1000.0,
    "color": "#00b894",
}
AURORA_L5 = {
    "freq_mhz":       1176.45,
    "chip_rate_mcps": 10.23,
    "code_length":   10230,
    "code_period_ms":  1.0,
    "cn0_min_dbhz":   25.0,
    "cn0_open_dbhz":  42.7,
    "altitude_km":   1000.0,
    "color": "#00cec9",
}
GPS_L1 = {
    "freq_mhz":       1575.42,
    "chip_rate_mcps":  1.023,
    "code_length":     1023,
    "code_period_ms":  1.0,
    "cn0_min_dbhz":   25.0,
    "cn0_open_dbhz":  35.0,      # GPS typical ~35 dBHz (LEO is +7 dBHz stronger)
    "altitude_km": 20200.0,
    "color": "#0984e3",
}

SIGNALS = {"AURORA_L1": AURORA_L1, "AURORA_L5": AURORA_L5, "GPS_L1": GPS_L1}


# ---------------------------------------------------------------------------
# Doppler model
# ---------------------------------------------------------------------------

def max_doppler_hz(freq_mhz: float, altitude_km: float) -> float:
    """Maximum Doppler shift (Hz) for LEO/MEO satellite."""
    r_orb = R_EARTH + altitude_km * 1000
    v_orb = math.sqrt(MU_EARTH / r_orb)
    return freq_mhz * 1e6 * v_orb / C_LIGHT


def doppler_rate_hz_per_s(freq_mhz: float, altitude_km: float) -> float:
    """Maximum Doppler rate of change near overhead pass (Hz/s)."""
    r_orb = R_EARTH + altitude_km * 1000
    v_orb = math.sqrt(MU_EARTH / r_orb)
    return freq_mhz * 1e6 * v_orb**2 / (C_LIGHT * r_orb)


def doppler_uncertainty_hz(
    freq_mhz: float, altitude_km: float,
    rx_clock_error_ppm: float = 1.0,  # typical TCXO: 1 ppm
) -> float:
    """Total Doppler uncertainty = orbital Doppler + local oscillator uncertainty."""
    orb_doppler = max_doppler_hz(freq_mhz, altitude_km)
    osc_error   = freq_mhz * 1e6 * rx_clock_error_ppm * 1e-6
    return orb_doppler + osc_error


# ---------------------------------------------------------------------------
# Acquisition search space
# ---------------------------------------------------------------------------

def acq_search_space(sig: Dict, rx_clock_error_ppm: float = 1.0) -> Dict:
    """Compute 2D acquisition search space (code phase x frequency)."""
    f_unc  = doppler_uncertainty_hz(sig["freq_mhz"], sig["altitude_km"], rx_clock_error_ppm)
    chips  = sig["code_length"]
    step_hz = sig["chip_rate_mcps"] * 1e3 / 2   # 1/2 chip bin width in Hz = chip_rate/code_len
    freq_bins = math.ceil(2 * f_unc / step_hz) + 1
    code_bins = chips * 2                          # half-chip steps

    total_cells = freq_bins * code_bins
    return {
        "freq_uncertainty_hz":  f_unc,
        "freq_bin_hz":          step_hz,
        "freq_bins":            freq_bins,
        "code_bins":            code_bins,
        "total_search_cells":   total_cells,
    }


# ---------------------------------------------------------------------------
# Dwell time and detection probability
# ---------------------------------------------------------------------------

def coherent_integration_limit_ms(doppler_rate_hz_s: float, freq_mhz: float) -> float:
    """
    Max coherent integration time before Doppler rate smears the signal.
    T_coh < sqrt(1 / doppler_rate) [seconds].
    """
    if doppler_rate_hz_s <= 0:
        return 20.0
    t_max_s = math.sqrt(1.0 / doppler_rate_hz_s)
    return min(t_max_s * 1000, 20.0)   # ms, cap at 20 ms


def detection_snr_db(cn0_dbhz: float, integration_ms: float) -> float:
    """Post-integration SNR in dB."""
    return cn0_dbhz + 10 * math.log10(integration_ms * 1e-3)


def ttff_cold_start_s(sig: Dict, rx_clock_ppm: float = 1.0,
                       parallel_channels: int = 32) -> Dict:
    """
    Estimate Time-To-First-Fix for cold start.
    Cold start: unknown position, clock, ephemeris, Doppler.
    """
    space  = acq_search_space(sig, rx_clock_ppm)
    dr_hz  = doppler_rate_hz_per_s(sig["freq_mhz"], sig["altitude_km"])
    t_coh  = coherent_integration_limit_ms(dr_hz, sig["freq_mhz"])

    # Number of non-coherent integrations for detection at min C/N0
    snr_per_coh = detection_snr_db(sig["cn0_open_dbhz"], t_coh)
    snr_needed  = 14.0   # dB, for Pd=0.99, Pfa=1e-3 with 2D search
    n_nc        = max(1, math.ceil((snr_needed - snr_per_coh) / 3.0))   # 3 dB per NC integ

    dwell_ms = t_coh * n_nc   # ms per cell

    # Total search time
    cells_per_channel = space["total_search_cells"] / parallel_channels
    acq_time_s = cells_per_channel * dwell_ms * 1e-3

    # Add time for ephem decode and fix
    ephem_decode_s = 18.0    # АВРОРА-T broadcasts every 10 s, need 1-2 frames
    fix_s          = 1.0     # after acquisition, fix is fast

    ttff = acq_time_s + ephem_decode_s + fix_s
    return {
        "freq_uncertainty_hz":  space["freq_uncertainty_hz"],
        "total_search_cells":   space["total_search_cells"],
        "coh_integration_ms":   t_coh,
        "nc_integrations":      n_nc,
        "dwell_ms":             dwell_ms,
        "parallel_channels":    parallel_channels,
        "acq_time_s":           acq_time_s,
        "ephem_decode_s":       ephem_decode_s,
        "ttff_cold_s":          ttff,
        "ttff_warm_s":          ephem_decode_s + fix_s + 2.0,   # known position
        "ttff_hot_s":           fix_s + 0.5,                    # all known
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_acquisition_analysis(output_dir: str, label: str,
                              rx_clock_ppm: float = 1.0,
                              parallel_ch: int = 32) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    for sig_name, sig in SIGNALS.items():
        dp = {
            "max_doppler_hz":     max_doppler_hz(sig["freq_mhz"], sig["altitude_km"]),
            "doppler_rate_hz_s":  doppler_rate_hz_per_s(sig["freq_mhz"], sig["altitude_km"]),
        }
        space = acq_search_space(sig, rx_clock_ppm)
        ttff  = ttff_cold_start_s(sig, rx_clock_ppm, parallel_ch)
        results[sig_name] = {**dp, "search": space, "ttff": ttff, "sig": sig}

    _plot_doppler_comparison(results, output_dir, label)
    _plot_ttff_comparison(results, output_dir, label)
    _save_acq_csv(results, output_dir, label)

    return {
        "results":        results,
        "rx_clock_ppm":   rx_clock_ppm,
        "parallel_ch":    parallel_ch,
    }


def _plot_doppler_comparison(results: Dict, output_dir: str, label: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    names   = list(results.keys())
    dopplers = [results[n]["max_doppler_hz"] / 1000 for n in names]   # kHz
    dr      = [results[n]["doppler_rate_hz_s"] for n in names]
    colors  = [SIGNALS[n]["color"] for n in names]

    ax = axes[0]
    bars = ax.bar(names, dopplers, color=colors, edgecolor="white", width=0.5)
    ax.bar_label(bars, fmt="%.1f кГц", padding=3, fontsize=9)
    ax.set_ylabel("Макс. доплеровский сдвиг (кГц)")
    ax.set_title("Макс. доплеровский сдвиг")
    ax.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    bars2 = ax2.bar(names, dr, color=colors, edgecolor="white", width=0.5)
    ax2.bar_label(bars2, fmt="%.1f Гц/с", padding=3, fontsize=9)
    ax2.set_ylabel("Скорость изменения доплеровского сдвига (Гц/с)")
    ax2.set_title("Макс. скорость изменения доплеровского сдвига")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(f"АВРОРА и GPS: доплеровский сдвиг [{label}]", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"acquisition_doppler_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_ttff_comparison(results: Dict, output_dir: str, label: str) -> None:
    names  = list(results.keys())
    cold   = [results[n]["ttff"]["ttff_cold_s"]  for n in names]
    warm   = [results[n]["ttff"]["ttff_warm_s"]  for n in names]
    hot    = [results[n]["ttff"]["ttff_hot_s"]   for n in names]
    colors = [SIGNALS[n]["color"] for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    w = 0.25
    b1 = ax.bar(x - w,   cold, w, label="Холодный старт", color="#e17055", edgecolor="white")
    b2 = ax.bar(x,       warm, w, label="Тёплый старт", color="#fdcb6e", edgecolor="white")
    b3 = ax.bar(x + w,   hot,  w, label="Горячий старт",  color="#00b894", edgecolor="white")
    ax.bar_label(b1, fmt="%.0f с", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.0f с", padding=2, fontsize=8)
    ax.bar_label(b3, fmt="%.1f с", padding=2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("TTFF (с)")
    ax.set_title(f"Время до первого решения (TTFF) [{label}]")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"acquisition_ttff_{label}.png"), dpi=150)
    plt.close(fig)


def _save_acq_csv(results: Dict, output_dir: str, label: str) -> None:
    path = os.path.join(output_dir, f"acquisition_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["signal", "max_doppler_khz", "doppler_rate_hz_s",
                    "freq_bins", "code_bins", "total_cells",
                    "coh_integration_ms", "ttff_cold_s", "ttff_warm_s", "ttff_hot_s"])
        for name, r in results.items():
            w.writerow([name,
                        f"{r['max_doppler_hz']/1000:.1f}",
                        f"{r['doppler_rate_hz_s']:.2f}",
                        r["search"]["freq_bins"],
                        r["search"]["code_bins"],
                        r["search"]["total_search_cells"],
                        f"{r['ttff']['coh_integration_ms']:.2f}",
                        f"{r['ttff']['ttff_cold_s']:.1f}",
                        f"{r['ttff']['ttff_warm_s']:.1f}",
                        f"{r['ttff']['ttff_hot_s']:.1f}"])


def print_acquisition_summary(label: str, result: Dict) -> None:
    sep = "=" * 72

    print(f"\n{sep}")
    print(f"  Поиск сигнала и TTFF -- {label}")
    print(f"  Часы приёмника: {result['rx_clock_ppm']} ppm  "
          f"Параллельных каналов: {result['parallel_ch']}")
    print(sep)

    for sig_name, r in result["results"].items():
        t = r["ttff"]
        s = r["search"]
        print(f"  [{sig_name}]")
        print(f"    Макс. доплер:      {r['max_doppler_hz']/1000:>8.1f} кГц")
        print(f"    Скорость доплера:  {r['doppler_rate_hz_s']:>8.1f} Гц/с")
        print(f"    Частотных ячеек:   {s['freq_bins']:>8}")
        print(f"    Кодовых ячеек:     {s['code_bins']:>8}")
        print(f"    Всего ячеек поиска:{s['total_search_cells']:>8,}")
        print(f"    Когер. интеграция: {t['coh_integration_ms']:>8.2f} мс")
        print(f"    На ячейку:         {t['dwell_ms']:>8.2f} мс")
        print(f"    TTFF холодный старт:{t['ttff_cold_s']:>7.1f} с")
        print(f"    TTFF тёплый старт: {t['ttff_warm_s']:>8.1f} с")
        print(f"    TTFF горячий старт:{t['ttff_hot_s']:>8.2f} с")
        print()
    print(sep)
