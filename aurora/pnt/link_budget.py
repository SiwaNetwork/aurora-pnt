"""
User Terminal Link Budget for АВРОРА.

Computes received signal power, C/N0, and navigation accuracy at the user
receiver for АВРОРА L1 (1575.42 MHz) and L5 (1176.45 MHz) signals.

Reference: IS-GPS-200, Galileo OS SIS ICD, ITU-R P.618, GNSS textbook
(Kaplan & Hegarty, 3rd ed.).
"""

import math
import os
import csv
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Physical constants
C_LIGHT   = 299_792_458.0   # m/s
K_BOLTZ   = 1.380649e-23    # J/K
R_EARTH_M = 6_371_000.0     # m

# ── АВРОРА signal parameters ──────────────────────────────────────────────────

AURORA_SIGNALS = {
    "L1": {
        "freq_mhz":   1575.42,
        "bw_mhz":     24.0,
        "chip_rate_mcps": 10.23,    # same as GPS L1C
        "modulation": "BPSK(10)",
        "tx_power_dbw": 13.0,       # 20 W EIRP-equivalent (sat TX)
        "tx_gain_dbi":  13.0,       # nadir-facing helix/patch array
        "polarization": "RHCP",
        "desc": "L1 GPS/Galileo compatible",
    },
    "L5": {
        "freq_mhz":   1176.45,
        "bw_mhz":     24.0,
        "chip_rate_mcps": 10.23,
        "modulation": "BPSK(10)",
        "tx_power_dbw": 13.0,
        "tx_gain_dbi":  13.0,
        "polarization": "RHCP",
        "desc": "L5 safety-of-life compatible",
    },
}

# АВРОРА satellite TX parameters (150 kg class)
AURORA_TX = {
    "tx_power_w":      20.0,        # 20 W RF power per signal
    "tx_losses_db":     1.5,        # feed + filter + diplexer losses
    "antenna_gain_dbi": 13.0,       # nadir patch array (3 dB beamwidth ~±65°)
    "pointing_loss_db":  0.5,       # max body-pointing error
}

# User terminal parameters
USER_TERMINALS = {
    "Handheld": {
        "antenna_gain_dbi": -3.0,   # small patch, RHCP
        "noise_fig_db":      3.5,   # typical LNA+frontend
        "rx_losses_db":      1.0,   # cable + connector
        "desc": "Smartphone / personal tracker",
    },
    "Survey": {
        "antenna_gain_dbi":  4.0,   # choke-ring or GNSS survey antenna
        "noise_fig_db":      1.5,   # low-noise receiver front-end
        "rx_losses_db":      0.5,
        "desc": "Geodetic survey receiver",
    },
    "Maritime": {
        "antenna_gain_dbi":  3.0,   # marine dome antenna
        "noise_fig_db":      2.5,
        "rx_losses_db":      1.0,
        "desc": "Ship / vessel navigation",
    },
    "Aviation": {
        "antenna_gain_dbi":  0.0,   # top-mounted blade antenna
        "noise_fig_db":      2.0,
        "rx_losses_db":      1.5,
        "desc": "Aircraft GNSS receiver (IFR)",
    },
}

# Atmospheric loss model (ITU-R P.618-14, zenith values)
ATMO = {
    "L1": {"troposphere_db": 0.5, "ionosphere_db": 1.0, "rain_1pct_db": 0.2},
    "L5": {"troposphere_db": 0.5, "ionosphere_db": 2.5, "rain_1pct_db": 0.1},
}

# Minimum required C/N0 for tracking / navigation
CN0_MIN_DB_HZ = {
    "tracking":    25.0,   # weak signal tracking threshold
    "navigation":  30.0,   # reliable PVT solution
    "rtk":         35.0,   # real-time kinematic positioning
}

# Altitude and min-elevation for АВРОРА LEO satellites
AURORA_ALT_M  = 1_000_000.0   # 1000 km
MIN_ELEV_DEG  = 10.0           # minimum elevation angle for service


def fspl_db(distance_m: float, freq_hz: float) -> float:
    """Free-space path loss (dB) at given distance and frequency."""
    return 20 * math.log10(4 * math.pi * distance_m * freq_hz / C_LIGHT)


def slant_range_m(altitude_m: float, elevation_deg: float) -> float:
    """
    Slant range from user to satellite at given elevation angle.
    Uses Earth-satellite geometry (law of cosines on sphere).
    """
    el_rad = math.radians(elevation_deg)
    r_sat  = R_EARTH_M + altitude_m
    # Nadir angle from satellite
    sin_nadir = (R_EARTH_M / r_sat) * math.cos(el_rad)
    nadir_rad = math.asin(sin_nadir)
    # Earth central angle
    central_rad = math.pi / 2 - el_rad - nadir_rad
    # Slant range via law of sines
    return R_EARTH_M * math.sin(central_rad) / math.cos(el_rad + nadir_rad - math.pi/2 + math.pi/2)


def slant_range_simple_m(altitude_m: float, elevation_deg: float) -> float:
    """Simplified slant range (flat Earth approx, good above 20°)."""
    el_rad = math.radians(elevation_deg)
    return altitude_m / math.sin(el_rad)


def noise_temp_k(noise_fig_db: float, t_sky_k: float = 150.0) -> float:
    """System noise temperature (K) from receiver noise figure + sky noise."""
    t_rx = 290.0 * (10 ** (noise_fig_db / 10) - 1)
    return t_rx + t_sky_k   # sky noise ~150 K for GNSS receivers


def compute_link_budget(
    signal_key: str,
    terminal_key: str,
    elevation_deg: float,
    altitude_m: float = AURORA_ALT_M,
) -> Dict:
    """
    Compute full one-way RF link budget (satellite → user).

    Returns dict with all budget components and C/N0 in dB-Hz.
    """
    sig  = AURORA_SIGNALS[signal_key]
    term = USER_TERMINALS[terminal_key]
    atmo = ATMO[signal_key]

    freq_hz = sig["freq_mhz"] * 1e6

    # Transmitter EIRP
    tx_power_dbw = 10 * math.log10(AURORA_TX["tx_power_w"])
    eirp_dbw     = (tx_power_dbw
                    - AURORA_TX["tx_losses_db"]
                    + AURORA_TX["antenna_gain_dbi"]
                    - AURORA_TX["pointing_loss_db"])

    # Path loss
    dist_m    = slant_range_simple_m(altitude_m, elevation_deg)
    fspl      = fspl_db(dist_m, freq_hz)
    atmo_loss = atmo["troposphere_db"] + atmo["ionosphere_db"]

    # Polarization mismatch loss (circular → circular = 0 dB, → linear = 3 dB)
    pol_loss_db = 0.0   # assume RHCP terminal

    # Received signal power at antenna port (dBW)
    rx_power_dbw = (eirp_dbw
                    - fspl
                    - atmo_loss
                    - pol_loss_db
                    + term["antenna_gain_dbi"]
                    - term["rx_losses_db"])

    # Noise
    t_sys    = noise_temp_k(term["noise_fig_db"])
    n0_dbw_hz = 10 * math.log10(K_BOLTZ * t_sys)   # dBW/Hz

    # C/N0
    cn0_db_hz = rx_power_dbw - n0_dbw_hz

    # Signal-to-noise over noise BW (carrier tracking loop ~10 Hz BW)
    snr_1hz_db = cn0_db_hz    # C/N0 already in 1 Hz bandwidth

    # Pseudorange noise (thermal): sigma_PR = lambda_chip / (2 * sqrt(2) * sqrt(T * C/N0))
    # Simplified: sigma_PR_m = chip_length_m / (2 * sqrt(cn0_linear * coherent_T))
    chip_len_m  = C_LIGHT / (sig["chip_rate_mcps"] * 1e6)
    cn0_linear  = 10 ** (cn0_db_hz / 10)
    coh_int_s   = 0.001   # 1 ms coherent integration
    sigma_pr_m  = chip_len_m / (2 * math.sqrt(2 * cn0_linear * coh_int_s))

    margin_tracking   = cn0_db_hz - CN0_MIN_DB_HZ["tracking"]
    margin_navigation = cn0_db_hz - CN0_MIN_DB_HZ["navigation"]

    return {
        "signal":           signal_key,
        "terminal":         terminal_key,
        "elevation_deg":    elevation_deg,
        "altitude_m":       altitude_m,
        "slant_range_km":   dist_m / 1000,
        "freq_mhz":         sig["freq_mhz"],
        # TX side
        "tx_power_dbw":     tx_power_dbw,
        "tx_losses_db":     AURORA_TX["tx_losses_db"],
        "tx_gain_dbi":      AURORA_TX["antenna_gain_dbi"],
        "eirp_dbw":         eirp_dbw,
        # Path
        "fspl_db":          fspl,
        "atmo_loss_db":     atmo_loss,
        "total_path_loss_db": fspl + atmo_loss,
        # RX side
        "rx_gain_dbi":      term["antenna_gain_dbi"],
        "rx_losses_db":     term["rx_losses_db"],
        "rx_power_dbw":     rx_power_dbw,
        "rx_power_dbm":     rx_power_dbw + 30,
        "t_sys_k":          t_sys,
        "n0_dbw_hz":        n0_dbw_hz,
        "cn0_db_hz":        cn0_db_hz,
        "sigma_pr_m":       sigma_pr_m,
        "chip_len_m":       chip_len_m,
        "margin_tracking_db":   margin_tracking,
        "margin_navigation_db": margin_navigation,
        "viable":           margin_navigation >= 0.0,
    }


def run_link_budget_analysis(
    output_dir: str,
    label: str,
    altitude_m: float = AURORA_ALT_M,
) -> Dict:
    """Run full link budget sweep and generate plots + CSV."""
    os.makedirs(output_dir, exist_ok=True)

    elevations = list(range(10, 91, 5))   # 10° to 90° in 5° steps
    signals    = list(AURORA_SIGNALS.keys())
    terminals  = list(USER_TERMINALS.keys())

    # ── Full result matrix ─────────────────────────────────────────────────────
    results = {}
    for sig in signals:
        for term in terminals:
            key = f"{sig}/{term}"
            results[key] = [
                compute_link_budget(sig, term, el, altitude_m)
                for el in elevations
            ]

    # ── Plots ──────────────────────────────────────────────────────────────────
    _plot_cn0_vs_elevation(results, elevations, signals, terminals, output_dir, label)
    _plot_sigma_pr(results, elevations, signals, terminals, output_dir, label)
    _plot_link_budget_waterfall(signals, terminals, altitude_m, output_dir, label)

    # ── CSV ───────────────────────────────────────────────────────────────────
    _save_link_budget_csv(results, output_dir, label)

    # ── Summary stats at key elevations (10°, 30°, 90°) ──────────────────────
    summary = {}
    for sig in signals:
        summary[sig] = {}
        for term in terminals:
            key = f"{sig}/{term}"
            pts = {r["elevation_deg"]: r for r in results[key]}
            summary[sig][term] = {el: pts[el] for el in [10, 30, 45, 90] if el in pts}

    return {
        "signals":    signals,
        "terminals":  terminals,
        "elevations": elevations,
        "results":    results,
        "summary":    summary,
        "altitude_m": altitude_m,
    }


def _plot_cn0_vs_elevation(results, elevations, signals, terminals, output_dir, label):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = {"Handheld": "#e17055", "Survey": "#00b894", "Maritime": "#0984e3", "Aviation": "#6c5ce7"}
    ls     = {"L1": "-", "L5": "--"}

    for ax_idx, sig in enumerate(signals):
        ax = axes[ax_idx]
        for term in terminals:
            key  = f"{sig}/{term}"
            cn0s = [r["cn0_db_hz"] for r in results[key]]
            ax.plot(elevations, cn0s, color=colors[term], ls=ls[sig],
                    lw=2, label=term, marker="o", ms=3)

        ax.axhline(CN0_MIN_DB_HZ["navigation"], ls=":", color="#e17055", lw=1.5,
                   label=f'Порог навигации ({CN0_MIN_DB_HZ["navigation"]} дБ-Гц)')
        ax.axhline(CN0_MIN_DB_HZ["tracking"], ls=":", color="#fdcb6e", lw=1.2,
                   label=f'Порог слежения ({CN0_MIN_DB_HZ["tracking"]} дБ-Гц)')
        ax.set_xlabel("Угол места (градусы)")
        ax.set_ylabel("C/N₀ (дБ-Гц)")
        ax.set_title(f"АВРОРА {sig} — C/N₀ от угла места")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim(10, 90)

    fig.suptitle(f"АВРОРА — Бюджет линии пользователя [{label}]")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"link_budget_cn0_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_sigma_pr(results, elevations, signals, terminals, output_dir, label):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = {"Handheld": "#e17055", "Survey": "#00b894", "Maritime": "#0984e3", "Aviation": "#6c5ce7"}

    for ax_idx, sig in enumerate(signals):
        ax = axes[ax_idx]
        for term in terminals:
            key    = f"{sig}/{term}"
            sigmas = [r["sigma_pr_m"] * 100 for r in results[key]]   # cm
            ax.plot(elevations, sigmas, color=colors[term], lw=2, label=term, marker="o", ms=3)

        ax.set_xlabel("Угол места (градусы)")
        ax.set_ylabel("Тепловой шум псевдодальности σ (см)")
        ax.set_title(f"АВРОРА {sig} — Шум псевдодальности σ")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim(10, 90)
        ax.set_ylim(bottom=0)

    fig.suptitle(f"АВРОРА — Тепловой шум псевдодальности [{label}]")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"link_budget_sigma_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_link_budget_waterfall(signals, terminals, altitude_m, output_dir, label):
    """Waterfall bar chart of budget components at 30° elevation, Survey terminal."""
    fig, axes = plt.subplots(1, len(signals), figsize=(5 * len(signals), 7))
    if len(signals) == 1:
        axes = [axes]

    for ax, sig in zip(axes, signals):
        r = compute_link_budget(sig, "Survey", 30.0, altitude_m)
        components = [
            ("Мощн. ПРД",      r["tx_power_dbw"],          "#74b9ff"),
            ("- Потери ПРД",  -r["tx_losses_db"],           "#fab1a0"),
            ("+ Усил. ПРД",    r["tx_gain_dbi"],            "#00b894"),
            ("- ИСРП",        -r["fspl_db"],                "#e17055"),
            ("- Атм. потери", -r["atmo_loss_db"],           "#fdcb6e"),
            ("+ Усил. ПРМ",    r["rx_gain_dbi"],            "#6c5ce7"),
            ("- Потери ПРМ",  -r["rx_losses_db"],           "#fd79a8"),
        ]
        names  = [c[0] for c in components]
        values = [c[1] for c in components]
        colors = [c[2] for c in components]

        x = range(len(names))
        bars = ax.bar(x, values, color=colors, edgecolor="white", lw=0.8)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, rotation=40, ha="right", fontsize=9)
        ax.set_ylabel("Вклад (дБ / дБВт)")
        ax.set_title(f"АВРОРА {sig} — Бюджет линии связи\n(Survey, 30° места, {altitude_m/1000:.0f} км)\n"
                     f"Мощн. ПРМ: {r['rx_power_dbm']:.1f} дБм  |  C/N₀: {r['cn0_db_hz']:.1f} дБ-Гц")
        ax.grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.5 if val >= 0 else -1.5),
                    f"{val:+.1f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(f"АВРОРА — Водопад бюджета линии [{label}]", fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"link_budget_waterfall_{label}.png"), dpi=150)
    plt.close(fig)


def _save_link_budget_csv(results, output_dir, label):
    path = os.path.join(output_dir, f"link_budget_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["signal", "terminal", "elevation_deg", "slant_range_km",
                    "eirp_dbw", "fspl_db", "atmo_db", "rx_power_dbm",
                    "cn0_db_hz", "sigma_pr_cm", "margin_nav_db", "viable"])
        for key, rows in results.items():
            for r in rows:
                w.writerow([
                    r["signal"], r["terminal"], r["elevation_deg"],
                    f"{r['slant_range_km']:.0f}",
                    f"{r['eirp_dbw']:.1f}", f"{r['fspl_db']:.1f}", f"{r['atmo_loss_db']:.1f}",
                    f"{r['rx_power_dbm']:.1f}", f"{r['cn0_db_hz']:.1f}",
                    f"{r['sigma_pr_m']*100:.2f}", f"{r['margin_navigation_db']:.1f}",
                    r["viable"],
                ])


def print_link_budget_summary(label: str, result: Dict) -> None:
    sep = "=" * 68

    print(f"\n{sep}")
    print(f"  User Terminal Link Budget -- {label}")
    print(sep)
    print(f"  Orbit altitude: {result['altitude_m']/1000:.0f} km")
    print(f"  Signals: {', '.join(result['signals'])}")
    print()

    for sig in result["signals"]:
        print(f"  Signal: АВРОРА {sig}  ({AURORA_SIGNALS[sig]['desc']})")
        print(f"  {'Terminal':<12}  {'El 10°':>8}  {'El 30°':>8}  {'El 45°':>8}  {'El 90°':>8}")
        print(f"  {'':-<12}  {'C/N0 (dB-Hz)':>38}")
        for term in result["terminals"]:
            pts = result["summary"][sig][term]
            vals = "  ".join(f"{pts[el]['cn0_db_hz']:>6.1f}" for el in [10, 30, 45, 90])
            print(f"  {term:<12}  {vals}")

        print()
        print(f"  Pseudorange thermal noise sigma (Survey, 30° el):")
        r30 = result["results"][f"{sig}/Survey"][4]   # index 4 = 30°
        print(f"    Chip length:     {r30['chip_len_m']*100:.1f} cm  "
              f"(fc={AURORA_SIGNALS[sig]['chip_rate_mcps']:.2f} Mcps)")
        print(f"    sigma_PR:        {r30['sigma_pr_m']*100:.2f} cm  (thermal only)")
        print(f"    C/N0:            {r30['cn0_db_hz']:.1f} dB-Hz")
        print(f"    Nav margin:      {r30['margin_navigation_db']:.1f} dB")
        print()

    print(f"  Minimum required C/N0: {CN0_MIN_DB_HZ['navigation']} dB-Hz (navigation)")
    print(f"  Handheld works above:  ~{_min_el_for_navigation(result, 'L1', 'Handheld'):.0f}° elevation (L1)")
    print(sep)


def _min_el_for_navigation(result, sig, term):
    """Find minimum elevation where navigation C/N0 is met."""
    for r in result["results"][f"{sig}/{term}"]:
        if r["margin_navigation_db"] >= 0:
            return r["elevation_deg"]
    return 90.0
