"""
Frequency Plan Analysis for AURORA PNT.

Models ITU frequency allocation, inter-system interference, and signal
compatibility with GPS/GALILEO L1+L5 and GLONASS G1+G2.

ITU references:
  - Radio Regulations Appendix 4 (coordination of LEO satellite systems)
  - ITU-R M.1902 (characteristics of radionavigation satellite systems)
  - ITU-R S.1719 (coordination of non-GSO systems)
"""

import math
import os
import csv
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Speed of light
C_LIGHT = 299_792_458.0   # m/s

# ---------------------------------------------------------------------------
# Frequency allocations (ITU Table of Frequency Allocations, § 5.328A etc.)
# ---------------------------------------------------------------------------

BANDS = {
    # AURORA proposed
    "AURORA_L1": {
        "center_mhz": 1575.42, "bw_mhz": 24.0,
        "service": "RNSS", "system": "AURORA",
        "color": "#00b894", "desc": "AURORA L1 (GPS/Galileo compatible)"
    },
    "AURORA_L5": {
        "center_mhz": 1176.45, "bw_mhz": 24.0,
        "service": "RNSS", "system": "AURORA",
        "color": "#00cec9", "desc": "AURORA L5 (GPS/Galileo compatible)"
    },
    # GPS
    "GPS_L1": {
        "center_mhz": 1575.42, "bw_mhz": 24.0,
        "service": "RNSS", "system": "GPS",
        "color": "#0984e3", "desc": "GPS L1 C/A + L1C + P(Y)"
    },
    "GPS_L2": {
        "center_mhz": 1227.60, "bw_mhz": 22.0,
        "service": "RNSS", "system": "GPS",
        "color": "#6c5ce7", "desc": "GPS L2 C + P(Y)"
    },
    "GPS_L5": {
        "center_mhz": 1176.45, "bw_mhz": 24.0,
        "service": "RNSS", "system": "GPS",
        "color": "#a29bfe", "desc": "GPS L5 safety-of-life"
    },
    # Galileo
    "GAL_E1": {
        "center_mhz": 1575.42, "bw_mhz": 32.0,
        "service": "RNSS", "system": "Galileo",
        "color": "#fdcb6e", "desc": "Galileo E1 (OS + PRS)"
    },
    "GAL_E5a": {
        "center_mhz": 1176.45, "bw_mhz": 24.0,
        "service": "RNSS", "system": "Galileo",
        "color": "#e17055", "desc": "Galileo E5a"
    },
    "GAL_E5b": {
        "center_mhz": 1207.14, "bw_mhz": 24.0,
        "service": "RNSS", "system": "Galileo",
        "color": "#fab1a0", "desc": "Galileo E5b"
    },
    # GLONASS
    "GLO_G1": {
        "center_mhz": 1602.00, "bw_mhz": 16.5,
        "service": "RNSS", "system": "GLONASS",
        "color": "#d63031", "desc": "GLONASS G1 FDMA (1598-1605 MHz)"
    },
    "GLO_G2": {
        "center_mhz": 1246.00, "bw_mhz": 11.0,
        "service": "RNSS", "system": "GLONASS",
        "color": "#e84393", "desc": "GLONASS G2 FDMA"
    },
    "GLO_G3": {
        "center_mhz": 1202.03, "bw_mhz": 20.0,
        "service": "RNSS", "system": "GLONASS",
        "color": "#fd79a8", "desc": "GLONASS G3 CDMA (future)"
    },
    # BeiDou
    "BDS_B1C": {
        "center_mhz": 1575.42, "bw_mhz": 32.0,
        "service": "RNSS", "system": "BeiDou",
        "color": "#55efc4", "desc": "BeiDou B1C"
    },
    "BDS_B2a": {
        "center_mhz": 1176.45, "bw_mhz": 24.0,
        "service": "RNSS", "system": "BeiDou",
        "color": "#00b894", "desc": "BeiDou B2a"
    },
    # IRNSS / NavIC
    "IRNSS_L5": {
        "center_mhz": 1176.45, "bw_mhz": 24.0,
        "service": "RNSS", "system": "NavIC",
        "color": "#74b9ff", "desc": "NavIC L5"
    },
}

# ITU-R protected zone for RNSS (1164-1215 MHz + 1215-1300 MHz + 1559-1610 MHz)
ITU_RNSS_PROTECTED = [
    (1164.0, 1215.0),
    (1215.0, 1300.0),
    (1559.0, 1610.0),
]

# ---------------------------------------------------------------------------
# Signal modulation parameters for AURORA L1 + L5
# ---------------------------------------------------------------------------

AURORA_SIGNALS = {
    "L1_BPSK": {
        "band": "AURORA_L1", "modulation": "BPSK(1)",
        "chip_rate_mcps": 1.023, "bw_3db_mhz": 2.046,
        "spreading_gain_db": 43.2, "desc": "L1 civilian C/A-compatible"
    },
    "L1_BOC": {
        "band": "AURORA_L1", "modulation": "BOC(1,1)",
        "chip_rate_mcps": 1.023, "bw_3db_mhz": 4.092,
        "spreading_gain_db": 43.2, "desc": "L1 precision BOC (lower cross-corr with GPS)"
    },
    "L5_BPSK": {
        "band": "AURORA_L5", "modulation": "BPSK(10)",
        "chip_rate_mcps": 10.23, "bw_3db_mhz": 20.46,
        "spreading_gain_db": 53.2, "desc": "L5 precision ranging"
    },
}

# ---------------------------------------------------------------------------
# Path loss and signal power model
# ---------------------------------------------------------------------------

def fspl_db(range_m: float, freq_mhz: float) -> float:
    """Free-space path loss in dB."""
    return 20 * math.log10(4 * math.pi * range_m * freq_mhz * 1e6 / C_LIGHT)


def received_power_dbw(
    tx_power_dbw: float,
    tx_gain_dbi: float,
    rx_gain_dbi: float,
    fspl_db_val: float,
    atm_loss_db: float = 0.5,
) -> float:
    return tx_power_dbw + tx_gain_dbi + rx_gain_dbi - fspl_db_val - atm_loss_db


def cn0_db(rx_power_dbw: float, noise_temp_k: float = 290.0, bw_hz: float = 1e6) -> float:
    """C/N0 in dBHz from received power."""
    noise_floor_dbw = 10 * math.log10(1.38e-23 * noise_temp_k * bw_hz)
    return rx_power_dbw - noise_floor_dbw


# ---------------------------------------------------------------------------
# Interference model
# ---------------------------------------------------------------------------

def spectral_overlap_fraction(center1_mhz: float, bw1_mhz: float,
                               center2_mhz: float, bw2_mhz: float) -> float:
    """
    Fraction of signal 2's bandwidth that overlaps with signal 1's bandwidth.
    Returns 0..1.
    """
    lo1, hi1 = center1_mhz - bw1_mhz/2, center1_mhz + bw1_mhz/2
    lo2, hi2 = center2_mhz - bw2_mhz/2, center2_mhz + bw2_mhz/2
    overlap = max(0.0, min(hi1, hi2) - max(lo1, lo2))
    return overlap / bw2_mhz if bw2_mhz > 0 else 0.0


def interference_cn0_degradation_db(
    n_interferers: int,
    intf_power_dbw: float,
    spectral_overlap: float,
    rx_bw_hz: float,
    prn_cross_corr_rejection_db: float = 30.0,
) -> float:
    """
    C/N0 degradation (dB) to a GNSS victim receiver from aggregate interference.

    Uses noise-rise model (ITU-R approach):
      ΔC/N0 = 10*log10(1 + I_eff / N_thermal)
    where I_eff is the aggregate interference power spectral density after
    PRN cross-correlation rejection.

    prn_cross_corr_rejection_db: ~30 dB for same chip-rate CDMA systems with
    different PRN families (1023-chip Gold codes give 10*log10(1023) = 30 dB).
    """
    if spectral_overlap <= 0 or n_interferers <= 0:
        return 0.0

    # Aggregate in-band interference power (per Hz)
    intf_agg_dbw    = intf_power_dbw + 10 * math.log10(n_interferers * spectral_overlap)
    intf_psd_dbw_hz = intf_agg_dbw - 10 * math.log10(rx_bw_hz)
    # Thermal noise PSD: N0 = kT @ 290 K = -204 dBW/Hz
    n0_thermal_dbw_hz = -204.0
    # Effective interference after PRN rejection
    intf_eff_dbw_hz = intf_psd_dbw_hz - prn_cross_corr_rejection_db
    # Noise rise ratio I_eff / N0
    inr_linear = 10 ** ((intf_eff_dbw_hz - n0_thermal_dbw_hz) / 10.0)
    # C/N0 degradation = 10*log10(1 + INR)
    degradation = 10 * math.log10(1.0 + inr_linear)
    return max(0.0, degradation)


def compute_aggregate_interference(
    n_leo_sats: int,
    altitude_m: float,
    tx_power_dbw: float = 16.0,    # 40 W
    tx_gain_dbi: float = 14.0,
    rx_gain_dbi: float = 3.0,
    victim_signal: str = "GPS_L1",
    interferer_band: str = "AURORA_L1",
) -> Dict:
    """
    Compute aggregate C/N0 degradation to a victim receiver from all AURORA LEO sats
    simultaneously visible at worst-case geometry.
    """
    v = BANDS[victim_signal]
    i = BANDS[interferer_band]

    # Spectral overlap
    overlap = spectral_overlap_fraction(
        v["center_mhz"], v["bw_mhz"],
        i["center_mhz"], i["bw_mhz"],
    )

    # Average visible sats from user perspective (typically 10-15% of constellation)
    n_visible = max(1, int(n_leo_sats * 0.12))

    # Minimum slant range (zenith, 1000 km altitude)
    min_range_m = altitude_m

    # Interference signal received from zenith satellite
    fspl = fspl_db(min_range_m, i["center_mhz"])
    intf_power = received_power_dbw(tx_power_dbw, tx_gain_dbi, rx_gain_dbi, fspl)

    # Victim receiver bandwidth (BPSK(1): 2.046 MHz; BPSK(10): 20.46 MHz)
    victim_bw_hz = v["bw_mhz"] * 1e6

    # PRN cross-correlation rejection: 30 dB for 1023-chip codes, 40 dB for 10230-chip
    chip_rate = 10.23e6 if "L5" in interferer_band else 1.023e6
    prn_rejection = 10 * math.log10(chip_rate / 1e3)  # ~30 dB or ~40 dB

    degradation = interference_cn0_degradation_db(
        n_visible, intf_power, overlap, victim_bw_hz, prn_rejection
    )

    return {
        "victim_signal":    victim_signal,
        "interferer_band":  interferer_band,
        "spectral_overlap": overlap,
        "n_visible_intf":   n_visible,
        "intf_power_dbw":   intf_power,
        "cn0_degradation_db": degradation,
        "compatible":       degradation < 0.25,  # ITU-R threshold: < 0.25 dB
    }


# ---------------------------------------------------------------------------
# Doppler shift analysis
# ---------------------------------------------------------------------------

def doppler_shift_khz(freq_mhz: float, velocity_m_s: float = 7800.0) -> float:
    """Maximum Doppler shift for LEO satellite at 1000 km, worst case."""
    return freq_mhz * 1e6 * velocity_m_s / C_LIGHT / 1e3  # Hz -> kHz


def doppler_rate_hz_per_s(freq_mhz: float, altitude_m: float = 1_000_000) -> float:
    """Maximum Doppler rate (Hz/s) near overhead pass."""
    mu = 3.986004418e14
    r  = 6_378_137 + altitude_m
    v  = math.sqrt(mu / r)
    # Max rate at overhead, d(Δf)/dt = f*v^2/(c*r)
    return freq_mhz * 1e6 * v**2 / (C_LIGHT * r)


# ---------------------------------------------------------------------------
# ITU coordination analysis
# ---------------------------------------------------------------------------

ITU_FILING_STEPS = [
    ("Advance Publication Information (API)", "Article 9, § 9.1",
     "Submit 7-10 years before launch; published in IFIC"),
    ("Request for Coordination (RfC)", "Article 9, § 9.3",
     "Trigger coordination if threshold exceeded; 4-year window"),
    ("Coordination Agreement", "Article 9, § 9.4-9.6",
     "Bilateral agreement with affected administrations"),
    ("Notification & Recording", "Article 11",
     "Submit not earlier than 7 years before, not later than due date"),
    ("Bring-Into-Use (BIU)", "Article 11.44",
     "7-year deadline from recording; partial BIU (1 sat) is sufficient"),
    ("Milestone declarations", "Res. 35 (Rev. WRC-19)",
     "Report orbital milestones to ITU every few years to retain priority"),
]

# Key WRC-19 decisions affecting LEO navigation
WRC_DECISIONS = [
    ("WRC-19 Res. 35", "5+2 year BIU deadline for GSO; LEO retains 7-year rule"),
    ("WRC-19 Res. 559", "Additional protection for RNSS in 1164-1215 MHz band"),
    ("WRC-23 preliminary", "Enhanced coordination for mega-constellations"),
    ("ITU-R M.1902-1", "Technical characteristics of RNSS for coordination purposes"),
]


# ---------------------------------------------------------------------------
# Full analysis runner
# ---------------------------------------------------------------------------

def run_frequency_plan_analysis(
    output_dir: str,
    label: str,
    n_sats: int = 180,
    altitude_m: float = 1_000_000,
    tx_power_dbw: float = 16.0,
    tx_gain_dbi: float = 14.0,
    rx_gain_dbi: float = 3.0,
) -> Dict:

    os.makedirs(output_dir, exist_ok=True)

    # 1. Frequency allocation overview
    alloc = {name: b for name, b in BANDS.items()}

    # 2. AURORA signal compatibility analysis
    compat = {}
    for sig_name, sig in AURORA_SIGNALS.items():
        band = BANDS[sig["band"]]
        # Check overlap with other systems at same center freq
        overlaps = {}
        for other_name, other in BANDS.items():
            if other["system"] == "AURORA":
                continue
            ov = spectral_overlap_fraction(
                band["center_mhz"], band["bw_mhz"],
                other["center_mhz"], other["bw_mhz"],
            )
            if ov > 0.01:
                overlaps[other_name] = round(ov, 3)
        compat[sig_name] = {
            "modulation":    sig["modulation"],
            "center_mhz":   band["center_mhz"],
            "bw_mhz":       band["bw_mhz"],
            "chip_rate":    sig["chip_rate_mcps"],
            "overlapping_systems": overlaps,
        }

    # 3. Interference to/from other GNSS
    interference = {}
    # AURORA L1 -> GPS L1 (are we degrading GPS?)
    interference["AURORA_L1_to_GPS_L1"] = compute_aggregate_interference(
        n_sats, altitude_m, tx_power_dbw, tx_gain_dbi, rx_gain_dbi,
        victim_signal="GPS_L1", interferer_band="AURORA_L1",
    )
    # AURORA L5 -> GPS L5
    interference["AURORA_L5_to_GPS_L5"] = compute_aggregate_interference(
        n_sats, altitude_m, tx_power_dbw, tx_gain_dbi, rx_gain_dbi,
        victim_signal="GPS_L5", interferer_band="AURORA_L5",
    )
    # GPS L1 -> AURORA L1 (does GPS interfere with us?)
    # GPS: ~31 sats, higher altitude 20200 km -> much weaker signal at user
    interference["GPS_L1_to_AURORA_L1"] = compute_aggregate_interference(
        31, 20_200_000, 14.0, 20.0, rx_gain_dbi,
        victim_signal="AURORA_L1", interferer_band="GPS_L1",
    )
    # GLONASS G1 vs AURORA L1 (adjacent channel, ~27 MHz separation -> no overlap)
    glo_overlap = spectral_overlap_fraction(1575.42, 24.0, 1602.0, 16.5)
    interference["GLO_G1_to_AURORA_L1"] = {
        "victim_signal":      "AURORA_L1",
        "interferer_band":    "GLO_G1",
        "spectral_overlap":   glo_overlap,
        "n_visible_intf":     0,
        "intf_power_dbw":     -999.0,
        "cn0_degradation_db": 0.0,
        "compatible":         True,
    }

    # 4. Doppler shift analysis
    doppler = {}
    for band_name in ["AURORA_L1", "AURORA_L5"]:
        f_mhz = BANDS[band_name]["center_mhz"]
        doppler[band_name] = {
            "center_mhz":         f_mhz,
            "max_doppler_khz":    doppler_shift_khz(f_mhz),
            "doppler_rate_hz_s":  doppler_rate_hz_per_s(f_mhz, altitude_m),
        }

    # 5. ITU protection zone compliance
    itu_compliance = []
    for band_name in ["AURORA_L1", "AURORA_L5"]:
        b = BANDS[band_name]
        lo = b["center_mhz"] - b["bw_mhz"] / 2
        hi = b["center_mhz"] + b["bw_mhz"] / 2
        in_zone = any(lo >= plo and hi <= phi for plo, phi in ITU_RNSS_PROTECTED)
        itu_compliance.append({
            "band":       band_name,
            "range_mhz":  f"{lo:.2f}-{hi:.2f}",
            "in_rnss_protected_zone": in_zone,
            "status":     "COMPLIANT" if in_zone else "REQUIRES COORDINATION",
        })

    # 6. Generate plots
    _plot_frequency_plan(output_dir, label)
    _plot_interference_summary(interference, output_dir, label)
    _plot_doppler(doppler, altitude_m, output_dir, label)

    # 7. Save CSVs
    _save_interference_csv(interference, output_dir, label)
    _save_itu_csv(itu_compliance, output_dir, label)

    return {
        "bands":            alloc,
        "aurora_signals":   AURORA_SIGNALS,
        "compatibility":    compat,
        "interference":     interference,
        "doppler":          doppler,
        "itu_compliance":   itu_compliance,
        "itu_filing_steps": ITU_FILING_STEPS,
        "wrc_decisions":    WRC_DECISIONS,
        "n_sats":           n_sats,
        "altitude_km":      altitude_m / 1000,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_frequency_plan(output_dir: str, label: str) -> None:
    """Frequency allocation diagram for RNSS bands."""
    fig, ax = plt.subplots(figsize=(14, 6))

    systems_order = ["GPS", "Galileo", "GLONASS", "BeiDou", "NavIC", "AURORA"]
    y_map = {s: i for i, s in enumerate(systems_order)}
    height = 0.5

    for band_name, b in BANDS.items():
        system = b["system"]
        y = y_map.get(system, len(systems_order))
        lo = b["center_mhz"] - b["bw_mhz"] / 2
        bw = b["bw_mhz"]
        alpha = 0.85 if system == "AURORA" else 0.55
        lw = 2 if system == "AURORA" else 0.8
        rect = plt.Rectangle(
            (lo, y - height/2), bw, height,
            color=b["color"], alpha=alpha,
            edgecolor="white", linewidth=lw,
        )
        ax.add_patch(rect)
        ax.text(b["center_mhz"], y, band_name.split("_")[1],
                ha="center", va="center", fontsize=7, fontweight="bold", color="white")

    # ITU RNSS protected zones
    for lo, hi in ITU_RNSS_PROTECTED:
        ax.axvspan(lo, hi, alpha=0.05, color="green", zorder=0)
        ax.text((lo+hi)/2, len(systems_order) + 0.4, f"ITU RNSS\n{lo:.0f}-{hi:.0f}",
                ha="center", fontsize=7, color="#2ecc71", alpha=0.7)

    ax.set_yticks(range(len(systems_order)))
    ax.set_yticklabels(systems_order)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_title(f"AURORA PNT — Frequency Allocation Plan [{label}]", fontsize=12)
    ax.set_xlim(1150, 1650)
    ax.set_ylim(-0.8, len(systems_order) + 1.2)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"freq_plan_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_interference_summary(interference: Dict, output_dir: str, label: str) -> None:
    """Bar chart of C/N0 degradation for each interference pair."""
    fig, ax = plt.subplots(figsize=(10, 5))
    pairs = list(interference.keys())
    degradations = [v.get("cn0_degradation_db", 0.0) for v in interference.values()]
    compatible   = [v.get("compatible", True)        for v in interference.values()]
    colors = ["#00b894" if c else "#d63031" for c in compatible]

    bars = ax.bar(range(len(pairs)), degradations, color=colors, edgecolor="white", width=0.6)
    ax.bar_label(bars, fmt="%.3f dB", padding=3, fontsize=9)
    ax.axhline(0.25, ls="--", color="#e17055", lw=1.5, label="ITU-R threshold 0.25 dB")
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels([p.replace("_to_", " →\n") for p in pairs], fontsize=8)
    ax.set_ylabel("C/N0 degradation (dB)")
    ax.set_title(f"AURORA PNT — Inter-system Interference [{label}]")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(max(degradations) * 1.3, 0.5))
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"interference_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_doppler(doppler: Dict, altitude_m: float, output_dir: str, label: str) -> None:
    """Doppler shift vs elevation angle for L1 and L5."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    elevations = list(range(0, 91, 5))

    mu = 3.986004418e14
    r_orb = 6_378_137 + altitude_m
    v_orb = math.sqrt(mu / r_orb)
    R_E   = 6_378_137.0

    for ax_idx, (band_name, d) in enumerate(doppler.items()):
        f_hz = d["center_mhz"] * 1e6
        shifts = []
        for el in elevations:
            el_rad = math.radians(el)
            # Doppler = f * v * cos(angle) / c
            # Angle between velocity vector and look direction varies with elevation
            cos_angle = math.cos(math.pi/2 + el_rad)  # worst case geometry approx
            doppler_hz = abs(f_hz * v_orb * math.cos(el_rad) / C_LIGHT)
            shifts.append(doppler_hz / 1000)  # kHz

        ax = axes[ax_idx]
        ax.plot(elevations, shifts, color=BANDS[band_name]["color"], lw=2)
        ax.fill_between(elevations, shifts, alpha=0.15, color=BANDS[band_name]["color"])
        ax.set_xlabel("Elevation angle (deg)")
        ax.set_ylabel("Doppler shift (kHz)")
        ax.set_title(f"{band_name} ({d['center_mhz']:.2f} MHz)\nMax rate: "
                     f"{d['doppler_rate_hz_s']:.1f} Hz/s")
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 90)

    fig.suptitle(f"AURORA PNT — Doppler Profile [{label}]", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"doppler_{label}.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def _save_interference_csv(interference: Dict, output_dir: str, label: str) -> None:
    path = os.path.join(output_dir, f"interference_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "pair", "victim", "interferer", "spectral_overlap",
            "cn0_degradation_db", "compatible"
        ])
        w.writeheader()
        for pair, v in interference.items():
            w.writerow({
                "pair":                pair,
                "victim":              v.get("victim_signal", ""),
                "interferer":          v.get("interferer_band", ""),
                "spectral_overlap":    f"{v.get('spectral_overlap', 0):.4f}",
                "cn0_degradation_db":  f"{v.get('cn0_degradation_db', 0):.4f}",
                "compatible":          v.get("compatible", True),
            })


def _save_itu_csv(itu_compliance: List[Dict], output_dir: str, label: str) -> None:
    path = os.path.join(output_dir, f"itu_compliance_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["band", "range_mhz",
                                           "in_rnss_protected_zone", "status"])
        w.writeheader()
        for row in itu_compliance:
            w.writerow(row)


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def print_frequency_plan_summary(label: str, result: Dict) -> None:
    sep = "=" * 70

    print(f"\n{sep}")
    print(f"  Frequency Plan Analysis -- {label}")
    print(sep)
    print(f"  Constellation: {result['n_sats']} sats @ {result['altitude_km']:.0f} km")
    print()

    print(f"  AURORA Signal Plan:")
    print(f"  {'Signal':<12} {'Modulation':<12} {'Center MHz':>12} {'BW MHz':>8} {'Rate Mcps':>10}")
    print("  " + "-" * 58)
    for sig_name, sig in result["aurora_signals"].items():
        band = result["bands"][sig["band"]]
        print(f"  {sig_name:<12} {sig['modulation']:<12} "
              f"{band['center_mhz']:>12.2f} {band['bw_mhz']:>8.1f} "
              f"{sig['chip_rate_mcps']:>10.3f}")
    print()

    print(f"  ITU RNSS Protected Zone Compliance:")
    print(f"  {'Band':<14} {'Range (MHz)':<20} {'Status'}")
    print("  " + "-" * 52)
    for c in result["itu_compliance"]:
        status_icon = "[OK]" if c["status"] == "COMPLIANT" else "[!!]"
        print(f"  {c['band']:<14} {c['range_mhz']:<20} {status_icon} {c['status']}")
    print()

    print(f"  Inter-system Interference (C/N0 degradation):")
    print(f"  {'Pair':<30} {'Overlap':>8} {'Degrad. dB':>12} {'Status':>12}")
    print("  " + "-" * 66)
    for pair, v in result["interference"].items():
        ov   = v.get("spectral_overlap", 0)
        deg  = v.get("cn0_degradation_db", 0)
        ok   = v.get("compatible", True)
        status = "[OK]" if ok else "[INTF]"
        print(f"  {pair:<30} {ov:>8.3f} {deg:>12.4f} dB {status:>12}")
    print()

    print(f"  Doppler Shifts (max, at 10 deg elevation):")
    print(f"  {'Band':<14} {'Freq (MHz)':>12} {'Max Doppler':>14} {'Max Rate':>12}")
    print("  " + "-" * 56)
    for band_name, d in result["doppler"].items():
        print(f"  {band_name:<14} {d['center_mhz']:>12.2f} "
              f"{d['max_doppler_khz']:>12.1f} kHz "
              f"{d['doppler_rate_hz_s']:>8.1f} Hz/s")
    print()

    print(f"  ITU Filing Roadmap:")
    for step, ref, desc in result["itu_filing_steps"]:
        print(f"    [{ref}] {step}")
        print(f"           {desc}")
    print()

    print(f"  Key WRC decisions:")
    for decision, note in result["wrc_decisions"]:
        print(f"    {decision}: {note}")
    print(sep)
