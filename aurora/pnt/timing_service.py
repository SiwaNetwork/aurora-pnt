"""
АВРОРА-T: Satellite Timing Service Protocol for АВРОРА.

АВРОРА satellites broadcast a precision timing signal (АВРОРА-T) embedded in
the navigation message. Ground receivers extract nanosecond-accurate UTC time
without GPS/GLONASS, using the receiver's own position fix to remove propagation
delay. The receiver then acts as a local PTP grandmaster or NTP stratum-1 source.

Architecture (borrowed from Satelles STL + extended):
  1. Master clock (Cs/Maser) at Zheleznogorsk MCS generates LPT
  2. ISL chain distributes LPT to all 300 satellites (sigma_ISL = sqrt(N)*ppb*T)
  3. Each satellite broadcasts АВРОРА-T frame every 10 s on L1/L5
  4. Ground receiver solves position -> range -> removes propagation delay
  5. Remaining error: ISL clock + receiver noise + atmospheric residual
  6. Receiver outputs: 1PPS (< 5 ns), PTP sync (IEEE 1588-2019), NTP stratum-1

Key advantage vs GPS timing:
  - +25 dB signal -> works indoors, in urban canyons, in jamming environments
  - LEO fast pass (~8 min) -> rapid re-sync even after brief outage
  - Sovereign: no dependency on US GPS or Russian GLONASS for timing service

Protocol comparison:
  - NTP  (RFC 5905): millisecond accuracy, UDP/IP, existing infrastructure
  - PTP  (IEEE 1588-2019): nanosecond accuracy, hardware timestamping, telecom
  - АВРОРА-T: the satellite-layer that feeds PTP/NTP grandmasters
  - TWSTT: two-way time transfer for MCS ground stations (highest accuracy)
"""

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_C = 299_792_458.0  # m/s

# ── PTP accuracy class thresholds (IEEE 1588-2019 clockAccuracy field) ──────
# These correspond to the BMCA clockAccuracy TLV values
PTP_CLASSES = [
    (25,         "PTP Class 25  (< 25 ns)   - Stratum-0 grandmaster"),
    (100,        "PTP Class 32  (< 100 ns)  - Primary reference"),
    (250,        "PTP Class 33  (< 250 ns)  - Secondary reference"),
    (1_000,      "PTP Class 34  (< 1 us)   - Telecom stratum-2"),
    (10_000,     "PTP Class 35  (< 10 us)  - Distribution"),
    (100_000,    "PTP Class 36  (< 100 us) - Edge"),
    (1_000_000,  "PTP Class 37  (< 1 ms)   - Basic")
]

# ── Clock types available for АВРОРА master station ───────────────────────────
CLOCK_TYPES = {
    "ocxo":  {"ppb": 1.0,    "label": "OCXO    (1 ppb)"},
    "rb":    {"ppb": 0.1,    "label": "Rb      (0.1 ppb)"},
    "cs":    {"ppb": 0.01,   "label": "Cs      (0.01 ppb)"},
    "maser": {"ppb": 0.001,  "label": "H-Maser (0.001 ppb)"},
}

# ── Fixed noise floor components (independent of master clock type) ───────────
RCVR_NOISE_NS     = 0.8   # Dual-freq receiver correlator noise
ATM_RESIDUAL_NS   = 0.17  # L1+L5 dual-freq atmospheric residual (~0.05 m / c)
MULTIPATH_NS      = 0.50  # Open-sky multipath on timing signal
PROP_RESIDUAL_PCT = 0.10  # Range-correction residual: 10% of position -> time

# ── АВРОРА-T frame specification ──────────────────────────────────────────────
AURORA_T_FRAME = {
    "broadcast_interval_s": 10,        # Frame every 10 s (vs GPS LNAV 30 s)
    "frame_bits": 500,                 # Total frame size
    "data_rate_bps": 50,               # Matches L1 C/A chip-rate subframe
    "fields": {
        "preamble":          24,
        "satellite_id":       8,
        "lpt_week":          16,
        "lpt_tow_ms":        20,       # Time of week, millisecond resolution
        "lpt_subsec_ns":     30,       # Sub-millisecond, nanosecond resolution
        "lpt_utc_offset_ns": 32,       # LPT-UTC, signed, 1 ns LSB
        "leap_seconds":       8,
        "clock_a0_ns":       32,       # Clock polynomial a0
        "clock_a1_ppb":      24,       # Clock polynomial a1 (drift)
        "clock_a2":          16,       # Clock polynomial a2 (drift rate)
        "sigma_t_log":        8,       # Timing accuracy estimate (log2 ns)
        "tesla_mac":        128,       # Navigation message authentication
        "crc24":             24,       # Frame integrity check
    },
    "notes": (
        "TESLA MAC: delayed key disclosure (Galileo OSNMA-compatible). "
        "Authentication latency = 1 frame (10 s). "
        "Keys broadcast in alternating frames; subscriber holds key for verification."
    ),
}

# ── Timing protocols that АВРОРА receiver feeds ───────────────────────────────
OUTPUT_PROTOCOLS = {
    "pps_1hz": {
        "name": "1PPS output",
        "standard": "PTTI 1000-2011",
        "accuracy_note": "Hardware edge, < 5 ns with Cs master",
    },
    "ptp": {
        "name": "PTP Grandmaster (IEEE 1588-2019)",
        "profile_telecom": "ITU-T G.8265.1 (frequency sync)",
        "profile_phase":   "ITU-T G.8275.1 (phase/time sync)",
        "sync_rate_hz": 16,           # Telecom profile default
        "delay_mechanism": "E2E",
        "domain_number": 44,          # Recommended for GNSS-disciplined GMs
    },
    "ntp": {
        "name": "NTP Stratum-1 (RFC 5905)",
        "refid": "AURA",              # NTP reference ID for АВРОРА
        "poll_interval_s": 16,
        "precision_ns": 2,            # clock_precision exponent
    },
}


def _ptp_class(sigma_total_ns: float) -> str:
    for threshold, label in PTP_CLASSES:
        if sigma_total_ns < threshold:
            return label
    return "PTP Class 255 (unknown)"


def compute_timing_budget(
    clock_key: str,
    n_isl_hops: int = 8,
    sync_interval_s: float = 60.0,
    uere_m: float = 3.10,
    dual_frequency: bool = True,
) -> dict:
    """
    Timing accuracy budget for a single clock type and operating mode.

    The receiver eliminates geometric propagation delay using its own position fix.
    Residual timing error comes from ISL clock distribution + receiver noise + atmosphere.

    Returns a dict with per-component errors and total sigma in nanoseconds.
    """
    ppb = CLOCK_TYPES[clock_key]["ppb"]

    # ISL chain: sqrt(N_hops) × ppb × T_sync [ns]
    sigma_isl_ns = math.sqrt(n_isl_hops) * ppb * sync_interval_s

    # Propagation residual: receiver position accuracy -> range uncertainty -> timing
    # Position error ≈ UERE; range correction residual ≈ 10% (due to multipath, etc.)
    sigma_prop_ns = (uere_m * PROP_RESIDUAL_PCT) / _C * 1e9

    sigma_rcvr_ns = RCVR_NOISE_NS if dual_frequency else 1.5
    sigma_atm_ns  = ATM_RESIDUAL_NS if dual_frequency else 2.0
    sigma_mp_ns   = MULTIPATH_NS

    sigma_total_ns = math.sqrt(
        sigma_isl_ns ** 2 +
        sigma_prop_ns ** 2 +
        sigma_rcvr_ns ** 2 +
        sigma_atm_ns  ** 2 +
        sigma_mp_ns   ** 2
    )

    return {
        "clock_key":       clock_key,
        "clock_label":     CLOCK_TYPES[clock_key]["label"],
        "sigma_isl_ns":    round(sigma_isl_ns, 3),
        "sigma_prop_ns":   round(sigma_prop_ns, 4),
        "sigma_rcvr_ns":   sigma_rcvr_ns,
        "sigma_atm_ns":    sigma_atm_ns,
        "sigma_mp_ns":     sigma_mp_ns,
        "sigma_total_ns":  round(sigma_total_ns, 3),
        "ptp_class":       _ptp_class(sigma_total_ns),
        "ntp_eligible":    sigma_total_ns < 1_000,  # < 1 us -> stratum-1 capable
    }


def compute_twstt_accuracy(
    ground_station_stability_ppb: float = 0.01,  # Cs at MCS
    round_trip_range_km: float = 5_000,           # Typical LEO-GS geometry
) -> dict:
    """
    Two-Way Satellite Time Transfer (TWSTT) accuracy estimate.

    Used for MCS-to-satellite synchronization. Both uplink and downlink
    propagation delays cancel in the two-way exchange, leaving only:
    - Asymmetry in equipment delays (hardware calibration residual)
    - Multipath on both links
    - Atmospheric asymmetry (< 0.1 ns for close geometry)
    """
    # Equipment calibration residual: typically 0.1–0.3 ns in modern systems
    sigma_equipment_ns = 0.2
    # Path asymmetry: up vs down can differ slightly (< 0.1 ns)
    sigma_asymmetry_ns = 0.1
    # Multipath both directions
    sigma_mp_ns = 0.3

    sigma_total_ns = math.sqrt(
        sigma_equipment_ns ** 2 +
        sigma_asymmetry_ns ** 2 +
        sigma_mp_ns ** 2
    )

    return {
        "method": "TWSTT (MCS ground stations only)",
        "sigma_equipment_ns":  sigma_equipment_ns,
        "sigma_asymmetry_ns":  sigma_asymmetry_ns,
        "sigma_mp_ns":         sigma_mp_ns,
        "sigma_total_ns":      round(sigma_total_ns, 3),
        "note": "TWSTT used only at MCS uplink stations; user terminals use one-way АВРОРА-T",
    }


# ── Miniature space-grade clock hardware specs ────────────────────────────────
SPACE_CLOCKS = {
    "ocxo": {
        "ppb":          1.0,
        "mass_g":       120,
        "power_w":      2.5,
        "label":        "OCXO",
        "example":      "NDK NVO-1210 / Rakon IT3207A",
        "trl":          9,
        "note":         "Standard satellite oscillator; very light and low-power",
    },
    "rb": {
        "ppb":          0.1,
        "mass_g":       400,
        "power_w":      10.0,
        "label":        "Miniature Rb",
        "example":      "Microsemi SA.31m / Jackson Labs Luna",
        "trl":          8,
        "note":         "Miniature Rb standard; flight-proven on LEO cubesats",
    },
    "cs": {
        "ppb":          0.01,
        "mass_g":       700,
        "power_w":      20.0,
        "label":        "Miniature Cs",
        "example":      "Symmetricom 4310B / Vremya-CH MHM-2010",
        "trl":          7,
        "note":         "Space-grade Cs standard; used on GLONASS-M class satellites",
    },
    "maser": {
        "ppb":          0.001,
        "mass_g":       20_000,
        "power_w":      55.0,
        "label":        "H-Maser",
        "example":      "Galileo PHM / GPS SSAF",
        "trl":          9,
        "note":         "NOT suitable for small LEO (too heavy); only for large platforms",
    },
}


def compute_isl_chain_sigma(hops: list[str], sync_interval_s: float = 60.0) -> float:
    """
    Compute ISL timing chain noise for a sequence of relay clocks.

    Each relay satellite introduces its own oscillator noise as it re-timestamps
    and re-transmits the correction. Terminal satellite is the receiver — its own
    clock does NOT enter the chain (only affects holdover).

    hops: list of clock keys for relay satellites ONLY (not the terminal).
    Example: ['cs', 'rb'] means MCS -> Cs relay -> Rb relay -> terminal satellite.

    Returns sigma_ISL in nanoseconds.
    """
    variance = sum((SPACE_CLOCKS[h]["ppb"] * sync_interval_s) ** 2 for h in hops)
    return math.sqrt(variance)


def compute_holdover(clock_key: str, target_ns: float = 10.0) -> dict:
    """
    How long a satellite maintains < target_ns timing error when ISL breaks.

    target_ns = 10 ns corresponds to ~3 m range error (position budget limit).
    """
    ppb = SPACE_CLOCKS[clock_key]["ppb"]
    holdover_s = target_ns / ppb
    return {
        "clock_key":     clock_key,
        "clock_label":   SPACE_CLOCKS[clock_key]["label"],
        "ppb":           ppb,
        "holdover_s":    holdover_s,
        "holdover_min":  holdover_s / 60,
        "target_ns":     target_ns,
        "target_range_m": target_ns * 3e8 / 1e9,
    }


def _autonomous_nav_limit(clock_key: str, target_m: float = 1.0) -> dict:
    """Time until clock drift accumulates target_m range error in autonomous navigation."""
    ppb = SPACE_CLOCKS[clock_key]["ppb"]
    drift_rate_m_per_s = ppb * 1e-9 * _C
    limit_s = target_m / drift_rate_m_per_s
    return {
        "clock_key":           clock_key,
        "drift_rate_m_per_s":  drift_rate_m_per_s,
        "drift_rate_m_per_hr": drift_rate_m_per_s * 3600,
        "limit_s":             limit_s,
        "limit_min":           limit_s / 60,
        "target_m":            target_m,
    }


def run_mixed_clock_analysis(
    output_dir: str,
    label: str = "phase4",
    n_planes: int = 15,
    n_sats_per_plane: int = 20,
    cs_per_plane: int = 1,
    rb_per_plane: int = 3,
    sync_interval_s: float = 60.0,
    ocxo_sync_interval_s: float = 10.0,
) -> dict:
    """
    Mixed-clock constellation analysis for АВРОРА Phase 4.

    Architecture:
      - cs_per_plane Cs satellites per orbital plane: timing anchors
        (nearest to MCS uplink geometry; receive TWSTT + relay to Rb)
      - rb_per_plane Rb satellites per plane: timing relays
        (relay corrections from Cs to OCXO neighbors)
      - rest: OCXO satellites as timing terminals (receive but do NOT relay)

    ISL chain from MCS to satellite type:
      OCXO terminal: MCS -> [Cs relay x1] -> [Rb relay x1] -> OCXO
      Rb relay:      MCS -> [Cs relay x1] -> Rb
      Cs anchor:     MCS -> Cs  (direct TWSTT or 1 ISL hop)

    OCXO timing budget = ISL chain sigma + OCXO self-drift during ocxo_sync_interval_s.
    To stay within PTP Class 25 (< 25 ns), OCXO requires ocxo_sync_interval_s ≤ ~24 s.
    Default: ocxo_sync_interval_s = 10 s → sigma ≈ 10 ns (Class 25 ✓).
    """
    os.makedirs(output_dir, exist_ok=True)

    n_total = n_planes * n_sats_per_plane
    n_cs    = n_planes * cs_per_plane
    n_rb    = n_planes * rb_per_plane
    n_ocxo  = n_total - n_cs - n_rb

    # ── ISL chain accuracy for each satellite tier ──────────────────────────
    # Cs anchor: 1 Cs hop from MCS (MCS itself is H-Maser/Cs ground clock ~0.001 ppb)
    sigma_cs_anchor = compute_isl_chain_sigma(["cs"], sync_interval_s)

    # Rb relay: Cs relay error + Rb's own drift during T_sync
    sigma_rb_relay  = compute_isl_chain_sigma(["cs", "rb"], sync_interval_s)

    # OCXO terminal: ISL chain error (Cs+Rb relays) + OCXO's own drift during ocxo_sync_interval_s
    # BUG FIX: previously missing OCXO self-drift term (10× underestimate at T=60 s)
    sigma_chain_to_ocxo = compute_isl_chain_sigma(["cs", "rb"], ocxo_sync_interval_s)
    ocxo_self_drift_ns  = SPACE_CLOCKS["ocxo"]["ppb"] * ocxo_sync_interval_s
    sigma_ocxo_term     = math.sqrt(sigma_chain_to_ocxo**2 + ocxo_self_drift_ns**2)

    # ── Holdover performance ────────────────────────────────────────────────
    target_ns = 10.0  # < 10 ns = < 3 m range error
    holdover = {k: compute_holdover(k, target_ns) for k in ["ocxo", "rb", "cs"]}

    # ── Mass and power budget ───────────────────────────────────────────────
    budget = {}
    for key in ["ocxo", "rb", "cs"]:
        c = SPACE_CLOCKS[key]
        count = {"ocxo": n_ocxo, "rb": n_rb, "cs": n_cs}[key]
        budget[key] = {
            "count":       count,
            "mass_kg":     count * c["mass_g"] / 1000,
            "power_w":     count * c["power_w"],
            "per_sat_g":   c["mass_g"],
            "per_sat_w":   c["power_w"],
        }

    # ── Autonomous navigation limits (clock-dominated) ─────────────────────
    auto_nav = {k: _autonomous_nav_limit(k, target_m=1.0) for k in ["ocxo", "rb", "cs"]}

    results = {
        "label": label,
        "n_total": n_total, "n_cs": n_cs, "n_rb": n_rb, "n_ocxo": n_ocxo,
        "n_planes": n_planes, "sats_per_plane": n_sats_per_plane,
        "sync_interval_s":      sync_interval_s,
        "ocxo_sync_interval_s": ocxo_sync_interval_s,
        "isl_chain": {
            "cs_anchor_sigma_ns":      round(sigma_cs_anchor, 3),
            "rb_relay_sigma_ns":       round(sigma_rb_relay, 3),
            "ocxo_terminal_sigma_ns":  round(sigma_ocxo_term, 3),
            "ocxo_self_drift_ns":      round(ocxo_self_drift_ns, 3),
            "ocxo_ptp_class":          _ptp_class(sigma_ocxo_term),
            "rb_ptp_class":            _ptp_class(sigma_rb_relay),
            "cs_ptp_class":            _ptp_class(sigma_cs_anchor),
        },
        "holdover":        {k: v for k, v in holdover.items()},
        "autonomous_nav":  auto_nav,
        "hardware_budget": budget,
    }

    json_path = os.path.join(output_dir, f"mixed_clock_{label}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    _plot_mixed_clock(results, output_dir, label)
    _print_mixed_clock_summary(results)

    return results


def _plot_mixed_clock(results: dict, output_dir: str, label: str):
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.suptitle(
        f"АВРОРА Mixed-Clock Constellation Analysis  "
        f"({results['n_total']} sats: {results['n_cs']} Cs + "
        f"{results['n_rb']} Rb + {results['n_ocxo']} OCXO)",
        fontsize=13, fontweight="bold"
    )

    # ── Plot 1: ISL chain accuracy per tier ──────────────────────────────────
    ax = axes[0]
    T_rb   = results["sync_interval_s"]
    T_ocxo = results["ocxo_sync_interval_s"]
    tiers  = [f"Cs anchor\n(T={T_rb:.0f}s)", f"Rb relay\n(T={T_rb:.0f}s)",
              f"OCXO terminal\n(T={T_ocxo:.0f}s)"]
    sigmas = [
        results["isl_chain"]["cs_anchor_sigma_ns"],
        results["isl_chain"]["rb_relay_sigma_ns"],
        results["isl_chain"]["ocxo_terminal_sigma_ns"],
    ]
    colors = ["#1565C0", "#2E7D32", "#F57F17"]
    bars = ax.bar(tiers, sigmas, color=colors, alpha=0.88)
    ax.axhline(25,   color="green",  linestyle="--", alpha=0.8, label="PTP Class 25 (25 ns)")
    ax.axhline(100,  color="orange", linestyle="--", alpha=0.6, label="PTP Class 32 (100 ns)")
    ax.axhline(1000, color="red",    linestyle="--", alpha=0.6, label="NTP stratum-1 (1 us)")
    for bar, val in zip(bars, sigmas):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.2, f"{val:.2f} ns",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("Timing error 1σ [ns]")
    ax.set_title("Timing accuracy by satellite tier\n(ISL chain + self-drift, normal operation)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(sigmas) * 1.5)

    # ── Plot 2: Holdover time at ISL break ───────────────────────────────────
    ax = axes[1]
    ho_keys    = ["ocxo", "rb", "cs"]
    ho_labels  = ["OCXO", "Rb", "Cs"]
    ho_times   = [results["holdover"][k]["holdover_min"] for k in ho_keys]
    ho_colors  = ["#F57F17", "#2E7D32", "#1565C0"]

    bars = ax.barh(ho_labels, ho_times, color=ho_colors, alpha=0.88)
    # Reference lines: ISL re-sync time ~10-30 s typical
    ax.axvline(0.17,  color="gray",   linestyle=":",  alpha=0.8, label="10 s (ISL packet loss)")
    ax.axvline(1.67,  color="orange", linestyle="--", alpha=0.8, label="100 s (ISL link fail)")
    ax.axvline(16.67, color="red",    linestyle="--", alpha=0.8, label="1000 s (ISL segment fail)")
    for bar, val in zip(bars, ho_times):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f} min ({val*60:.0f} s)",
                ha="left", va="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Holdover time before > 3 m timing error [min]")
    ax.set_title(f"Holdover capability at ISL break\n(target < {results['holdover']['ocxo']['target_ns']:.0f} ns = "
                 f"< {results['holdover']['ocxo']['target_range_m']:.0f} m range error)")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)

    # ── Plot 3: Mass + power cost per constellation ──────────────────────────
    ax = axes[2]
    bud    = results["hardware_budget"]
    tiers3 = ["OCXO\n(all)", "Rb\n(relays)", "Cs\n(anchors)"]
    masses = [bud["ocxo"]["mass_kg"], bud["rb"]["mass_kg"], bud["cs"]["mass_kg"]]
    powers = [bud["ocxo"]["power_w"],  bud["rb"]["power_w"],  bud["cs"]["power_w"]]
    x      = np.arange(len(tiers3))
    w      = 0.38

    ax2r = ax.twinx()
    bars_m = ax.bar(x - w/2, masses, w, color=["#F57F17","#2E7D32","#1565C0"],
                    alpha=0.82, label="Mass (kg)")
    bars_p = ax2r.bar(x + w/2, powers, w, color=["#FFB74D","#81C784","#64B5F6"],
                      alpha=0.82, label="Power (W)")
    ax.set_ylabel("Total clock mass [kg]", color="#555")
    ax2r.set_ylabel("Total clock power [W]", color="#555")
    ax.set_title("Clock hardware budget\n(total for constellation)")
    ax.set_xticks(x)
    ax.set_xticklabels(tiers3)
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars_m, masses):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3, f"{val:.1f} kg",
                ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars_p, powers):
        ax2r.text(bar.get_x() + bar.get_width() / 2, val + 0.3, f"{val:.0f} W",
                  ha="center", va="bottom", fontsize=8)

    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2r.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, f"mixed_clock_{label}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def _print_mixed_clock_summary(results: dict):
    sep = "=" * 72
    print(f"\n{sep}")
    print("  АВРОРА MIXED-CLOCK CONSTELLATION ANALYSIS")
    print(sep)

    n = results
    print(f"\n  Constellation: {n['n_total']} satellites "
          f"({n['n_planes']} planes x {n['sats_per_plane']} sats)")
    print(f"  Clock tiers:   {n['n_cs']} Cs anchors + "
          f"{n['n_rb']} Rb relays + {n['n_ocxo']} OCXO terminals")

    isl = results["isl_chain"]
    T_rb   = results["sync_interval_s"]
    T_ocxo = results["ocxo_sync_interval_s"]
    print(f"\n  ISL Timing Accuracy (normal operation):")
    print(f"    Cs anchor  (T={T_rb:.0f}s, 1 Cs hop)           : "
          f"{isl['cs_anchor_sigma_ns']:.3f} ns  -> {isl['cs_ptp_class']}")
    print(f"    Rb relay   (T={T_rb:.0f}s, Cs+Rb chain)         : "
          f"{isl['rb_relay_sigma_ns']:.3f} ns  -> {isl['rb_ptp_class']}")
    print(f"    OCXO term  (T={T_ocxo:.0f}s, chain+self drift)   : "
          f"{isl['ocxo_terminal_sigma_ns']:.3f} ns  -> {isl['ocxo_ptp_class']}")
    print(f"    OCXO self-drift component            : "
          f"{isl['ocxo_self_drift_ns']:.3f} ns  (1 ppb x {T_ocxo:.0f}s)")

    print(f"\n  Holdover (ISL disrupted, target < {results['holdover']['ocxo']['target_ns']:.0f} ns = "
          f"< {results['holdover']['ocxo']['target_range_m']:.0f} m):")
    for key in ["ocxo", "rb", "cs"]:
        ho = results["holdover"][key]
        print(f"    {ho['clock_label']:<18}: {ho['holdover_s']:>6.0f} s "
              f"({ho['holdover_min']:>5.1f} min)")

    auto = results["autonomous_nav"]
    print(f"\n  Autonomous navigation (1 m range error limit, clock-dominated):")
    for key in ["ocxo", "rb", "cs"]:
        a = auto[key]
        print(f"    {SPACE_CLOCKS[key]['label']:<18}: drift {a['drift_rate_m_per_hr']:>7.2f} m/hr "
              f"-> {a['limit_s']:>6.0f} s ({a['limit_min']:.1f} min) to 1 m error")
    print(f"    Ephemeris (no ISL OD)  :  0.50 m/hr -> 7200 s (120 min, 2.0 hr) limit")
    print(f"    Ephemeris (with ISL OD): ~0.02 m/hr -> 180000 s (~50 hr = 2.1 days) limit")
    print(f"    => OCXO/Rb: clock-limited (seconds-minutes); Cs: ~5.6 min; with ISL OD: ephemeris-limited")

    bud = results["hardware_budget"]
    print(f"\n  Hardware budget (full {results['n_total']}-satellite constellation):")
    total_mass_kg  = sum(bud[k]["mass_kg"]  for k in bud)
    total_power_w  = sum(bud[k]["power_w"]  for k in bud)
    print(f"    {'Type':<12} {'Count':>6} {'Mass/sat':>10} {'Total mass':>12} {'Total power':>12}")
    print(f"    {'-'*55}")
    for key in ["ocxo", "rb", "cs"]:
        c  = SPACE_CLOCKS[key]
        b  = bud[key]
        print(f"    {c['label']:<12} {b['count']:>6} {c['mass_g']:>8} g "
              f"  {b['mass_kg']:>8.1f} kg   {b['power_w']:>8.0f} W")
    print(f"    {'TOTAL':<12} {results['n_total']:>6}            "
          f"  {total_mass_kg:>8.1f} kg   {total_power_w:>8.0f} W")
    print(f"    Per-satellite avg: {total_mass_kg*1000/results['n_total']:.0f} g clock mass, "
          f"{total_power_w/results['n_total']:.1f} W clock power")

    print(f"\n  Recommended hardware:")
    for key in ["ocxo", "rb", "cs"]:
        c = SPACE_CLOCKS[key]
        print(f"    {c['label']:<15}: {c['example']}")

    print(sep)


def run_timing_service_analysis(
    output_dir: str,
    label: str = "phase4",
    n_isl_hops: int = 8,
    sync_interval_s: float = 60.0,
    uere_autonomous_m: float = 3.10,
    uere_combined_m: float = 1.69,
) -> dict:
    """
    Full АВРОРА-T timing service analysis:
      - Timing accuracy budget for all clock types × two operating modes
      - PTP class achievable
      - TWSTT for ground stations
      - Protocol parameters
      - Plots + JSON output
    """
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "label": label,
        "n_isl_hops": n_isl_hops,
        "sync_interval_s": sync_interval_s,
        "aurora_t_frame": AURORA_T_FRAME,
        "output_protocols": OUTPUT_PROTOCOLS,
    }

    # Compute budgets for all clocks × both modes
    budgets = {}
    for clock_key in CLOCK_TYPES:
        budgets[clock_key] = {
            "autonomous": compute_timing_budget(
                clock_key, n_isl_hops, sync_interval_s, uere_autonomous_m
            ),
            "combined": compute_timing_budget(
                clock_key, n_isl_hops, sync_interval_s, uere_combined_m
            ),
        }
    results["budgets"] = budgets
    results["twstt"] = compute_twstt_accuracy()

    # Save JSON
    json_path = os.path.join(output_dir, f"timing_service_{label}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    _plot_timing_accuracy(budgets, output_dir, label)
    _plot_protocol_stack(output_dir, label)
    _print_timing_summary(results)

    return results


def _plot_timing_accuracy(budgets: dict, output_dir: str, label: str):
    """
    Two subplots:
      Left:  autonomous mode - stacked bar (ISL vs rest), log scale
      Right: combined mode   - same
    Reference lines for PTP classes.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    fig.suptitle("АВРОРА-T Timing Service - Accuracy Budget by Clock Type",
                 fontsize=14, fontweight="bold")

    keys   = list(CLOCK_TYPES.keys())
    xlabels = [CLOCK_TYPES[k]["label"] for k in keys]
    x = np.arange(len(keys))
    width = 0.55

    modes = [
        ("autonomous", "Autonomous LPT (no GLONASS)"),
        ("combined",   "Combined LEO+GLONASS"),
    ]
    colors = {
        "isl":   "#1565C0",
        "other": "#FB8C00",
    }

    for ax, (mode_key, mode_title) in zip(axes, modes):
        isl_vals   = [budgets[k][mode_key]["sigma_isl_ns"]   for k in keys]
        total_vals = [budgets[k][mode_key]["sigma_total_ns"] for k in keys]
        other_vals = [t - i for t, i in zip(total_vals, isl_vals)]

        ax.bar(x, isl_vals,   width, label="ISL clock chain",    color=colors["isl"],   alpha=0.88)
        ax.bar(x, other_vals, width, label="Prop + Rcvr + Atm",  color=colors["other"], alpha=0.88,
               bottom=isl_vals)

        # PTP reference lines
        ref_lines = [(25, "green", "PTP <25 ns (Class 25)"),
                     (100, "darkorange", "PTP <100 ns (Class 32)"),
                     (1000, "red", "NTP stratum-1 limit (1 us)")]
        for val, color, lbl in ref_lines:
            ax.axhline(y=val, color=color, linestyle="--", linewidth=1.2, alpha=0.75, label=lbl)

        ax.set_yscale("log")
        ax.set_ylim(0.1, 50_000)
        ax.set_title(mode_title, fontsize=11)
        ax.set_xlabel("Master clock type")
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=8)
        ax.legend(fontsize=7.5, loc="upper right")
        ax.grid(axis="y", alpha=0.3)

        for i, total in enumerate(total_vals):
            ax.text(i, total * 1.4, f"{total:.1f} ns",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    axes[0].set_ylabel("Timing error 1σ [ns] (log scale)")
    plt.tight_layout()

    path = os.path.join(output_dir, f"timing_accuracy_{label}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def _plot_protocol_stack(output_dir: str, label: str):
    """
    Horizontal diagram showing АВРОРА-T protocol stack and timing chain:
    MCS -> ISL -> Satellite -> АВРОРА-T broadcast -> Receiver -> PTP/NTP/1PPS
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle("АВРОРА-T Protocol Chain - From Master Clock to PTP Grandmaster",
                 fontsize=13, fontweight="bold", y=0.98)

    blocks = [
        (0.3, "MCS\n(Zheleznogorsk)\nCs/H-Maser\nLPT master", "#1565C0", "white"),
        (2.7, "ISL chain\n√N × ppb × T\n8 hops\n1.70 ns (Cs)", "#0288D1", "white"),
        (5.1, "LEO satellite\nAURORA-T frame\n10 s interval\n500 bits L1/L5", "#2E7D32", "white"),
        (7.5, "Ground\nreceiver\nPosition fix\n-> range sub.", "#6A1B9A", "white"),
        (9.9, "1PPS output\nPTP GM\nNTP S1\n< 5 ns (Cs)", "#B71C1C", "white"),
    ]

    for x0, text, bg, fg in blocks:
        rect = plt.Rectangle((x0, 1.2), 2.0, 2.6, linewidth=1.5,
                              edgecolor="white", facecolor=bg, zorder=2)
        ax.add_patch(rect)
        ax.text(x0 + 1.0, 2.5, text, ha="center", va="center",
                fontsize=8.5, color=fg, fontweight="bold", zorder=3, linespacing=1.4)

    # Arrows
    for x_arrow in [2.3, 4.7, 7.1, 9.5]:
        ax.annotate("", xy=(x_arrow + 0.4, 2.5), xytext=(x_arrow, 2.5),
                    arrowprops=dict(arrowstyle="->", color="#37474F", lw=2.0), zorder=4)

    # Labels on arrows
    arrow_labels = ["TWSTT\n±0.37 ns", "L-band\nbroadcast", "range\ncorrection", "IEEE 1588\nRFC 5905"]
    arrow_x = [2.35, 4.75, 7.15, 9.55]
    for x_label, txt in zip(arrow_x, arrow_labels):
        ax.text(x_label + 0.2, 3.15, txt, ha="center", va="bottom",
                fontsize=7, color="#37474F", style="italic")

    # Bottom note
    ax.text(7.0, 0.5, (
        "Signal advantage: +25 dB vs GPS -> indoor / urban canyon / jamming resilience   |   "
        "TESLA MAC: NMA authentication (Galileo OSNMA-compatible)   |   "
        "Sovereign: no US GPS or ГЛОНАСС dependency"
    ), ha="center", va="center", fontsize=7.5, color="#455A64", style="italic")

    path = os.path.join(output_dir, f"timing_protocol_stack_{label}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def _print_timing_summary(results: dict):
    budgets = results["budgets"]
    sep = "=" * 72

    print(f"\n{sep}")
    print("  АВРОРА-T TIMING SERVICE - ACCURACY SUMMARY")
    print(sep)

    for mode_key, mode_label in [
        ("autonomous", "Autonomous LPT (no GLONASS)"),
        ("combined",   "Combined LEO+GLONASS"),
    ]:
        print(f"\n  Mode: {mode_label}")
        print(f"  {'Clock type':<22} {'s_ISL':>9} {'s_total':>9} {'Classification'}")
        print("  " + "-" * 68)
        for clock_key in CLOCK_TYPES:
            b = budgets[clock_key][mode_key]
            print(f"  {b['clock_label']:<22} {b['sigma_isl_ns']:>7.2f} ns "
                  f"{b['sigma_total_ns']:>7.2f} ns  {b['ptp_class']}")

    tw = results["twstt"]
    print(f"\n  TWSTT (MCS ground stations, two-way): {tw['sigma_total_ns']:.3f} ns")
    print(f"    Equipment residual: {tw['sigma_equipment_ns']} ns | "
          f"Path asymmetry: {tw['sigma_asymmetry_ns']} ns | "
          f"Multipath: {tw['sigma_mp_ns']} ns")

    frame = results["aurora_t_frame"]
    print(f"\n  АВРОРА-T Frame Protocol:")
    print(f"    Broadcast interval : {frame['broadcast_interval_s']} s (vs GPS LNAV 30 s)")
    print(f"    Frame size         : {frame['frame_bits']} bits @ {frame['data_rate_bps']} bps")
    print(f"    Authentication     : TESLA MAC ({frame['fields']['tesla_mac']} bits, "
          f"10 s key disclosure)")

    proto = results["output_protocols"]
    print(f"\n  Output Protocols:")
    print(f"    1PPS   : {proto['pps_1hz']['accuracy_note']}")
    print(f"    PTP GM : {proto['ptp']['name']} - sync {proto['ptp']['sync_rate_hz']} Hz, "
          f"refID per satellite EUI-64")
    print(f"    NTP S1 : {proto['ntp']['name']} - refid='{proto['ntp']['refid']}', "
          f"poll {proto['ntp']['poll_interval_s']} s")

    print(f"\n{sep}")
