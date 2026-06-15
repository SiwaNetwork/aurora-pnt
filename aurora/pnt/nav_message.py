"""
Navigation Message Structure and Bit Budget for АВРОРА.

Models the navigation data content, bit allocation, frame structure, and
Time To First Fix (TTFF) for the АВРОРА L1/L5 signals.

Design goals:
  - GPS/Galileo compatible message structure (LNAV/FNAV inspired)
  - OSNMA-style navigation message authentication (NMA)
  - Ionospheric corrections (NeQuick-G model, Klobuchar fallback)
  - ISL-derived clock corrections at higher update rate than ground-only GNSS
  - TTFF < 30 s with valid almanac (cold start < 60 s)

References:
  IS-GPS-200N (ICD-GPS-LNAV / CNAV)
  Galileo OS SIS ICD 2.1 (FNAV / INAV)
  ESA OSNMA SIS ICD 1.1
  АВРОРА design baseline
"""

import math
import os
import csv
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Signal parameters ─────────────────────────────────────────────────────────
AURORA_DATA_RATE_BPS = {
    "L1": 500,    # bits/second (secondary code + data overlay, like GPS CNAV)
    "L5": 500,    # bits/second (dataless + pilot channels excluded from budget)
}

# ── Navigation message field definitions ─────────────────────────────────────
# Each field: (name, bits, resolution, range, description)
# Modelled after GPS CNAV + Galileo FNAV + АВРОРА extensions

EPHEMERIS_FIELDS = [
    # Time
    ("TOW",             20,  1,        "s",    "Time of week (truncated)"),
    ("week_number",     13,  1,        "weeks","GPS week number (extended)"),
    ("IODC",            10,  1,        "",     "Issue of data, clock"),
    ("IODE",             8,  1,        "",     "Issue of data, ephemeris"),
    # Orbit (Keplerian + corrections)
    ("sqrt_A",          32,  2**-19,   "m^0.5","Square root of semi-major axis"),
    ("delta_n",         16,  2**-43,   "sc/s", "Mean motion correction"),
    ("M0",              32,  2**-31,   "sc",   "Mean anomaly at reference time"),
    ("e",               32,  2**-33,   "",     "Eccentricity"),
    ("omega",           32,  2**-31,   "sc",   "Argument of perigee"),
    ("OMEGA0",          32,  2**-31,   "sc",   "Right ascension of ascending node"),
    ("i0",              32,  2**-31,   "sc",   "Inclination angle at t_oe"),
    ("OMEGA_dot",       24,  2**-43,   "sc/s", "Rate of right ascension"),
    ("idot",            14,  2**-43,   "sc/s", "Rate of inclination angle"),
    ("Cuc",             16,  2**-29,   "rad",  "Lat correction cosine, arg of lat"),
    ("Cus",             16,  2**-29,   "rad",  "Lat correction sine, arg of lat"),
    ("Crc",             16,  2**-5,    "m",    "Radius correction cosine"),
    ("Crs",             16,  2**-5,    "m",    "Radius correction sine"),
    ("Cic",             16,  2**-29,   "rad",  "Inclination correction cosine"),
    ("Cis",             16,  2**-29,   "rad",  "Inclination correction sine"),
    ("t_oe",            16,  300,      "s",    "Reference time, ephemeris"),
    # Fit interval flag + health
    ("health",           8,  1,        "",     "Signal health / SV health word"),
    ("fit_interval",     1,  1,        "",     "Fit interval flag (4/6 h)"),
]

CLOCK_FIELDS = [
    ("t_oc",            16,  300,      "s",    "Clock data reference time"),
    ("af0",             22,  2**-31,   "s",    "Clock bias"),
    ("af1",             16,  2**-43,   "s/s",  "Clock drift"),
    ("af2",              8,  2**-55,   "s/s^2","Clock drift rate"),
    ("TGD_L1",           8,  2**-31,   "s",    "L1 group delay (intersignal)"),
    ("TGD_L5",           8,  2**-31,   "s",    "L5 group delay"),
    ("ISL_correction",  16,  2**-35,   "s",    "ISL-derived clock correction (АВРОРА ext)"),
    ("ISL_age",          6,  1,        "s",    "Age of ISL correction (seconds)"),
]

IONO_FIELDS = [
    # NeQuick-G style (3 coefficients, like Galileo)
    ("ai0",             11,  2**-2,    "sfu",  "NeQuick-G alpha coefficient 0"),
    ("ai1",             11,  2**-8,    "sfu/deg","NeQuick-G alpha coefficient 1"),
    ("ai2",             14,  2**-15,   "sfu/deg^2","NeQuick-G alpha coefficient 2"),
    ("iono_flags",       5,  1,        "",     "Ionospheric disturbance flags"),
    # Klobuchar fallback (8 coefficients, GPS-compatible)
    ("alpha0",           8,  2**-30,   "s",    "Klobuchar alpha 0"),
    ("alpha1",           8,  2**-27,   "s/sc", "Klobuchar alpha 1"),
    ("alpha2",           8,  2**-24,   "s/sc^2","Klobuchar alpha 2"),
    ("alpha3",           8,  2**-24,   "s/sc^3","Klobuchar alpha 3"),
    ("beta0",            8,  2**11,    "s",    "Klobuchar beta 0"),
    ("beta1",            8,  2**14,    "s/sc", "Klobuchar beta 1"),
    ("beta2",            8,  2**16,    "s/sc^2","Klobuchar beta 2"),
    ("beta3",            8,  2**16,    "s/sc^3","Klobuchar beta 3"),
]

TIMING_FIELDS = [
    ("GGTO",            32,  2**-35,   "s",    "GPS-to-АВРОРА time offset"),
    ("GGTO_week",       13,  1,        "weeks","Reference week for GGTO"),
    ("GGTO_tow",        16,  300,      "s",    "Reference TOW for GGTO"),
    ("leap_second",      8,  1,        "s",    "Current UTC leap second count"),
    ("aurora_timescale", 4,  1,        "",     "АВРОРА timescale flags"),
]

AUTH_FIELDS = [
    # OSNMA-style: hash-based chain authentication
    ("TESLA_root_hash",    256, 1, "bits", "Root hash of TESLA chain"),
    ("TESLA_key",          128, 1, "bits", "Current TESLA key (revealed)"),
    ("ECDSA_sig_part",     128, 1, "bits", "ECDSA-P256 signature fragment"),
    ("NMA_status",           4, 1, "",     "NMA chain status flags"),
    ("NMA_chain_id",         8, 1, "",     "Authentication chain ID"),
]

ALMANAC_FIELDS_PER_SAT = [
    ("e_alm",           16,  2**-21, "",     "Eccentricity (almanac)"),
    ("delta_i",         16,  2**-19, "sc",   "Inclination offset from nominal"),
    ("OMEGA_dot_alm",   16,  2**-38, "sc/s", "RAAN rate (almanac)"),
    ("sqrt_A_alm",      24,  2**-11, "m^0.5","sqrt semi-major axis (almanac)"),
    ("OMEGA0_alm",      24,  2**-23, "sc",   "RAAN at reference epoch"),
    ("omega_alm",       24,  2**-23, "sc",   "Argument of perigee (almanac)"),
    ("M0_alm",          24,  2**-23, "sc",   "Mean anomaly (almanac)"),
    ("af0_alm",         11,  2**-20, "s",    "Clock correction (coarse)"),
    ("af1_alm",         11,  2**-37, "s/s",  "Clock rate (coarse)"),
    ("health_alm",       4,  1,     "",     "Health status (almanac)"),
    ("svid",             8,  1,     "",     "Satellite ID"),
]

# Frame structure (inspired by GPS CNAV-2 / Galileo FNAV)
FRAME_STRUCTURE = {
    "subframe_bits":    300,     # bits per subframe (like GPS)
    "preamble_bits":     8,      # sync preamble
    "message_id_bits":  6,       # message type
    "tow_bits":         17,      # time-of-week in subframe
    "crc_bits":         24,      # CRC-24Q (same as GPS CNAV)
    "fec_overhead":     0.5,     # 1/2 rate FEC (rate-1/2 convolutional or LDPC)
}

AURORA_CONSTELLATION = {
    "n_sats":          180,   # Phase 4
    "n_planes":         15,
    "sats_per_plane":   12,
}


def bits_per_field_group(fields: List) -> int:
    return sum(f[1] for f in fields)


def almanac_bits_total(n_sats: int) -> int:
    bits_per_sat = sum(f[1] for f in ALMANAC_FIELDS_PER_SAT)
    return bits_per_sat * n_sats


def effective_data_bits(raw_bits: int, fec_overhead: float = 0.5) -> int:
    """Useful payload bits after FEC encoding."""
    return int(raw_bits * (1 - fec_overhead))


def ttff_analysis(
    data_rate_bps: int,
    subframe_bits: int,
    fec_overhead: float,
    n_sats: int,
) -> Dict:
    """
    Estimate Time To First Fix (TTFF) based on message delivery time.

    Assumptions:
    - Hot start: ephemeris + clock valid; need only TOW synchronisation
    - Warm start: almanac valid, ephemeris expired; need fresh ephemeris
    - Cold start: no valid data; need full almanac + ephemeris for ≥4 sats
    """
    payload_bps = data_rate_bps * (1 - fec_overhead)

    # Time to receive one subframe
    subframe_s  = subframe_bits / data_rate_bps

    # Ephemeris delivery time (one satellite's complete ephemeris)
    eph_bits    = bits_per_field_group(EPHEMERIS_FIELDS) + bits_per_field_group(CLOCK_FIELDS)
    eph_bits   += subframe_bits * 3   # typically 3 subframes in GPS-style
    eph_time_s  = eph_bits / payload_bps

    # Almanac delivery time (all satellites)
    alm_bits    = almanac_bits_total(n_sats)
    alm_time_s  = alm_bits / payload_bps

    # TTFF scenarios
    hot_ttff_s   = 2.0               # TOW sync + position fix
    warm_ttff_s  = eph_time_s + 5.0  # fresh ephemeris for visible sats + fix
    cold_ttff_s  = alm_time_s + warm_ttff_s + 10.0  # full almanac + eph + fix

    return {
        "data_rate_bps":   data_rate_bps,
        "payload_bps":     payload_bps,
        "subframe_s":      subframe_s,
        "eph_bits":        eph_bits,
        "eph_time_s":      eph_time_s,
        "alm_bits_total":  alm_bits,
        "alm_time_s":      alm_time_s,
        "hot_ttff_s":      hot_ttff_s,
        "warm_ttff_s":     warm_ttff_s,
        "cold_ttff_s":     cold_ttff_s,
        "n_sats":          n_sats,
    }


def message_capacity_analysis(data_rate_bps: int = 500) -> Dict:
    """
    Analyse total navigation message capacity and field allocation.
    """
    eph_bits   = bits_per_field_group(EPHEMERIS_FIELDS)
    clk_bits   = bits_per_field_group(CLOCK_FIELDS)
    iono_bits  = bits_per_field_group(IONO_FIELDS)
    time_bits  = bits_per_field_group(TIMING_FIELDS)
    auth_bits  = bits_per_field_group(AUTH_FIELDS)
    alm_bits   = almanac_bits_total(AURORA_CONSTELLATION["n_sats"])

    total_core  = eph_bits + clk_bits + iono_bits + time_bits
    total_all   = total_core + auth_bits + alm_bits

    # Delivery time at 500 bps with 1/2 FEC
    fec  = FRAME_STRUCTURE["fec_overhead"]
    pl   = data_rate_bps * (1 - fec)

    return {
        "data_rate_bps":       data_rate_bps,
        "fec_overhead":        fec,
        "payload_bps":         pl,
        "fields": {
            "ephemeris_bits":  eph_bits,
            "clock_bits":      clk_bits,
            "ionosphere_bits": iono_bits,
            "timing_bits":     time_bits,
            "auth_bits":       auth_bits,
            "almanac_bits":    alm_bits,
        },
        "total_core_bits":     total_core,
        "total_all_bits":      total_all,
        "eph_delivery_s":      eph_bits / pl,
        "clk_delivery_s":      clk_bits / pl,
        "iono_delivery_s":     iono_bits / pl,
        "auth_delivery_s":     auth_bits / pl,
        "alm_delivery_s":      alm_bits / pl,
        "total_delivery_s":    total_all / pl,
        "update_interval_s":   30.0,   # target: full message cycle ≤ 30 s
    }


def run_nav_message_analysis(
    output_dir: str,
    label: str,
    data_rate_bps: int = 500,
    n_sats: int = AURORA_CONSTELLATION["n_sats"],
) -> Dict:
    """Run full navigation message analysis."""
    os.makedirs(output_dir, exist_ok=True)

    capacity = message_capacity_analysis(data_rate_bps)
    ttff     = ttff_analysis(
        data_rate_bps,
        FRAME_STRUCTURE["subframe_bits"],
        FRAME_STRUCTURE["fec_overhead"],
        n_sats,
    )

    # Data rate sensitivity sweep
    rates     = [50, 100, 250, 500, 1000, 2000]
    ttff_cold = [ttff_analysis(r, FRAME_STRUCTURE["subframe_bits"],
                               FRAME_STRUCTURE["fec_overhead"], n_sats)["cold_ttff_s"]
                 for r in rates]
    ttff_warm = [ttff_analysis(r, FRAME_STRUCTURE["subframe_bits"],
                               FRAME_STRUCTURE["fec_overhead"], n_sats)["warm_ttff_s"]
                 for r in rates]

    _plot_bit_allocation(capacity, output_dir, label)
    _plot_ttff_vs_rate(rates, ttff_cold, ttff_warm, output_dir, label)
    _save_nav_csv(capacity, ttff, output_dir, label)

    return {
        "capacity":    capacity,
        "ttff":        ttff,
        "rates":       rates,
        "ttff_cold":   ttff_cold,
        "ttff_warm":   ttff_warm,
        "n_sats":      n_sats,
        "data_rate_bps": data_rate_bps,
    }


def _plot_bit_allocation(capacity: Dict, output_dir: str, label: str):
    fields = capacity["fields"]
    names  = ["Эфемериды", "Часы", "Ионосфера", "Время/GGTO", "Аутентиф. (OSNMA)", "Альманах"]
    sizes  = [fields["ephemeris_bits"], fields["clock_bits"], fields["ionosphere_bits"],
              fields["timing_bits"],    fields["auth_bits"],   fields["almanac_bits"]]
    colors = ["#00b894", "#0984e3", "#6c5ce7", "#fdcb6e", "#e17055", "#fd79a8"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Горизонтальный bar в лог-масштабе вместо пирога: при доле Альманаха ~96 %
    # круговая диаграмма нечитаема (подписи долей сталкиваются), а лог-бар
    # показывает все компоненты, различающиеся на 2 порядка.
    order    = sorted(range(len(sizes)), key=lambda i: sizes[i])
    s_names  = [names[i]  for i in order]
    s_sizes  = [sizes[i]  for i in order]
    s_colors = [colors[i] for i in order]
    total    = sum(sizes)
    axes[0].barh(s_names, s_sizes, color=s_colors, edgecolor="white", linewidth=0.6)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Биты сообщения (лог. шкала)")
    axes[0].grid(axis="x", alpha=0.3, which="both")
    axes[0].set_xlim(right=max(s_sizes) * 4)
    for i, sz in enumerate(s_sizes):
        axes[0].text(sz * 1.15, i, f"{sz} б · {100*sz/total:.0f}%", va="center", fontsize=9)
    axes[0].set_title(f"Распределение битов навигационного сообщения\n"
                      f"Всего: {total:,} бит  ({total/8/1024:.1f} кБ)")

    # Bar: delivery time
    del_times = [
        capacity["eph_delivery_s"],
        capacity["clk_delivery_s"],
        capacity["iono_delivery_s"],
        capacity["total_delivery_s"] - capacity["eph_delivery_s"]
                 - capacity["clk_delivery_s"] - capacity["iono_delivery_s"],
    ]
    bar_names  = ["Эфемериды", "Часы", "Ионосфера", "Аутент.+Альманах"]
    bar_colors = ["#00b894", "#0984e3", "#6c5ce7", "#fd79a8"]
    axes[1].barh(bar_names, del_times, color=bar_colors, edgecolor="white")
    axes[1].axvline(30.0, ls="--", color="#e17055", lw=1.5, label="Цель 30 с")
    axes[1].set_xlabel("Время доставки (с) при нагрузке 500 bps")
    axes[1].set_title("Время доставки компонентов сообщения")
    axes[1].legend()
    axes[1].grid(axis="x", alpha=0.3)
    for i, (t, n) in enumerate(zip(del_times, bar_names)):
        axes[1].text(t + 0.5, i, f"{t:.1f} с", va="center", fontsize=9)

    fig.suptitle(f"АВРОРА — Структура навигационного сообщения [{label}]")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"nav_message_bits_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_ttff_vs_rate(rates, ttff_cold, ttff_warm, output_dir, label):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogx(rates, [t for t in ttff_cold], "o-", color="#e17055", lw=2, label="Холодный старт")
    ax.semilogx(rates, [t for t in ttff_warm], "s-", color="#0984e3", lw=2, label="Тёплый старт")
    ax.axhline(60.0,  ls="--", color="#e17055", lw=1.2, label="Цель 60 с (холодный)")
    ax.axhline(30.0,  ls="--", color="#0984e3", lw=1.2, label="Цель 30 с (тёплый)")
    ax.axvline(500.0, ls=":", color="#6c5ce7",  lw=1.2, label="АВРОРА 500 bps")
    ax.set_xlabel("Скорость данных (bps)")
    ax.set_ylabel("TTFF (секунды)")
    ax.set_title(f"АВРОРА — Время первого фикса от скорости данных [{label}]")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"nav_ttff_{label}.png"), dpi=150)
    plt.close(fig)


def _save_nav_csv(capacity: Dict, ttff: Dict, output_dir: str, label: str):
    path = os.path.join(output_dir, f"nav_message_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["component", "bits", "delivery_s"])
        f_map = capacity["fields"]
        t_map = {
            "ephemeris": capacity["eph_delivery_s"],
            "clock":     capacity["clk_delivery_s"],
            "ionosphere": capacity["iono_delivery_s"],
            "auth":      capacity["auth_delivery_s"],
            "almanac":   capacity["alm_delivery_s"],
        }
        for name, bits in zip(
            ["ephemeris", "clock", "ionosphere", "timing", "auth", "almanac"],
            [f_map["ephemeris_bits"], f_map["clock_bits"], f_map["ionosphere_bits"],
             f_map["timing_bits"], f_map["auth_bits"], f_map["almanac_bits"]],
        ):
            t = t_map.get(name)
            w.writerow([name, bits, f"{t:.1f}" if t is not None else "-"])
        w.writerow(["", "", ""])
        w.writerow(["ttff_scenario", "seconds"])
        w.writerow(["hot_start",  f"{ttff['hot_ttff_s']:.1f}"])
        w.writerow(["warm_start", f"{ttff['warm_ttff_s']:.1f}"])
        w.writerow(["cold_start", f"{ttff['cold_ttff_s']:.1f}"])


def print_nav_message_summary(label: str, result: Dict) -> None:
    sep = "=" * 68
    cap  = result["capacity"]
    ttff = result["ttff"]
    f    = cap["fields"]

    print(f"\n{sep}")
    print(f"  Navigation Message Structure -- {label}")
    print(sep)
    print(f"  Data rate:      {cap['data_rate_bps']} bps  "
          f"(payload: {cap['payload_bps']:.0f} bps after 1/2 FEC)")
    print()
    print(f"  Bit budget:")
    print(f"    Ephemeris:    {f['ephemeris_bits']:>6} bits  ({cap['eph_delivery_s']:.1f} s)")
    print(f"    Clock:        {f['clock_bits']:>6} bits  ({cap['clk_delivery_s']:.1f} s)  [+ ISL correction]")
    print(f"    Ionosphere:   {f['ionosphere_bits']:>6} bits  ({cap['iono_delivery_s']:.1f} s)  [NeQuick-G + Klobuchar]")
    print(f"    Timing/GGTO:  {f['timing_bits']:>6} bits")
    print(f"    Auth (OSNMA): {f['auth_bits']:>6} bits  ({cap['auth_delivery_s']:.1f} s)  [TESLA chain]")
    print(f"    Almanac:      {f['almanac_bits']:>6} bits  ({cap['alm_delivery_s']:.1f} s)  [{result['n_sats']} sats]")
    print(f"    ─────────────────────────────────")
    print(f"    TOTAL:        {cap['total_all_bits']:>6} bits  = {cap['total_all_bits']/8/1024:.1f} kB")
    print(f"    Full cycle:   {cap['total_delivery_s']:.1f} s  "
          f"({'OK' if cap['total_delivery_s'] <= 120 else 'EXCEEDS 2 min target'})")
    print()
    print(f"  Frame structure (GPS LNAV/CNAV inspired):")
    print(f"    Subframe:     {FRAME_STRUCTURE['subframe_bits']} bits")
    print(f"    Preamble:     {FRAME_STRUCTURE['preamble_bits']} bits")
    print(f"    Message ID:   {FRAME_STRUCTURE['message_id_bits']} bits  ({2**FRAME_STRUCTURE['message_id_bits']} message types)")
    print(f"    CRC:          {FRAME_STRUCTURE['crc_bits']} bits  (CRC-24Q)")
    print(f"    FEC:          1/2 rate  (rate-1/2 conv. or LDPC)")
    print()
    print(f"  Time To First Fix:")
    print(f"    Hot start:    {ttff['hot_ttff_s']:.0f} s   (valid almanac + ephemeris)")
    print(f"    Warm start:   {ttff['warm_ttff_s']:.0f} s   (almanac valid, re-acquire eph)")
    print(f"    Cold start:   {ttff['cold_ttff_s']:.0f} s   (no prior data)")
    print()
    print(f"  ISL clock correction extension:")
    print(f"    Extra bits:   {next(b for n,b,*_ in CLOCK_FIELDS if n=='ISL_correction') + next(b for n,b,*_ in CLOCK_FIELDS if n=='ISL_age')} bits per satellite")
    print(f"    Update rate:  every ISL sync interval (< 60 s)")
    print(f"    Benefit:      reduces user clock correction age vs ground-only")
    print(sep)
