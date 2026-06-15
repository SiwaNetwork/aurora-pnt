"""
Anti-Jamming Analysis for АВРОРА.

Models J/S (jammer-to-signal) ratio and effective jamming radius for
АВРОРА LEO vs GPS MEO. Quantifies the +25 dB signal advantage of LEO.
"""

import math
import os
import csv
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_LIGHT = 299_792_458.0

# System parameters
SYSTEMS = {
    "GPS_L1": {
        "altitude_km":    20_200.0,
        "freq_mhz":       1575.42,
        "tx_power_dbw":   14.3,      # ~27 W EIRP
        "tx_gain_dbi":    13.0,
        "chip_rate_mcps":  1.023,
        "proc_gain_db":   43.2,      # 10*log10(1.023e6)
        "modulation":     "BPSK(1)",
        "color": "#0984e3",
    },
    "GPS_L5": {
        "altitude_km":    20_200.0,
        "freq_mhz":       1176.45,
        "tx_power_dbw":   14.3,
        "tx_gain_dbi":    13.0,
        "chip_rate_mcps": 10.23,
        "proc_gain_db":   53.2,
        "modulation":     "BPSK(10)",
        "color": "#6c5ce7",
    },
    "AURORA_L1_BPSK": {
        "altitude_km":     1_000.0,
        "freq_mhz":       1575.42,
        "tx_power_dbw":   16.0,      # 40 W
        "tx_gain_dbi":    14.0,
        "chip_rate_mcps":  1.023,
        "proc_gain_db":   43.2,
        "modulation":     "BPSK(1)",
        "color": "#00b894",
    },
    "AURORA_L1_BOC": {
        "altitude_km":     1_000.0,
        "freq_mhz":       1575.42,
        "tx_power_dbw":   16.0,
        "tx_gain_dbi":    14.0,
        "chip_rate_mcps":  1.023,
        "proc_gain_db":   43.2,      # BOC has same chip rate but better spectral separation
        "modulation":     "BOC(1,1)",
        "color": "#00cec9",
    },
    "AURORA_L5": {
        "altitude_km":     1_000.0,
        "freq_mhz":       1176.45,
        "tx_power_dbw":   16.0,
        "tx_gain_dbi":    14.0,
        "chip_rate_mcps": 10.23,
        "proc_gain_db":   53.2,
        "modulation":     "BPSK(10)",
        "color": "#55efc4",
    },
}

# Jammer models
JAMMERS = {
    "Носимый (0.1 Вт)": {
        "power_w": 0.1,  "gain_dbi": 0.0,  "type": "broadband"},
    "Портативный (1 Вт)":   {
        "power_w": 1.0,  "gain_dbi": 3.0,  "type": "broadband"},
    "Возимый (10 Вт)":   {
        "power_w": 10.0, "gain_dbi": 6.0,  "type": "broadband"},
    "Стационарный (100 Вт)":    {
        "power_w": 100.0,"gain_dbi": 10.0, "type": "broadband"},
    "DRFM-спуфер":     {
        "power_w": 5.0,  "gain_dbi": 6.0,  "type": "spoof"},
}


def fspl_db(range_m: float, freq_mhz: float) -> float:
    return 20 * math.log10(4 * math.pi * range_m * freq_mhz * 1e6 / C_LIGHT)


def signal_power_dbw(sys: Dict, elevation_deg: float = 10.0) -> float:
    """Received signal power from satellite at given elevation."""
    alt_m    = sys["altitude_km"] * 1000
    r_earth  = 6_371_000.0
    el_rad   = math.radians(elevation_deg)
    # Slant range for given elevation (simplified spherical Earth)
    sin_rho  = r_earth / (r_earth + alt_m)
    rho      = math.asin(sin_rho)
    eta      = math.pi / 2 - el_rad - rho
    slant_m  = (r_earth + alt_m) * math.sin(eta) / math.cos(el_rad)
    slant_m  = max(slant_m, alt_m)  # can't be less than altitude

    pl = fspl_db(slant_m, sys["freq_mhz"])
    return sys["tx_power_dbw"] + sys["tx_gain_dbi"] - pl


def jammer_power_at_rx_dbw(jammer: Dict, range_m: float, freq_mhz: float) -> float:
    pj_dbw = 10 * math.log10(jammer["power_w"])
    pl     = fspl_db(range_m, freq_mhz)
    return pj_dbw + jammer["gain_dbi"] - pl


def js_ratio_db(signal_dbw: float, jammer_dbw: float, proc_gain_db: float) -> float:
    """Effective J/S after processing gain. Positive = jammer wins."""
    return jammer_dbw - signal_dbw - proc_gain_db


def effective_jamming_radius_m(
    sys: Dict, jammer: Dict, elevation_deg: float = 10.0
) -> float:
    """
    Maximum ground range at which jammer can disrupt reception.
    J/S_effective = 0 dB is the threshold (jammer noise = signal power after PG).
    """
    sig_dbw = signal_power_dbw(sys, elevation_deg)
    # Need: Pj + Gj - FSPL(r) - proc_gain >= sig_dbw
    # => FSPL(r) <= Pj + Gj - sig_dbw - proc_gain
    max_fspl = (10 * math.log10(jammer["power_w"]) + jammer["gain_dbi"]
                - sig_dbw - sys["proc_gain_db"])
    if max_fspl <= 0:
        return 0.0
    # FSPL = 20*log10(4*pi*r*f/c) => r = c/(4*pi*f) * 10^(FSPL/20)
    freq_hz = sys["freq_mhz"] * 1e6
    r_m = (C_LIGHT / (4 * math.pi * freq_hz)) * 10**(max_fspl / 20)
    return r_m


def run_anti_jam_analysis(
    output_dir: str,
    label: str,
    elevation_deg: float = 10.0,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    # 1. Signal power comparison
    sig_powers = {name: signal_power_dbw(sys, elevation_deg)
                  for name, sys in SYSTEMS.items()}

    # 2. J/S ratio at 1 km for each jammer / system pair (reference distance)
    js_table = {}
    ref_range_m = 1000.0
    for sys_name, sys in SYSTEMS.items():
        sig = sig_powers[sys_name]
        for jam_name, jammer in JAMMERS.items():
            pj_at_rx = jammer_power_at_rx_dbw(jammer, ref_range_m, sys["freq_mhz"])
            js = js_ratio_db(sig, pj_at_rx, sys["proc_gain_db"])
            js_table[(sys_name, jam_name)] = {
                "signal_dbw": sig,
                "jammer_dbw": pj_at_rx,
                "js_eff_db":  js,
                "jammed":     js > 0,
            }

    # 3. Effective jamming radius per system per jammer
    jam_radii = {}
    for sys_name, sys in SYSTEMS.items():
        jam_radii[sys_name] = {}
        for jam_name, jammer in JAMMERS.items():
            r_m = effective_jamming_radius_m(sys, jammer, elevation_deg)
            jam_radii[sys_name][jam_name] = r_m / 1000  # km

    # 4. LEO advantage: АВРОРА vs GPS for each jammer
    advantage = {}
    for jam_name in JAMMERS:
        r_gps    = jam_radii.get("GPS_L1", {}).get(jam_name, 0)
        r_aurora = jam_radii.get("AURORA_L1_BPSK", {}).get(jam_name, 0)
        if r_gps > 0 and r_aurora > 0:
            reduction = (1 - r_aurora / r_gps) * 100
        elif r_gps > 0:
            reduction = 100.0
        else:
            reduction = 0.0
        advantage[jam_name] = {
            "r_gps_km":        r_gps,
            "r_aurora_km":     r_aurora,
            "radius_reduction_pct": reduction,
            "area_reduction_pct":   (1 - (r_aurora / r_gps)**2) * 100 if r_gps > 0 else 100,
        }

    # 5. Plots
    _plot_signal_comparison(sig_powers, output_dir, label)
    _plot_jamming_radii(jam_radii, output_dir, label)
    _plot_advantage(advantage, output_dir, label)

    # 6. CSV
    _save_js_csv(js_table, jam_radii, advantage, output_dir, label)

    return {
        "signal_powers":    sig_powers,
        "js_table":         js_table,
        "jam_radii_km":     jam_radii,
        "advantage":        advantage,
        "elevation_deg":    elevation_deg,
    }


def _plot_signal_comparison(sig_powers: Dict, output_dir: str, label: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    names  = list(sig_powers.keys())
    powers = [sig_powers[n] for n in names]
    colors = [SYSTEMS[n]["color"] for n in names]
    bars   = ax.bar(names, powers, color=colors, edgecolor="white", width=0.55)
    ax.bar_label(bars, fmt="%.1f дБВт", padding=3, fontsize=9)
    ax.axhline(min(powers), ls="--", color="#e17055", lw=0.8, alpha=0.7)
    ax.set_ylabel("Принимаемая мощность сигнала (дБВт)")
    ax.set_title(f"Мощность сигнала на приёмнике [{label}]")
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"antijam_signal_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_jamming_radii(jam_radii: Dict, output_dir: str, label: str) -> None:
    jammers  = list(JAMMERS.keys())
    systems  = ["GPS_L1", "GPS_L5", "AURORA_L1_BPSK", "AURORA_L1_BOC", "AURORA_L5"]
    systems  = [s for s in systems if s in jam_radii]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(jammers))
    width = 0.15
    for i, sys_name in enumerate(systems):
        radii = [jam_radii[sys_name].get(j, 0) for j in jammers]
        bars  = ax.bar(x + i * width, radii, width,
                       label=sys_name, color=SYSTEMS[sys_name]["color"],
                       edgecolor="white")

    ax.set_xticks(x + width * (len(systems) - 1) / 2)
    ax.set_xticklabels(jammers, fontsize=8, rotation=10)
    ax.set_ylabel("Эффективный радиус подавления (км)")
    ax.set_title(f"Радиус подавления по системе и типу джаммера [{label}]")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"antijam_radius_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_advantage(advantage: Dict, output_dir: str, label: str) -> None:
    jammers   = list(advantage.keys())
    reductions = [advantage[j]["radius_reduction_pct"] for j in jammers]
    area_reductions = [advantage[j]["area_reduction_pct"] for j in jammers]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    bars = ax.bar(range(len(jammers)), reductions,
                  color="#00b894", edgecolor="white", width=0.5)
    ax.bar_label(bars, fmt="%.0f%%", padding=3, fontsize=9)
    ax.set_xticks(range(len(jammers)))
    ax.set_xticklabels(jammers, rotation=10, ha="right", fontsize=9)
    ax.set_ylabel("Снижение радиуса подавления (%)")
    ax.set_title("АВРОРА и GPS: снижение радиуса подавления")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    bars2 = ax2.bar(range(len(jammers)), area_reductions,
                    color="#6c5ce7", edgecolor="white", width=0.5)
    ax2.bar_label(bars2, fmt="%.0f%%", padding=3, fontsize=9)
    ax2.set_xticks(range(len(jammers)))
    ax2.set_xticklabels(jammers, rotation=10, ha="right", fontsize=9)
    ax2.set_ylabel("Снижение площади подавления (%)")
    ax2.set_title("АВРОРА и GPS: снижение площади подавления")
    ax2.set_ylim(0, 110)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(f"АВРОРА: преимущество в помехозащищённости [{label}]", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"antijam_advantage_{label}.png"), dpi=150)
    plt.close(fig)


def _save_js_csv(js_table, jam_radii, advantage, output_dir, label) -> None:
    path = os.path.join(output_dir, f"antijam_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "jammer", "signal_dbw", "jammer_dbw_at_1km",
                    "js_eff_db", "jammed_at_1km", "jam_radius_km"])
        for (sys_name, jam_name), js in js_table.items():
            r_km = jam_radii.get(sys_name, {}).get(jam_name, 0)
            w.writerow([sys_name, jam_name,
                        f"{js['signal_dbw']:.2f}", f"{js['jammer_dbw']:.2f}",
                        f"{js['js_eff_db']:.2f}", js["jammed"],
                        f"{r_km:.2f}"])


def print_anti_jam_summary(label: str, result: Dict) -> None:
    sep = "=" * 72

    print(f"\n{sep}")
    print(f"  Анализ помехозащищённости -- {label}  "
          f"(угол места {result['elevation_deg']} град)")
    print(sep)
    print(f"  Мощность сигнала на приёмнике:")
    print(f"  {'Система':<22} {'Высота':>9} {'Сигнал дБВт':>11} {'к GPS L1':>10}")
    print("  " + "-" * 56)
    gps_ref = result["signal_powers"].get("GPS_L1", -160.0)
    for sys_name, pwr in result["signal_powers"].items():
        alt = SYSTEMS[sys_name]["altitude_km"]
        diff = pwr - gps_ref
        print(f"  {sys_name:<22} {alt:>8.0f}км {pwr:>10.1f}  {diff:>+9.1f} дБ")

    print()
    print(f"  Эффективный радиус подавления (км) по системе и джаммеру:")
    jammers = list(JAMMERS.keys())
    print(f"  {'System':<22} " + "  ".join(f"{j[:10]:>12}" for j in jammers))
    print("  " + "-" * (22 + 14 * len(jammers)))
    for sys_name in result["jam_radii_km"]:
        radii = [result["jam_radii_km"][sys_name].get(j, 0) for j in jammers]
        row   = f"  {sys_name:<22} " + "  ".join(f"{r:>12.1f}" for r in radii)
        print(row)

    print()
    print(f"  Преимущество АВРОРА L1 над GPS L1:")
    print(f"  {'Джаммер':<22} {'Радиус GPS':>12} {'Радиус АВРОРА':>14} "
          f"{'Радиус -':>10} {'Площадь -':>8}")
    print("  " + "-" * 70)
    for jam_name, adv in result["advantage"].items():
        print(f"  {jam_name:<22} {adv['r_gps_km']:>10.1f}км "
              f"{adv['r_aurora_km']:>12.1f}км "
              f"{adv['radius_reduction_pct']:>9.0f}% "
              f"{adv['area_reduction_pct']:>7.0f}%")
    print(sep)
