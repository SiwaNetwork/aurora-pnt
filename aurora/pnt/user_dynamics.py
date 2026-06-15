"""
User Dynamics and Moving Platform Accuracy for АВРОРА.

Models positioning accuracy for different vehicle types:
  - Aviation (en-route, approach)
  - Maritime (ocean, coastal, harbour)
  - Land vehicle (highway, urban)
  - Pedestrian (open sky, urban canyon)

Accounts for Doppler dynamics, multipath environment, signal blockage
geometry, and kinematic filter performance.

References:
  ICAO Annex 10 (GNSS accuracy requirements).
  IMO Resolution MSC.112(73) (maritime GNSS).
  EU-GNSS / Galileo Commercial Service SDD v1.1.
"""

import math
import os
import csv
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Physical constants
C_LIGHT = 299_792_458.0   # m/s
R_EARTH = 6_371_000.0     # m

# ── АВРОРА constellation parameters ──────────────────────────────────────────
AURORA_ALT_M = 1_000_000.0
AURORA_ORBITAL_V = math.sqrt(3.986004418e14 / (R_EARTH + AURORA_ALT_M))  # m/s

# ── Platform definitions ──────────────────────────────────────────────────────
PLATFORMS = {
    "Авиация (эшелон)": {
        "speed_mps":   250.0,    # m/s cruise speed
        "accel_mps2":  1.0,      # max maneuver acceleration
        "max_rate_dps": 3.0,     # max attitude rate (degrees/second)
        "sky_coverage":  0.6,    # fraction of sky visible (pitch + roll)
        "multipath_m":   0.5,    # typical multipath error (m)
        "n_visible_leo": 10,     # typical visible АВРОРА sats
        "pdop_typical":  2.5,
        "gnss_req_h_m":  220.0,  # ICAO RNP4 horizontal (m, 95%)
        "gnss_req_v_m":  50.0,   # en-route vertical (m, 95%)
        "color": "#0984e3",
    },
    "Авиация (заход)": {
        "speed_mps":   75.0,
        "accel_mps2":  0.5,
        "max_rate_dps": 1.0,
        "sky_coverage":  0.5,
        "multipath_m":   1.0,
        "n_visible_leo": 8,
        "pdop_typical":  3.0,
        "gnss_req_h_m":  40.0,   # APV-I approach (40 m, 95%)
        "gnss_req_v_m":  50.0,
        "color": "#6c5ce7",
    },
    "Судоходство (океан)": {
        "speed_mps":   10.0,     # ~20 knots
        "accel_mps2":  0.05,
        "max_rate_dps": 0.1,
        "sky_coverage":  0.85,
        "multipath_m":   1.5,
        "n_visible_leo": 12,
        "pdop_typical":  2.0,
        "gnss_req_h_m":  10.0,
        "gnss_req_v_m":  1000.0,
        "color": "#00b894",
    },
    "Судоходство (гавань)": {
        "speed_mps":   2.0,
        "accel_mps2":  0.02,
        "max_rate_dps": 0.05,
        "sky_coverage":  0.6,
        "multipath_m":   3.0,
        "n_visible_leo": 8,
        "pdop_typical":  3.5,
        "gnss_req_h_m":  1.0,
        "gnss_req_v_m":  1000.0,
        "color": "#fdcb6e",
    },
    "Автомобиль (трасса)": {
        "speed_mps":   30.0,
        "accel_mps2":  3.0,
        "max_rate_dps": 5.0,
        "sky_coverage":  0.7,
        "multipath_m":   2.0,
        "n_visible_leo": 9,
        "pdop_typical":  2.8,
        "gnss_req_h_m":  5.0,
        "gnss_req_v_m":  10.0,
        "color": "#e17055",
    },
    "Автомобиль (город)": {
        "speed_mps":   10.0,
        "accel_mps2":  2.0,
        "max_rate_dps": 10.0,
        "sky_coverage":  0.3,
        "multipath_m":  10.0,    # urban canyon
        "n_visible_leo": 5,
        "pdop_typical":  6.0,
        "gnss_req_h_m":  5.0,
        "gnss_req_v_m":  10.0,
        "color": "#fab1a0",
    },
}

# ── UERE components by frequency mode ────────────────────────────────────────
UERE = {
    "L1 (одночастотный)": {
        "sv_clock_m":  0.50,    # satellite clock
        "sv_orbit_m":  0.80,    # orbit determination
        "iono_m":      4.50,    # Klobuchar model residual (typical)
        "tropo_m":     0.50,    # Hopfield model residual
        "noise_m":     0.30,    # code tracking noise
        "color": "#e17055",
    },
    "L1+L5 (двухчастотный)": {
        "sv_clock_m":  0.50,
        "sv_orbit_m":  0.80,
        "iono_m":      0.10,    # dual-freq IF residual (~0.1 m)
        "tropo_m":     0.50,
        "noise_m":     0.60,    # IF combination noise amplification × √(1.26)
        "color": "#0984e3",
    },
}


def doppler_shift_hz(platform_speed_mps: float, sat_speed_mps: float,
                     freq_hz: float, geometry_factor: float = 0.5) -> float:
    """
    Maximum Doppler shift: relative velocity × geometry_factor / lambda.
    geometry_factor ≈ 0.5 for typical LEO pass (cos of angle between
    velocity vectors averaged over pass).
    """
    v_rel = platform_speed_mps + sat_speed_mps
    return v_rel * geometry_factor * freq_hz / C_LIGHT


def dynamic_tracking_noise_m(
    accel_mps2: float,
    loop_bw_hz: float = 5.0,
    freq_hz: float = 1575.42e6,
) -> float:
    """
    Carrier-phase tracking noise due to platform dynamics.
    Dynamic stress error = accel / (2π × loop_bw)²  ×  c/f  [m].
    """
    wavelength = C_LIGHT / freq_hz
    stress_err_cycles = accel_mps2 / ((2 * math.pi * loop_bw_hz) ** 2)
    return stress_err_cycles * wavelength


def position_accuracy_m(
    platform_key: str,
    uere_key: str,
    include_multipath: bool = True,
) -> Dict:
    """
    Compute 1σ and 95% position accuracy for a given platform and UERE scenario.
    """
    p = PLATFORMS[platform_key]
    u = UERE[uere_key]

    # UERE budget
    uere_components = {
        "Спутник (часы)": u["sv_clock_m"],
        "Спутник (орбита)": u["sv_orbit_m"],
        "Ионосфера": u["iono_m"],
        "Тропосфера": u["tropo_m"],
        "Шум кода": u["noise_m"],
    }

    # Platform-specific errors
    multipath_m = p["multipath_m"] if include_multipath else 0.0
    loop_noise_m = dynamic_tracking_noise_m(p["accel_mps2"])
    blockage_factor = 1.0 / p["sky_coverage"]   # degraded geometry from blockage

    uere_components["Многолучёвость"] = multipath_m
    uere_components["Дин. слежение"] = loop_noise_m

    uere_total = math.sqrt(sum(v**2 for v in uere_components.values()))

    # PDOP includes blockage degradation
    pdop = p["pdop_typical"] * blockage_factor

    # Position error
    h_1sigma = uere_total * pdop / math.sqrt(2)   # horizontal (split PDOP)
    v_1sigma = uere_total * pdop * 1.5             # vertical (typically ~1.5× worse)
    h_95 = h_1sigma * 2.45   # 95% from 2D normal (chi-squared factor)
    v_95 = v_1sigma * 1.96   # 95% from 1D normal

    # Doppler
    doppler_hz = doppler_shift_hz(p["speed_mps"], AURORA_ORBITAL_V,
                                  1575.42e6, geometry_factor=0.5)

    return {
        "platform":      platform_key,
        "uere_key":      uere_key,
        "uere_total_m":  uere_total,
        "components":    uere_components,
        "pdop":          pdop,
        "h_1sigma_m":    h_1sigma,
        "v_1sigma_m":    v_1sigma,
        "h_95_m":        h_95,
        "v_95_m":        v_95,
        "h_req_m":       p["gnss_req_h_m"],
        "v_req_m":       p["gnss_req_v_m"],
        "h_margin_m":    p["gnss_req_h_m"] - h_95,
        "h_ok":          h_95 <= p["gnss_req_h_m"],
        "doppler_hz":    doppler_hz,
        "loop_noise_m":  loop_noise_m,
        "multipath_m":   multipath_m,
    }


def run_user_dynamics_analysis(output_dir: str, label: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    uere_keys = list(UERE.keys())
    results = {}
    for uk in uere_keys:
        results[uk] = {pk: position_accuracy_m(pk, uk)
                       for pk in PLATFORMS}

    _plot_accuracy_by_platform(results, output_dir, label)
    _plot_uere_breakdown(results, output_dir, label)
    _plot_doppler(output_dir, label)
    _save_dynamics_csv(results, output_dir, label)

    return {
        "results":    results,
        "uere_keys":  uere_keys,
        "platforms":  list(PLATFORMS.keys()),
    }


def _plot_accuracy_by_platform(results, output_dir, label):
    uere_keys = list(results.keys())
    platforms  = list(PLATFORMS.keys())

    fig, axes = plt.subplots(1, len(uere_keys), figsize=(7 * len(uere_keys), 7), sharey=False)
    if len(uere_keys) == 1:
        axes = [axes]

    for ax, uk in zip(axes, uere_keys):
        h95  = [results[uk][pk]["h_95_m"] for pk in platforms]
        reqs = [PLATFORMS[pk]["gnss_req_h_m"] for pk in platforms]
        colors = [PLATFORMS[pk]["color"] for pk in platforms]
        ok_colors = ["#00b894" if results[uk][pk]["h_ok"] else "#e17055"
                     for pk in platforms]

        x = np.arange(len(platforms))
        bars = ax.bar(x, h95,  color=colors, edgecolor="white", alpha=0.85, label="CEP-95 (м)")
        ax.scatter(x, reqs, marker="D", color="black", zorder=5, s=60, label="Требование")
        ax.set_xticks(x)
        ax.set_xticklabels([p[:18] for p in platforms], rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Горизонтальная погрешность 95% (м)")
        ax.set_title(uk, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        for bar, ok_c in zip(bars, ok_colors):
            bar.set_edgecolor(ok_c)
            bar.set_linewidth(2)

    fig.suptitle(f"АВРОРА — Точность позиционирования по типу платформы [{label}]",
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"dynamics_accuracy_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_uere_breakdown(results, output_dir, label):
    uk = "L1+L5 (двухчастотный)"
    platforms = list(PLATFORMS.keys())

    components = ["Спутник (часы)", "Спутник (орбита)", "Ионосфера",
                  "Тропосфера", "Шум кода", "Многолучёвость", "Дин. слежение"]
    comp_colors = ["#0984e3", "#6c5ce7", "#e17055", "#fdcb6e",
                   "#00b894", "#fab1a0", "#74b9ff"]

    data = {}
    for comp in components:
        data[comp] = []
        for pk in platforms:
            val = results[uk][pk]["components"].get(comp, 0.0)
            data[comp].append(val)

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(platforms))
    bottoms = np.zeros(len(platforms))

    for comp, color in zip(components, comp_colors):
        vals = np.array(data[comp])
        ax.bar(x, vals**2, bottom=bottoms, color=color, edgecolor="white",
               alpha=0.85, label=comp)
        bottoms += vals**2

    # Plot total UERE as diamonds
    totals = [results[uk][pk]["uere_total_m"] for pk in platforms]
    ax2 = ax.twinx()
    ax2.plot(x, totals, "D--", color="black", ms=8, lw=2, label="UERE суммарный")
    ax2.set_ylabel("UERE суммарный (м)", color="black")

    ax.set_xticks(x)
    ax.set_xticklabels([p[:18] for p in platforms], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Вклад в дисперсию (м²)")
    ax.set_title(f"АВРОРА — Бюджет UERE по источникам и платформам [{label}]\n(L1+L5 двухчастотный)")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"dynamics_uere_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_doppler(output_dir, label):
    speed_range = np.linspace(0, 300, 200)   # m/s
    f_l1 = 1575.42e6

    fig, ax = plt.subplots(figsize=(10, 5))
    for sat_v in [0, AURORA_ORBITAL_V]:
        doppler = [(v + sat_v) * 0.5 * f_l1 / C_LIGHT for v in speed_range]
        lbl = f"Только платформа" if sat_v == 0 else f"Платформа + спутник ({AURORA_ORBITAL_V/1000:.1f} км/с)"
        ax.plot(speed_range, doppler, lw=2, label=lbl)

    for pk, p in PLATFORMS.items():
        dop = doppler_shift_hz(p["speed_mps"], AURORA_ORBITAL_V, f_l1, 0.5)
        ax.scatter([p["speed_mps"]], [dop], s=80, color=p["color"],
                   zorder=5, label=f"{pk[:16]} ({p['speed_mps']:.0f} м/с)")

    ax.set_xlabel("Скорость платформы (м/с)")
    ax.set_ylabel("Сдвиг Доплера L1 (Гц)")
    ax.set_title(f"АВРОРА — Доплеровский сдвиг L1 [{label}]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"dynamics_doppler_{label}.png"), dpi=150)
    plt.close(fig)


def _save_dynamics_csv(results, output_dir, label):
    path = os.path.join(output_dir, f"user_dynamics_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["uere_mode", "platform", "uere_m", "pdop", "h_95_m",
                    "v_95_m", "h_req_m", "h_ok", "doppler_hz"])
        for uk, plats in results.items():
            for pk, r in plats.items():
                w.writerow([uk, pk, f"{r['uere_total_m']:.2f}", f"{r['pdop']:.2f}",
                            f"{r['h_95_m']:.2f}", f"{r['v_95_m']:.2f}",
                            r["h_req_m"], r["h_ok"], f"{r['doppler_hz']:.1f}"])


def print_user_dynamics_summary(label: str, result: Dict) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  User Dynamics and Platform Accuracy -- {label}")
    print(sep)

    for uk in result["uere_keys"]:
        print(f"\n  Режим: {uk}")
        print(f"  {'Платформа':<28} {'UERE':>7} {'PDOP':>6} "
              f"{'H-95% (м)':>10} {'Треб. (м)':>10}  Статус")
        print(f"  {'':─<68}")
        for pk in result["platforms"]:
            r = result["results"][uk][pk]
            ok = "OK" if r["h_ok"] else "REVIEW"
            print(f"  {pk:<28} {r['uere_total_m']:>6.2f}  {r['pdop']:>5.1f}  "
                  f"{r['h_95_m']:>9.1f}  {r['h_req_m']:>9.0f}   [{ok}]")

    print(sep)
