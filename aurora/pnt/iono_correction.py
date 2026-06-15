"""
Ionospheric Correction Analysis for АВРОРА.

Compares residual ionospheric delay after applying:
  1. Klobuchar (GPS-compatible single-freq, 8 coefficients)
  2. NeQuick-G (Galileo model, 3 az-coefficients)
  3. Dual-frequency L1+L5 ionosphere-free combination

Models TEC diurnal variation, solar activity dependence, and the
correction accuracy as a function of latitude and solar conditions.

References:
  Klobuchar (1987) IEEE Trans. AES; IS-GPS-200 (ICD, 2022).
  Galileo OS SIS ICD 2.1 — NeQuick-G.
  ITU-R P.531-14 (ionospheric propagation).
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
C_LIGHT    = 299_792_458.0   # m/s
F_L1_HZ    = 1575.42e6       # Hz
F_L5_HZ    = 1176.45e6       # Hz

# Ionospheric parameters
TECU = 1e16   # TEC unit → electrons/m²
K_IONO = 40.3   # m/TECU at 1 Hz (plasma dispersion constant)

# ── Signal frequencies ────────────────────────────────────────────────────────
AURORA_SIGNALS = {
    "L1": {"freq_hz": F_L1_HZ, "wavelength_m": C_LIGHT / F_L1_HZ},
    "L5": {"freq_hz": F_L5_HZ, "wavelength_m": C_LIGHT / F_L5_HZ},
}

# ── Typical TEC values by solar condition and latitude ────────────────────────
# TEC in TECU (1 TECU = 10^16 el/m²)
TEC_CONDITIONS = {
    "Спокойное (мин. солн.)":  {"lat_tec": {0: 5, 20: 8, 40: 6, 60: 4, 90: 2},  "solar_index": 70},
    "Среднее":                  {"lat_tec": {0: 20, 20: 35, 40: 25, 60: 12, 90: 5}, "solar_index": 130},
    "Активное (макс. солн.)":   {"lat_tec": {0: 60, 20: 80, 40: 50, 60: 20, 90: 8}, "solar_index": 220},
    "Бури (K>5)":               {"lat_tec": {0: 100, 20: 120, 40: 80, 60: 40, 90: 15}, "solar_index": 250},
}

# ── Model correction accuracy ─────────────────────────────────────────────────
# Residual error as fraction of actual delay, by model and condition
MODEL_ACCURACY = {
    "Клобухар (L1)": {
        "residual_fraction": 0.50,   # removes ~50% of delay on average
        "rms_fraction":      0.60,   # RMS residual / TEC
        "latency_s":         0,
        "color": "#e17055",
        "desc": "GPS-compatible 8-coefficient model",
    },
    "NeQuick-G (L1)": {
        "residual_fraction": 0.30,
        "rms_fraction":      0.40,
        "latency_s":         0,
        "color": "#0984e3",
        "desc": "Galileo 3-coefficient model (broadcast)",
    },
    "L1+L5 ИБ (ионосф. комб.)": {
        "residual_fraction": 0.002,  # ~0.2% residual due to higher-order terms
        "rms_fraction":      0.003,
        "latency_s":         0,
        "color": "#00b894",
        "desc": "Dual-frequency ionosphere-free combination",
    },
    "SBAS/SDCM": {
        "residual_fraction": 0.15,
        "rms_fraction":      0.20,
        "latency_s":         6.0,
        "color": "#6c5ce7",
        "desc": "Ground-based augmentation grid (≤6 s latency)",
    },
}

AURORA_ALT_M = 1_000_000.0


def iono_delay_m(tec_tecu: float, freq_hz: float) -> float:
    """Ionospheric delay (metres) for given TEC and signal frequency."""
    tec_el_m2 = tec_tecu * TECU
    return K_IONO * tec_el_m2 / freq_hz**2


def iono_free_combination_noise_m(sigma_l1_m: float, sigma_l5_m: float) -> float:
    """
    Noise amplification of dual-frequency ionosphere-free combination.

    IF = (f1² · L1 - f2² · L5) / (f1² - f2²)
    Noise: σ_IF = sqrt((f1/(f1²-f2²))² + (f2/(f1²-f2²))²) × σ_code
    """
    f1, f2 = F_L1_HZ, F_L5_HZ
    denom  = f1**2 - f2**2
    alpha  = f1**2 / denom
    beta   = f2**2 / denom
    return math.sqrt(alpha**2 * sigma_l1_m**2 + beta**2 * sigma_l5_m**2)


def tec_at_elevation(tec_zenith: float, elevation_deg: float) -> float:
    """
    Slant TEC as function of elevation angle (mapping function).
    STEC = TEC_zenith / sin(elevation)  (simplified thin-shell model)
    """
    el_rad = math.radians(max(elevation_deg, 5.0))   # avoid division by zero
    return tec_zenith / math.sin(el_rad)


def residual_error_m(
    tec_tecu: float,
    model_key: str,
    freq_key: str = "L1",
    elevation_deg: float = 30.0,
) -> float:
    """
    Residual ionospheric error after applying correction model.

    Returns RMS residual in metres.
    """
    model = MODEL_ACCURACY[model_key]
    stec  = tec_at_elevation(tec_tecu, elevation_deg)
    full_delay_m = iono_delay_m(stec, AURORA_SIGNALS[freq_key]["freq_hz"])

    if "ИБ" in model_key or "IF" in model_key or "ionosph" in model_key.lower():
        # Dual-freq: residual due to higher-order terms only
        code_noise_m = 0.30   # code ranging noise
        sigma_l5_m   = 0.30
        # IF combination amplifies noise
        if_noise_m = iono_free_combination_noise_m(code_noise_m, sigma_l5_m)
        return max(full_delay_m * model["residual_fraction"], if_noise_m)
    else:
        return full_delay_m * model["rms_fraction"]


def run_iono_analysis(output_dir: str, label: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    elevations = list(range(5, 91, 5))
    lat_deg    = list(range(0, 91, 10))

    # ── Delay vs TEC for both signals ─────────────────────────────────────────
    tec_range = np.linspace(0, 120, 300)
    delay_l1  = [iono_delay_m(t * TECU / TECU, F_L1_HZ) for t in tec_range]
    delay_l5  = [iono_delay_m(t * TECU / TECU, F_L5_HZ) for t in tec_range]

    # ── Residuals by model and condition ──────────────────────────────────────
    results = {}
    for cond_key, cond in TEC_CONDITIONS.items():
        results[cond_key] = {}
        for model_key in MODEL_ACCURACY:
            errs = []
            for el in elevations:
                tec = cond["lat_tec"].get(30, 20)   # midlatitude reference
                err = residual_error_m(tec, model_key, "L1", el)
                errs.append(err)
            results[cond_key][model_key] = errs

    # ── Latitude profile ──────────────────────────────────────────────────────
    lat_profile = {}
    for cond_key, cond in TEC_CONDITIONS.items():
        lat_tec = [np.interp(lat, list(cond["lat_tec"].keys()),
                              list(cond["lat_tec"].values()))
                   for lat in lat_deg]
        lat_profile[cond_key] = lat_tec

    # ── IF noise amplification ────────────────────────────────────────────────
    code_noise_range = np.linspace(0.1, 1.0, 50)
    if_noise = [iono_free_combination_noise_m(s, s) for s in code_noise_range]

    _plot_iono_delay(tec_range, delay_l1, delay_l5, output_dir, label)
    _plot_residuals_vs_elevation(elevations, results, output_dir, label)
    _plot_lat_tec_profile(lat_deg, lat_profile, output_dir, label)
    _plot_model_comparison(results, output_dir, label)
    _save_iono_csv(elevations, results, output_dir, label)

    # Summary stats at 30° elevation, middle condition
    cond_ref = "Среднее"
    summary = {}
    for mk in MODEL_ACCURACY:
        tec  = TEC_CONDITIONS[cond_ref]["lat_tec"].get(30, 20)
        err  = residual_error_m(tec, mk, "L1", 30.0)
        summary[mk] = {"err_m": err, "err_ns": err / C_LIGHT * 1e9}

    return {
        "results":      results,
        "lat_profile":  lat_profile,
        "if_noise":     if_noise,
        "code_noise":   code_noise_range.tolist(),
        "summary":      summary,
        "elevations":   elevations,
    }


def _plot_iono_delay(tec_range, delay_l1, delay_l5, output_dir, label):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(tec_range, delay_l1, color="#e17055", lw=2, label="L1 (1575 МГц)")
    ax.plot(tec_range, delay_l5, color="#0984e3", lw=2, label="L5 (1176 МГц)")
    ax.axvline(20,  ls=":", color="#6c5ce7", lw=1.2, label="Среднее TEC (20 TECU)")
    ax.axvline(80,  ls=":", color="#fdcb6e", lw=1.2, label="Бури (80 TECU)")
    ax.set_xlabel("TEC (TECU, 1 TECU = 10¹⁶ эл/м²)")
    ax.set_ylabel("Ионосферная задержка (м)")
    ax.set_title(f"АВРОРА — Ионосферная задержка от TEC [{label}]")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"iono_delay_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_residuals_vs_elevation(elevations, results, output_dir, label):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax_i, (cond_key, cond_res) in enumerate(results.items()):
        ax = axes[ax_i]
        for mk, errs in cond_res.items():
            c = MODEL_ACCURACY[mk]["color"]
            ax.plot(elevations, errs, color=c, lw=2, label=mk)
        ax.axhline(0.50, ls="--", color="gray", lw=0.8, label="0.5 м порог")
        ax.set_xlabel("Угол места (°)")
        ax.set_ylabel("Остаточная ошибка (м, σ)")
        ax.set_title(cond_key)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.set_xlim(5, 90)

    fig.suptitle(f"АВРОРА — Остаточная ионосферная ошибка [{label}]", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"iono_residuals_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_lat_tec_profile(lat_deg, lat_profile, output_dir, label):
    fig, ax = plt.subplots(figsize=(10, 5))
    for cond_key, tec_vals in lat_profile.items():
        color = {"Спокойное (мин. солн.)": "#00b894",
                 "Среднее": "#0984e3",
                 "Активное (макс. солн.)": "#e17055",
                 "Бури (K>5)": "#6c5ce7"}.get(cond_key, "black")
        ax.plot(lat_deg, tec_vals, lw=2, label=cond_key, color=color)
    ax.set_xlabel("Широта (°)")
    ax.set_ylabel("Вертикальный TEC (TECU)")
    ax.set_title(f"АВРОРА — Широтный профиль TEC по условиям [{label}]")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"iono_lat_tec_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_model_comparison(results, output_dir, label):
    cond_keys  = list(results.keys())
    model_keys = list(MODEL_ACCURACY.keys())
    el_ref     = 6   # index → 30° elevation in elevations list (5,10,...,30)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(cond_keys))
    w = 0.18
    for i, mk in enumerate(model_keys):
        errs = [results[ck][mk][el_ref] for ck in cond_keys]
        ax.bar(x + i * w, errs, w, label=mk, color=MODEL_ACCURACY[mk]["color"],
               edgecolor="white")
    ax.set_xticks(x + w * (len(model_keys) - 1) / 2)
    ax.set_xticklabels(cond_keys, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Остаточная ошибка при 30° (м, σ)")
    ax.set_title(f"АВРОРА — Сравнение моделей коррекции ионосферы [{label}]")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"iono_model_compare_{label}.png"), dpi=150)
    plt.close(fig)


def _save_iono_csv(elevations, results, output_dir, label):
    path = os.path.join(output_dir, f"iono_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "model", "elevation_deg", "residual_m"])
        for cond_key, cond_res in results.items():
            for mk, errs in cond_res.items():
                for el, err in zip(elevations, errs):
                    w.writerow([cond_key, mk, el, f"{err:.3f}"])


def print_iono_summary(label: str, result: Dict) -> None:
    sep = "=" * 68
    print(f"\n{sep}")
    print(f"  Ionospheric Correction Analysis -- {label}")
    print(sep)
    print(f"  Сигналы: L1 ({F_L1_HZ/1e6:.2f} МГц), L5 ({F_L5_HZ/1e6:.2f} МГц)")
    print()
    print(f"  Задержки при среднем TEC=20 TECU, угол места 30°:")
    tec_ref = 20
    for sig_key, sig in AURORA_SIGNALS.items():
        stec = tec_at_elevation(tec_ref, 30.0)
        d = iono_delay_m(stec, sig["freq_hz"])
        print(f"    {sig_key}: {d:.2f} м  ({d/C_LIGHT*1e9:.2f} нс)")
    print()
    print(f"  Остаточная ошибка при среднем TEC, угол места 30°:")
    for mk, s in result["summary"].items():
        print(f"    {mk:<35} {s['err_m']:>6.3f} м  ({s['err_ns']:.3f} нс)")
    print()
    if_amp = iono_free_combination_noise_m(0.30, 0.30)
    print(f"  Амплификация шума L1+L5 ИБ: {if_amp:.2f} м  (код σ=0.30 м)")
    print(sep)
