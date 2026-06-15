"""
Пользовательский приёмник АВРОРА: требования и характеристики.

Анализирует:
- Максимальный доплер и скорость изменения доплера (LEO vs MEO)
- Минимальная полоса PLL для слежения без потери синфазности
- Ошибка PLL vs C/N₀ и полоса
- Требуемое число каналов vs размер созвездия
- Классы приёмников: геодезический, авиационный, автомобильный, носимый

Ссылки:
  Kaplan & Hegarty (2017) — Understanding GPS/GNSS, Ch. 5, 7.
  Ward, Betz, Hegarty (2006) — Satellite Signal Acquisition, Tracking and Data Demodulation.
  RTCA DO-229E (2017) — MOPS for GPS/WAAS.
  IS-GPS-200L (2020) — GPS Interface Specification.
"""

import math, os, csv
from typing import Dict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Физические константы и параметры АВРОРА ──────────────────────────────────
C_MS       = 299792458.0
F_L1_HZ    = 1575.42e6
F_L5_HZ    = 1176.45e6
H_KM       = 1000.0
V_ORB_MS   = 7354.0
R_EARTH_KM = 6371.0

# ── Сравнительные системы ──────────────────────────────────────────────────────
SYSTEMS = {
    "GPS L1 C/A\n(MEO 20 200 км)":    {"freq": F_L1_HZ, "v_ms": 3870.0, "h_km": 20200.0, "color": "#e17055"},
    "Galileo E1\n(MEO 23 222 км)":    {"freq": F_L1_HZ, "v_ms": 3640.0, "h_km": 23222.0, "color": "#0984e3"},
    "ГЛОНАСС L1\n(MEO 19 100 км)":   {"freq": 1602e6,  "v_ms": 3950.0, "h_km": 19100.0, "color": "#6c5ce7"},
    "АВРОРА L1\n(LEO 1000 км)":       {"freq": F_L1_HZ, "v_ms": V_ORB_MS, "h_km": H_KM,  "color": "#00b894"},
    "АВРОРА L5\n(LEO 1000 км)":       {"freq": F_L5_HZ, "v_ms": V_ORB_MS, "h_km": H_KM,  "color": "#74b9ff"},
}

# ── Классы приёмников ─────────────────────────────────────────────────────────
RECEIVER_CLASSES = {
    "Геодезический / PPP": {
        "channels": 256, "track_bw_hz": 10.0,
        "acq_sens_dbhz": 28.0, "track_sens_dbhz": 20.0,
        "ant_gain_dbi": 5.0, "dual_freq": True,
        "price_range": "$3 000–10 000", "color": "#0984e3",
    },
    "Авиационный (DO-229)": {
        "channels": 64, "track_bw_hz": 15.0,
        "acq_sens_dbhz": 30.0, "track_sens_dbhz": 25.0,
        "ant_gain_dbi": 3.0, "dual_freq": True,
        "price_range": "$5 000–15 000", "color": "#e17055",
    },
    "Автомобильный": {
        "channels": 32, "track_bw_hz": 18.0,
        "acq_sens_dbhz": 33.0, "track_sens_dbhz": 28.0,
        "ant_gain_dbi": 0.0, "dual_freq": False,
        "price_range": "$20–200", "color": "#00b894",
    },
    "Носимый (смартфон)": {
        "channels": 16, "track_bw_hz": 20.0,
        "acq_sens_dbhz": 35.0, "track_sens_dbhz": 30.0,
        "ant_gain_dbi": -2.0, "dual_freq": False,
        "price_range": "$2–10", "color": "#6c5ce7",
    },
}


def doppler_max_hz(freq_hz: float, v_ms: float) -> float:
    return v_ms / C_MS * freq_hz


def doppler_rate_hz_s(freq_hz: float, h_km: float, v_ms: float) -> float:
    """Скорость изменения доплера (Гц/с). a_max = v²/r."""
    r_m = (R_EARTH_KM + h_km) * 1e3
    return (v_ms**2 / r_m) / C_MS * freq_hz


def pll_bw_min_hz(fdr: float, order: int = 3) -> float:
    """
    Минимальная полоса петли PLL для слежения без ухода фазы (3-й порядок).
    Правило: ω_n³ / (a₃·ω_n³) > ḟ_D → BW_n > (ḟ_D)^(1/3) (рад/с).
    """
    omega_n = (3.0 * fdr * 2 * math.pi) ** (1.0 / 3.0)
    return omega_n / (2 * math.pi)


def pll_range_noise_cm(bw_hz: float, cn0_dbhz: float) -> float:
    """Ошибка дальности из-за шума PLL (1σ, см)."""
    cn0 = 10 ** (cn0_dbhz / 10)
    sigma_phi_rad = math.sqrt(bw_hz / cn0)
    wavelength_m = C_MS / F_L1_HZ
    return sigma_phi_rad / (2 * math.pi) * wavelength_m * 100


def channels_needed(n_sats: int, n_freq: int = 2) -> int:
    """Число каналов для обработки видимых спутников АВРОРА."""
    f_vis = (1 - math.cos(math.radians(52.9))) / 2   # ~0.136 полусферы
    n_vis = max(1, int(n_sats * f_vis * 2))           # × 2: запас 100%
    return n_vis * n_freq


def acq_bins(freq_hz: float, v_ms: float,
             df_step: float = 500.0,
             code_chips: int = 10230,
             dtau_chip: float = 0.5) -> Dict:
    fd_max = doppler_max_hz(freq_hz, v_ms)
    n_freq_cold = int(2 * fd_max / df_step)
    n_freq_warm = int(2 * 100 / df_step)   # альманах → ±100 Гц
    n_code = int(code_chips / dtau_chip)
    return {
        "fd_max_hz":   fd_max,
        "bins_cold":   n_freq_cold * n_code,
        "bins_warm":   n_freq_warm * n_code,
        "n_freq_cold": n_freq_cold,
        "n_freq_warm": n_freq_warm,
    }


def run_user_segment_analysis(output_dir: str, label: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    fdr_l1  = doppler_rate_hz_s(F_L1_HZ, H_KM, V_ORB_MS)
    fdr_l5  = doppler_rate_hz_s(F_L5_HZ, H_KM, V_ORB_MS)
    bw_min  = pll_bw_min_hz(fdr_l1)
    bins_l1 = acq_bins(F_L1_HZ, V_ORB_MS)

    results = {
        "doppler_max_l1_khz":   doppler_max_hz(F_L1_HZ, V_ORB_MS) / 1e3,
        "doppler_max_l5_khz":   doppler_max_hz(F_L5_HZ, V_ORB_MS) / 1e3,
        "doppler_rate_l1_hz_s": fdr_l1,
        "doppler_rate_l5_hz_s": fdr_l5,
        "pll_bw_min_hz":        bw_min,
        "acq_bins_cold":        bins_l1["bins_cold"],
        "acq_bins_warm":        bins_l1["bins_warm"],
        "channels_foc_l1l5":    channels_needed(300, 2),
    }

    _plot_doppler(output_dir, label)
    _plot_channels(output_dir, label)
    _plot_pll_noise(bw_min, output_dir, label)
    _plot_receiver_classes(output_dir, label)
    _save_csv(results, output_dir, label)
    return results


def _plot_doppler(output_dir, label):
    names = list(SYSTEMS.keys())
    fds  = [doppler_max_hz(SYSTEMS[n]["freq"], SYSTEMS[n]["v_ms"]) / 1e3 for n in names]
    fdrs = [doppler_rate_hz_s(SYSTEMS[n]["freq"], SYSTEMS[n]["h_km"], SYSTEMS[n]["v_ms"])
            for n in names]
    cols = [SYSTEMS[n]["color"] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, vals, ylabel, title_sfx in [
        (ax1, fds,  "Макс. доплер ±|f_D| (кГц)", "Максимальный доплер"),
        (ax2, fdrs, "Скорость изменения доплера ḟ_D (Гц/с)", "Динамика доплера"),
    ]:
        bars = ax.bar(names, vals, color=cols, edgecolor="white", width=0.55)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals)*0.01,
                    f"{v:.1f}", ha="center", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(f"АВРОРА — {title_sfx} [{label}]")
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"user_doppler_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_channels(output_dir, label):
    n_arr = np.arange(1, 320, 4)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(n_arr, [channels_needed(n, 2) for n in n_arr],
            color="#0984e3", lw=2.5, label="L1+L5 (двухчастотный)")
    ax.plot(n_arr, [channels_needed(n, 1) for n in n_arr],
            color="#00b894", lw=2.5, ls="--", label="L1 only")
    for n_sat, phase in [(18, "Ф1"), (60, "Ф2"), (180, "Ф3"), (300, "Ф4")]:
        ax.axvline(n_sat, ls=":", color="#b2bec3", lw=1)
        ax.text(n_sat + 2, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] else 150,
                phase, fontsize=8, color="#636e72")
    for n_chan, cls, col in [(16, "Смартфон", "#fab1a0"), (32, "Авто", "#fdcb6e"),
                              (64, "Авиац.", "#e17055"), (256, "Геодез.", "#6c5ce7")]:
        ax.axhline(n_chan, ls=":", color=col, lw=1.2, label=f"{cls} ({n_chan} кан.)")
    ax.set_xlabel("Число спутников в созвездии")
    ax.set_ylabel("Требуемое число каналов приёмника")
    ax.set_title(f"АВРОРА — требования к числу каналов [{label}]")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"user_channels_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_pll_noise(bw_min, output_dir, label):
    bw_arr = np.linspace(1, 50, 300)
    cn0_levels = [35, 40, 45, 50, 55, 62]
    colors = ["#fab1a0", "#fdcb6e", "#55efc4", "#00cec9", "#0984e3", "#6c5ce7"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for cn0, col in zip(cn0_levels, colors):
        noise = np.array([pll_range_noise_cm(bw, cn0) for bw in bw_arr])
        ax.plot(bw_arr, noise, color=col, lw=2, label=f"C/N₀ = {cn0} дБ·Гц")

    ax.axvline(bw_min, ls="--", color="#d63031", lw=1.8,
               label=f"BW_min АВРОРА LEO = {bw_min:.1f} Гц")
    ax.axvline(5.0, ls=":", color="#b2bec3", lw=1.3, label="BW_min MEO = 5 Гц")
    ax.axhline(1.0, ls=":", color="#2d3436", lw=1.0, label="1 см (PPP цель)")
    ax.set_xlabel("Полоса петли PLL (Гц)")
    ax.set_ylabel("Ошибка дальности 1σ (см)")
    ax.set_ylim(0, 35)
    ax.set_title(f"АВРОРА — шум PLL-слежения vs ширина полосы [{label}]")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"user_pll_noise_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_receiver_classes(output_dir, label):
    classes = list(RECEIVER_CLASSES.keys())
    cols = [RECEIVER_CLASSES[c]["color"] for c in classes]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (key, ylabel) in zip(axes, [
        ("channels",       "Число каналов"),
        ("acq_sens_dbhz",  "C/N₀ порог захвата (дБ·Гц)"),
        ("track_sens_dbhz","C/N₀ порог слежения (дБ·Гц)"),
    ]):
        vals = [RECEIVER_CLASSES[c][key] for c in classes]
        bars = ax.barh(classes, vals, color=cols, edgecolor="white", height=0.55)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + max(vals) * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    str(v), va="center", fontsize=10)
        ax.set_xlabel(ylabel)
        ax.grid(axis="x", alpha=0.3)

    # АВРОРА C/N₀ достижимый — вертикальная линия
    axes[1].axvline(55.8, ls="--", color="#00b894", lw=1.5,
                    label="АВРОРА C/N₀ при ε=10°")
    axes[1].legend(fontsize=8)
    axes[2].axvline(55.8, ls="--", color="#00b894", lw=1.5)

    plt.suptitle(f"АВРОРА — Классы пользовательских приёмников [{label}]",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"user_classes_{label}.png"), dpi=150)
    plt.close(fig)


def _save_csv(results, output_dir, label):
    path = os.path.join(output_dir, f"user_segment_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["parameter", "value", "unit"])
        for k, v in results.items():
            w.writerow([k, f"{v:.2f}" if isinstance(v, float) else v, ""])
        for cls, p in RECEIVER_CLASSES.items():
            w.writerow([f"class_{cls}_channels", p["channels"], "кан."])
            w.writerow([f"class_{cls}_dual_freq", p["dual_freq"], ""])


def print_user_segment_summary(label: str, results: Dict) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  User Segment Analysis -- {label}")
    print(sep)
    print(f"  Доплер L1 max:        ±{results['doppler_max_l1_khz']:.1f} кГц")
    print(f"  Доплер L5 max:        ±{results['doppler_max_l5_khz']:.1f} кГц")
    print(f"  Скорость доплера L1:  {results['doppler_rate_l1_hz_s']:.1f} Гц/с")
    print(f"  Мин. полоса PLL:      {results['pll_bw_min_hz']:.1f} Гц")
    print(f"  Ячеек поиска (холод): {results['acq_bins_cold']:,}")
    print(f"  Ячеек поиска (тёпл.): {results['acq_bins_warm']:,}")
    print(f"  Каналов (FOC L1+L5):  {results['channels_foc_l1l5']}")
    print(sep)
