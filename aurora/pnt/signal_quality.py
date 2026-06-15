"""
Точность навигационного сигнала АВРОРА: SISA, SISRA, URE, UERE.

Метрики по аналогии с Galileo SIS ICD (OS-SIS-ICD) и GPS SPS PS:
  - SISA  (Signal In Space Accuracy) — точность эфемеридного прогноза
  - SISRA (SIS Ranging Accuracy) — RMS дальностной ошибки
  - SISMA (SIS Monitoring Accuracy)
  - URE   (User Range Error) — суммарная дальностная ошибка у пользователя
  - UERE  (User Equivalent Range Error) — проекция в позицию
  - Breakdown бюджета ошибок по источникам

Ссылки:
  Galileo OS-SIS-ICD (2021) Issue 2.0 — Signal In Space ICD.
  GPS SPS Performance Standard (2020) 5th Edition.
  Kaplan & Hegarty (2017) — Understanding GPS/GNSS: Principles and Applications.
  RTCA DO-229E (2017) — MOPS for GPS/WAAS.
"""

import math, os, csv
from typing import Dict, List
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Бюджет ошибок сигнала (1-sigma, метры) ───────────────────────────────────
# Сравнение: GPS, Galileo, АВРОРА (LEO 1000 км)

SYSTEMS: Dict[str, Dict] = {
    "GPS Block III": {
        "eph_radial_m":    0.08,
        "eph_along_m":     0.15,
        "eph_cross_m":     0.15,
        "clock_m":         0.20,
        "iono_m":          4.00,   # L1-only, Klobuchar остаток
        "tropo_m":         0.10,
        "multipath_m":     0.30,
        "receiver_noise_m":0.20,
        "color": "#e17055",
    },
    "ГЛОНАСС-K2": {
        "eph_radial_m":    0.10,   # FDMA: чуть хуже GPS по эфемеридам
        "eph_along_m":     0.20,
        "eph_cross_m":     0.20,
        "clock_m":         0.25,   # Cs ГЛОНАСС-K2; ГЛОНАСС-M: 0.40
        "iono_m":          3.50,   # L1OF only; L1+L2 dual: 0.30
        "tropo_m":         0.10,
        "multipath_m":     0.30,
        "receiver_noise_m":0.20,
        "color": "#6c5ce7",
    },
    "Galileo E1/E5a": {
        "eph_radial_m":    0.06,
        "eph_along_m":     0.12,
        "eph_cross_m":     0.12,
        "clock_m":         0.15,
        "iono_m":          2.50,
        "tropo_m":         0.10,
        "multipath_m":     0.25,
        "receiver_noise_m":0.15,
        "color": "#0984e3",
    },
    "АВРОРА L1/L5 (LEO)": {
        "eph_radial_m":    0.05,
        "eph_along_m":     0.10,
        "eph_cross_m":     0.10,
        "clock_m":         0.10,
        "iono_m":          0.08,   # dual-freq, NeQuick-2 остаток
        "tropo_m":         0.05,
        "multipath_m":     0.20,
        "receiver_noise_m":0.15,
        "color": "#00b894",
    },
}

# Проекционный коэффициент дальность→позицию
ORBITAL_FACTORS = {
    "radial":  0.98,   # радиальная ошибка сильно проецируется на дальность
    "along":   0.08,
    "cross":   0.08,
}


def ure(system_name: str) -> float:
    """
    User Range Error (URE) — RMS комбинация эфемеридной и часовой ошибок.
    URE = sqrt(σ_r² + (σ_a/n)² + (σ_c/n)² + σ_clk²)
    где n = 49 (orbital factor для along/cross ≈ 1/n).
    """
    s = SYSTEMS[system_name]
    n_fac = 49   # типовой Galileo/GPS
    return math.sqrt(
        s["eph_radial_m"]**2 +
        (s["eph_along_m"] / math.sqrt(n_fac))**2 +
        (s["eph_cross_m"] / math.sqrt(n_fac))**2 +
        s["clock_m"]**2
    )


def sisra(system_name: str) -> float:
    """SISRA = URE (одинаково по определению для SIS-only)."""
    return ure(system_name)


def sisa(system_name: str) -> float:
    """
    SISA — сигма прогноза SIS для интервала обновления навигационного сообщения.
    Добавляет маргинал прогноза эфемерид ~10%.
    """
    return ure(system_name) * 1.10


def uere_total(system_name: str) -> float:
    """UERE = RSS всех источников ошибок дальности."""
    s = SYSTEMS[system_name]
    return math.sqrt(
        ure(system_name)**2 +
        s["iono_m"]**2 +
        s["tropo_m"]**2 +
        s["multipath_m"]**2 +
        s["receiver_noise_m"]**2
    )


def uere_dual_freq(system_name: str) -> float:
    """UERE при dual-freq (ионосфера убрана dual-freq комбинацией)."""
    s = SYSTEMS[system_name]
    iono_residual = s["iono_m"] * 0.04  # остаток ~4% после IF комбинации
    return math.sqrt(
        ure(system_name)**2 +
        iono_residual**2 +
        s["tropo_m"]**2 +
        s["multipath_m"]**2 +
        s["receiver_noise_m"]**2
    )


def run_signal_quality_analysis(output_dir: str, label: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    for name in SYSTEMS:
        results[name] = {
            "ure_m":         ure(name),
            "sisa_m":        sisa(name),
            "sisra_m":       sisra(name),
            "uere_l1_m":     uere_total(name),
            "uere_dual_m":   uere_dual_freq(name),
        }

    _plot_ure_budget(results, output_dir, label)
    _plot_uere_breakdown(output_dir, label)
    _plot_sisa_comparison(results, output_dir, label)
    _save_csv(results, output_dir, label)

    return results


def _plot_ure_budget(results, output_dir, label):
    """Столбчатая диаграмма URE/SISA/UERE по системам."""
    systems = list(results.keys())
    metrics = ["ure_m", "sisa_m", "uere_dual_m"]
    metric_labels = ["URE (м)", "SISA (м)", "UERE dual-freq (м)"]
    x = np.arange(len(systems))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
        vals = [results[s][metric] for s in systems]
        bars = ax.bar(x + i * width - width, vals, width,
                      label=mlabel, alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=10)
    ax.set_ylabel("Ошибка дальности 1σ (м)")
    ax.set_title(f"АВРОРА — URE / SISA / UERE по системам [{label}]")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"signal_ure_budget_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_uere_breakdown(output_dir, label):
    """Вклад каждого источника в UERE для АВРОРА."""
    aurora = SYSTEMS["АВРОРА L1/L5 (LEO)"]
    sources = {
        "Эфемериды\n(радиал.)": aurora["eph_radial_m"],
        "Эфемериды\n(вдоль/попереч)": math.sqrt(aurora["eph_along_m"]**2 + aurora["eph_cross_m"]**2) / 7,
        "Часы": aurora["clock_m"],
        "Ионосфера\n(dual-freq)": aurora["iono_m"] * 0.04,
        "Тропосфера": aurora["tropo_m"],
        "Многолучёв.": aurora["multipath_m"],
        "Шум прим.": aurora["receiver_noise_m"],
    }
    names = list(sources.keys())
    vals  = list(sources.values())
    total = math.sqrt(sum(v**2 for v in vals))
    pcts  = [v**2 / total**2 * 100 for v in vals]

    colors = ["#0984e3", "#74b9ff", "#6c5ce7", "#00cec9", "#00b894", "#fdcb6e", "#fab1a0"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Горизонтальные бары (1-sigma вклад)
    bars = ax2.barh(names, vals, color=colors, edgecolor="white", height=0.6)
    ax2.axvline(total, ls="--", color="#2d3436", lw=1.5, label=f"Суммарный UERE = {total:.3f} м")
    for bar, v in zip(bars, vals):
        ax2.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                 f"{v:.3f} м", va="center", fontsize=9)
    ax2.set_xlabel("Ошибка 1σ (м)")
    ax2.set_title(f"Вклад источников в UERE АВРОРА [{label}]")
    ax2.legend(fontsize=9)
    ax2.grid(axis="x", alpha=0.3)

    # Пирог — доля дисперсии
    wedges, texts, autotexts = ax1.pie(pcts, labels=names, colors=colors,
                                        autopct="%1.1f%%", startangle=90, pctdistance=0.75)
    for t in texts:
        t.set_fontsize(8)
    for at in autotexts:
        at.set_fontsize(8)
    ax1.set_title(f"Доля дисперсии UERE АВРОРА [{label}]")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"signal_uere_breakdown_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_sisa_comparison(results, output_dir, label):
    """SISA vs время обновления навигационного сообщения."""
    update_intervals_min = np.linspace(1, 120, 200)

    fig, ax = plt.subplots(figsize=(10, 6))
    for sys_name, res in results.items():
        base_sisa = res["sisa_m"]
        color = SYSTEMS[sys_name]["color"]
        # SISA растёт с корнем времени (случайное блуждание часов)
        sisa_vs_time = [base_sisa * math.sqrt(1 + (t / 30.0)) for t in update_intervals_min]
        ax.plot(update_intervals_min, sisa_vs_time, color=color, lw=2, label=sys_name)

    ax.axhline(0.50, ls="--", color="#2d3436", lw=1.2, label="0.5 м (Galileo SISA-D порог)")
    ax.axhline(0.20, ls=":", color="#00b894",  lw=1.2, label="0.2 м (цель АВРОРА)")
    ax.axvline(30.0, ls="--", color="#b2bec3", lw=1.2, label="30 мин (стандарт. обновление)")
    ax.set_xlabel("Интервал обновления навигационного сообщения (мин)")
    ax.set_ylabel("SISA (м, 1σ)")
    ax.set_title(f"АВРОРА — SISA vs интервал обновления [{label}]")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"signal_sisa_vs_update_{label}.png"), dpi=150)
    plt.close(fig)


def _save_csv(results, output_dir, label):
    path = os.path.join(output_dir, f"signal_quality_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "ure_m", "sisa_m", "sisra_m", "uere_l1_m", "uere_dual_m"])
        for name, r in results.items():
            w.writerow([name,
                        f"{r['ure_m']:.4f}",
                        f"{r['sisa_m']:.4f}",
                        f"{r['sisra_m']:.4f}",
                        f"{r['uere_l1_m']:.4f}",
                        f"{r['uere_dual_m']:.4f}"])


def print_signal_quality_summary(label: str, results: Dict) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  Signal Quality Analysis -- {label}")
    print(sep)
    print(f"  {'Система':<25} {'URE':>7} {'SISA':>7} {'UERE L1':>9} {'UERE DF':>9}")
    print(f"  {'':─<64}")
    for name, r in results.items():
        print(f"  {name:<25} {r['ure_m']:>6.3f}м  {r['sisa_m']:>6.3f}м  "
              f"{r['uere_l1_m']:>7.3f}м  {r['uere_dual_m']:>7.3f}м")
    print(sep)
