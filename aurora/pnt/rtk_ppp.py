"""
PPP-RTK архитектура и анализ точности АВРОРА.

Анализирует:
- Точность PPP-RTK vs расстояния до опорной станции
- Время сходимости: классический PPP, PPP-RTK, АВРОРА PPP, АВРОРА PPP-RTK
- Архитектура сети опорных станций RSN (Reference Station Network)
- Бюджет задержки конец-в-конец (latency budget)

Ссылки:
  Teunissen & Khodabandeh (2015) — Review and Principles of PPP-RTK. J. Geodesy.
  Wübbena et al. (2005) — PPP-RTK: Precise Point Positioning Using State-Space Repr.
  Galileo HAS ICD (2022) — High Accuracy Service Interface Control Document. EC.
  Zhang et al. (2020) — LEO constellation-augmented GNSS for rapid PPP convergence. GPS Sol.
"""

import math, os, csv
from typing import Dict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Сравниваемые сервисы позиционирования ─────────────────────────────────────
SERVICES = {
    "GPS RTK": {
        "base_acc_cm":      2.0,
        "dep_cm_per_km":    1.0,    # зависимость точности от базы
        "max_base_km":      50.0,
        "conv_s":           60.0,
        "latency_ms":       50.0,
        "color": "#e17055", "ls": "-",  "marker": "o",
    },
    "GPS PPP (классич.)": {
        "base_acc_cm":      5.0,
        "dep_cm_per_km":    0.0,
        "max_base_km":      10000.0,
        "conv_s":           1500.0,
        "latency_ms":       200.0,
        "color": "#fdcb6e", "ls": "--", "marker": "s",
    },
    "Galileo PPP (HAS)": {
        "base_acc_cm":      2.0,
        "dep_cm_per_km":    0.0,
        "max_base_km":      10000.0,
        "conv_s":           300.0,
        "latency_ms":       120.0,
        "color": "#0984e3", "ls": "-.", "marker": "^",
    },
    "АВРОРА PPP": {
        "base_acc_cm":      1.5,
        "dep_cm_per_km":    0.0,
        "max_base_km":      10000.0,
        "conv_s":           30.0,
        "latency_ms":       50.0,
        "color": "#00b894", "ls": "-",  "marker": "D",
    },
    "АВРОРА PPP-RTK": {
        "base_acc_cm":      0.5,
        "dep_cm_per_km":    0.03,   # 0,03 см/км — почти не зависит от базы
        "max_base_km":      3000.0,
        "conv_s":           5.0,
        "latency_ms":       30.0,
        "color": "#6c5ce7", "ls": "-",  "marker": "*",
    },
}

# ── Бюджет задержки конец-в-конец ────────────────────────────────────────────
LATENCY_BUDGET = {
    "Сигнал АВРОРА → Земля": 3.3,
    "Обработка RSN станции":  5.0,
    "RSN → Сервер (оптика)": 20.0,
    "Сервер: расчёт SSR":    10.0,
    "SSR → Пользователь":   30.0,
    "Приёмник: обработка":    2.0,
}

RSN_SPACING_KM = 300.0   # целевой шаг сети RSN для России


def acc_vs_baseline(svc: str, base_km: np.ndarray) -> np.ndarray:
    s = SERVICES[svc]
    return s["base_acc_cm"] + s["dep_cm_per_km"] * base_km


def run_rtk_ppp_analysis(output_dir: str, label: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    conv_times = {n: s["conv_s"] for n, s in SERVICES.items()}
    total_lat  = sum(LATENCY_BUDGET.values())

    results = {
        "convergence_times_s": conv_times,
        "latency_total_ms":    total_lat,
        "rsn_spacing_km":      RSN_SPACING_KM,
    }

    _plot_accuracy_vs_baseline(output_dir, label)
    _plot_convergence(conv_times, output_dir, label)
    _plot_latency(output_dir, label)
    _plot_rsn_map(output_dir, label)
    _save_csv(results, output_dir, label)
    return results


def _plot_accuracy_vs_baseline(output_dir, label):
    bases = np.linspace(0, 500, 600)
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, s in SERVICES.items():
        mask = bases <= s["max_base_km"]
        acc  = acc_vs_baseline(name, bases[mask])
        ax.plot(bases[mask], acc,
                color=s["color"], lw=2.5, ls=s["ls"], label=name)

    ax.axhline(0.5,  ls=":",  color="#6c5ce7", lw=1.3, label="0,5 см (АВРОРА PPP-RTK цель)")
    ax.axhline(2.0,  ls=":",  color="#00b894", lw=1.0, label="2 см (геодезия)")
    ax.axhline(10.0, ls=":",  color="#fdcb6e", lw=1.0, label="10 см (авиация LPV-200)")
    ax.axhline(50.0, ls=":",  color="#dfe6e9", lw=1.0, label="50 см (авто)")
    ax.set_xlabel("Расстояние до ближайшей RSN-станции (км)")
    ax.set_ylabel("Горизонтальная точность 1σ (см)")
    ax.set_title(f"АВРОРА PPP-RTK — точность vs расстояние до станции [{label}]")
    ax.set_ylim(0, 55)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"rtk_accuracy_vs_dist_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_convergence(conv_times, output_dir, label):
    names  = list(conv_times.keys())
    t_min  = [conv_times[n] / 60 for n in names]
    colors = [SERVICES[n]["color"] for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(names, t_min, color=colors, edgecolor="white", height=0.55)
    for bar, t, name in zip(bars, t_min, names):
        lbl = f"{t*60:.0f} с" if t < 0.5 else f"{t:.1f} мин"
        ax.text(bar.get_width() + max(t_min) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                lbl, va="center", fontsize=10)
    ax.axvline(5 / 60, ls="--", color="#6c5ce7", lw=1.5,
               label="5 с (АВРОРА PPP-RTK)")
    ax.axvline(1.0,    ls=":",  color="#00b894", lw=1.3,
               label="1 мин (АВРОРА PPP)")
    ax.set_xlabel("Время сходимости (мин)")
    ax.set_title(f"Время сходимости — сравнение сервисов [{label}]")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"rtk_convergence_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_latency(output_dir, label):
    components = list(LATENCY_BUDGET.keys())
    values     = list(LATENCY_BUDGET.values())
    total      = sum(values)
    colors     = ["#00b894", "#0984e3", "#6c5ce7", "#fdcb6e", "#e17055", "#74b9ff"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    cum = np.cumsum([0] + values[:-1])
    bars = ax1.barh(components, values, left=cum, color=colors,
                    edgecolor="white", height=0.6)
    for bar, v in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_y() + bar.get_height() / 2,
                 f"{v:.0f}", ha="center", va="center", fontsize=9)
    ax1.axvline(total, ls="--", color="#2d3436", lw=1.5,
                label=f"Итого: {total:.0f} мс")
    ax1.set_xlabel("Задержка (мс)")
    ax1.set_title(f"Бюджет задержки конец-в-конец [{label}]")
    ax1.legend(fontsize=9)
    ax1.grid(axis="x", alpha=0.3)

    wedges, texts, ats = ax2.pie(values, labels=components, colors=colors,
                                  autopct="%1.0f%%", startangle=90, pctdistance=0.8)
    for t in texts: t.set_fontsize(8)
    for at in ats: at.set_fontsize(8)
    ax2.set_title(f"Доля компонент задержки\n(итого {total:.0f} мс)")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"rtk_latency_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_rsn_map(output_dir, label):
    np.random.seed(42)
    # МКС 21 станций + 8 кандидатов
    mks_lons = [37.6, 60.6, 82.9, 104.3, 131.9, 150.8,
                20.5, 50.4, 73.1, 98.7, 120.4, 143.2,
                29.9, 77.1, 109.2, 134.5, 38.0, 91.5, 162.0, 55.0, 115.0]
    mks_lats = [55.8, 53.2, 54.9, 52.1, 62.0, 59.6,
                68.5, 64.8, 69.4, 66.5, 63.8, 61.7,
                74.6, 72.9, 71.5, 73.8, 82.4, 80.1, 70.5, 68.0, 64.0]
    cand_lons = [-82.4, -35.0, 28.0, 130.9, 80.3, 106.7, 125.6, -68.3]
    cand_lats = [23.1,  -8.0, -26.2, -12.5, 13.1,  10.8, -33.9, -53.1]

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set_facecolor("#dfe6e9")

    # Покрытие RSN (стилизованное)
    for lon, lat in zip(mks_lons, mks_lats):
        circle = plt.Circle((lon, lat), RSN_SPACING_KM / 2 / 111,
                             color="#00b894", alpha=0.12, zorder=1)
        ax.add_patch(circle)
        ax.plot(lon, lat, "o", color="#00b894", ms=7, zorder=3,
                markeredgecolor="white", markeredgewidth=0.8)

    for lon, lat in zip(cand_lons, cand_lats):
        circle = plt.Circle((lon, lat), RSN_SPACING_KM / 2 / 111,
                             color="#fdcb6e", alpha=0.12, zorder=1)
        ax.add_patch(circle)
        ax.plot(lon, lat, "^", color="#fdcb6e", ms=9, zorder=3,
                markeredgecolor="white", markeredgewidth=0.8)

    ax.set_xlim(-100, 175)
    ax.set_ylim(-60, 90)
    ax.axhline(75, ls="--", color="#6c5ce7", lw=1.3, label="Наклонение АВРОРА 75°")
    ax.axhline(-75, ls="--", color="#6c5ce7", lw=1.3)
    ax.set_xlabel("Долгота (°)")
    ax.set_ylabel("Широта (°)")
    ax.set_title(
        f"АВРОРА PPP-RTK — Сеть опорных станций RSN [{label}]\n"
        f"● МКС (21 ст.) ▲ Кандидаты (8 ст.)  |  Шаг RSN = {RSN_SPACING_KM:.0f} км"
    )
    from matplotlib.patches import Patch
    legend_el = [
        Patch(facecolor="#00b894", alpha=0.5, label=f"МКС станция (зона ±{RSN_SPACING_KM/2:.0f} км)"),
        Patch(facecolor="#fdcb6e", alpha=0.5, label="Кандидат (зона)"),
    ]
    ax.legend(handles=legend_el + [
        plt.Line2D([0], [0], ls="--", color="#6c5ce7", lw=1.3, label="Наклонение 75°")
    ], fontsize=9)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"rtk_rsn_map_{label}.png"), dpi=150)
    plt.close(fig)


def _save_csv(results, output_dir, label):
    path = os.path.join(output_dir, f"rtk_ppp_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["service", "conv_time_s", "conv_time_min",
                    "base_acc_cm", "latency_ms"])
        for name, t in results["convergence_times_s"].items():
            s = SERVICES[name]
            w.writerow([name, f"{t:.1f}", f"{t/60:.2f}",
                        f"{s['base_acc_cm']:.2f}", f"{s['latency_ms']:.0f}"])
        w.writerow(["latency_total_ms", f"{results['latency_total_ms']:.1f}", "", "", ""])
        w.writerow(["rsn_spacing_km",   f"{results['rsn_spacing_km']:.0f}",   "", "", ""])


def print_rtk_ppp_summary(label: str, results: Dict) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  PPP-RTK Analysis -- {label}")
    print(sep)
    print(f"  {'Сервис':<28} {'Сход. (с)':>10} {'Сход. (мин)':>12} {'Точн. (см)':>12}")
    print(f"  {'':─<64}")
    for name, t in results["convergence_times_s"].items():
        s = SERVICES[name]
        print(f"  {name:<28} {t:>9.1f}с {t/60:>11.2f}  {s['base_acc_cm']:>10.1f}")
    print(f"  Задержка (конец-в-конец): {results['latency_total_ms']:.0f} мс")
    print(f"  Шаг RSN: {results['rsn_spacing_km']:.0f} км")
    print(sep)
