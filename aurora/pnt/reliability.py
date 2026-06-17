"""
Надёжность и MTBF подсистем АВРОРА.

Модели: MIL-HDBK-217F / ECSS-Q-ST-30-C (класс S, геостационарная → LEO).
Рассчитывает:
  - MTBF каждой подсистемы, надёжность R(t) = e^(-λt) за 7 лет
  - Системная надёжность одного спутника (последовательная структура)
  - Markov-модель деградации созвездия (300 спутников, Walker Delta)
  - Необходимое число запасных спутников для поддержания 270 рабочих
  - Вероятность отказа плоскости (20 спутников в плоскости)

Ссылки:
  ECSS-Q-ST-30C (2008) — Dependability.
  MIL-HDBK-217F (1991) — Reliability Prediction of Electronic Equipment.
  Wertz & Larson (2011) — Space Mission Engineering: The New SMAD.
"""

import math, os, csv
from typing import Dict, List, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── Параметры миссии ──────────────────────────────────────────────────────────
MISSION_HOURS = 7.0 * 8760.0   # 7 лет в часах
N_SATS_TOTAL  = 300             # полное созвездие
N_SATS_MIN    = 270             # минимум для IOC-качества сервиса
N_PLANES      = 15              # орбитальных плоскостей
SATS_PER_PLANE = N_SATS_TOTAL // N_PLANES   # 20 спутников/плоскость

# ── MTBF подсистем (часы) — ECSS-Q-ST-30, класс S ────────────────────────────
SUBSYSTEMS: Dict[str, Dict] = {
    "CSAC (SA.45s)":     {"mtbf_h": 100_000, "weight": 1, "critical": True,  "color": "#e17055"},
    "space-Rb (Quantum-18)": {"mtbf_h": 100_000, "weight": 1, "critical": True,  "color": "#fdcb6e"},
    "Бортовой компьютер":{"mtbf_h": 300_000, "weight": 1, "critical": True,  "color": "#0984e3"},
    "ADCS":              {"mtbf_h": 100_000, "weight": 1, "critical": True,  "color": "#6c5ce7"},
    "Ресивер L1/L5":     {"mtbf_h": 150_000, "weight": 1, "critical": True,  "color": "#00cec9"},
    "ISL трансивер":     {"mtbf_h": 120_000, "weight": 1, "critical": False, "color": "#a29bfe"},
    "Солнечная батарея": {"mtbf_h": 150_000, "weight": 1, "critical": False, "color": "#ffeaa7"},
    "АКБ (Li-Ion)":      {"mtbf_h": 50_000,  "weight": 1, "critical": False, "color": "#fab1a0"},
    "Двигатель (ХГ)":    {"mtbf_h": 500_000, "weight": 1, "critical": False, "color": "#dfe6e9"},
    "Терморегуляция":    {"mtbf_h": 400_000, "weight": 1, "critical": False, "color": "#b2bec3"},
    "Структура":         {"mtbf_h": 1_000_000, "weight": 1,"critical": False, "color": "#636e72"},
}


def r_subsystem(name: str, hours: float = MISSION_HOURS) -> float:
    """Надёжность подсистемы R(t) = exp(-t/MTBF)."""
    mtbf = SUBSYSTEMS[name]["mtbf_h"]
    return math.exp(-hours / mtbf)


def r_satellite(hours: float = MISSION_HOURS) -> float:
    """Надёжность одного спутника (последовательная структура критических подсистем)."""
    r = 1.0
    for name, p in SUBSYSTEMS.items():
        if p["critical"]:
            r *= r_subsystem(name, hours)
    return r


def constellation_reliability(n_total: int, n_min: int, hours: float = MISSION_HOURS) -> float:
    """
    Вероятность того, что ≥ n_min спутников из n_total работают через hours часов.
    Биномиальная модель (независимые отказы).
    """
    r = r_satellite(hours)
    prob = 0.0
    for k in range(n_min, n_total + 1):
        binom = math.comb(n_total, k)
        prob += binom * (r ** k) * ((1 - r) ** (n_total - k))
    return prob


def spares_needed(n_total: int, hours: float = MISSION_HOURS,
                  confidence: float = 0.95) -> int:
    """
    Минимальное число запасных спутников для обеспечения confidence
    вероятности что > n_total - n_spare работоспособны.
    Использует логарифмическое суммирование для устойчивости к переполнению.
    """
    r = r_satellite(hours)
    lam = n_total * (1 - r)
    # Накопленная функция Пуассона через log-сумму
    log_lam = math.log(lam) if lam > 0 else -math.inf
    log_prob_k = -lam  # log P(X=0)
    cum = math.exp(log_prob_k)
    k = 0
    while cum < confidence and k < n_total:
        k += 1
        log_prob_k += log_lam - math.log(k)
        cum += math.exp(log_prob_k)
    return k


def plane_reliability(n_plane: int, n_min_plane: int,
                      hours: float = MISSION_HOURS) -> float:
    """Надёжность одной орбитальной плоскости (≥ n_min_plane работают)."""
    r = r_satellite(hours)
    prob = 0.0
    for k in range(n_min_plane, n_plane + 1):
        prob += math.comb(n_plane, k) * (r ** k) * ((1 - r) ** (n_plane - k))
    return prob


def markov_fleet_degradation(n_total: int, hours_range: np.ndarray) -> np.ndarray:
    """
    Ожидаемое число работающих спутников как функция времени.
    E[N_working(t)] = n_total × R_sat(t).
    """
    return np.array([n_total * r_satellite(h) for h in hours_range])


def run_reliability_analysis(output_dir: str, label: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    # Временная шкала
    years_range = np.linspace(0, 8, 300)
    hours_range = years_range * 8760.0

    # Надёжность подсистем за 7 лет
    subsys_r7 = {name: r_subsystem(name) for name in SUBSYSTEMS}
    subsys_mtbf = {name: SUBSYSTEMS[name]["mtbf_h"] for name in SUBSYSTEMS}

    # R(t) спутника
    r_sat_curve = np.array([r_satellite(h) for h in hours_range])

    # Деградация созвездия
    fleet_curve = markov_fleet_degradation(N_SATS_TOTAL, hours_range)

    # Запасные
    spares = spares_needed(N_SATS_TOTAL)

    # Надёжность плоскости
    plane_r = plane_reliability(SATS_PER_PLANE, SATS_PER_PLANE - 2)

    # Системная надёжность созвездия ≥ 270 через 7 лет
    constel_r7 = constellation_reliability(N_SATS_TOTAL, N_SATS_MIN)

    _plot_subsystem_mtbf(subsys_mtbf, subsys_r7, output_dir, label)
    _plot_satellite_r_vs_time(years_range, r_sat_curve, output_dir, label)
    _plot_fleet_degradation(years_range, fleet_curve, output_dir, label)
    _plot_spares_analysis(output_dir, label)
    _plot_plane_reliability(output_dir, label)
    _save_csv(subsys_r7, subsys_mtbf, spares, constel_r7, plane_r, output_dir, label)

    return {
        "subsys_r7":    subsys_r7,
        "subsys_mtbf":  subsys_mtbf,
        "r_sat_7yr":    r_satellite(),
        "fleet_7yr":    fleet_curve[-1],
        "spares":       spares,
        "constel_r7":   constel_r7,
        "plane_r":      plane_r,
    }


def _plot_subsystem_mtbf(mtbf_dict, r7_dict, output_dir, label):
    names = list(mtbf_dict.keys())
    mtbfs = [mtbf_dict[n] / 1000 for n in names]  # тыс. часов
    r7s   = [r7_dict[n] * 100 for n in names]       # %
    colors = [SUBSYSTEMS[n]["color"] for n in names]

    x = np.arange(len(names))
    fig, ax1 = plt.subplots(figsize=(13, 6))
    ax2 = ax1.twinx()
    bars = ax1.bar(x, mtbfs, color=colors, alpha=0.8, edgecolor="white", label="MTBF (тыс. ч)")
    ax2.plot(x, r7s, "o-", color="#2d3436", lw=2, ms=6, label="R(7 лет) %")
    ax2.axhline(80, ls="--", color="#e17055", lw=1, alpha=0.7, label="80% порог")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("MTBF (тыс. часов)")
    ax2.set_ylabel("Надёжность R(7 лет), %")
    ax2.set_ylim(0, 110)
    ax1.set_title(f"АВРОРА — MTBF и надёжность подсистем за 7 лет [{label}]")
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(loc="upper left", fontsize=9)
    ax2.legend(loc="upper right", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, r7s):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f"{val:.0f}%", ha="center", fontsize=7, color="#2d3436")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"reliability_subsystem_mtbf_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_satellite_r_vs_time(years_range, r_curve, output_dir, label):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(years_range, r_curve * 100, color="#0984e3", lw=2.5, label="R_sat(t)")
    ax.axvline(7.0, ls="--", color="#e17055", lw=1.5, label="7 лет (срок миссии)")
    r7 = r_satellite()
    ax.scatter([7.0], [r7 * 100], color="#e17055", s=80, zorder=5)
    ax.annotate(f"R(7 лет) = {r7*100:.1f}%", xy=(7.0, r7 * 100),
                xytext=(5.5, r7 * 100 + 4), fontsize=10,
                arrowprops=dict(arrowstyle="->", color="#2d3436"))
    ax.set_xlabel("Время (лет)")
    ax.set_ylabel("Надёжность спутника R(t), %")
    ax.set_title(f"АВРОРА — Надёжность одного спутника vs время [{label}]")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"reliability_sat_r_vs_time_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_fleet_degradation(years_range, fleet_curve, output_dir, label):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(years_range, fleet_curve, color="#00b894", lw=2.5, label="E[работающих спутников]")
    ax.axhline(N_SATS_MIN, ls="--", color="#e17055", lw=1.5,
               label=f"Минимум {N_SATS_MIN} спутников (IOC)")
    ax.axhline(N_SATS_TOTAL, ls=":", color="#b2bec3", lw=1.2,
               label=f"Полное созвездие ({N_SATS_TOTAL})")
    ax.axvline(7.0, ls="--", color="#636e72", lw=1.2, alpha=0.7)
    ax.fill_between(years_range, fleet_curve, N_SATS_MIN,
                    where=fleet_curve >= N_SATS_MIN,
                    alpha=0.15, color="#00b894", label="Зона IOC-сервиса")
    ax.fill_between(years_range, fleet_curve, N_SATS_MIN,
                    where=fleet_curve < N_SATS_MIN,
                    alpha=0.25, color="#e17055", label="Зона деградации")
    ax.set_xlabel("Время (лет)")
    ax.set_ylabel("Ожидаемое число работающих спутников")
    ax.set_title(f"АВРОРА — Деградация созвездия [{label}]")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"reliability_fleet_degradation_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_spares_analysis(output_dir, label):
    """Число необходимых запасных спутников для разных уровней надёжности."""
    confidence_levels = [0.80, 0.90, 0.95, 0.99]
    spares_vals = [spares_needed(N_SATS_TOTAL, confidence=c) for c in confidence_levels]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors_s = ["#74b9ff", "#0984e3", "#fdcb6e", "#e17055"]
    bars = ax.bar([f"{c*100:.0f}%" for c in confidence_levels],
                  spares_vals, color=colors_s, edgecolor="white", width=0.5)
    for bar, val in zip(bars, spares_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(val), ha="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("Уровень достоверности")
    ax.set_ylabel("Число запасных спутников")
    ax.set_title(f"АВРОРА — Необходимые запасы по уровню надёжности [{label}]")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"reliability_spares_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_plane_reliability(output_dir, label):
    """Надёжность плоскости как функция минимального числа рабочих спутников."""
    min_working_range = range(10, SATS_PER_PLANE + 1)
    r_plane_vals = [plane_reliability(SATS_PER_PLANE, m) * 100 for m in min_working_range]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(list(min_working_range), r_plane_vals, color="#6c5ce7", lw=2.5, marker="o", ms=5)
    ax.axvline(SATS_PER_PLANE - 2, ls="--", color="#fdcb6e", lw=1.5,
               label=f"{SATS_PER_PLANE-2} рабочих (минус 2 запас)")
    ax.axhline(95, ls=":", color="#00b894", lw=1.2, label="95% требование")
    ax.set_xlabel(f"Минимальное число рабочих спутников в плоскости (из {SATS_PER_PLANE})")
    ax.set_ylabel("Надёжность плоскости, %")
    ax.set_title(f"АВРОРА — Надёжность орбитальной плоскости через 7 лет [{label}]")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"reliability_plane_{label}.png"), dpi=150)
    plt.close(fig)


def _save_csv(subsys_r7, subsys_mtbf, spares, constel_r7, plane_r, output_dir, label):
    path = os.path.join(output_dir, f"reliability_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subsystem", "mtbf_hours", "r_7yr_pct", "critical"])
        for name in subsys_mtbf:
            w.writerow([name, subsys_mtbf[name],
                        f"{subsys_r7[name]*100:.2f}",
                        SUBSYSTEMS[name]["critical"]])
        w.writerow([])
        w.writerow(["metric", "value"])
        w.writerow(["r_satellite_7yr", f"{r_satellite()*100:.2f}%"])
        w.writerow(["constellation_r7yr_ge270", f"{constel_r7*100:.3f}%"])
        w.writerow(["plane_reliability_7yr", f"{plane_r*100:.2f}%"])
        w.writerow(["spares_needed_95pct", spares])


def print_reliability_summary(label: str, result: Dict) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  Reliability Analysis -- {label}")
    print(sep)
    print(f"  {'Подсистема':<25} {'MTBF':>10} {'R(7 лет)':>10} {'Критич.'}")
    print(f"  {'':─<66}")
    for name in result["subsys_mtbf"]:
        r = result["subsys_r7"][name]
        mtbf = result["subsys_mtbf"][name]
        crit = "ДА" if SUBSYSTEMS[name]["critical"] else "—"
        print(f"  {name:<25} {mtbf:>8,} ч  {r*100:>7.1f}%  {crit}")
    print()
    print(f"  R(спутник, 7 лет):          {result['r_sat_7yr']*100:.1f}%")
    print(f"  R(созвездие ≥ 270, 7 лет):  {result['constel_r7']*100:.2f}%")
    print(f"  Запасных спутников (95%):   {result['spares']}")
    print(f"  R(плоскость ≥ 18, 7 лет):   {result['plane_r']*100:.1f}%")
    print(sep)
