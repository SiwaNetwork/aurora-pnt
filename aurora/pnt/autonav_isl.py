"""
Автономная навигация созвездия по межспутниковым измерениям (AutoNav / ISL).

Анализирует:
- Псевдодальность по Ka-линии ISL: Δρ_ij = |r_i - r_j| + c·Δδt, σ_range ≈ 0.05 м
- Проблему ранговой недостаточности созвездия "только ISL": неустранимый
  поворот созвездия как твёрдого тела (3 степени свободы — ориентация)
  без хотя бы одного наземного якоря или абсолютной привязки ориентации
- Рост ошибки эфемерид без наземного контакта: ε(t) ≈ ε0 + k·t
  (ε0≈0.1 м, k≈1.0 м/сутки) и его ограничение при периодической
  наземной привязке (раз в 1/3/7 суток)
- Сравнение точности орбиты/часов: наземный режим vs автономный

Ссылки:
  Ananda et al. (1990) — Autonomous navigation of GPS (AutoNav). ION GPS.
  Rajan (2002) — Highly Autonomous GPS using crosslink ranging. ION GPS.
  Fernández (2011) — Inter-satellite ranging and rank deficiency. Acta Astron.
  Sánchez et al. (2020) — Autonomous orbit determination for LEO PNT. NAVIGATION.
"""

import math, os, csv
from typing import Dict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = ["#e17055", "#fdcb6e", "#0984e3", "#00b894",
           "#6c5ce7", "#74b9ff", "#dfe6e9", "#2d3436"]

C_LIGHT = 299792458.0   # м/с
ISL_FREQ_GHZ = 26.0     # Ka-диапазон ISL
ISL_RANGE_SIGMA_M = 0.05

# Walker Delta 300/15: 15 плоскостей × 20 спутников
N_PLANES = 15
N_PER_PLANE = 20

# ── Рост ошибки эфемерид ─────────────────────────────────────────────────────
EPS0_M = 0.1            # начальная ошибка эфемерид, м
DRIFT_M_PER_DAY = 1.0   # дрейф без наземного контакта, м/сутки

# Сценарии периодичности наземной привязки (сутки между привязками; None=нет)
ANCHOR_SCENARIOS = {
    "Без наземной привязки":     {"period_d": None, "color": "#e17055"},
    "Наземная привязка раз в 7 сут": {"period_d": 7.0, "color": "#fdcb6e"},
    "Наземная привязка раз в 3 сут": {"period_d": 3.0, "color": "#0984e3"},
    "Наземная привязка раз в 1 сут": {"period_d": 1.0, "color": "#00b894"},
}

# ── Наблюдаемость: степени свободы ───────────────────────────────────────────
# Только ISL: неустранимы 3 DOF поворота созвездия (rigid-body rotation).
# Масштаб/трансляция фиксируются законами динамики; абсолютная ориентация — нет.
DOF_TOTAL = 6           # 3 трансляции + 3 поворота (привязка к ИСК)
DOF_RANK_DEFICIT_ISL_ONLY = 3   # неустранимый поворот созвездия

# ── Сравнение точности: наземный vs автономный (м) ──────────────────────────
ACCURACY_COMPARISON = {
    "Наземный (POD)":          {"orbit_7d": 0.05, "orbit_30d": 0.05,
                                "clock_7d": 0.04, "clock_30d": 0.04},
    "Автономный (ISL+якорь)":  {"orbit_7d": 0.30, "orbit_30d": 0.50,
                                "clock_7d": 0.15, "clock_30d": 0.25},
    "Автономный (только ISL)": {"orbit_7d": 7.0, "orbit_30d": 30.0,
                                "clock_7d": 2.0, "clock_30d": 8.0},
}


def _eph_error(t_days: np.ndarray, period_d) -> np.ndarray:
    """ε(t)=ε0+k·t; при наземной привязке раз в period_d ошибка сбрасывается."""
    if period_d is None:
        return EPS0_M + DRIFT_M_PER_DAY * t_days
    t_since = np.mod(t_days, period_d)
    return EPS0_M + DRIFT_M_PER_DAY * t_since


def run_autonav_analysis(output_dir: str, label: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    eph_at = {}
    for name, s in ANCHOR_SCENARIOS.items():
        p = s["period_d"]
        eph_at[name] = {
            "d1":  float(_eph_error(np.array([1.0]), p)[0]),
            "d7":  float(_eph_error(np.array([7.0]), p)[0]),
            "d14": float(_eph_error(np.array([14.0]), p)[0]),
        }

    results = {
        "isl_range_sigma_m": ISL_RANGE_SIGMA_M,
        "isl_freq_ghz": ISL_FREQ_GHZ,
        "drift_m_per_day": DRIFT_M_PER_DAY,
        "eps0_m": EPS0_M,
        "dof_total": DOF_TOTAL,
        "rank_deficit_isl_only": DOF_RANK_DEFICIT_ISL_ONLY,
        "eph_error_m": eph_at,
        "accuracy_comparison": ACCURACY_COMPARISON,
    }

    _plot_eph_growth(output_dir, label)
    _plot_rank(output_dir, label)
    _plot_isl_geometry(output_dir, label)
    _plot_vs_ground(output_dir, label)
    _save_csv(results, output_dir, label)
    return results


def _plot_eph_growth(output_dir, label):
    t = np.linspace(0, 14, 1400)
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, s in ANCHOR_SCENARIOS.items():
        eps = _eph_error(t, s["period_d"])
        ax.plot(t, eps, color=s["color"], lw=2.2, label=name)
    ax.axhline(1.0, ls=":", color="#6c5ce7", lw=1.3, label="1 м (требование)")
    ax.set_xlabel("Время без переопределения POD (сутки)")
    ax.set_ylabel("Ошибка эфемерид (м)")
    ax.set_title(f"AutoNav — рост ошибки эфемерид (дрейф "
                 f"{DRIFT_M_PER_DAY:.1f} м/сут) [{label}]")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 14.5)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"autonav_eph_growth_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_rank(output_dir, label):
    configs = ["Только ISL", "ISL + 1 наземный якорь", "ISL + абс. ориентация"]
    observable = [DOF_TOTAL - DOF_RANK_DEFICIT_ISL_ONLY, DOF_TOTAL, DOF_TOTAL]
    unobservable = [DOF_RANK_DEFICIT_ISL_ONLY, 0, 0]

    x = np.arange(len(configs))
    fig, ax = plt.subplots(figsize=(11, 6))
    b1 = ax.bar(x, observable, 0.5, label="Наблюдаемые DOF",
                color="#00b894", edgecolor="white")
    b2 = ax.bar(x, unobservable, 0.5, bottom=observable,
                label="Ненаблюдаемые DOF (поворот созвездия)",
                color="#e17055", edgecolor="white")
    for i in range(len(configs)):
        ax.text(x[i], observable[i] / 2, f"{observable[i]}",
                ha="center", va="center", fontsize=11, color="white",
                fontweight="bold")
        if unobservable[i] > 0:
            ax.text(x[i], observable[i] + unobservable[i] / 2,
                    f"{unobservable[i]}", ha="center", va="center",
                    fontsize=11, color="white", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=10)
    ax.set_ylabel("Степени свободы привязки к ИСК")
    ax.set_ylim(0, DOF_TOTAL + 1)
    ax.set_title(f"AutoNav — ранговая недостаточность созвездия "
                 f"(дефицит ранга = {DOF_RANK_DEFICIT_ISL_ONLY}) [{label}]")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"autonav_rank_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_isl_geometry(output_dir, label):
    """Топология ISL опорного КА в Walker 300/15 (15 плоскостей × 20 КА).

    Решётка «плоскость × фаза». Опорный КА (плоскость 7, фаза 10) и его
    4 ISL-связи: внутриплоскостные (±1 фаза, та же плоскость) и
    межплоскостные (±1 плоскость, ближайшая фаза). Дальности считаются
    из геометрии Walker на h=1000 км:
      R = R_E + h; хорда при угле θ:  d = 2R·sin(θ/2).
      внутриплоскостной шаг по орбите: 360/20 = 18°;
      межплоскостной разнос соседних плоскостей: 360/15 = 24°.
    """
    RE_KM = 6378.137
    H_KM = 1000.0
    R = RE_KM + H_KM                       # км, радиус орбитального кольца

    def chord(theta_deg):
        return 2.0 * R * math.sin(math.radians(theta_deg) / 2.0)

    intra_deg = 360.0 / N_PER_PLANE        # 18° между соседями в плоскости
    inter_deg = 360.0 / N_PLANES           # 24° между соседними плоскостями
    d_intra = chord(intra_deg)             # ~2306 км
    d_inter = chord(inter_deg)             # ~3070 км

    sat_plane = 7
    sat_phase = 10

    INTRA_C = "#00b894"
    INTER_C = "#0984e3"
    REF_C = "#6c5ce7"
    WRAP_C = "#e17055"

    fig, ax = plt.subplots(figsize=(13, 8))

    # ── решётка всех КА ──────────────────────────────────────────────────
    pg, fg = np.meshgrid(np.arange(N_PLANES), np.arange(N_PER_PLANE))
    ax.scatter(pg.ravel(), fg.ravel(), s=22, color="#b2bec3",
               edgecolor="white", linewidth=0.3, zorder=1)

    # ── опорный КА ───────────────────────────────────────────────────────
    ax.scatter([sat_plane], [sat_phase], s=320, color=REF_C,
               edgecolor="white", linewidth=1.5, zorder=6)
    ax.annotate(f"Опорный КА\n(плоск. {sat_plane}, фаза {sat_phase})",
                xy=(sat_plane, sat_phase), xytext=(sat_plane - 4.2, sat_phase + 4.0),
                fontsize=9.5, fontweight="bold", color=REF_C,
                arrowprops=dict(arrowstyle="->", color=REF_C, lw=1.2))

    def link(x0, y0, x1, y1, color, wrap=False):
        style = "arc3,rad=0.32" if wrap else "arc3,rad=0.0"
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="<|-|>", color=color,
                                    lw=3.0 if not wrap else 2.2,
                                    connectionstyle=style,
                                    linestyle="--" if wrap else "-"),
                    zorder=4)

    # ── внутриплоскостные ISL (±1 фаза, та же плоскость) ────────────────
    for dph in (-1, 1):
        nph = (sat_phase + dph) % N_PER_PLANE
        wrap = (sat_phase + dph) != nph        # перенос фаза 0<->19
        link(sat_plane, sat_phase, sat_plane, nph, INTRA_C, wrap=wrap)
        ax.scatter([sat_plane], [nph], s=120, color=INTRA_C,
                   edgecolor="white", linewidth=1.0, zorder=5)
    ax.annotate(f"≈ {d_intra:.0f} км",
                xy=(sat_plane + 0.15, sat_phase + 0.5), fontsize=9,
                color=INTRA_C, fontweight="bold")

    # ── межплоскостные ISL (±1 плоскость, ближайшая фаза) ───────────────
    for dp in (-1, 1):
        npl = (sat_plane + dp) % N_PLANES
        link(sat_plane, sat_phase, npl, sat_phase, INTER_C)
        ax.scatter([npl], [sat_phase], s=120, color=INTER_C,
                   edgecolor="white", linewidth=1.0, zorder=5)
    ax.annotate(f"≈ {d_inter:.0f} км",
                xy=(sat_plane + 1.0, sat_phase - 1.4), fontsize=9,
                color=INTER_C, fontweight="bold")

    # ── иллюстрация переноса фазы 0 <-> 19 (кольцевая топология) ─────────
    ax.annotate("", xy=(sat_plane, 0), xytext=(sat_plane, N_PER_PLANE - 1),
                arrowprops=dict(arrowstyle="<|-|>", color=WRAP_C, lw=1.6,
                                linestyle="--",
                                connectionstyle="arc3,rad=0.45"), zorder=3)
    ax.annotate("перенос фазы 0↔19\n(кольцо орбиты)",
                xy=(sat_plane + 0.6, N_PER_PLANE / 2.0),
                fontsize=8.5, color=WRAP_C, style="italic")

    # ── легенда ──────────────────────────────────────────────────────────
    from matplotlib.lines import Line2D
    legend_el = [
        Line2D([0], [0], color=REF_C, marker="o", ls="", ms=11,
               markeredgecolor="white", label="Опорный КА (плоск.7, фаза 10)"),
        Line2D([0], [0], color=INTRA_C, lw=3.0,
               label=f"Внутриплоскостные ISL ±1 фаза  (≈{d_intra:.0f} км)"),
        Line2D([0], [0], color=INTER_C, lw=3.0,
               label=f"Межплоскостные ISL ±1 плоскость  (≈{d_inter:.0f} км)"),
        Line2D([0], [0], color=WRAP_C, lw=1.6, ls="--",
               label="Кольцевой перенос фазы 0↔19"),
    ]
    ax.legend(handles=legend_el, fontsize=9, loc="upper right",
              framealpha=0.95)

    # ── текстовый блок-сводка ────────────────────────────────────────────
    txt = ("Сводка ISL опорного КА:\n"
           f"• число ISL-связей на КА: 4\n"
           f"• тип: 2 внутриплоскостные + 2 межплоскостные\n"
           f"• дальность внутри плоскости: ≈ {d_intra:.0f} км ({intra_deg:.0f}°)\n"
           f"• дальность между плоскостями: ≈ {d_inter:.0f} км ({inter_deg:.0f}°)\n"
           f"• радиус кольца R = R_E+{H_KM:.0f} = {R:.0f} км")
    ax.text(0.015, 0.025, txt, transform=ax.transAxes, fontsize=8.8,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f6fa",
                      edgecolor="#b2bec3", alpha=0.95))

    ax.set_xlabel("Номер орбитальной плоскости (0…14)")
    ax.set_ylabel("Фазовый индекс КА в плоскости (0…19)")
    ax.set_title(f"AutoNav — топология ISL опорного КА "
                 f"(Walker {N_PLANES*N_PER_PLANE}/{N_PLANES}) [{label}]",
                 fontsize=12)
    ax.set_xlim(-1.5, N_PLANES + 0.5)
    ax.set_ylim(-1.5, N_PER_PLANE + 4.0)
    ax.set_xticks(np.arange(0, N_PLANES))
    ax.set_yticks(np.arange(0, N_PER_PLANE, 2))
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"autonav_isl_geometry_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_vs_ground(output_dir, label):
    modes = list(ACCURACY_COMPARISON.keys())
    metrics = [("orbit_7d", "Орбита 7 сут"), ("orbit_30d", "Орбита 30 сут"),
               ("clock_7d", "Часы 7 сут"), ("clock_30d", "Часы 30 сут")]
    colors = ["#00b894", "#0984e3", "#fdcb6e", "#e17055"]

    x = np.arange(len(modes))
    width = 0.2

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (key, mlabel) in enumerate(metrics):
        vals = [ACCURACY_COMPARISON[m][key] for m in modes]
        bars = ax.bar(x + (i - 1.5) * width, vals, width,
                      label=mlabel, color=colors[i], edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v * 1.1,
                    f"{v:g}", ha="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(modes, fontsize=10)
    ax.set_ylabel("Ошибка 1σ (м, лог. шкала)")
    ax.set_title(f"AutoNav — точность: наземный vs автономный режим [{label}]")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y", which="both")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"autonav_vs_ground_{label}.png"), dpi=150)
    plt.close(fig)


def _save_csv(results, output_dir, label):
    path = os.path.join(output_dir, f"autonav_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "eph_err_1d_m", "eph_err_7d_m", "eph_err_14d_m"])
        for name, e in results["eph_error_m"].items():
            w.writerow([name, f"{e['d1']:.3f}", f"{e['d7']:.3f}",
                        f"{e['d14']:.3f}"])
        w.writerow([])
        w.writerow(["param", "value"])
        w.writerow(["isl_range_sigma_m", f"{results['isl_range_sigma_m']:.3f}"])
        w.writerow(["isl_freq_ghz", f"{results['isl_freq_ghz']:.1f}"])
        w.writerow(["drift_m_per_day", f"{results['drift_m_per_day']:.2f}"])
        w.writerow(["dof_total", results["dof_total"]])
        w.writerow(["rank_deficit_isl_only", results["rank_deficit_isl_only"]])


def print_autonav_summary(label: str, results: Dict) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  AutoNav (ISL-based autonomous navigation) -- {label}")
    print(sep)
    print(f"  ISL: Ka {results['isl_freq_ghz']:.0f} ГГц, "
          f"sigma_range = {results['isl_range_sigma_m']*100:.0f} см")
    print(f"  Дрейф эфемерид без привязки: {results['drift_m_per_day']:.1f} м/сут "
          f"(eps0={results['eps0_m']:.1f} м)")
    print(f"  Степени свободы: всего {results['dof_total']}, "
          f"дефицит ранга (поворот созвездия) = "
          f"{results['rank_deficit_isl_only']}")
    print(f"  {'':-<66}")
    print(f"  {'Сценарий':<34}{'1 сут':>9}{'7 сут':>9}{'14 сут':>9}")
    print(f"  {'':-<66}")
    for name, e in results["eph_error_m"].items():
        print(f"  {name:<34}{e['d1']:>8.2f}м{e['d7']:>8.2f}м{e['d14']:>8.2f}м")
    print(f"  {'':-<66}")
    no_anchor = results["eph_error_m"]["Без наземной привязки"]
    rate = (no_anchor["d14"] - no_anchor["d1"]) / 13.0
    print(f"  Рост без наземного якоря: ~{rate:.2f} м/сут "
          f"(цель ~1 м/сут) {'OK' if 0.8 <= rate <= 1.2 else '?'}")
    print(sep)


if __name__ == "__main__":
    r = run_autonav_analysis("results/autonav", "phase5")
    print_autonav_summary("phase5", r)
