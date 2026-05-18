"""
Модель стоимости жизненного цикла (LCC) системы AURORA PNT.

Рассчитывает:
- NRE (неповторяющиеся затраты): проектирование + квалификация
- Повторяющиеся затраты на серию из 300 спутников с кривой обучения
- Стоимость запусков (кластерные/выделенные носители)
- CAPEX наземного сегмента
- OPEX в год (миссия 7 лет, расширенный сценарий 15 лет)
- Анализ чувствительности LCC (торнадо-диаграмма)

Все стоимости — оценочные параметрические (порядок величины), USD млн ($M);
для справки приведён эквивалент в ₽ по курсу ~90 ₽/$.

Ссылки:
  Wertz, Everett & Puschell (2011) — Space Mission Engineering: The New SMAD.
    Microcosm Press. Гл. 11 (cost modelling), кривая обучения Райта.
  NASA (2015) — Cost Estimating Handbook (CEH), v4.0. NASA HQ.
  Wright T.P. (1936) — Factors Affecting the Cost of Airplanes. J. Aero. Sci.
  ESA/ECSS-M-ST-60C (2008) — Cost and schedule management.
"""

import sys, math, os, csv
from typing import Dict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Windows cp1251 -> UTF-8: безопасный вывод кириллицы и спецсимволов (₽, ≈, Δ)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PALETTE = ["#e17055", "#fdcb6e", "#0984e3", "#00b894",
           "#6c5ce7", "#74b9ff", "#dfe6e9", "#2d3436"]

# ── Система (общие параметры созвездия) ───────────────────────────────────────
N_SATS        = 300          # спутников всего
N_PLANES      = 15
SAT_WET_KG    = 140.0        # масса спутника (мокрая), кг
DESIGN_LIFE_Y = 7            # проектный срок службы, лет
RUB_PER_USD   = 90.0         # курс для справки

# ── Параметрические оценки стоимости ($M) ─────────────────────────────────────
COST_ASSUMPTIONS = {
    "NRE_design_qual_Musd":   120.0,   # проектирование + квалификация (NRE)
    "recurring_T1_Musd":      4.0,     # стоимость первого лётного спутника (T1)
    "learning_b":             0.85,    # коэффициент кривой обучения (Райт)
    "n_launches":             25,      # число пусков (кластерные носители)
    "launch_cost_per_Musd":   12.0,    # стоимость одного пуска, $M -> 25*12=300
    "ground_capex_Musd":      80.0,    # CAPEX наземного сегмента
    "opex_per_year_Musd":     40.0,    # OPEX в год
}


def unit_cost(n: int, t1: float, b: float) -> float:
    """Стоимость n-го серийного изделия по кривой обучения Райта: T1 * n^log2(b)."""
    exp = math.log(b) / math.log(2.0)
    return t1 * (n ** exp)


def cumulative_recurring(n_sats: int, t1: float, b: float) -> float:
    """Суммарная повторяющаяся стоимость серии из n_sats спутников ($M)."""
    return float(sum(unit_cost(i, t1, b) for i in range(1, n_sats + 1)))


def lcc(years: int, a: Dict) -> Dict:
    nre      = a["NRE_design_qual_Musd"]
    recur    = cumulative_recurring(N_SATS, a["recurring_T1_Musd"], a["learning_b"])
    launch   = a["n_launches"] * a["launch_cost_per_Musd"]
    ground   = a["ground_capex_Musd"]
    opex     = a["opex_per_year_Musd"] * years
    total    = nre + recur + launch + ground + opex
    return {
        "NRE": nre, "Серия (300 КА)": recur, "Запуски": launch,
        "Наземный сегмент": ground, f"OPEX x{years} лет": opex,
        "_total": total, "_years": years,
    }


def run_cost_analysis(output_dir: str, label: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    a = COST_ASSUMPTIONS

    lcc7  = lcc(DESIGN_LIFE_Y, a)
    lcc15 = lcc(15, a)
    recur = cumulative_recurring(N_SATS, a["recurring_T1_Musd"], a["learning_b"])

    results = {
        "assumptions":        dict(a),
        "components_7y":       {k: v for k, v in lcc7.items() if not k.startswith("_")},
        "lcc_7y_Musd":         lcc7["_total"],
        "lcc_15y_Musd":        lcc15["_total"],
        "recurring_total_Musd": recur,
        "cost_per_sat_Musd":   lcc7["_total"] / N_SATS,
        "cost_per_year_7y_Musd": lcc7["_total"] / DESIGN_LIFE_Y,
        "lcc_7y_Brub":         lcc7["_total"] * RUB_PER_USD / 1000.0,
    }

    _plot_cost_breakdown(lcc7, output_dir, label)
    _plot_learning_curve(a, output_dir, label)
    _plot_lcc_vs_years(a, output_dir, label)
    _plot_cost_sensitivity(a, output_dir, label)
    _save_csv(results, lcc7, lcc15, output_dir, label)
    return results


def _plot_cost_breakdown(lcc7, output_dir, label):
    comps  = [k for k in lcc7 if not k.startswith("_")]
    vals   = [lcc7[k] for k in comps]
    total  = lcc7["_total"]
    colors = PALETTE[:len(comps)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    bars = ax1.bar(comps, vals, color=colors, edgecolor="white")
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + total * 0.01,
                 f"${v:.0f}M", ha="center", fontsize=10)
    ax1.set_ylabel("Стоимость ($M)")
    ax1.set_title(f"Структура LCC (7 лет) — компоненты [{label}]")
    ax1.tick_params(axis="x", rotation=20)
    ax1.grid(axis="y", alpha=0.3)

    w, t, at = ax2.pie(vals, labels=comps, colors=colors, autopct="%1.0f%%",
                       startangle=90, pctdistance=0.78)
    for x in t: x.set_fontsize(9)
    for x in at: x.set_fontsize(9)
    ax2.set_title(f"Доля компонент LCC\n(итого ${total:.0f}M ≈ {total*RUB_PER_USD/1000:.1f} млрд ₽)")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"cost_breakdown_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_learning_curve(a, output_dir, label):
    t1, b = a["recurring_T1_Musd"], a["learning_b"]
    n = np.arange(1, N_SATS + 1)
    per_unit = np.array([unit_cost(int(i), t1, b) for i in n])
    cum = np.cumsum(per_unit)

    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax1.plot(n, per_unit, color=PALETTE[0], lw=2.5,
             label=f"Стоимость n-го КА (b={b})")
    ax1.axhline(t1, ls=":", color=PALETTE[7], lw=1.2,
                label=f"T1 (первый КА) = ${t1:.1f}M")
    ax1.axhline(per_unit[-1], ls="--", color=PALETTE[3], lw=1.2,
                label=f"КА №300 = ${per_unit[-1]:.2f}M")
    ax1.set_xlabel("Номер серийного спутника")
    ax1.set_ylabel("Стоимость одного КА ($M)", color=PALETTE[0])
    ax1.tick_params(axis="y", labelcolor=PALETTE[0])
    ax1.set_title(f"Кривая обучения Райта — серия 300 КА [{label}]")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(n, cum, color=PALETTE[2], lw=2.5, ls="-.",
             label="Накопленная стоимость серии")
    ax2.set_ylabel("Накопленная стоимость серии ($M)", color=PALETTE[2])
    ax2.tick_params(axis="y", labelcolor=PALETTE[2])

    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, fontsize=9, loc="center right")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"learning_curve_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_lcc_vs_years(a, output_dir, label):
    yrs = [5, 7, 10, 15]
    totals = [lcc(y, a)["_total"] for y in yrs]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar([str(y) for y in yrs], totals,
                  color=[PALETTE[3], PALETTE[2], PALETTE[4], PALETTE[0]],
                  edgecolor="white", width=0.55)
    for b, v in zip(bars, totals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(totals) * 0.01,
                f"${v:.0f}M", ha="center", fontsize=11)
    ax.axvline(1, ls="--", color=PALETTE[7], lw=1.3,
               label=f"Проектный срок {DESIGN_LIFE_Y} лет")
    ax.set_xlabel("Срок миссии (лет)")
    ax.set_ylabel("Совокупная LCC ($M)")
    ax.set_title(f"Совокупная LCC vs срок миссии [{label}]")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"lcc_vs_years_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_cost_sensitivity(a, output_dir, label):
    base = lcc(DESIGN_LIFE_Y, a)["_total"]
    factors = {
        "T1 (стоим. 1-го КА)":   "recurring_T1_Musd",
        "Кривая обучения b":     "learning_b",
        "Стоимость пуска":       "launch_cost_per_Musd",
        "OPEX в год":            "opex_per_year_Musd",
        "NRE":                   "NRE_design_qual_Musd",
    }
    names, los, his = [], [], []
    for nm, key in factors.items():
        a_lo = dict(a); a_hi = dict(a)
        a_lo[key] = a[key] * 0.7
        a_hi[key] = a[key] * 1.3
        lo = lcc(DESIGN_LIFE_Y, a_lo)["_total"]
        hi = lcc(DESIGN_LIFE_Y, a_hi)["_total"]
        names.append(nm); los.append(lo); his.append(hi)

    spans = [abs(h - l) for h, l in zip(his, los)]
    order = np.argsort(spans)
    names = [names[i] for i in order]
    los   = [los[i]   for i in order]
    his   = [his[i]   for i in order]

    fig, ax = plt.subplots(figsize=(11, 6))
    y = np.arange(len(names))
    for i in range(len(names)):
        ax.barh(y[i], los[i] - base, left=base, color=PALETTE[2],
                edgecolor="white", height=0.55)
        ax.barh(y[i], his[i] - base, left=base, color=PALETTE[0],
                edgecolor="white", height=0.55)
    ax.axvline(base, ls="--", color=PALETTE[7], lw=1.5,
               label=f"База LCC 7 лет = ${base:.0f}M")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("LCC 7 лет ($M)")
    ax.set_title(f"Чувствительность LCC к параметрам ±30% (торнадо) [{label}]")
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=PALETTE[2], label="−30% параметра"),
        Patch(facecolor=PALETTE[0], label="+30% параметра"),
        plt.Line2D([0], [0], ls="--", color=PALETTE[7], label=f"База ${base:.0f}M"),
    ], fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"cost_sensitivity_{label}.png"), dpi=150)
    plt.close(fig)


def _save_csv(results, lcc7, lcc15, output_dir, label):
    path = os.path.join(output_dir, f"cost_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["параметр", "значение", "ед."])
        for k, v in results["components_7y"].items():
            w.writerow([f"Компонента LCC: {k}", f"{v:.1f}", "$M"])
        w.writerow(["LCC всего (7 лет)",  f"{results['lcc_7y_Musd']:.1f}",  "$M"])
        w.writerow(["LCC всего (15 лет)", f"{results['lcc_15y_Musd']:.1f}", "$M"])
        w.writerow(["LCC 7 лет (справ.)", f"{results['lcc_7y_Brub']:.2f}",  "млрд ₽"])
        w.writerow(["Серия 300 КА (повтор.)", f"{results['recurring_total_Musd']:.1f}", "$M"])
        w.writerow(["Стоимость на 1 КА (7 лет)", f"{results['cost_per_sat_Musd']:.2f}", "$M"])
        w.writerow(["Стоимость в год (7 лет)",   f"{results['cost_per_year_7y_Musd']:.1f}", "$M/год"])
        w.writerow([])
        w.writerow(["Допущение", "значение", "ед."])
        for k, v in results["assumptions"].items():
            w.writerow([k, f"{v}", ""])


def print_cost_summary(label: str, results: Dict) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  Life-Cycle Cost (LCC) -- {label}")
    print(sep)
    print(f"  {'Компонента':<26} {'$M':>12}")
    print(f"  {'':-<40}")
    for k, v in results["components_7y"].items():
        print(f"  {k:<26} {v:>12.1f}")
    print(f"  {'':-<40}")
    print(f"  LCC всего (7 лет):   ${results['lcc_7y_Musd']:.1f}M  "
          f"(≈ {results['lcc_7y_Brub']:.1f} млрд ₽)")
    print(f"  LCC всего (15 лет):  ${results['lcc_15y_Musd']:.1f}M")
    print(f"  Серия 300 КА:        ${results['recurring_total_Musd']:.1f}M")
    print(f"  Стоимость на 1 КА:   ${results['cost_per_sat_Musd']:.2f}M")
    print(f"  Стоимость в год:     ${results['cost_per_year_7y_Musd']:.1f}M/год")
    print(sep)
