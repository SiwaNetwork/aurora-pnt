"""
Архитектура наземного сегмента (MCS / TT&C) системы AURORA PNT.

Рассчитывает:
- Элементы: MCS (центр управления, резервированный), станции TT&C (5),
  станции мониторинга/RSN (21), линии передачи данных
- Бюджет задержки конец-в-конец (мс): даунлинк, обработка TT&C,
  линия станция→MCS, обработка MCS (OD + часы), загрузка расписания, аплинк
- Готовность MCS при резервировании N+1: A = 1-(1-R)^n
- Пропускная способность TT&C: требуемые контакты/сут для 300 КА vs доступные

Система: 300 КА, 15 плоскостей, h=1000 км, i=75°, проектный срок 7 лет.

Ссылки:
  Wertz, Everett & Puschell (2011) — The New SMAD. Гл. 15 (ground systems),
    Гл. 19 (reliability, redundancy, availability).
  GPS SPS Performance Standard (2020) — наземный сегмент OCS/MCS.
  ECSS-Q-ST-30C — надёжность и резервирование (модель N+1).
"""

import sys, math, os, csv
from typing import Dict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Windows cp1251 -> UTF-8: безопасный вывод кириллицы и спецсимволов
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PALETTE = ["#e17055", "#fdcb6e", "#0984e3", "#00b894",
           "#6c5ce7", "#74b9ff", "#dfe6e9", "#2d3436"]

N_SATS    = 300
N_PLANES  = 15
N_TTC     = 5            # станций TT&C
N_MONITOR = 21           # станций мониторинга / RSN
R_SINGLE  = 0.98         # готовность одиночной нитки MCS

# ── Бюджет задержки конец-в-конец (мс) ────────────────────────────────────────
LATENCY_BUDGET = {
    "Даунлинк КА → станция":    3.3,
    "Обработка TT&C-станции":   5.0,
    "Станция → MCS (оптика)":  20.0,
    "MCS: расчёт OD + часы":   50.0,
    "Загрузка расписания":     15.0,
    "Аплинк → КА":              3.3,
}

# ── Параметры пропускной способности TT&C ─────────────────────────────────────
CONTACTS_PER_SAT_PER_DAY = 4.0       # требуемых сеансов на 1 КА в сутки
PASS_DURATION_MIN        = 10.0      # средняя длительность сеанса, мин
STATION_AVAIL_HOURS      = 20.0      # эфф. рабочих часов станции в сутки


def mcs_availability(n_strings: int, r: float = R_SINGLE) -> float:
    """Готовность MCS при n параллельных нитках (резерв N+...): A = 1-(1-R)^n."""
    return 1.0 - (1.0 - r) ** n_strings


def run_ground_segment_analysis(output_dir: str, label: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    total_lat = sum(LATENCY_BUDGET.values())

    redundancy = {
        "1 (без резерва)":  mcs_availability(1),
        "1+1 (горячий)":    mcs_availability(2),
        "2+1 (двойной)":    mcs_availability(3),
    }

    contacts_needed = N_SATS * CONTACTS_PER_SAT_PER_DAY                 # сеансов/сут
    passes_per_station = STATION_AVAIL_HOURS * 60.0 / PASS_DURATION_MIN # сеансов/сут на станцию
    contacts_available = N_TTC * passes_per_station
    margin = contacts_available - contacts_needed
    margin_pct = margin / contacts_needed * 100.0

    results = {
        "latency_components_ms": dict(LATENCY_BUDGET),
        "latency_total_ms":      total_lat,
        "mcs_redundancy":        redundancy,
        "n_ttc":                 N_TTC,
        "n_monitor":             N_MONITOR,
        "contacts_needed_day":   contacts_needed,
        "contacts_available_day": contacts_available,
        "ttc_margin_day":        margin,
        "ttc_margin_pct":        margin_pct,
    }

    _plot_architecture(output_dir, label)
    _plot_latency(total_lat, output_dir, label)
    _plot_redundancy(redundancy, contacts_needed, contacts_available,
                     output_dir, label)
    _save_csv(results, output_dir, label)
    return results


def _box(ax, x, y, w, h, text, color):
    from matplotlib.patches import FancyBboxPatch
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                       linewidth=1.5, edgecolor=PALETTE[7],
                       facecolor=color, alpha=0.85)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=10, fontweight="bold", color="white")
    return (x + w / 2, y + h / 2)


def _plot_architecture(output_dir, label):
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    ax.axis("off")

    c1 = _box(ax, 0.5, 2.3, 2.2, 1.4,
              f"Созвездие\n{N_SATS} КА\n{N_PLANES} плоск.", PALETTE[2])
    c2 = _box(ax, 3.6, 2.3, 2.2, 1.4,
              f"TT&C\n{N_TTC} станций", PALETTE[4])
    c3 = _box(ax, 6.7, 2.3, 2.2, 1.4,
              "MCS\n(центр упр.)\nрезерв 1+1", PALETTE[0])
    c4 = _box(ax, 9.8, 2.3, 2.6, 1.4,
              "Пользователи\n(SSR / PNT)", PALETTE[3])
    _box(ax, 3.6, 0.4, 5.3, 1.0,
         f"Сеть мониторинга / RSN — {N_MONITOR} станций", PALETTE[5])

    def arrow(p, q):
        ax.annotate("", xy=q, xytext=p,
                    arrowprops=dict(arrowstyle="-|>", lw=2.0,
                                    color=PALETTE[7]))
    arrow((2.7, 3.0), (3.6, 3.0))
    arrow((5.8, 3.0), (6.7, 3.0))
    arrow((8.9, 3.0), (9.8, 3.0))
    ax.annotate("", xy=(7.8, 2.3), xytext=(6.2, 1.4),
                arrowprops=dict(arrowstyle="-|>", lw=1.8,
                                color=PALETTE[5], ls="--"))
    ax.text(2.9, 3.25, "даунлинк/аплинк", fontsize=8, color=PALETTE[7])
    ax.text(5.95, 3.25, "оптика", fontsize=8, color=PALETTE[7])
    ax.text(8.95, 3.25, "SSR", fontsize=8, color=PALETTE[7])

    ax.set_title(f"Архитектура наземного сегмента AURORA PNT "
                 f"(КА → TT&C → MCS → Пользователи) [{label}]",
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"gs_architecture_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_latency(total_lat, output_dir, label):
    comps  = list(LATENCY_BUDGET.keys())
    vals   = list(LATENCY_BUDGET.values())
    colors = PALETTE[:len(comps)]
    cum    = np.cumsum([0] + vals[:-1])

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(comps, vals, left=cum, color=colors,
                   edgecolor="white", height=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2,
                b.get_y() + b.get_height() / 2,
                f"{v:.1f}", ha="center", va="center", fontsize=9,
                color="white", fontweight="bold")
    ax.axvline(total_lat, ls="--", color=PALETTE[7], lw=1.5,
               label=f"Итого: {total_lat:.1f} мс")
    ax.invert_yaxis()
    ax.set_xlabel("Накопленная задержка (мс)")
    ax.set_title(f"Бюджет задержки конец-в-конец наземного сегмента [{label}]")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"gs_latency_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_redundancy(redundancy, c_needed, c_avail, output_dir, label):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    names = list(redundancy.keys())
    avs   = [redundancy[n] for n in names]
    unav  = [(1 - a) * 100 for a in avs]
    bars = ax1.bar(names, [a * 100 for a in avs],
                   color=[PALETTE[0], PALETTE[1], PALETTE[3]],
                   edgecolor="white", width=0.55)
    for b, a, u in zip(bars, avs, unav):
        ax1.text(b.get_x() + b.get_width() / 2, a * 100 + 0.02,
                 f"{a*100:.3f}%\n({u*24*365/100*1000:.0f} мин/год недост.)"
                 if False else f"{a*100:.3f}%",
                 ha="center", fontsize=10)
    ax1.set_ylim(95, 100.05)
    ax1.set_ylabel("Готовность MCS (%)")
    ax1.set_title(f"Готовность MCS vs уровень резервирования\n"
                  f"(R одной нитки = {R_SINGLE}) [{label}]")
    ax1.grid(axis="y", alpha=0.3)

    bars2 = ax2.bar(["Требуется", "Доступно"],
                    [c_needed, c_avail],
                    color=[PALETTE[0], PALETTE[3]],
                    edgecolor="white", width=0.5)
    for b, v in zip(bars2, [c_needed, c_avail]):
        ax2.text(b.get_x() + b.get_width() / 2, v + max(c_needed, c_avail) * 0.01,
                 f"{v:.0f}", ha="center", fontsize=11)
    margin_pct = (c_avail - c_needed) / c_needed * 100
    ax2.set_ylabel("Сеансы TT&C в сутки")
    ax2.set_title(f"Пропускная способность TT&C ({N_TTC} ст.)\n"
                  f"запас = {margin_pct:+.0f}%")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"gs_redundancy_{label}.png"), dpi=150)
    plt.close(fig)


def _save_csv(results, output_dir, label):
    path = os.path.join(output_dir, f"ground_segment_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Компонента задержки", "мс"])
        for k, v in results["latency_components_ms"].items():
            w.writerow([k, f"{v:.1f}"])
        w.writerow(["Задержка ИТОГО", f"{results['latency_total_ms']:.1f}"])
        w.writerow([])
        w.writerow(["Резервирование MCS", "готовность", "недоступность %"])
        for k, v in results["mcs_redundancy"].items():
            w.writerow([k, f"{v:.5f}", f"{(1-v)*100:.4f}"])
        w.writerow([])
        w.writerow(["параметр", "значение", "ед."])
        w.writerow(["Станций TT&C",        results["n_ttc"], "шт"])
        w.writerow(["Станций мониторинга", results["n_monitor"], "шт"])
        w.writerow(["Контактов нужно/сут",  f"{results['contacts_needed_day']:.0f}", "сеанс/сут"])
        w.writerow(["Контактов доступно/сут", f"{results['contacts_available_day']:.0f}", "сеанс/сут"])
        w.writerow(["Запас TT&C",          f"{results['ttc_margin_pct']:+.0f}", "%"])


def print_ground_segment_summary(label: str, results: Dict) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  Ground Segment (MCS / TT&C) -- {label}")
    print(sep)
    print(f"  {'Компонента задержки':<28} {'мс':>8}")
    print(f"  {'':-<38}")
    for k, v in results["latency_components_ms"].items():
        print(f"  {k:<28} {v:>8.1f}")
    print(f"  {'Задержка ИТОГО':<28} {results['latency_total_ms']:>8.1f} мс")
    print(f"  {'':-<38}")
    print("  Готовность MCS по резервированию:")
    for k, v in results["mcs_redundancy"].items():
        print(f"    {k:<20} A = {v:.5f}  ({(1-v)*100:.4f}% недост.)")
    print(f"  TT&C: нужно {results['contacts_needed_day']:.0f} сеанс/сут, "
          f"доступно {results['contacts_available_day']:.0f} "
          f"(запас {results['ttc_margin_pct']:+.0f}%)")
    print(f"  Станций: TT&C={results['n_ttc']}, мониторинг={results['n_monitor']}")
    print(sep)
