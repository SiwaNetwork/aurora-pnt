"""
Программный график АВРОРА 2026-2040.

Анализирует:
- 13 фаз программы (концепт, эскиз, ТП, ОКР, производство, запуски,
  ВЭ, штатная эксплуатация)
- 7 ключевых вех (M1-M7)
- Критический путь: ОКР → производство Phase 0/1 → запуски → ВЭ Ф4
- Профиль численности персонала (50 ОКР → 200 серия → 100 эксплуатация)

Привязка: месяц 0 = январь 2026.

Ссылки:
  ECSS-M-ST-10C (2009) — Project planning and implementation. ECSS.
  PMI PMBoK 7th Ed. (2021) — Project Management Body of Knowledge.
  ГОСТ Р 56136-2014 — Управление жизненным циклом продукции
                      военного назначения.
"""

import sys, os, csv
from typing import Dict, List, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Палитра проекта ──────────────────────────────────────────────────────────
PALETTE = ["#e17055", "#fdcb6e", "#0984e3", "#00b894",
           "#6c5ce7", "#74b9ff", "#dfe6e9", "#2d3436"]

START_YEAR = 2026  # месяц 0 = январь 2026


# ── Фазы программы: (start_month, duration_months, color, status) ───────────
PHASES: Dict[str, Tuple[int, int, str, str]] = {
    "1. Концептуальный проект (выполнен)":        (0,    6, "#00b894", "done"),
    "2. Эскизный проект":                          (3,    9, "#74b9ff", "done"),
    "3. Технический проект (ТЕКУЩАЯ ФАЗА)":        (9,   12, "#fdcb6e", "active"),
    "4. ОКР: рабочая документация":                (18,  18, "#e17055", "planned"),
    "5. ОКР: квалификационные испытания":          (30,  12, "#e17055", "planned"),
    "6. Производство опытной партии (Phase 0/1)":  (36,  18, "#6c5ce7", "planned"),
    "7. Запуск демонстратора (Ф0, 3 КА)":          (52,   1, "#0984e3", "planned"),
    "8. ЛКИ + валидация":                          (53,   6, "#0984e3", "planned"),
    "9. Серия Ф1-Ф2 (90 КА)":                       (54,  24, "#6c5ce7", "planned"),
    "10. Запуски Ф1-Ф2 (5 пусков)":                (66,  18, "#0984e3", "planned"),
    "11. Серия Ф3-Ф4 (210 КА)":                    (78,  36, "#6c5ce7", "planned"),
    "12. Запуски Ф3-Ф4 (10 пусков)":               (90,  30, "#0984e3", "planned"),
    "13. Штатная эксплуатация Ф4":                 (120, 84, "#00b894", "planned"),
}


# ── Вехи программы ──────────────────────────────────────────────────────────
MILESTONES: List[Tuple[str, int, str]] = [
    ("M1: Защита ТП",                     21, "#fdcb6e"),
    ("M2: Готовность к ОКР",              30, "#e17055"),
    ("M3: Запуск Ф0 (3 КА)",              52, "#0984e3"),
    ("M4: ВЭ Ф1 (24 КА)",                 78, "#6c5ce7"),
    ("M5: ВЭ Ф3 (180 КА, оп. фаза)",     102, "#6c5ce7"),
    ("M6: ВЭ Ф4 (300 КА полное)",        120, "#00b894"),
    ("M7: Конец проектного срока (7 лет)", 204, "#2d3436"),
]


# ── Критический путь (последовательность работ) ─────────────────────────────
CRITICAL_PATH: List[Tuple[str, int, int, str]] = [
    ("ОКР: РД",                    18,  18, "#e17055"),
    ("ОКР: КИ",                    30,  12, "#e17055"),
    ("Произв. Phase 0/1",          36,  18, "#6c5ce7"),
    ("Запуск Ф0",                  52,   1, "#0984e3"),
    ("ЛКИ + валидация",            53,   6, "#0984e3"),
    ("Серия Ф1-Ф2",                54,  24, "#6c5ce7"),
    ("Запуски Ф1-Ф2",              66,  18, "#0984e3"),
    ("Серия Ф3-Ф4",                78,  36, "#6c5ce7"),
    ("Запуски Ф3-Ф4",              90,  30, "#0984e3"),
    ("ВЭ Ф4 (штатная экспл.)",    120,  84, "#00b894"),
]


def month_to_year(m: int) -> float:
    return START_YEAR + m / 12.0


def month_to_year_str(m: int) -> str:
    y = START_YEAR + m // 12
    mo = m % 12
    return f"{y}.{mo + 1:02d}"


def run_schedule_analysis(output_dir: str, label: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    phases_list = []
    for name, (start, dur, color, status) in PHASES.items():
        end = start + dur
        phases_list.append({
            "name":   name,
            "start":  start,
            "dur":    dur,
            "end":    end,
            "color":  color,
            "status": status,
            "year_start": month_to_year(start),
            "year_end":   month_to_year(end),
        })

    total_months = max(p["end"] for p in phases_list)

    results = {
        "phases":         phases_list,
        "milestones":     MILESTONES,
        "critical_path":  CRITICAL_PATH,
        "total_months":   total_months,
        "total_years":    total_months / 12.0,
        "start_year":     START_YEAR,
        "end_year":       month_to_year(total_months),
        "n_phases":       len(phases_list),
        "n_milestones":   len(MILESTONES),
    }

    _plot_gantt(phases_list, output_dir, label)
    _plot_milestones(MILESTONES, total_months, output_dir, label)
    _plot_critical_path(CRITICAL_PATH, MILESTONES, output_dir, label)
    _plot_workforce(total_months, output_dir, label)
    _save_csv(phases_list, output_dir, label)
    return results


# ── График 1: диаграмма Ганта ────────────────────────────────────────────────
def _plot_gantt(phases: List[Dict], output_dir: str, label: str) -> None:
    fig, ax = plt.subplots(figsize=(15, 8))

    names  = [p["name"] for p in phases]
    y_pos  = np.arange(len(names))

    for y, p in zip(y_pos, phases):
        edge = "#2d3436"
        hatch = None
        if p["status"] == "done":
            hatch = "//"
            edge  = "#00b894"
        elif p["status"] == "active":
            edge = "#e17055"
        ax.barh(y, p["dur"], left=p["start"], color=p["color"],
                edgecolor=edge, lw=1.2, height=0.65, hatch=hatch)
        # подпись длительности и года
        ax.text(p["start"] + p["dur"] / 2, y,
                f"{p['dur']} мес.",
                ha="center", va="center", fontsize=8,
                color="#2d3436", fontweight="bold")
        # год начала/конца справа
        ax.text(p["end"] + 1, y,
                f"{month_to_year_str(p['start'])} → {month_to_year_str(p['end'])}",
                va="center", fontsize=7.5, color="#2d3436")

    # вехи поверх Ганта
    for m_name, m_month, m_color in MILESTONES:
        ax.axvline(m_month, ls=":", color=m_color, lw=1.0, alpha=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()

    # тики X — годы
    total = max(p["end"] for p in phases) + 4
    year_ticks = np.arange(0, total + 12, 12)
    ax.set_xticks(year_ticks)
    ax.set_xticklabels([str(START_YEAR + int(t // 12)) for t in year_ticks],
                       fontsize=9)
    ax.set_xlim(-2, total + 12)
    ax.set_xlabel("Год")
    ax.set_title(
        f"АВРОРА — Программный график (диаграмма Ганта)  [{label}]\n"
        f"13 фаз, длительность {total - 4} мес. = {(total - 4) / 12:.1f} года")
    ax.grid(axis="x", alpha=0.3)

    # легенда статусов
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="#00b894", hatch="//", edgecolor="#00b894",
              label="Выполнено"),
        Patch(facecolor="#fdcb6e", edgecolor="#e17055", label="Активная фаза"),
        Patch(facecolor="#74b9ff", edgecolor="#2d3436", label="Планируется"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"schedule_gantt_{label}.png"), dpi=150)
    plt.close(fig)


# ── График 2: вехи на временной шкале ────────────────────────────────────────
def _plot_milestones(milestones, total_months: int,
                     output_dir: str, label: str) -> None:
    fig, ax = plt.subplots(figsize=(15, 5))

    total = max(m[1] for m in milestones) + 6
    ax.axhline(0, color="#2d3436", lw=1.5, zorder=1)

    # верх/низ чередование подписей
    for i, (name, m, color) in enumerate(milestones):
        y_text = 0.5 if i % 2 == 0 else -0.5
        va     = "bottom" if i % 2 == 0 else "top"
        # ромб
        ax.plot(m, 0, marker="D", ms=18, color=color,
                markeredgecolor="white", mew=1.8, zorder=3)
        # стойка
        ax.plot([m, m], [0, y_text * 0.85], color=color, lw=1.2, zorder=2)
        # подпись
        ax.text(m, y_text,
                f"{name}\n{month_to_year_str(m)}  (м. {m})",
                ha="center", va=va, fontsize=9, color="#2d3436",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=color, lw=1.0))

    # тики X — годы
    year_ticks = np.arange(0, total + 12, 12)
    ax.set_xticks(year_ticks)
    ax.set_xticklabels([str(START_YEAR + int(t // 12)) for t in year_ticks],
                       fontsize=9)
    ax.set_xlim(-4, total + 4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_yticks([])
    ax.set_xlabel("Год")
    ax.set_title(
        f"АВРОРА — Вехи программы (M1-M7)  [{label}]\n"
        f"M1 = защита ТП (м. 21),  M6 = ВЭ Ф4 полное (м. 120),  "
        f"M7 = конец проектного срока (м. 204)")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"schedule_milestones_{label}.png"),
                dpi=150)
    plt.close(fig)


# ── График 3: критический путь ───────────────────────────────────────────────
def _plot_critical_path(crit, milestones,
                        output_dir: str, label: str) -> None:
    fig, ax = plt.subplots(figsize=(15, 6.5))

    y_pos = np.arange(len(crit))
    for y, (name, start, dur, color) in zip(y_pos, crit):
        ax.barh(y, dur, left=start, color=color,
                edgecolor="#2d3436", lw=1.4, height=0.55)
        ax.text(start + dur / 2, y, f"{dur} мес.",
                ha="center", va="center", fontsize=8.5,
                color="white", fontweight="bold")

    # связи зависимости (стрелки)
    for i in range(len(crit) - 1):
        _, s1, d1, _ = crit[i]
        _, s2, _, _  = crit[i + 1]
        x0, y0 = s1 + d1, i
        x1, y1 = s2, i + 1
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="#e17055",
                                    lw=1.6, alpha=0.85,
                                    connectionstyle="arc3,rad=0.15"))

    # вехи на критическом пути
    for m_name, m_month, m_color in milestones:
        ax.axvline(m_month, ls=":", color=m_color, lw=1.1, alpha=0.55)
        ax.text(m_month, len(crit) - 0.3, m_name.split(":")[0],
                rotation=90, va="top", ha="right",
                fontsize=8, color=m_color)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([c[0] for c in crit], fontsize=9.5)
    ax.invert_yaxis()
    total = max(c[1] + c[2] for c in crit) + 6
    year_ticks = np.arange(0, total + 12, 12)
    ax.set_xticks(year_ticks)
    ax.set_xticklabels([str(START_YEAR + int(t // 12)) for t in year_ticks],
                       fontsize=9)
    ax.set_xlim(0, total + 6)
    ax.set_xlabel("Год")
    ax.set_title(
        f"АВРОРА — Критический путь программы  [{label}]\n"
        f"ОКР → производство Phase 0/1 → запуски → ВЭ Ф4   "
        f"(длительность {total - 6} мес. = {(total - 6) / 12:.1f} года)")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"schedule_critical_path_{label}.png"),
                dpi=150)
    plt.close(fig)


# ── График 4: профиль численности персонала ──────────────────────────────────
def _plot_workforce(total_months: int, output_dir: str, label: str) -> None:
    months = np.arange(0, total_months + 1)
    work   = np.zeros_like(months, dtype=float)

    # Профиль:
    #  0-9    : концепт + эскиз     = 25 чел.
    #  9-18   : ТП                   = 40 чел.
    # 18-42   : ОКР                  = 50 чел.
    # 36-78   : производство Ф0/Ф1   = +120 чел.  (макс. на пике)
    # 78-120  : серии Ф3/Ф4          = +150 чел.
    # 120+    : штатная эксплуатация = 100 чел.
    for i, m in enumerate(months):
        v = 0.0
        if m < 9:               v += 25.0
        elif m < 18:            v += 40.0
        elif m < 42:            v += 50.0
        # производство опытной партии и серии
        if 36 <= m < 78:        v += 120.0
        if 78 <= m < 120:       v += 150.0
        # запуски и ЛКИ
        if 52 <= m < 120:       v += 30.0
        # эксплуатация
        if m >= 120:            v += 100.0
        work[i] = v

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.fill_between(months, 0, work, color="#74b9ff", alpha=0.45,
                    label="Численность персонала")
    ax.plot(months, work, color="#0984e3", lw=2.2)

    # фазовые границы
    annotations = [
        (0,   "Концепт\n+ эскиз"),
        (12,  "ТП"),
        (30,  "ОКР"),
        (50,  "Производство\nФ0 / Ф1"),
        (90,  "Серия + запуски\nФ3 / Ф4"),
        (150, "Штатная\nэксплуатация"),
    ]
    for m, txt in annotations:
        ax.axvline(m, ls=":", color="#2d3436", lw=0.7, alpha=0.4)
        ax.text(m + 2, max(work) * 0.93, txt, fontsize=8.5, color="#2d3436")

    # вехи
    for m_name, m_month, m_color in MILESTONES:
        ax.axvline(m_month, ls="--", color=m_color, lw=0.8, alpha=0.5)

    year_ticks = np.arange(0, total_months + 12, 12)
    ax.set_xticks(year_ticks)
    ax.set_xticklabels([str(START_YEAR + int(t // 12)) for t in year_ticks],
                       fontsize=9)
    ax.set_xlim(0, total_months)
    ax.set_ylim(0, max(work) * 1.10)
    ax.set_xlabel("Год")
    ax.set_ylabel("Численность персонала (чел.)")
    ax.set_title(
        f"АВРОРА — Профиль численности персонала  [{label}]\n"
        f"Пик = {max(work):.0f} чел.,  ср. = {work.mean():.0f} чел.,  "
        f"эксплуатация = 100 чел.")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"schedule_workforce_{label}.png"),
                dpi=150)
    plt.close(fig)


# ── CSV ──────────────────────────────────────────────────────────────────────
def _save_csv(phases: List[Dict], output_dir: str, label: str) -> None:
    path = os.path.join(output_dir, f"schedule_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["фаза", "старт_месяц", "длительность_мес",
                    "конец_месяц", "год_начала", "год_конца", "статус"])
        for p in phases:
            w.writerow([p["name"], p["start"], p["dur"], p["end"],
                        month_to_year_str(p["start"]),
                        month_to_year_str(p["end"]),
                        p["status"]])
        w.writerow([])
        w.writerow(["веха", "месяц", "год"])
        for name, m, _ in MILESTONES:
            w.writerow([name, m, month_to_year_str(m)])


# ── Текстовый отчёт ──────────────────────────────────────────────────────────
def print_schedule_summary(label: str, results: Dict) -> None:
    sep = "=" * 78
    print(f"\n{sep}")
    print(f"  АВРОРА -- Programme Schedule  --  {label}")
    print(sep)
    print(f"  Старт программы:    {results['start_year']}.01 (месяц 0)")
    print(f"  Конец программы:    {month_to_year_str(results['total_months'])} "
          f"(месяц {results['total_months']})")
    print(f"  Общая длительность: {results['total_months']} мес. "
          f"= {results['total_years']:.1f} года")
    print(f"  Число фаз:          {results['n_phases']}")
    print(f"  Число вех:          {results['n_milestones']}")
    print(f"\n  Фазы программы:")
    print(f"  {'№':<3}{'фаза':<48}{'старт':>8}{'длит.':>8}{'статус':>10}")
    print(f"  {'─' * 76}")
    for i, p in enumerate(results["phases"], 1):
        nm = p["name"][:46]
        print(f"  {i:<3}{nm:<48}"
              f"{month_to_year_str(p['start']):>8}"
              f"{p['dur']:>6} м.{p['status']:>10}")
    print(f"\n  Вехи (M1-M7):")
    for name, m, _ in results["milestones"]:
        print(f"    {name:<40} месяц {m:>3}  ({month_to_year_str(m)})")
    print(f"\n  Критический путь:")
    crit_total = max(c[1] + c[2] for c in results["critical_path"])
    print(f"    {len(results['critical_path'])} работ, "
          f"длительность {crit_total} мес. = {crit_total / 12:.1f} года")
    print(sep)
