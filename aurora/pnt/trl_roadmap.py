"""
Маршрут зрелости АВРОРА: фазы развёртывания → рост TRL → закрытие критических
замечаний аудита (К-1…К-5). Для презентаций инвестору и РКС/Роскосмосу.

Светлая тема. Верхняя дорожка — TRL-кривая с маркерами фаз; нижняя — полосы
закрытия рисков на той же оси времени.
"""

import sys, os
from typing import Dict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

INK = "#1a2330"

# (год, фаза, КА, TRL к концу фазы, подпись)
PHASES = [
    (2026.5, "Ф0", 3,   3, "демо 3×16U, 87°"),
    (2027.5, "Ф1", 12,  5, "12 КА: ISL, LPT, дисциплина часов"),
    (2029.0, "Ф2", 90,  6, "90 КА, 75°: РФ 82%"),
    (2031.0, "Ф3", 180, 7, "180 КА: РФ 100%"),
    (2033.5, "Ф4", 300, 9, "300 КА: глобально, FOC"),
]

# (метка, старт, конец, цвет, текст)
RISKS = [
    ("К-1", 2026.0, 2028.5, "#d6604d", "CSAC: инж. образец → квалификация на Ф0–1"),
    ("К-2", 2026.5, 2028.0, "#e08214", "POD / фазовая ISL — лётная проверка Ф0–1"),
    ("К-4", 2026.0, 2029.0, "#b2882a", "Спектр Сервиса Б: API МСЭ → ГКРЧ → координация"),
    ("К-3", 2026.0, 2027.0, "#1a9850", "Экономика восполнения — закрыто в LCC (§51.3)"),
    ("К-5", 2026.0, 2026.6, "#2166ac", "Конфиг-дисциплина — закрыто (линтер check_tp)"),
]


def run_trl_roadmap(output_dir: str, label: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    fig, (axT, axR) = plt.subplots(2, 1, figsize=(13.5, 8.4),
                                   gridspec_kw={"height_ratios": [1.5, 1.0]})
    fig.patch.set_facecolor("#ffffff")

    # ── Верх: TRL-кривая + фазы ──
    axT.set_facecolor("#ffffff")
    yrs = [p[0] for p in PHASES]; trl = [p[3] for p in PHASES]
    axT.step([2026.0] + yrs, [2.5] + trl, where="post", color="#2166ac", lw=2.6, zorder=3)
    axT.scatter(yrs, trl, s=120, color="#2166ac", edgecolors="white", linewidths=1.4, zorder=4)
    axT.axhspan(1, 4, color="#d6604d", alpha=0.07); axT.axhspan(4, 7, color="#fdcb6e", alpha=0.10)
    axT.axhspan(7, 9.5, color="#1a9850", alpha=0.08)
    axT.text(2025.7, 2.5, "TRL 2–3\n(сейчас)", fontsize=8, color="#a33", va="center")
    for x, name, n, t, note in PHASES:
        axT.annotate(f"{name}: {n} КА\n{note}", (x, t),
                     textcoords="offset points", xytext=(6, 10), fontsize=8.2,
                     color=INK, fontweight="bold")
    axT.set_ylim(1, 10); axT.set_xlim(2025.5, 2035)
    axT.set_ylabel("TRL (готовность технологии)", color=INK, fontsize=10)
    axT.set_yticks(range(1, 10))
    axT.grid(alpha=0.25); axT.tick_params(colors=INK)
    axT.set_title("АВРОРА — маршрут зрелости: фазы развёртывания → рост TRL → закрытие рисков",
                  color=INK, fontsize=12.5, fontweight="bold")

    # ── Низ: полосы закрытия рисков ──
    axR.set_facecolor("#ffffff")
    for i, (mark, s, e, col, txt) in enumerate(RISKS):
        y = len(RISKS) - 1 - i
        axR.add_patch(FancyBboxPatch((s, y - 0.32), e - s, 0.64,
                                     boxstyle="round,pad=0.0,rounding_size=0.08",
                                     facecolor=col, edgecolor="white", alpha=0.85, zorder=3))
        axR.text(s + 0.05, y, f"{mark}", color="white", fontsize=9.5,
                 fontweight="bold", va="center", zorder=4)
        axR.text(e + 0.1, y, txt, color=INK, fontsize=8.5, va="center", zorder=4)
        axR.scatter([e], [y], marker=">", s=60, color=col, zorder=5)
    axR.set_ylim(-0.7, len(RISKS) - 0.3); axR.set_xlim(2025.5, 2035)
    axR.set_yticks([]); axR.tick_params(colors=INK)
    axR.set_xlabel("Год", color=INK, fontsize=10)
    axR.grid(axis="x", alpha=0.25)
    axR.set_title("Закрытие критических замечаний аудита (К-1…К-5)",
                  color=INK, fontsize=11, fontweight="bold")
    for ax in (axT, axR):
        for yv in yrs:
            ax.axvline(yv, ls=":", color="#9aa6b2", lw=0.7, alpha=0.6)

    fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.07, hspace=0.28)
    path = os.path.join(output_dir, f"trl_roadmap_{label}.png")
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"image": path}


def print_trl_roadmap_summary(label: str, r: Dict) -> None:
    print(f"\n  TRL roadmap -- {label}")
    print(f"    Image: {r['image']}")
