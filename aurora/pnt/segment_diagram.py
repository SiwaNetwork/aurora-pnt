"""
Структурная (блок-)схема сегментов системы АВРОРА для §3.1:
космический сегмент (КА + ISL) → пользовательский сегмент → наземный сегмент (МКС),
с подписанными связями (нав-сигнал L1/L5, S-диапазон TT&C, поправки СДКМ/время).

Светлая тема, аккуратные прямоугольники и стрелки (matplotlib).
"""

import sys, os
from typing import Dict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

INK = "#1a2330"


def _band(ax, x, y, w, h, title, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.02",
                                facecolor=color, edgecolor=INK, linewidth=1.4, alpha=0.18, zorder=1))
    ax.text(x + 0.012, y + h - 0.028, title, fontsize=11, fontweight="bold",
            color=INK, ha="left", va="top", zorder=3)


def _box(ax, cx, cy, w, h, text, color, fs=9.5):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.004,rounding_size=0.015",
                                facecolor=color, edgecolor=INK, linewidth=1.1, zorder=4))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            color="white", fontweight="bold", zorder=5)


def _arrow(ax, p0, p1, color, text=None, two=False, off=0.0, fs=8.5):
    style = "<|-|>" if two else "-|>"
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=16,
                                 color=color, linewidth=1.8, zorder=2,
                                 shrinkA=2, shrinkB=2))
    if text:
        mx, my = (p0[0] + p1[0]) / 2 + off, (p0[1] + p1[1]) / 2
        ax.text(mx, my, text, ha="center", va="center", fontsize=fs, color=color,
                fontweight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))


def run_segment_diagram(output_dir: str, label: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12.5, 8.6))
    fig.patch.set_facecolor("#ffffff"); ax.set_facecolor("#ffffff")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    C_SPACE, C_USER, C_GROUND = "#2166ac", "#1a9850", "#d6803a"

    # ── Космический сегмент ──
    _band(ax, 0.06, 0.70, 0.88, 0.26, "КОСМИЧЕСКИЙ СЕГМЕНТ", C_SPACE)
    sat_y = 0.80
    sx = [0.24, 0.50, 0.76]
    for i, x in enumerate(sx):
        _box(ax, x, sat_y, 0.15, 0.085, ["КА-1", "КА-2", "КА-N"][i], C_SPACE)
    _arrow(ax, (sx[0] + 0.075, sat_y), (sx[1] - 0.075, sat_y), C_SPACE, "ISL Ka", two=True)
    _arrow(ax, (sx[1] + 0.075, sat_y), (sx[2] - 0.075, sat_y), C_SPACE, "ISL Ka", two=True)
    ax.text(0.86, sat_y, "…", fontsize=16, color=INK, ha="center", va="center")

    # ── Пользовательский сегмент ──
    _band(ax, 0.06, 0.40, 0.88, 0.18, "ПОЛЬЗОВАТЕЛЬСКИЙ СЕГМЕНТ", C_USER)
    uy = 0.455
    for x, t in zip([0.20, 0.40, 0.60, 0.80],
                    ["Авиация", "Морской", "Автомобиль", "Геодезия"]):
        _box(ax, x, uy, 0.16, 0.075, t, C_USER, fs=9)

    # ── Наземный сегмент ──
    _band(ax, 0.06, 0.06, 0.88, 0.20, "НАЗЕМНЫЙ СЕГМЕНТ (МКС АВРОРА)", C_GROUND)
    gy = 0.13
    for x, t in zip([0.24, 0.50, 0.76],
                    ["21 станция\nслежения", "ЦУП", "СДКМ-\nинтерфейс"]):
        _box(ax, x, gy, 0.20, 0.10, t, C_GROUND, fs=9)

    # ── Связи между сегментами ──
    # нав-сигнал космос → пользователь
    _arrow(ax, (0.40, 0.695), (0.40, 0.585), C_SPACE,
           "L1/L5 нав-сигнал\n(Сервис А/Б)", off=-0.0)
    # время/СДКМ-поправки наземный → пользователь
    _arrow(ax, (0.62, 0.265), (0.62, 0.395), C_GROUND,
           "СДКМ-поправки,\nSHIWA TIME", off=0.0)
    # S-диапазон TT&C космос ↔ наземный (правый край)
    _arrow(ax, (0.88, 0.70), (0.88, 0.26), C_GROUND,
           "S-диапазон\nTM/TC, эфемериды", two=True, off=0.07)

    ax.set_title("Рисунок — Структурная схема сегментов системы АВРОРА",
                 fontsize=12.5, color=INK, fontweight="bold", pad=8)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.02)
    path = os.path.join(output_dir, f"segment_diagram_{label}.png")
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"image": path}


def print_segment_diagram_summary(label: str, r: Dict) -> None:
    print(f"\n  Segment diagram -- {label}")
    print(f"    Image: {r['image']}")
