"""
Блок-схема навигационной полезной нагрузки АВРОРА и частотно-временно́го
обеспечения (ЧВО) — для технического диалога с РКС (домен нав-ПН/ЧВО).

Цепочка: ЧВО (CSAC/space-Rb → ансамбль → UTC(SU)) → формирование нав-сообщения
(эфемериды, часы, TGD, GGTO, крипто ГОСТ) → формирование сигнала (коды → модулятор)
→ Сервис А (L1/L5) и Сервис Б (L-полоса) → антенны. ISL Ka замыкает дисциплину
часов и автономное определение орбит. Интерфейс с ГЛОНАСС/UTC(SU).
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


def _stage(ax, x, y, w, h, title, lines, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.02",
                                facecolor=color, edgecolor=INK, linewidth=1.4, alpha=0.16, zorder=1))
    ax.text(x + w / 2, y + h - 0.03, title, ha="center", va="top", fontsize=10.5,
            fontweight="bold", color=INK, zorder=3)
    for i, ln in enumerate(lines):
        ax.text(x + 0.014, y + h - 0.075 - i * 0.042, ln, ha="left", va="top",
                fontsize=8.6, color=INK, zorder=3)


def _small(ax, cx, cy, w, h, text, color, fs=8.8):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.003,rounding_size=0.012",
                                facecolor=color, edgecolor=INK, linewidth=1.0, zorder=4))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color="white",
            fontweight="bold", zorder=5)


def _arrow(ax, p0, p1, color, text=None, fs=8.2, two=False):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="<|-|>" if two else "-|>",
                                 mutation_scale=15, color=color, linewidth=1.8,
                                 zorder=2, shrinkA=2, shrinkB=2))
    if text:
        ax.text((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + 0.02, text, ha="center",
                va="bottom", fontsize=fs, color=color, fontweight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))


def run_nav_payload_diagram(output_dir: str, label: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13.5, 7.6))
    fig.patch.set_facecolor("#ffffff"); ax.set_facecolor("#ffffff")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    C_CT, C_MSG, C_SIG, C_A, C_B, C_ISL, C_GL = (
        "#2166ac", "#6a51a3", "#1a9850", "#0984e3", "#d6604d", "#7f8c9a", "#b8860b")

    # Стадия 1 — ЧВО
    _stage(ax, 0.04, 0.55, 0.26, 0.34, "Частотно-временно́е\nобеспечение (ЧВО)", [
        "• CSAC — все 300 КА (ШИВА)",
        "• space-Rb Quantum-18 —",
        "   ~15 якорей (ШИВА)",
        "• ансамблевая шкала (§28.6)",
        "• привязка к UTC(SU)",
        "   (H-мазер Ч1-1008)",
    ], C_CT)

    # Стадия 2 — формирование нав-сообщения
    _stage(ax, 0.37, 0.55, 0.26, 0.34, "Формирование\nнав-сообщения (ANAV)", [
        "• эфемериды (POD, §47)",
        "• часы a₀,a₁,a₂ + TGD/DCB",
        "• GGTO АВРОРА↔ГЛОНАСС",
        "• ионо-модель, EOP",
        "• аутентификация ГОСТ",
        "   (TESLA-Стрибог, §16)",
    ], C_MSG)

    # Стадия 3 — формирование сигнала
    _stage(ax, 0.70, 0.55, 0.26, 0.34, "Формирование сигнала", [
        "• коды Weil / Extended Memory",
        "• модулятор BOC/TMBOC,",
        "   BPSK(10) — §6",
        "• LDPC(1/2) + CRC-32",
        "• разделение Сервис А / Б",
    ], C_SIG)

    _arrow(ax, (0.30, 0.72), (0.37, 0.72), INK, "опорн.\nшкала")
    _arrow(ax, (0.63, 0.72), (0.70, 0.72), INK, "ANAV\nданные")

    # Сервис А и Б → антенны
    _small(ax, 0.80, 0.40, 0.30, 0.072, "Сервис А: L1/L5 · УМ 5/3 Вт · изо-flux антенна", C_A, fs=8.6)
    _small(ax, 0.80, 0.29, 0.30, 0.072, "Сервис Б: L-полоса · УМ 30 Вт · фаз. решётка 8 дБи", C_B, fs=8.6)
    _arrow(ax, (0.83, 0.55), (0.81, 0.44), C_SIG)
    _arrow(ax, (0.78, 0.55), (0.79, 0.33), C_SIG)

    # ISL Ka — снизу, замыкает дисциплину и POD
    _small(ax, 0.205, 0.30, 0.30, 0.10,
           "ISL Ka (×2): TWSTT — дисциплина часов;\nдальнометрия — автономное определение орбит", C_ISL, fs=8.6)
    _arrow(ax, (0.17, 0.35), (0.13, 0.55), C_ISL, "дисциплина")
    _arrow(ax, (0.27, 0.35), (0.45, 0.55), C_ISL, "POD")

    # Интерфейс с ГЛОНАСС
    _small(ax, 0.52, 0.13, 0.66, 0.09,
           "Интерфейс с ГЛОНАСС / UTC(SU):  датум ПЗ-90.11 · общая шкала UTC(SU) · "
           "GGTO АВРОРА↔ГЛОНАСС → комбинир. обработка (§29, §64)", C_GL, fs=8.0)
    _arrow(ax, (0.50, 0.55), (0.52, 0.18), C_GL)
    _arrow(ax, (0.17, 0.55), (0.30, 0.18), C_GL)

    ax.set_title("Рисунок — Навигационная ПН и частотно-временно́е обеспечение АВРОРА",
                 fontsize=12.5, color=INK, fontweight="bold", pad=8)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.02)
    path = os.path.join(output_dir, f"nav_payload_diagram_{label}.png")
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"image": path}


def print_nav_payload_diagram_summary(label: str, r: Dict) -> None:
    print(f"\n  Nav-payload diagram -- {label}")
    print(f"    Image: {r['image']}")
