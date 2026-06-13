"""
Пути повышения точности позиционирования AURORA PNT.

Базовый бюджет UERE (двухчастотный, §37.3) показывает, что точность ограничивают
НЕ шум псевдодальности, а орбитальное определение (транслируемая эфемерида) и
тропосфера. Модель количественно оценивает выигрыш от последовательного внедрения:

  S1  SSR/PPP-коррекции орбиты и часов (state-space) + фазовая ISL-дальность:
      радиальная эфемерида 0,45 → 0,10 м, вдоль/поперёк 0,15 → 0,05 м;
  S2  оценка тропосферы как состояния фильтра + ГНСС-метео аугментация:
      тропосфера 0,35 → 0,10 м;
  S3  расширение space-Rb за пределы 15 якорей + более частая ISL-синхронизация:
      остаток часов 0,20 → 0,10 м;
  S4  фазовый PPP-RTK с разрешением неоднозначности — отдельный «этаж» точности,
      не сводимый к RSS псевдодальности (сантиметровый уровень после сходимости).

Горизонтальная точность: H-95 = 2 · HDOP · UERE (§12.3). HDOP взят для
комбинированного с ГЛОНАСС режима (§5, §63.4).

References:
  Wübbena et al. (2005), PPP-RTK SSR; Teunissen & Khodabandeh (2015), PPP-RTK;
  Ge et al. (2018), LeGNSS; §37.3, §44, §47 настоящего ТП.
"""

import os
import csv
import math
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Базовый бюджет UERE (1σ, м), двухчастотный — из §37.3 ──────────────────────
# Порядок: радиальн. эфемер., вдоль/попер. эфемер., часы, ионосфера,
#          тропосфера, многолучёвость, шум приёмника.
LABELS = ["Эфем. радиал.", "Эфем. в/п", "Часы", "Ионосфера",
          "Тропосфера", "Многолуч.", "Шум"]
BASELINE = [0.45, 0.15, 0.20, 0.05, 0.35, 0.20, 0.15]

HDOP_COMB = 0.9   # комбинированный с ГЛОНАСС режим (репрезентативный)
K_H95     = 2.0   # H-95 = 2 · HDOP · UERE

# Сценарии: какие компоненты и до какого значения снижаются (кумулятивно)
SCENARIOS = [
    ("S0. Базовый (трансл. эфемерида)", {}),
    ("S1. + SSR/PPP орбита+часы, фазовая ISL", {0: 0.10, 1: 0.05}),
    ("S2. + оценка тропосферы (ZWD) и метео", {0: 0.10, 1: 0.05, 4: 0.10}),
    ("S3. + space-Rb сверх 15 якорей, ISL-синхр.", {0: 0.10, 1: 0.05, 4: 0.10, 2: 0.10}),
]

# Фазовый PPP-RTK — отдельный «этаж» (горизонт., после сходимости)
PPP_RTK_H95_M = 0.15


def _rss(vec: List[float]) -> float:
    return math.sqrt(sum(v * v for v in vec))


def compute() -> Dict:
    rows = []
    for name, overrides in SCENARIOS:
        vec = list(BASELINE)
        for idx, val in overrides.items():
            vec[idx] = val
        uere = _rss(vec)
        h95 = K_H95 * HDOP_COMB * uere
        rows.append({"name": name, "uere": uere, "h95": h95, "vec": vec})
    return {"rows": rows, "ppp_rtk_h95": PPP_RTK_H95_M, "hdop": HDOP_COMB}


def run_accuracy_paths_analysis(output_dir: str, label: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    res = compute()
    rows = res["rows"]

    # ── График: два подграфика — UERE и H-95 по сценариям ─────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    names = [r["name"].split(".")[0] for r in rows]   # S0..S3
    full_names = [r["name"] for r in rows]
    ueres = [r["uere"] for r in rows]
    h95s = [r["h95"] for r in rows]
    cols = ["#b2bec3", "#74b9ff", "#0984e3", "#0652DD"]

    b1 = ax1.bar(names, ueres, color=cols, edgecolor="white")
    for rect, v in zip(b1, ueres):
        ax1.text(rect.get_x() + rect.get_width() / 2, v + 0.012,
                 f"{v:.2f}", ha="center", fontweight="bold")
    ax1.set_ylabel("UERE, 1σ (м)")
    ax1.set_title("Бюджет псевдодальности UERE по сценариям")
    ax1.set_ylim(0, max(ueres) * 1.2)
    ax1.grid(axis="y", alpha=0.3)

    b2 = ax2.bar(names, h95s, color=cols, edgecolor="white")
    for rect, v in zip(b2, h95s):
        ax2.text(rect.get_x() + rect.get_width() / 2, v + 0.02,
                 f"{v:.2f}", ha="center", fontweight="bold")
    ax2.axhline(res["ppp_rtk_h95"], ls="--", color="#00b894", lw=1.5)
    ax2.text(len(names) - 1, res["ppp_rtk_h95"] + 0.03,
             f"PPP-RTK (фаза) ≈ {res['ppp_rtk_h95']:.2f} м",
             ha="right", color="#00b894", fontsize=9, fontweight="bold")
    ax2.set_ylabel("H-95 (м), комбинир. режим, HDOP=0,9")
    ax2.set_title("Горизонтальная точность H-95")
    ax2.set_ylim(0, max(h95s) * 1.25)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(f"AURORA PNT — пути повышения точности позиционирования [{label}]",
                 fontsize=13, fontweight="bold")
    # легенда соответствия S0..S3 → полные названия
    fig.text(0.5, -0.02, "   |   ".join(full_names), ha="center", fontsize=8,
             color="#2d3436")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(os.path.join(output_dir, f"accuracy_paths_{label}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── CSV ──────────────────────────────────────────────────────────────────
    with open(os.path.join(output_dir, f"accuracy_paths_{label}.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "UERE_m", "H95_m"])
        for r in rows:
            w.writerow([r["name"], f"{r['uere']:.3f}", f"{r['h95']:.3f}"])
        w.writerow(["S4. PPP-RTK (фаза, после сходимости)", "-",
                    f"{res['ppp_rtk_h95']:.3f}"])

    print(f"  Пути повышения точности -- {label}")
    for r in rows:
        print(f"    {r['name']:46s} UERE {r['uere']:.2f} м | H-95 {r['h95']:.2f} м")
    print(f"    S4. PPP-RTK (фаза)                              "
          f"            H-95 ≈ {res['ppp_rtk_h95']:.2f} м")
    print(f"    Итого: UERE {rows[0]['uere']:.2f}→{rows[-1]['uere']:.2f} м, "
          f"H-95 {rows[0]['h95']:.2f}→{rows[-1]['h95']:.2f} м (псевдодальн.), "
          f"→ {res['ppp_rtk_h95']:.2f} м (фаза)")
    return res


if __name__ == "__main__":
    run_accuracy_paths_analysis("results/accuracy_paths", "phase4")
