"""
Sankey-диаграммы потоков бюджета:
  1. Дисперсия UERE (двухчастотный, §37.3) — вклады в σ²_UERE (доли суммируются).
  2. Временна́я цепочка синхронизации (§8.5) — вклады дисперсии ошибки времени
     худшего CSAC-терминала (6 ISL-хопов): осциллятор ⊕ ISL-шум ⊕ релятивистика.

Дисперсии (σ²) складываются линейно, поэтому Sankey корректен именно для них.
"""

import sys, os
from typing import Dict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def run_sankey_budget(output_dir: str, label: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(15.5, 7.6))
    fig.patch.set_facecolor("#0a0f1a")

    # ── UERE: доли дисперсии (§37.3) ──
    ax1 = fig.add_subplot(1, 2, 1); ax1.axis("off"); ax1.set_facecolor("#0a0f1a")
    comps = [
        ("Эфемериды R", 45, "#e17055"),
        ("Тропосфера", 27, "#fdcb6e"),
        ("Часы CSAC", 9, "#00b894"),
        ("Многолучёв.", 9, "#6c5ce7"),
        ("Эфемериды A/C", 5, "#0984e3"),
        ("Шум ПРМ", 5, "#74b9ff"),
        ("Ионосфера", 0.5, "#b2bec3"),
    ]
    flows = [c[1] for c in comps] + [-sum(c[1] for c in comps)]
    labels = [f"{c[0]} {c[1]}%" for c in comps] + ["σ²_UERE\n(0,49 м²)"]
    orient = [1, 1, 1, -1, -1, -1, 1, 0]
    sk1 = Sankey(ax=ax1, unit="%", scale=1.0 / 100, offset=0.18,
                 head_angle=130, shoulder=0.02, gap=0.28)
    sk1.add(flows=flows, labels=labels, orientations=orient,
            pathlengths=[0.25] * len(flows),
            facecolor="#3a6ea5", edgecolor="#cfd8e3", trunklength=2.2)
    d1 = sk1.finish()
    for t in d1[0].texts:
        t.set_color("#e8eef5"); t.set_fontsize(8.5)
    ax1.set_title("Поток дисперсии UERE (двухчаст., §37.3)\n"
                  "доминируют эфемериды и тропосфера",
                  color="white", fontsize=12)

    # ── Временна́я цепочка: дисперсия ошибки времени (§8.5) ──
    ax2 = fig.add_subplot(1, 2, 2); ax2.axis("off"); ax2.set_facecolor("#0a0f1a")
    # вклады (нс²): осц. CSAC 0,95²; ISL 6 хопов (√6·1,0)²=6,0; релятив. 0,2²
    osc, isl, rel = 0.95**2, 6.0, 0.2**2
    tot = osc + isl + rel
    parts = [("Осциллятор\nCSAC (0,95 нс)", osc, "#00b894"),
             ("ISL-шум\n6 хопов (2,45 нс)", isl, "#e17055"),
             ("Релятивистика\n(0,2 нс)", rel, "#fdcb6e")]
    fl2 = [round(100 * p[1] / tot, 1) for p in parts]
    flows2 = fl2 + [-sum(fl2)]
    labels2 = [f"{p[0]} {100*p[1]/tot:.0f}%" for p in parts] + \
              [f"σ²_t терминала\n(2,7 нс) < 5 нс"]
    sk2 = Sankey(ax=ax2, unit="%", scale=1.0 / 100, offset=0.22,
                 head_angle=130, shoulder=0.02, gap=0.30)
    sk2.add(flows=flows2, labels=labels2, orientations=[1, 0, -1, 0],
            pathlengths=[0.3] * len(flows2),
            facecolor="#a5683a", edgecolor="#cfd8e3", trunklength=2.4)
    d2 = sk2.finish()
    for t in d2[0].texts:
        t.set_color("#e8eef5"); t.set_fontsize(8.5)
    ax2.set_title("Поток дисперсии ошибки времени (§8.5)\n"
                  "доминирует ISL-шум, не нестабильность CSAC",
                  color="white", fontsize=12)

    fig.suptitle("АВРОРА — Sankey бюджетов: дисперсия UERE и временно́й цепочки",
                 color="white", fontsize=13, y=0.99)
    fig.subplots_adjust(left=0.04, right=0.96, top=0.86, bottom=0.04, wspace=0.08)
    path = os.path.join(output_dir, f"sankey_budget_{label}.png")
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"image": path, "uere_top": "Эфемериды R 45%",
            "timing_top": f"ISL-шум {100*isl/tot:.0f}%"}


def print_sankey_budget_summary(label: str, r: Dict) -> None:
    print(f"\n  Sankey budget -- {label}")
    print(f"    UERE доминанта:  {r['uere_top']}")
    print(f"    Время доминанта: {r['timing_top']}")
    print(f"    Image: {r['image']}")
