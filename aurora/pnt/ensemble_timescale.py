"""
Распределённая шкала времени АВРОРА (SHIWA TIME-Space): ансамблевый эталон.

Идея развития: вместо дорогого атомного стандарта на КАЖДОМ из 300 КА —
протокол SHIWA TIME-Space синхронизирует группировку по ISL в единую
АНСАМБЛЕВУЮ (составную) шкалу. Дешёвые чип-цезиевые CSAC-терминалы на всех КА
дисциплинируются по немногим якорям space-Rb (~15) и наземному H-мазеру.

Составная шкала по теории композитных часов (обратно-дисперсионное взвешивание)
устойчивее любого отдельного дешёвого стандарта:
    σ_ens²(τ) = 1 / ( N_term/σ_csac²(τ) + N_anch/σ_rb²(τ) )
Это даёт системную шкалу уровня space-Rb БЕЗ space-Rb на каждом КА, что кратно
снижает стоимость часового сегмента группировки.

SHIWA TIME-Space обрабатывает асимметрию ISL (релятивистика/Саньяк, движение)
двусторонним обменом метками (TWSTT, §8.3) + детерминированной коррекцией по
эфемеридам; наземный SHIWA TIME (§28.4) — «счастливым пакетом» в IP-сетях.
Общий слой — распределённый ансамбль/консенсус без единого грандмастера.

References: Allan (1966); Brown (1991) композитные часы GPS; §8 (часы), §9 (ISL).
"""

import os
import csv
import math
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── ADEV-модели (белый FM σ0·τ^-1/2 с фликер-полом) ────────────────────────────
CLOCKS = {
    "CSAC (чип-цезий)":   {"sigma0": 3e-10, "floor": 1e-11, "col": "#e17055"},
    "space-Rb (якорь)":   {"sigma0": 1e-11, "floor": 1e-12, "col": "#0984e3"},
    "H-мазер (земля)":    {"sigma0": 1.5e-13, "floor": 8e-15, "col": "#6c5ce7"},
}
N_TERM = 300     # CSAC-терминалов (все КА)
N_ANCH = 15      # space-Rb якорей

# Относительная стоимость (ед.): space-Rb ≈ 10× CSAC
COST = {"csac": 1.0, "space_rb": 10.0}


def adev(sigma0: float, floor: float, tau: float) -> float:
    return math.hypot(sigma0 / math.sqrt(tau), floor)


def ensemble_adev(tau: float) -> float:
    """Составная шкала: обратно-дисперсионное взвешивание терминалов и якорей."""
    s_csac = adev(CLOCKS["CSAC (чип-цезий)"]["sigma0"],
                  CLOCKS["CSAC (чип-цезий)"]["floor"], tau)
    s_rb = adev(CLOCKS["space-Rb (якорь)"]["sigma0"],
                CLOCKS["space-Rb (якорь)"]["floor"], tau)
    inv_var = N_TERM / s_csac**2 + N_ANCH / s_rb**2
    return 1.0 / math.sqrt(inv_var)


def compute() -> Dict:
    taus = [10**(e / 2) for e in range(0, 11)]   # 1 … 10^5 с
    rows = []
    for t in taus:
        rows.append({
            "tau": t,
            "csac": adev(3e-10, 1e-11, t),
            "rb": adev(1e-11, 1e-12, t),
            "ensemble": ensemble_adev(t),
        })
    # Экономика: все space-Rb vs распределённый ансамбль
    cost_all_rb = N_TERM * COST["space_rb"]
    cost_distributed = N_TERM * COST["csac"] + N_ANCH * COST["space_rb"]
    saving = 1.0 - cost_distributed / cost_all_rb
    return {"rows": rows,
            "cost_all_rb": cost_all_rb, "cost_distributed": cost_distributed,
            "saving": saving,
            "ens_1s": ensemble_adev(1.0)}


def run_ensemble_timescale_analysis(output_dir: str, label: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    res = compute()
    rows = res["rows"]
    taus = [r["tau"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.3))

    # ── Панель 1: ADEV — одиночные часы vs ансамбль ──────────────────────────
    ax1.loglog(taus, [r["csac"] for r in rows], "o-", color="#e17055", lw=2,
               label="Одиночный CSAC (чип-цезий)")
    ax1.loglog(taus, [r["rb"] for r in rows], "s-", color="#0984e3", lw=2,
               label="Одиночный space-Rb")
    ax1.loglog(taus, [r["ensemble"] for r in rows], "^-", color="#00b894", lw=2.5,
               label="Ансамбль SHIWA TIME (300 CSAC + 15 Rb)")
    ax1.set_xlabel("Интервал усреднения τ, с")
    ax1.set_ylabel("Девиация Аллана σ_y(τ)")
    ax1.set_title("Распределённая шкала устойчивее одиночного дешёвого стандарта")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, which="both", alpha=0.3)

    # ── Панель 2: экономика часового сегмента ────────────────────────────────
    labels = ["space-Rb\nна всех 300", "Ансамбль\n(300 CSAC + 15 Rb)"]
    vals = [res["cost_all_rb"], res["cost_distributed"]]
    bars = ax2.bar(labels, vals, color=["#b2bec3", "#00b894"], edgecolor="white")
    for rect, v in zip(bars, vals):
        ax2.text(rect.get_x() + rect.get_width() / 2, v + 30,
                 f"{v:.0f} ед.", ha="center", fontweight="bold")
    ax2.text(1, res["cost_distributed"] * 0.5,
             f"−{res['saving']*100:.0f}%", ha="center", fontsize=16,
             fontweight="bold", color="#0a7a52")
    ax2.set_ylabel("Относительная стоимость часового сегмента, ед.")
    ax2.set_title("Экономика: ансамбль вместо дорогих стандартов на всех КА")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(f"АВРОРА — SHIWA TIME-Space: распределённый ансамблевый эталон [{label}]",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(output_dir, f"ensemble_timescale_{label}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(output_dir, f"ensemble_timescale_{label}.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tau_s", "csac_adev", "rb_adev", "ensemble_adev"])
        for r in rows:
            w.writerow([f"{r['tau']:.3g}", f"{r['csac']:.3e}",
                        f"{r['rb']:.3e}", f"{r['ensemble']:.3e}"])

    print(f"  SHIWA TIME-Space (ансамбль) -- {label}")
    print(f"    Ансамбль σ_y(1с) = {res['ens_1s']:.2e} — уровень space-Rb "
          f"при дешёвых CSAC-терминалах")
    print(f"    Стоимость часового сегмента: все space-Rb {res['cost_all_rb']:.0f} ед. → "
          f"ансамбль {res['cost_distributed']:.0f} ед. (−{res['saving']*100:.0f}%)")
    return res


if __name__ == "__main__":
    run_ensemble_timescale_analysis("results/ensemble_timescale", "phase4")
