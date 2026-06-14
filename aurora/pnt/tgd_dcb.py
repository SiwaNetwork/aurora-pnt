"""
Межчастотная аппаратная задержка TGD/DCB для AURORA PNT.

В тракте КА сигналы L1 и L5 проходят разные цепи и приобретают разную групповую
задержку. Разность — TGD (Timing Group Delay) / DCB (Differential Code Bias) —
обязательна к учёту:

  - бортовые часы транслируются относительно ионосферосвободной (IF) комбинации,
    поэтому ДВУХЧАСТОТНЫЙ пользователь компенсирует TGD автоматически; остаётся
    лишь остаток калибровки и температурного дрейфа межчастотного смещения;
  - ОДНОЧАСТОТНЫЙ пользователь применяет транслируемый параметр TGD из навигацион-
    ного сообщения (§7); его остаток больше (квантование + временна́я нестабильность).

Модель оценивает вклад TGD/DCB в бюджет UERE (§37.3) для обоих типов приёмника при
заданной точности наземной калибровки DCB.

References:
  IS-GPS-200 (параметр T_GD); Montenbruck et al. (2014), DCB estimation;
  §6.1 (IF-комбинация, α1=2,261, α2=1,261), §37.3 (бюджет UERE).
"""

import os
import csv
import math
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = 299_792_458.0          # скорость света, м/с
NS = 1e-9

# Коэффициенты IF-комбинации L1/L5 (§6.1)
ALPHA1 = 2.261
ALPHA2 = 1.261

# ── Параметры TGD/DCB ─────────────────────────────────────────────────────────
TGD_NOMINAL_NS = 5.0       # типовая аппаратная разность L1/L5, нс
DCB_CAL_NS     = 0.10      # остаток наземной калибровки DCB, нс (1σ)
TGD_BROADCAST_RESID_NS = 0.30   # остаток одночастотного пользователя (квант.+дрейф)


def budget() -> Dict:
    # Двухчастотный: TGD сокращается в IF; остаётся остаток калибровки,
    # усиленный IF-комбинацией межчастотного смещения.
    dual_resid_ns = math.hypot(ALPHA1, ALPHA2) * DCB_CAL_NS  # нс
    dual_m = dual_resid_ns * NS * C

    # Одночастотный: применяет транслируемый TGD, остаток — квант.+нестабильность
    single_m = TGD_BROADCAST_RESID_NS * NS * C

    tgd_range_m = TGD_NOMINAL_NS * NS * C   # величина самой задержки (для контекста)
    return {
        "tgd_nominal_ns": TGD_NOMINAL_NS, "tgd_range_m": tgd_range_m,
        "dual_resid_ns": dual_resid_ns, "dual_uere_m": dual_m,
        "single_uere_m": single_m,
        "dcb_cal_ns": DCB_CAL_NS,
    }


def run_tgd_dcb_analysis(output_dir: str, label: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    b = budget()

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    cats = ["Величина TGD\n(до учёта)", "Одночастотный\n(трансл. TGD)",
            "Двухчастотный\n(остаток калибр.)"]
    vals = [b["tgd_range_m"], b["single_uere_m"], b["dual_uere_m"]]
    cols = ["#b2bec3", "#e17055", "#00b894"]
    bars = ax.bar(cats, vals, color=cols, edgecolor="white")
    for rect, v in zip(bars, vals):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.02,
                f"{v:.2f} м", ha="center", fontweight="bold")
    ax.axhline(0.70, ls="--", color="#0652DD", lw=1.2)
    ax.text(2.4, 0.72, "UERE dual ≈ 0,70 м", color="#0652DD", fontsize=8, ha="right")
    ax.set_ylabel("Вклад в дальностную ошибку, м (1σ)")
    ax.set_title(f"AURORA PNT — вклад TGD/DCB в UERE [{label}]\n"
                 f"(TGD ≈ {b['tgd_nominal_ns']:.0f} нс; калибровка DCB "
                 f"{b['dcb_cal_ns']:.2f} нс 1σ)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(vals) * 1.25)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"tgd_dcb_{label}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(output_dir, f"tgd_dcb_{label}.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["tgd_nominal_ns", f"{b['tgd_nominal_ns']:.2f}"])
        w.writerow(["tgd_range_m", f"{b['tgd_range_m']:.3f}"])
        w.writerow(["single_uere_m", f"{b['single_uere_m']:.3f}"])
        w.writerow(["dual_resid_ns", f"{b['dual_resid_ns']:.3f}"])
        w.writerow(["dual_uere_m", f"{b['dual_uere_m']:.3f}"])

    print(f"  TGD/DCB -- {label}")
    print(f"    Величина TGD ≈ {b['tgd_nominal_ns']:.0f} нс ({b['tgd_range_m']:.2f} м) — обязательна к учёту")
    print(f"    Одночастотный (трансл. TGD): остаток {b['single_uere_m']:.2f} м")
    print(f"    Двухчастотный (остаток калибровки DCB {b['dcb_cal_ns']:.2f} нс): "
          f"{b['dual_uere_m']:.3f} м (входит в часовой член 0,20 м §37.3)")
    return b


if __name__ == "__main__":
    run_tgd_dcb_analysis("results/tgd_dcb", "phase4")
