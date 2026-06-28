"""
Экономика hosted-варианта: размещение нав-ПН АВРОРЫ на 106 КА ГОНЕЦ-М1 в сравнении
с выделенной группировкой АВРОРЫ (300 КА). Часть исследования АВРОРА-ГОНЕЦ-001
(НЕ часть основного ТП). Опирается на затраты cost_model.py (§51 ТП).

ВАЖНО: это оценка ПЕРВОГО ПОРЯДКА. Ключевое допущение — 106 КА ГОНЕЦ-М1 строятся и
запускаются для связи В ЛЮБОМ СЛУЧАЕ, а АВРОРА доплачивает только за инкремент
полезной нагрузки и хостинг. Если АВРОРЕ придётся финансировать саму группировку
(официально у ГОНЕЦ-М1 28 КА, не 106), экономия резко сокращается — см. вывод.

Запуск:  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python aurora/pnt/gonets_econ.py
"""

import sys, os, importlib.util
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PALETTE = ["#e17055", "#fdcb6e", "#0984e3", "#00b894",
           "#6c5ce7", "#74b9ff", "#dfe6e9", "#2d3436"]

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("cm", os.path.join(_here, "cost_model.py"))
cm = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(cm)

N_HOST = 106  # КА-носителей ГОНЕЦ-М1 (оценка заказчика)

# ── Затраты выделенной АВРОРЫ (из cost_model / §51 ТП), млрд ₽ ────────────────
A = cm.COST_ASSUMPTIONS
DED_LCC7 = cm.lcc(7, A)["_total"] / 1000.0          # ≈ 113,4

# ── Допущения hosted-варианта (млрд ₽), первый порядок ────────────────────────
# Обоснование величин — в комментариях; все помечены как предварительные.
HOSTED = {
    "NRE: адаптация нав-ПН к бусу ГОНЕЦ":  5.0,   # < 10,8 АВРОРЫ — нет разработки платформы
    "Нав-ПН + CSAC × 106 (серия)":         6.0,   # ≈ 57 млн ₽/КА: только ПН (L-ПРД+антенна+CSAC), без буса/ISL
    "Интеграция/хостинг × 106":            2.1,   # ≈ 20 млн ₽/КА accommodation на чужом бусе
    "Доп. выведение (масса ПН)":           2.0,   # инкремент к пускам ГОНЕЦ (rideshare), не отдельные пуски
    "Наземный сегмент (нав-контур)":       6.0,   # < 7,2 АВРОРЫ — часть инфраструктуры ГОНЕЦ переиспользуется
}
HOSTED_OPEX_YR = 1.5                              # < 3,6 АВРОРЫ — нет своей платформы
HOSTED_CAPEX = sum(HOSTED.values())
HOSTED_LCC7  = HOSTED_CAPEX + HOSTED_OPEX_YR * 7

# Сценарий «АВРОРА финансирует расширение ГОНЕЦ 28→106» (пессимистичный):
# добавляется стоимость 78 доп. бусов + их выведение (грубо по экономике АВРОРЫ ~172 млн/КА).
EXPAND_SATS = 106 - 28
EXPAND_COST = EXPAND_SATS * 0.172                 # ≈ 13,4 млрд ₽ (буст+пуск за КА)
HOSTED_LCC7_EXPAND = HOSTED_LCC7 + EXPAND_COST


def report():
    print("Экономика: выделенная АВРОРА (300 КА) vs hosted на ГОНЕЦ-М1 (106 КА)\n")
    print(f"  Выделенная АВРОРА LCC-7:        {DED_LCC7:6.1f} млрд ₽ (§51 ТП)")
    print(f"  CAPEX выделенной (без OPEX):    {DED_LCC7 - A['opex_per_year_Mrub']*7/1000:6.1f} млрд ₽")
    print("\n  Hosted-вариант (носители строятся для связи; платим инкремент):")
    for k, v in HOSTED.items():
        print(f"    {k:38s} {v:5.1f}")
    print(f"    {'OPEX × 7 лет (1,5/год)':38s} {HOSTED_OPEX_YR*7:5.1f}")
    print(f"    {'—'*44}")
    print(f"    {'CAPEX hosted':38s} {HOSTED_CAPEX:5.1f}")
    print(f"    {'LCC-7 hosted':38s} {HOSTED_LCC7:5.1f}  "
          f"(× {DED_LCC7/HOSTED_LCC7:.1f} дешевле выделенной)")
    print(f"\n  Пессим. (АВРОРА финансирует ГОНЕЦ 28→106: +{EXPAND_COST:.1f} млрд):")
    print(f"    LCC-7 hosted+расширение:       {HOSTED_LCC7_EXPAND:5.1f} млрд ₽  "
          f"(× {DED_LCC7/HOSTED_LCC7_EXPAND:.1f} дешевле)")


def plot(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    cats = ["Выделенная\nАВРОРА (300 КА)", "Hosted на ГОНЕЦ\n(106 КА)",
            "Hosted + расширение\nГОНЕЦ 28→106"]
    capex = [DED_LCC7 - A['opex_per_year_Mrub']*7/1000, HOSTED_CAPEX,
             HOSTED_CAPEX + EXPAND_COST]
    opex  = [A['opex_per_year_Mrub']*7/1000, HOSTED_OPEX_YR*7, HOSTED_OPEX_YR*7]
    ax.bar(cats, capex, color=PALETTE[2], edgecolor="white", label="CAPEX")
    ax.bar(cats, opex, bottom=capex, color=PALETTE[1], edgecolor="white",
           label="OPEX × 7 лет")
    for i, (c, o) in enumerate(zip(capex, opex)):
        ax.text(i, c + o + 2, f"{c+o:.0f} млрд ₽", ha="center", fontsize=11,
                fontweight="bold")
    ax.set_ylabel("LCC за 7 лет (млрд ₽)")
    ax.set_title("Стоимость: выделенная АВРОРА vs hosted на ГОНЕЦ-М1 (оценка 1-го порядка)")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "gonets_econ_compare.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"\n  График: {path}")


def main():
    report()
    out = os.path.abspath(os.path.join(_here, "..", "..", "results", "gonets"))
    plot(out)


if __name__ == "__main__":
    main()
