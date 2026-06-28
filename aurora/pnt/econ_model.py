"""
Экономическая модель сервисов АВРОРА (вход для ТЭО, АВРОРА-ТЭО-001).

Считает «снизу вверх» коммерческую выручку по четырём сервисам на основе
РЕАЛЬНЫХ тарифов зарубежных аналогов (см. блок TARIFFS), сводит её с затратной
частью LCC (§51 ТП, модуль cost_model) в годовой денежный поток и вычисляет
показатели эффективности: NPV, IRR, простой и дисконтированный срок окупаемости.
Три сценария проникновения: пессимистический / базовый / оптимистический.

Все суммы — в РУБЛЯХ (млрд ₽, если не указано иное). Доходные оценки имеют
статус ЭКСПЕРТНОГО ПРОГНОЗА: тарифы — фактические (источники ниже), число
платящих абонентов — допущение, подлежащее уточнению маркетинговым исследованием.

Источники тарифов и рынка (веб, 2026):
  - u-blox PointPerfect: $15–50/мес за устройство (PPP-RTK поправки).
    https://www.u-blox.com/en/PointPerfect-usage-based-plans
  - Swift Navigation Skylark: от $29/мес за устройство; Nx RTK от $15/мес (5-летн.
    предоплата) — перепроверено июнь 2026. https://www.swiftnav.com/products/skylark
  - NovAtel TerraStar-L $580/год; TerraStar-C Pro ≈ $1750/год за машину.
    https://terrastar.net/services/terrastar-service-options
  - Iridium STL (Satellite Time & Location, тайминг + анти-спуфинг, LEO):
    целевая выручка сервиса >$100 млн/год к 2030 — прямой аналог Сервиса Б.
    https://investor.iridium.com/2024-04-02-...-Satellite-Time-and-Location-STL
  - Рынок Assured PNT: $0,9 млрд (2025) → $3,0 млрд (2030), CAGR ≈ 28%.
    https://www.marketsandmarkets.com/Market-Reports/satellite-positioning-navigation-timing-pnt-market-67260062.html
  - Глобальный рынок GNSS (EUSPA 2024): €300 млрд (2024) → €580 млрд (2034),
    ~80% — добавленные сервисы к 2034. https://www.euspa.europa.eu/.../eo-gnss-market-report
  - Xona Space (Pulsar, 258 КА): подписочная модель с уровнями сервиса (B2B+оборона).
    https://www.xonaspace.com/pulsar
  - EFT-CORS (РФ): RTK 50 400 ₽/год за приёмник (валидация внутреннего ARPU).
    https://eft-cors.ru/prices
  - Qianxun SI (КНР): единый нац. оператор точн. позиционирования, >390 млн клиентов,
    2 800 базовых станций — ориентир масштаба экспортного рынка. https://en.qxwz.com/
"""

import sys, os, csv
from typing import Dict, List
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PALETTE = ["#e17055", "#fdcb6e", "#0984e3", "#00b894",
           "#6c5ce7", "#74b9ff", "#dfe6e9", "#2d3436"]

RUB_PER_USD = 90.0   # согласован с cost_model

# ── Фактические тарифы операторов высокоточного позиционирования ($/год/устр.) ──
# Зарубежные сервисы коррекций + РОССИЙСКИЙ оператор EFT-CORS (для валидации ARPU).
# Перепроверено в июне 2026 (вживую). Skylark снижен: вендор даёт вход $29/мес;
# Nx RTK — от $15/мес на 5-летней предоплате (ранее в модели были $495/$699 годовых).
TARIFFS_USD_PER_YEAR = {
    "u-blox PointPerfect (мин.)":   180,    # $15/мес
    "u-blox PointPerfect (макс.)":  600,    # $50/мес
    "Swift Skylark":                348,    # $29/мес (swiftnav.com, 2026)
    "Swift Skylark Nx RTK":         180,    # от $15/мес на 5-летней предоплате (2026)
    "NovAtel TerraStar-L":          580,    # не перепроверен в 2026 (значение 2026-ранее)
    "NovAtel TerraStar-C Pro":      1_750,  # не перепроверен в 2026
    "EFT-CORS RTK (РФ)":            560,    # 50 400 ₽/год — подтверждён (eft-cors.ru, 2026)
}
# Опорный тариф высокоточного позиционирования, ₽/год/устройство.
# ПОДТВЕРЖДЁН фактическим тарифом РФ: EFT-CORS RTK = 50 400 ₽/год (≈ опорному ARPU);
# совпадает с медианой Skylark ($495) / TerraStar-L ($580).
ARPU_POSITIONING_RUB = 50_000.0          # ≈ $555/год

# Масштаб экспортного рынка (контекст, не тариф): Qianxun SI (КНР) — единый
# национальный оператор точного позиционирования: >390 млн клиентов, >1 млрд
# пользователей, 2 800 базовых станций, привлёк $816 млн (en.qxwz.com / PitchBook).
# Показывает потолок спроса единого рынка; экспорт АВРОРЫ берётся как малая доля.
QIANXUN_CUSTOMERS = 390e6

# Прямой сервисный аналог Сервиса Б (тайминг+анти-спуфинг), целевая выручка
IRIDIUM_STL_TARGET_USD_PER_YEAR = 100e6  # >$100M/год к 2030

# Экспортный ARPU (СНГ + дружественные рынки) — консервативно на уровне внутреннего
EXPORT_ARPU_RUB = 50_000.0

# ── Сценарии проникновения (выручка зрелой фазы по сегментам, млрд ₽/год) ──────
# positioning: число платящих устройств × ARPU_POSITIONING_RUB.
SCENARIOS = {
    "Пессимистический": {
        "pos_devices":  30_000,   # → 1,5 млрд ₽/год
        "timing":       1.0,      # Timing-as-a-Service (5G/ЦОД/энергетика/финтех)
        "auth":         0.5,      # аутентификация / анти-спуфинг (Сервис Б, КИИ/оборона)
        "arctic":       0.5,      # Арктика / Севморпуть / спецсервис
        "export_devices": 10_000, # экспорт: СНГ+друж. устройства × EXPORT_ARPU → 0,5 млрд
    },
    "Базовый": {
        "pos_devices":  100_000,  # → 5,0 млрд ₽/год
        "timing":       3.0,
        "auth":         4.0,
        "arctic":       1.0,
        "export_devices": 50_000, # → 2,5 млрд ₽/год (≈ 0,013% масштаба Qianxun)
    },
    "Оптимистический": {
        "pos_devices":  200_000,  # → 10,0 млрд ₽/год
        "timing":       5.0,
        "auth":         7.0,
        "arctic":       2.0,
        "export_devices": 150_000, # → 7,5 млрд ₽/год (≈ 0,04% масштаба Qianxun)
    },
}

# ── Затраты (из §51 ТП / cost_model), млрд ₽ ──────────────────────────────────
CAPEX_BRUB = {                 # капитальные, до глобального FOC (без OPEX/восполн.)
    "NRE":                     10.8,
    "Демонстраторы Ф0–1":      0.82,
    "Серия 300 КА + часы":     31.35,  # 29,85 (серия с обучением) + 1,50 (часы) — §51.2
    "Запуски":                 27.0,
    "Наземный сегмент":        7.2,
}
CAPEX_TOTAL_BRUB = sum(CAPEX_BRUB.values())   # ≈ 77,17 млрд ₽ (сверено с cost_model §51)
OPEX_BRUB_YEAR    = 3.6                        # §51
REPL_FULL_BRUB_YEAR = 5.16                     # 30 КА/год × 0,172 млрд ₽ (§51.3, полная гр.)

# ── Профиль развёртывания: доля группировки и зрелости выручки по годам ────────
# t — индекс года программы (1..15), старт 2026,5. Привязка к §30.2.
DEPLOY_FRACTION = {  # доля действующей группировки (для OPEX-восполнения)
    1: 0.01, 2: 0.04, 3: 0.30, 4: 0.30, 5: 0.60, 6: 0.60, 7: 0.85, 8: 1.0,
}
REVENUE_RAMP = {     # доля зрелой выручки (привязана к покрытию РФ / вводу нав.)
    1: 0.00, 2: 0.02, 3: 0.05, 4: 0.10, 5: 0.40, 6: 0.50, 7: 0.60, 8: 0.75,
    9: 0.85, 10: 0.95, 11: 1.0,
}
HORIZON_Y = 15
DISCOUNT_R = 0.12


def _frac(table: Dict[int, float], t: int) -> float:
    """Значение ступенчатого профиля на год t (последнее известное — далее)."""
    keys = sorted(table)
    val = table[keys[0]]
    for k in keys:
        if k <= t:
            val = table[k]
    return val


def maturity_revenue(scn: Dict) -> Dict[str, float]:
    """Выручка зрелой фазы по сегментам (млрд ₽/год)."""
    pos = scn["pos_devices"] * ARPU_POSITIONING_RUB / 1e9
    exp = scn["export_devices"] * EXPORT_ARPU_RUB / 1e9
    return {
        "Позиционирование PPP-RTK": pos,
        "Тайминг (TaaS)":           scn["timing"],
        "Аутентификация/анти-спуфинг": scn["auth"],
        "Арктика/спец":             scn["arctic"],
        "Экспорт (СНГ+друж.)":      exp,
    }


def annual_cashflow(scn: Dict) -> Dict[str, np.ndarray]:
    """Годовые потоки за горизонт: выручка, затраты, чистый и дисконтированный."""
    mat = sum(maturity_revenue(scn).values())
    years = np.arange(1, HORIZON_Y + 1)
    revenue = np.array([mat * _frac(REVENUE_RAMP, int(t)) for t in years])

    # CAPEX распределён по фазам ввода (профиль развёртывания до FOC, t=1..8)
    capex_profile = np.array([0.18, 0.10, 0.22, 0.10, 0.15, 0.08, 0.10, 0.07]
                             + [0.0] * (HORIZON_Y - 8))
    capex = CAPEX_TOTAL_BRUB * capex_profile

    opex = np.array([OPEX_BRUB_YEAR * (0.5 + 0.5 * _frac(DEPLOY_FRACTION, int(t)))
                     for t in years])  # наземный OPEX нарастает к FOC
    repl = np.array([REPL_FULL_BRUB_YEAR * _frac(DEPLOY_FRACTION, int(t))
                     for t in years])

    net = revenue - capex - opex - repl
    disc = np.array([net[i] / (1 + DISCOUNT_R) ** (i + 1) for i in range(HORIZON_Y)])
    return {"years": years, "revenue": revenue, "capex": capex, "opex": opex,
            "repl": repl, "net": net, "disc": disc,
            "cum_net": np.cumsum(net), "cum_disc": np.cumsum(disc),
            "maturity": mat}


def irr(net: np.ndarray) -> float:
    """IRR денежного потока (год 0 — без вложений; t=1..N) бисекцией."""
    def npv(rate):
        return sum(net[i] / (1 + rate) ** (i + 1) for i in range(len(net)))
    lo, hi = -0.9, 2.0
    if npv(lo) * npv(hi) > 0:
        return float("nan")  # знак не меняется — IRR не определён в диапазоне
    for _ in range(200):
        mid = (lo + hi) / 2
        if npv(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def payback_year(cum: np.ndarray) -> float:
    """Год, в котором накопленный поток впервые ≥ 0 (NaN если не достигнут)."""
    for i, v in enumerate(cum):
        if v >= 0:
            return float(i + 1)
    return float("nan")


def evaluate(scn: Dict) -> Dict:
    cf = annual_cashflow(scn)
    return {
        "maturity_Brub":   cf["maturity"],
        "npv_Brub":        float(cf["cum_disc"][-1]),
        "irr":             irr(cf["net"]),
        "payback_simple":  payback_year(cf["cum_net"]),
        "payback_disc":    payback_year(cf["cum_disc"]),
        "cf":              cf,
    }


# ── Графики ───────────────────────────────────────────────────────────────────
def _plot_tariffs(output_dir, label):
    names = list(TARIFFS_USD_PER_YEAR)
    usd = list(TARIFFS_USD_PER_YEAR.values())
    rub = [v * RUB_PER_USD / 1000 for v in usd]  # тыс ₽/год

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(names, rub, color=PALETTE[2], edgecolor="white")
    for b, u, r in zip(bars, usd, rub):
        ax.text(r + max(rub) * 0.01, b.get_y() + b.get_height() / 2,
                f"{r:.0f} тыс ₽  (${u})", va="center", fontsize=9)
    ax.axvline(ARPU_POSITIONING_RUB / 1000, ls="--", color=PALETTE[0], lw=1.6,
               label=f"Опорный ARPU АВРОРА = {ARPU_POSITIONING_RUB/1000:.0f} тыс ₽/год")
    ax.set_xlabel("Годовой тариф за устройство (тыс ₽)")
    ax.set_title(f"Тарифы операторов высокоточного позиционирования (РФ и зарубежные) [{label}]")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"econ_tariffs_benchmark_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_revenue_segments(output_dir, label):
    scns = list(SCENARIOS)
    segs = list(maturity_revenue(SCENARIOS[scns[0]]))
    data = {s: maturity_revenue(SCENARIOS[s]) for s in scns}

    fig, ax = plt.subplots(figsize=(11, 6))
    bottom = np.zeros(len(scns))
    for j, seg in enumerate(segs):
        vals = np.array([data[s][seg] for s in scns])
        ax.bar(scns, vals, bottom=bottom, label=seg,
               color=PALETTE[j], edgecolor="white")
        bottom += vals
    for i, s in enumerate(scns):
        tot = sum(data[s].values())
        ax.text(i, tot + 0.3, f"{tot:.1f} млрд ₽/год", ha="center", fontsize=10,
                fontweight="bold")
    ax.set_ylabel("Выручка зрелой фазы (млрд ₽/год)")
    ax.set_title(f"Выручка по сегментам и сценариям проникновения [{label}]")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"econ_revenue_segments_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_cashflow(results, output_dir, label):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    colors = {"Пессимистический": PALETTE[0], "Базовый": PALETTE[2],
              "Оптимистический": PALETTE[3]}
    for name, r in results.items():
        yrs = r["cf"]["years"]
        ax1.plot(yrs, r["cf"]["cum_disc"], lw=2.5, marker="o", ms=4,
                 color=colors[name],
                 label=f"{name}: NPV={r['npv_Brub']:.1f} млрд ₽")
    ax1.axhline(0, ls="--", color=PALETTE[7], lw=1.3)
    ax1.set_xlabel("Год программы (1 = 2026,5)")
    ax1.set_ylabel("Накопленный дисконтированный поток (млрд ₽)")
    ax1.set_title(f"Кумулятивный NPV (r={int(DISCOUNT_R*100)}%) [{label}]")
    ax1.legend(fontsize=9, loc="lower right")
    ax1.grid(alpha=0.3)

    base = results["Базовый"]["cf"]
    ax2.bar(base["years"] - 0.2, base["revenue"], width=0.4,
            color=PALETTE[3], label="Выручка (базовый)")
    ax2.bar(base["years"] + 0.2, base["capex"] + base["opex"] + base["repl"],
            width=0.4, color=PALETTE[0], label="Затраты (CAPEX+OPEX+восполн.)")
    ax2.set_xlabel("Год программы")
    ax2.set_ylabel("млрд ₽/год")
    ax2.set_title("Годовые потоки — базовый сценарий")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"econ_cashflow_npv_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_integral(results, output_dir, label):
    """Честный мост ценности (базовый сценарий, NPV-горизонт 15 лет):
    коммерческий NPV (< 0) → + импортозамещение часов (реально избегаемая
    закупка) → остаточный экономический разрыв, который закрывается НЕ
    коммерчески: бюджетным мандатом, стратегической ценностью и предотвращённым
    ущербом от подавления/подмены GNSS (последний здесь НЕ монетизирован).
    """
    base_npv = results["Базовый"]["npv_Brub"]
    import_sub = 18.9            # избегаемая закупка западных часов, §51.3 ТП
    gap = base_npv + import_sub  # остаточный разрыв (обычно < 0)

    fig, ax = plt.subplots(figsize=(11, 6))
    # 1) коммерческий NPV (отрицательный)
    ax.bar("Коммерческий\nNPV (базовый)", base_npv, color=PALETTE[0],
           edgecolor="white")
    ax.text(0, base_npv / 2, f"{base_npv:.1f}", ha="center", va="center",
            fontsize=11, fontweight="bold", color="white")
    # 2) вклад импортозамещения (мост вверх от base_npv)
    ax.bar("Импортозамещение\nчасов (+)", import_sub, bottom=base_npv,
           color=PALETTE[3], edgecolor="white")
    ax.text(1, base_npv + import_sub + 1, f"+{import_sub:.1f}", ha="center", fontsize=10)
    # 3) остаточный экономический разрыв
    ax.bar("Остаточный разрыв\n(стратег./бюджет)", gap, color=PALETTE[4],
           edgecolor="white")
    ax.text(2, gap - 3, f"{gap:.1f}", ha="center", fontsize=10, fontweight="bold")

    ax.axhline(0, ls="-", color=PALETTE[7], lw=1.2)
    ax.set_ylabel("NPV-эквивалент за 15 лет (млрд ₽)")
    ax.set_title(f"Мост ценности: коммерция + импортозамещение → остаточный "
                 f"разрыв [{label}]")
    ax.annotate("закрывается стратегической ценностью\nи предотвращённым ущербом "
                "(не монетизирован)",
                xy=(2, gap), xytext=(1.1, gap - 18), fontsize=9,
                arrowprops=dict(arrowstyle="->", color=PALETTE[7]))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"econ_integral_value_{label}.png"), dpi=150)
    plt.close(fig)


def _save_csv(results, output_dir, label):
    path = os.path.join(output_dir, f"econ_summary_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Сценарий", "Выручка зрелой фазы, млрд ₽/год",
                    "NPV(15 лет, r=12%), млрд ₽", "IRR", "Срок окуп. (простой), лет",
                    "Срок окуп. (дисконт.), лет"])
        for name, r in results.items():
            w.writerow([name, f"{r['maturity_Brub']:.1f}", f"{r['npv_Brub']:.1f}",
                        ("—" if np.isnan(r["irr"]) else f"{r['irr']*100:.1f}%"),
                        ("> гориз." if np.isnan(r["payback_simple"]) else f"{r['payback_simple']:.0f}"),
                        ("> гориз." if np.isnan(r["payback_disc"]) else f"{r['payback_disc']:.0f}")])
    return path


def run_econ_analysis(output_dir: str, label: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    results = {name: evaluate(scn) for name, scn in SCENARIOS.items()}
    _plot_tariffs(output_dir, label)
    _plot_revenue_segments(output_dir, label)
    _plot_cashflow(results, output_dir, label)
    _plot_integral(results, output_dir, label)
    csv_path = _save_csv(results, output_dir, label)

    print(f"[econ] CAPEX (до FOC) = {CAPEX_TOTAL_BRUB:.1f} млрд ₽; "
          f"экспл. зрелой фазы ≈ {OPEX_BRUB_YEAR + REPL_FULL_BRUB_YEAR:.1f} млрд ₽/год")
    for name, r in results.items():
        irr_s = "—" if np.isnan(r["irr"]) else f"{r['irr']*100:.1f}%"
        ps = "> гориз." if np.isnan(r["payback_simple"]) else f"{r['payback_simple']:.0f} лет"
        pd_ = "> гориз." if np.isnan(r["payback_disc"]) else f"{r['payback_disc']:.0f} лет"
        print(f"[econ] {name:18s}: выручка {r['maturity_Brub']:5.1f} млрд ₽/год | "
              f"NPV {r['npv_Brub']:7.1f} млрд ₽ | IRR {irr_s:>6s} | "
              f"окуп. прост. {ps:>9s} | дискон. {pd_:>9s}")
    print(f"[econ] CSV: {csv_path}")
    return results


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "..", "..", "results", "econ")
    run_econ_analysis(os.path.abspath(out), "phase5")
