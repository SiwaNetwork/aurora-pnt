"""
Монте-Карло анализ сквозного бюджета ошибок PVT для AURORA PNT.

Связывает все частные бюджеты ошибок (часы спутника, эфемериды/SISRE,
ионосфера, тропосфера, многолучёвость, шум приёмника) в единый бюджет
UERE по трём сервисным уровням (PPP-RTK / PPP / одночастотный),
и через коэффициенты DOP плотного LEO-созвездия (300 КА) переводит
UERE в горизонтальную/вертикальную ошибку местоопределения методом
Монте-Карло (N = 10000 розыгрышей).

Параметры системы: Walker Delta 300/15, h = 1000 км, i = 75°, L1+L5.
Скорректированный бюджет линии: C/N0 = 52,6 дБ-Гц в зените.

Ссылки:
  Kaplan & Hegarty (2017) — Understanding GPS/GNSS: Principles and Applications, 3rd ed.
  Misra & Enge (2011) — Global Positioning System: Signals, Measurements, Performance.
  RTCA DO-229E (2016) — MOPS for GPS/SBAS Airborne Equipment (UERE budget).
  Teunissen & Montenbruck (2017) — Springer Handbook of Global Navigation Satellite Systems.
"""

import math, os, csv
from typing import Dict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Палитра проекта ──────────────────────────────────────────────────────────
PALETTE = ["#e17055", "#fdcb6e", "#0984e3", "#00b894",
           "#6c5ce7", "#74b9ff", "#dfe6e9", "#2d3436"]

# ── Компоненты бюджета ошибок UERE (1σ, метры) по сервисным уровням ──────────
#   Порядок ключей фиксирован — используется в торнадо-диаграмме.
ERROR_BUDGET = {
    "PPP-RTK": {
        "Остаточная ошибка часов КА":      0.02,
        "Эфемериды (SISRE)":               0.05,
        "Ионосфера (двухчаст. остаток)":   0.01,
        "Тропосфера (моделирована)":       0.04,
        "Многолучёвость (открытое небо)":  0.10,
        "Шум приёмника":                   0.05,
    },
    "PPP": {
        "Остаточная ошибка часов КА":      0.10,
        "Эфемериды (SISRE)":               0.15,
        "Ионосфера (двухчаст. остаток)":   0.02,
        "Тропосфера (моделирована)":       0.04,
        "Многолучёвость (открытое небо)":  0.10,
        "Шум приёмника":                   0.10,
    },
    "Одночастотный": {
        "Остаточная ошибка часов КА":      0.30,
        "Эфемериды (SISRE)":               0.50,
        "Ионосфера (Klobuchar 50%)":       0.80,
        "Тропосфера (моделирована)":       0.04,
        "Многолучёвость (город)":          0.30,
        "Шум приёмника":                   0.30,
    },
}

# ── Коэффициенты DOP — номинал плотного LEO AURORA (300 КА) ─────────────────
HDOP_NOM = 1.1
VDOP_NOM = 1.8
PDOP_NOM = 2.1

TIER_COLORS = {
    "PPP-RTK":        "#6c5ce7",
    "PPP":            "#0984e3",
    "Одночастотный":  "#e17055",
}

N_MC = 10000              # число розыгрышей Монте-Карло
ELEV_MASKS = [5, 10, 15, 20]   # углы маски возвышения, град


def _uere(tier: str) -> float:
    """RSS всех компонент бюджета ошибок -> UERE (м, 1σ)."""
    comps = ERROR_BUDGET[tier]
    return math.sqrt(sum(v * v for v in comps.values()))


def _mask_hdop_factor(mask_deg: float) -> float:
    """Деградация HDOP с ростом маски возвышения: 1 + (mask-10)/40."""
    return 1.0 + (mask_deg - 10.0) / 40.0


def run_pvt_montecarlo_analysis(output_dir: str, label: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(2026)

    results: Dict = {"tiers": {}, "hdop": HDOP_NOM,
                     "vdop": VDOP_NOM, "pdop": PDOP_NOM}

    mc_data = {}   # tier -> dict(horiz, vert)
    for tier in ERROR_BUDGET:
        comps = ERROR_BUDGET[tier]
        sigmas = np.array(list(comps.values()))
        # Каждая компонента ~ N(0, sigma_i); агрегируем в ошибку дальности
        draws = rng.normal(0.0, 1.0, size=(N_MC, sigmas.size)) * sigmas
        range_err = draws.sum(axis=1)              # суммарная ошибка псевдодальности
        uere_emp = range_err.std()

        # Геометрическое преобразование: ошибка по осям ~ DOP * UERE
        ue = _uere(tier)
        ex = rng.normal(0.0, HDOP_NOM / math.sqrt(2) * ue, N_MC)
        ey = rng.normal(0.0, HDOP_NOM / math.sqrt(2) * ue, N_MC)
        ez = rng.normal(0.0, VDOP_NOM * ue, N_MC)
        horiz = np.hypot(ex, ey)
        vert = np.abs(ez)

        cep50 = np.percentile(horiz, 50)
        cep95 = np.percentile(horiz, 95)
        vert95 = np.percentile(vert, 95)

        results["tiers"][tier] = {
            "uere_m":    ue,
            "uere_emp":  uere_emp,
            "cep50_m":   cep50,
            "cep95_m":   cep95,
            "vert95_m":  vert95,
        }
        mc_data[tier] = {"horiz": horiz, "vert": vert}

    # CEP95 vs маска возвышения
    mask_cep95 = {}
    for tier in ERROR_BUDGET:
        ue = _uere(tier)
        row = []
        for m in ELEV_MASKS:
            hdop_m = HDOP_NOM * _mask_hdop_factor(m)
            ex = rng.normal(0.0, hdop_m / math.sqrt(2) * ue, N_MC)
            ey = rng.normal(0.0, hdop_m / math.sqrt(2) * ue, N_MC)
            row.append(np.percentile(np.hypot(ex, ey), 95))
        mask_cep95[tier] = row
    results["mask_cep95"] = mask_cep95
    results["elev_masks"] = ELEV_MASKS

    _plot_error_cdf(mc_data, output_dir, label)
    _plot_uere_tornado(output_dir, label)
    _plot_vs_elevation_mask(mask_cep95, output_dir, label)
    _plot_error_box(mc_data, output_dir, label)
    _save_csv(results, output_dir, label)
    return results


def _plot_error_cdf(mc_data, output_dir, label):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    for tier, d in mc_data.items():
        c = TIER_COLORS[tier]
        for ax, key in ((ax1, "horiz"), (ax2, "vert")):
            x = np.sort(d[key])
            y = np.arange(1, x.size + 1) / x.size * 100.0
            ax.plot(x, y, color=c, lw=2.4, label=tier)

    for ax in (ax1, ax2):
        ax.axhline(50, ls=":", color="#dfe6e9", lw=1.2)
        ax.axhline(95, ls=":", color="#2d3436", lw=1.0)
        ax.set_ylabel("Накопленная вероятность (%)")
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_xscale("log")
    ax1.set_xlabel("Горизонтальная ошибка (м)")
    ax1.set_title(f"CDF горизонтальной ошибки PVT [{label}]")
    ax2.set_xlabel("Вертикальная ошибка (м)")
    ax2.set_title(f"CDF вертикальной ошибки PVT [{label}]")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"pvt_error_cdf_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_uere_tornado(output_dir, label):
    tier = "PPP-RTK"
    comps = ERROR_BUDGET[tier]
    names = list(comps.keys())
    sig = np.array(list(comps.values()))
    order = np.argsort(sig)              # снизу вверх по возрастанию
    names = [names[i] for i in order]
    sig = sig[order]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(names))]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(names, sig, color=colors, edgecolor="white", height=0.6)
    for bar, v in zip(bars, sig):
        ax.text(bar.get_width() + max(sig) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.3f} м", va="center", fontsize=9)
    ue = _uere(tier)
    ax.axvline(ue, ls="--", color="#2d3436", lw=1.5,
               label=f"UERE (RSS) = {ue:.3f} м")
    ax.set_xlabel("Вклад в бюджет ошибок, 1σ (м)")
    ax.set_title(f"Торнадо-диаграмма UERE — уровень {tier} [{label}]")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"uere_tornado_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_vs_elevation_mask(mask_cep95, output_dir, label):
    fig, ax = plt.subplots(figsize=(11, 6))
    for tier, vals in mask_cep95.items():
        ax.plot(ELEV_MASKS, vals, "o-", color=TIER_COLORS[tier],
                lw=2.4, ms=8, label=tier)
        for m, v in zip(ELEV_MASKS, vals):
            ax.annotate(f"{v:.2f}", (m, v), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xlabel("Угол маски возвышения (°)")
    ax.set_ylabel("Горизонтальная CEP95 (м)")
    ax.set_title(f"CEP95 vs маска возвышения [{label}]")
    ax.set_xticks(ELEV_MASKS)
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"pvt_vs_elevation_mask_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_error_box(mc_data, output_dir, label):
    tiers = list(mc_data.keys())
    data = [mc_data[t]["horiz"] for t in tiers]
    colors = [TIER_COLORS[t] for t in tiers]

    fig, ax = plt.subplots(figsize=(11, 6))
    bp = ax.boxplot(data, vert=True, patch_artist=True,
                    showfliers=False, widths=0.55,
                    medianprops=dict(color="#2d3436", lw=2))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax.set_xticklabels(tiers)
    ax.set_ylabel("Горизонтальная ошибка (м)")
    ax.set_yscale("log")
    ax.set_title(f"Распределение горизонтальной ошибки по уровням [{label}]")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"pvt_error_box_{label}.png"), dpi=150)
    plt.close(fig)


def _save_csv(results, output_dir, label):
    path = os.path.join(output_dir, f"pvt_montecarlo_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tier", "uere_m", "uere_emp_m",
                    "cep50_m", "cep95_m", "vert95_m",
                    "hdop", "vdop", "pdop"])
        for tier, r in results["tiers"].items():
            w.writerow([tier,
                        f"{r['uere_m']:.4f}",
                        f"{r['uere_emp']:.4f}",
                        f"{r['cep50_m']:.4f}",
                        f"{r['cep95_m']:.4f}",
                        f"{r['vert95_m']:.4f}",
                        f"{results['hdop']:.2f}",
                        f"{results['vdop']:.2f}",
                        f"{results['pdop']:.2f}"])
        w.writerow([])
        w.writerow(["tier"] + [f"cep95_mask{m}deg_m" for m in results["elev_masks"]])
        for tier, vals in results["mask_cep95"].items():
            w.writerow([tier] + [f"{v:.4f}" for v in vals])


def print_pvt_montecarlo_summary(label: str, results: Dict) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  PVT Monte-Carlo Error Budget -- {label}")
    print(sep)
    print(f"  DOP (плотный LEO 300 КА): HDOP={results['hdop']:.2f}  "
          f"VDOP={results['vdop']:.2f}  PDOP={results['pdop']:.2f}")
    print(f"  Розыгрышей Монте-Карло: N = {N_MC}")
    print(f"  {'Уровень':<16} {'UERE(м)':>9} {'CEP50(м)':>10} "
          f"{'CEP95(м)':>10} {'Vert95(м)':>11}")
    print(f"  {'-' * 60}")
    for tier, r in results["tiers"].items():
        print(f"  {tier:<16} {r['uere_m']:>9.3f} {r['cep50_m']:>10.3f} "
              f"{r['cep95_m']:>10.3f} {r['vert95_m']:>11.3f}")
    print(f"  {'-' * 60}")
    print("  Проверка: PPP-RTK CEP95 ~ 0,1-0,3 м; одночастотный ~ 1-3 м")
    print(sep)


if __name__ == "__main__":
    r = run_pvt_montecarlo_analysis("results/pvt_montecarlo", "phase5")
    print_pvt_montecarlo_summary("phase5", r)
