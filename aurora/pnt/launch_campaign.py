"""
Кампания запусков и стратегия развёртывания созвездия АВРОРА.

Рассчитывает:
- Манифест пусков: малый РН (20 КА) vs средний РН (40 КА), 300 КА
- Фазирование плоскостей по RAAN дифференциальным дрейфом J2:
  выведение на низкую парковочную орбиту → быстрая прецессия узла →
  набор ΔΩ между плоскостями → подъём на рабочую орбиту
- Бюджет ΔV: подъём орбиты (Гоман 300→1000 км), штраф фазирования,
  доведение/циркуляризация
- Сроки заполнения Walker Delta vs темп пусков (4/год, 8/год)

Система: 300 КА, 15 плоскостей (20 КА/плоскость), h=1000 км, i=75°,
масса КА 140 кг, проектный срок 7 лет.

Ссылки:
  Vallado D.A. (2013) — Fundamentals of Astrodynamics and Applications, 4th ed.
    Microcosm Press. (J2-прецессия узла, Гомановский переход, ΔV).
  Wertz, Everett & Puschell (2011) — The New SMAD. Гл. 7 (orbit phasing).
  Walker J.G. (1984) — Satellite constellations. J. British Interpl. Soc.
"""

import sys, math, os, csv
from typing import Dict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Windows cp1251 -> UTF-8: безопасный вывод кириллицы и спецсимволов (ΔV, Ω̇, →)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PALETTE = ["#e17055", "#fdcb6e", "#0984e3", "#00b894",
           "#6c5ce7", "#74b9ff", "#dfe6e9", "#2d3436"]

# ── Константы и параметры орбиты ──────────────────────────────────────────────
MU      = 398600.4418      # км^3/с^2, грав. параметр Земли
RE      = 6378.137         # км, экв. радиус
J2      = 1.08263e-3
N_SATS   = 300
N_PLANES = 15
SAT_WET_KG = 140.0
INC_DEG  = 75.0
H_OPER   = 1000.0          # км, рабочая высота
H_PARK   = 300.0           # км, парковочная/дрейфовая высота
DRAAN_BETWEEN_DEG = 360.0 / N_PLANES   # 24° между соседними плоскостями

LV_OPTIONS = {
    "Малый РН (20 КА/пуск)":   20,
    "Средний РН (40 КА/пуск)": 40,
}


def raan_dot_deg_per_day(h_km: float, inc_deg: float) -> float:
    """Скорость прецессии RAAN от J2: Ω̇ = -1.5 n J2 (Re/p)^2 cos i (круг. орбита)."""
    a = RE + h_km
    n = math.sqrt(MU / a ** 3)            # рад/с
    i = math.radians(inc_deg)
    raan_dot = -1.5 * n * J2 * (RE / a) ** 2 * math.cos(i)   # рад/с
    return math.degrees(raan_dot) * 86400.0                  # град/сут


def hohmann_dv(h1_km: float, h2_km: float) -> Dict:
    """ΔV Гомановского перехода между круговыми орбитами h1->h2 (м/с)."""
    r1 = RE + h1_km
    r2 = RE + h2_km
    v1 = math.sqrt(MU / r1)
    v2 = math.sqrt(MU / r2)
    a_t = 0.5 * (r1 + r2)
    vp = math.sqrt(MU * (2.0 / r1 - 1.0 / a_t))   # перигей переходной
    va = math.sqrt(MU * (2.0 / r2 - 1.0 / a_t))   # апогей переходной
    dv1 = abs(vp - v1) * 1000.0
    dv2 = abs(v2 - va) * 1000.0
    return {"dv1": dv1, "dv2": dv2, "total": dv1 + dv2}


def run_launch_campaign_analysis(output_dir: str, label: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    manifest = {}
    for nm, per in LV_OPTIONS.items():
        n_launch = math.ceil(N_SATS / per)
        manifest[nm] = {"per_launch": per, "n_launch": n_launch}

    # дифференциальный дрейф RAAN: разница темпов прецессии на парковочной vs рабочей
    rd_park = raan_dot_deg_per_day(H_PARK, INC_DEG)
    rd_oper = raan_dot_deg_per_day(H_OPER, INC_DEG)
    d_rate  = rd_park - rd_oper                       # град/сут (отн.)
    days_per_plane = abs(DRAAN_BETWEEN_DEG / d_rate)  # сут на разнос в 24°

    # бюджет ΔV
    hoh = hohmann_dv(H_PARK, H_OPER)
    raan_phasing_penalty = 30.0    # м/с — оценка на коррекции в ходе дрейфа
    circularisation      = 15.0    # м/с — финальное доведение
    dv_budget = {
        "Подъём орбиты (перигей)": hoh["dv1"],
        "Подъём орбиты (апогей)":  hoh["dv2"],
        "Штраф RAAN-фазирования":  raan_phasing_penalty,
        "Циркуляризация/доведение": circularisation,
    }
    dv_total = sum(dv_budget.values())

    # сроки заполнения: малый РН, темп 4/год и 8/год
    n_launch_small = manifest["Малый РН (20 КА/пуск)"]["n_launch"]
    months_full = {
        "4 пуска/год": n_launch_small / 4.0 * 12.0,
        "8 пусков/год": n_launch_small / 8.0 * 12.0,
    }

    results = {
        "manifest":           manifest,
        "raan_dot_park":      rd_park,
        "raan_dot_oper":      rd_oper,
        "raan_diff_rate":     d_rate,
        "days_per_plane_24deg": days_per_plane,
        "dv_budget_mps":      dv_budget,
        "dv_total_mps":       dv_total,
        "months_to_full":     months_full,
        "n_launch_small":     n_launch_small,
        "n_launch_medium":    manifest["Средний РН (40 КА/пуск)"]["n_launch"],
    }

    _plot_launch_manifest(manifest, output_dir, label)
    _plot_raan_phasing(d_rate, days_per_plane, output_dir, label)
    _plot_deployment_timeline(n_launch_small, output_dir, label)
    _plot_dv_budget(dv_budget, dv_total, output_dir, label)
    _save_csv(results, output_dir, label)
    return results


def _plot_launch_manifest(manifest, output_dir, label):
    names    = list(manifest.keys())
    n_launch = [manifest[n]["n_launch"] for n in names]
    per      = [manifest[n]["per_launch"] for n in names]
    sats_per_plane = N_SATS // N_PLANES

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    bars = ax1.bar(names, n_launch, color=[PALETTE[2], PALETTE[0]],
                   edgecolor="white", width=0.5)
    for b, v, p in zip(bars, n_launch, per):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.3,
                 f"{v} пусков\n({p} КА/пуск)", ha="center", fontsize=10)
    ax1.set_ylabel("Число пусков для 300 КА")
    ax1.set_title(f"Манифест пусков — варианты РН [{label}]")
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(["КА/плоскость"], [sats_per_plane], color=PALETTE[3],
            edgecolor="white", width=0.4)
    ax2.text(0, sats_per_plane + 0.3, f"{sats_per_plane} КА",
             ha="center", fontsize=12)
    ax2.set_ylabel("Спутников на плоскость")
    ax2.set_title(f"Walker Delta: {N_PLANES} плоскостей × "
                  f"{sats_per_plane} КА = {N_SATS}")
    ax2.set_ylim(0, sats_per_plane * 1.4)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"launch_manifest_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_raan_phasing(d_rate, days_per_plane, output_dir, label):
    """Фазирование 15 плоскостей дифференциальным J2-дрейфом.

    Каждая плоскость k должна набрать относительный RAAN = k·24°.
    Все КА после общего пуска находятся на низкой дрейфовой орбите
    (~300 км), где Ω̇ отличается от рабочей (1000 км) на |Δ| °/сут.
    Плоскость k остаётся на дрейфовой орбите, пока не накопит k·24°,
    затем поднимается на рабочую орбиту и фиксирует свой RAAN (плато).
    Последняя плоскость (k=14) набирает 14·24°=336° — это и есть
    реальная длительность кампании ≈ 336/|Δ| ≈ 520 сут.
    """
    rate = abs(d_rate)                              # °/сут, относит. дрейф
    targets = np.arange(N_PLANES) * DRAAN_BETWEEN_DEG   # 0,24,...,336°
    t_full = targets[-1] / rate                     # сут до последней плоскости
    one_step_days = DRAAN_BETWEEN_DEG / rate        # ~37 сут на один шаг 24°

    days = np.linspace(0, t_full * 1.05, 1200)

    fig, ax = plt.subplots(figsize=(12, 7))
    cmap = plt.get_cmap("viridis")

    for k in range(N_PLANES):
        target = targets[k]
        # на дрейфовой орбите RAAN растёт линейно до target, затем плато
        raan = np.minimum(days * rate, target)
        col = cmap(k / (N_PLANES - 1))
        ax.plot(days, raan, lw=1.4, color=col, alpha=0.95, zorder=2)
        # маркер момента подъёма на рабочую орбиту (выход на плато)
        t_done = target / rate
        ax.plot(t_done, target, "o", color=col, ms=5, zorder=3,
                markeredgecolor="white", markeredgewidth=0.6)

    # подписи только для первой и последней плоскости
    ax.annotate(f"Плоскость 1\n(24°, ~{one_step_days:.0f} сут)",
                xy=(targets[1] / rate, targets[1]),
                xytext=(t_full * 0.16, 70), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#2d3436", lw=1.0))
    ax.annotate(f"Плоскость 14\n(336°, ~{t_full:.0f} сут)",
                xy=(t_full, targets[-1]),
                xytext=(t_full * 0.50, 250), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#2d3436", lw=1.0))

    # опорная линия полного разноса 336°
    ax.axhline(targets[-1], ls="--", color="#e17055", lw=1.6, zorder=1,
               label=f"Полный разнос {N_PLANES} плоскостей "
                     f"({targets[-1]:.0f}° = 14×24°)")
    # вертикаль полного развёртывания
    ax.axvline(t_full, ls=":", color="#636e72", lw=1.3, zorder=1,
               label=f"Полное фазирование ≈ {t_full:.0f} сут")

    ax.set_xlabel("Время с момента общего пуска (сут)")
    ax.set_ylabel("Относительный RAAN плоскости (°)")
    ax.set_xlim(0, t_full * 1.05)
    ax.set_ylim(0, 360)
    ax.set_yticks(np.arange(0, 361, 24))
    ax.set_title(
        f"Фазирование 15 плоскостей дифференциальным J2-дрейфом [{label}]\n"
        f"Ω̇_дрейф−Ω̇_раб = {rate:.3f}°/сут, шаг 24° "
        f"(~{one_step_days:.0f} сут/шаг), полное развёртывание "
        f"≈ {t_full:.0f} сут", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)

    # колорбар по индексу плоскости
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=0, vmax=N_PLANES - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.015)
    cbar.set_label("Индекс орбитальной плоскости k (цель = k×24°)",
                   fontsize=9)
    cbar.set_ticks(np.arange(0, N_PLANES, 2))

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"raan_phasing_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_deployment_timeline(n_launch_small, output_dir, label):
    fig, ax = plt.subplots(figsize=(11, 6))
    for idx, (cad_name, per_year) in enumerate(
            (("4 пуска/год", 4), ("8 пусков/год", 8))):
        m_full = n_launch_small / per_year * 12.0
        months = np.linspace(0, m_full, 300)
        launches_done = np.floor(months / 12.0 * per_year)
        fill = np.minimum(launches_done * 20 / N_SATS * 100, 100)
        ax.plot(months, fill, color=PALETTE[idx * 3], lw=2.5,
                label=f"{cad_name} → {m_full:.0f} мес")
    ax.axhline(100, ls=":", color=PALETTE[7], lw=1.3, label="100% (15 плоскостей)")
    ax.set_xlabel("Месяцы от первого пуска")
    ax.set_ylabel("Заполнение созвездия (%)")
    ax.set_title(f"Сроки развёртывания vs темп пусков (малый РН) [{label}]")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"deployment_timeline_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_dv_budget(dv_budget, dv_total, output_dir, label):
    comps  = list(dv_budget.keys())
    vals   = list(dv_budget.values())
    colors = PALETTE[:len(comps)]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(comps, vals, color=colors, edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + dv_total * 0.01,
                f"{v:.0f}", ha="center", fontsize=10)
    ax.axhline(dv_total, ls="--", color=PALETTE[7], lw=1.5,
               label=f"Итого ΔV = {dv_total:.0f} м/с")
    ax.set_ylabel("ΔV (м/с)")
    ax.set_title(f"Бюджет ΔV развёртывания (парковка {H_PARK:.0f} → "
                 f"{H_OPER:.0f} км) [{label}]")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"dv_raise_budget_{label}.png"), dpi=150)
    plt.close(fig)


def _save_csv(results, output_dir, label):
    path = os.path.join(output_dir, f"launch_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Вариант РН", "КА/пуск", "Число пусков"])
        for nm, m in results["manifest"].items():
            w.writerow([nm, m["per_launch"], m["n_launch"]])
        w.writerow([])
        w.writerow(["параметр", "значение", "ед."])
        w.writerow(["Ω̇ на парковке", f"{results['raan_dot_park']:.4f}", "°/сут"])
        w.writerow(["Ω̇ на рабочей",  f"{results['raan_dot_oper']:.4f}", "°/сут"])
        w.writerow(["Отн. скорость дрейфа", f"{results['raan_diff_rate']:.4f}", "°/сут"])
        w.writerow(["Дней на разнос 24°", f"{results['days_per_plane_24deg']:.1f}", "сут"])
        w.writerow([])
        w.writerow(["Компонента ΔV", "м/с"])
        for k, v in results["dv_budget_mps"].items():
            w.writerow([k, f"{v:.1f}"])
        w.writerow(["ΔV ИТОГО", f"{results['dv_total_mps']:.1f}"])
        w.writerow([])
        w.writerow(["Темп пусков", "мес до полного созвездия"])
        for k, v in results["months_to_full"].items():
            w.writerow([k, f"{v:.1f}"])


def print_launch_campaign_summary(label: str, results: Dict) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  Launch Campaign & Deployment -- {label}")
    print(sep)
    print(f"  {'Вариант РН':<28} {'КА/пуск':>8} {'Пусков':>8}")
    print(f"  {'':-<46}")
    for nm, m in results["manifest"].items():
        print(f"  {nm:<28} {m['per_launch']:>8} {m['n_launch']:>8}")
    print(f"  Ω̇ парковка {results['raan_dot_park']:.3f}°/сут | "
          f"рабочая {results['raan_dot_oper']:.3f}°/сут")
    print(f"  Отн. дрейф RAAN: {results['raan_diff_rate']:.3f}°/сут "
          f"→ {results['days_per_plane_24deg']:.0f} сут на 24°")
    print(f"  {'ΔV компонента':<28} {'м/с':>8}")
    print(f"  {'':-<38}")
    for k, v in results["dv_budget_mps"].items():
        print(f"  {k:<28} {v:>8.0f}")
    print(f"  ΔV ИТОГО:                   {results['dv_total_mps']:>8.0f} м/с")
    for k, v in results["months_to_full"].items():
        print(f"  {k:<14}: {v:.0f} мес до полного созвездия")
    print(sep)
