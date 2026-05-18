"""
Производство и сборка-интеграция-испытания (AIT) серии 300 спутников AURORA PNT.

Рассчитывает:
- Поток AIT одного спутника по стадиям (Gantt)
- Такт-время (takt) для программы и число параллельных AIT-линий
- Кривую обучения по трудозатратам AIT
- Темп производства при 1/2/3/4 линиях (накопленный выпуск)
- График развёртывания: производство vs пусковые партии (20 КА/пуск)

Все длительности и темпы — оценочные параметрические для серийной фабрики
малых КА (concurrent / pulse-line производство, по аналогии OneWeb/Starlink).

Ссылки:
  Wertz, Everett & Puschell (2011) — Space Mission Engineering: The New SMAD.
    Microcosm Press. Гл. 19 (производство, AIT, serial production lines).
  ECSS-E-ST-10-03C (2012) — Testing (AIT flow: TVAC, vibration).
  Crawford J.R. (1944) — Learning curve в серийном производстве.
"""

import sys, math, os, csv
from typing import Dict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Windows cp1251 -> UTF-8: безопасный вывод кириллицы и спецсимволов
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PALETTE = ["#e17055", "#fdcb6e", "#0984e3", "#00b894",
           "#6c5ce7", "#74b9ff", "#dfe6e9", "#2d3436"]

N_SATS   = 300
N_PLANES = 15
SATS_PER_LAUNCH = 20

# ── Поток AIT одного КА (длительности по стадиям, дни) ────────────────────────
AIT_STAGES = {
    "Интеграция КА":                 10,
    "Функциональные испытания":       7,
    "Термовакуум + вибрация (TVAC)": 14,
    "Финальная сдача + отгрузка":     4,
}
AIT_DAYS_TOTAL = sum(AIT_STAGES.values())   # 35 дней на 1 КА (поток)

TARGET_PROGRAM_MONTHS = 36          # целевая длительность сборки серии
DAYS_PER_MONTH        = 30.0
LEARNING_B_AIT        = 0.90        # кривая обучения по трудозатратам AIT
WORK_DAYS_PER_MONTH   = 21.0        # рабочих дней в месяце на одной линии


def takt_time_days() -> float:
    """Такт-время: сколько дней должно сходить с линии на 1 КА для цели 36 мес."""
    return TARGET_PROGRAM_MONTHS * DAYS_PER_MONTH / N_SATS


def lines_needed() -> int:
    """Число параллельных AIT-линий, чтобы пропускная способность >= цели."""
    # каждая линия выпускает 1 КА за «бутылочное горлышко» стадии (TVAC=14 дн)
    bottleneck = max(AIT_STAGES.values())
    takt = takt_time_days()
    return max(1, math.ceil(bottleneck / takt))


def throughput_sats_per_month(n_lines: int) -> float:
    """Темп выпуска (КА/мес) при n_lines линиях; узкое место — TVAC-стадия."""
    bottleneck = max(AIT_STAGES.values())   # дней на КА на одной линии
    return n_lines * WORK_DAYS_PER_MONTH / bottleneck


def months_to_complete(n_lines: int) -> float:
    return N_SATS / throughput_sats_per_month(n_lines)


def run_production_analysis(output_dir: str, label: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    takt   = takt_time_days()
    n_line = lines_needed()
    rates  = {n: throughput_sats_per_month(n) for n in (1, 2, 3, 4)}
    months = {n: months_to_complete(n) for n in (1, 2, 3, 4)}

    results = {
        "ait_stages_days":      dict(AIT_STAGES),
        "ait_days_total":       AIT_DAYS_TOTAL,
        "takt_time_days":       takt,
        "lines_needed":         n_line,
        "throughput_per_month": rates,
        "months_to_300":        months,
        "months_to_300_chosen": months[n_line],
        "learning_b_ait":       LEARNING_B_AIT,
        "target_program_months": TARGET_PROGRAM_MONTHS,
    }

    _plot_ait_flow(output_dir, label)
    _plot_production_rate(output_dir, label)
    _plot_factory_throughput(rates, output_dir, label)
    _plot_deployment_schedule(n_line, output_dir, label)
    _save_csv(results, output_dir, label)
    return results


def _plot_ait_flow(output_dir, label):
    stages = list(AIT_STAGES.keys())
    durs   = list(AIT_STAGES.values())
    starts = np.cumsum([0] + durs[:-1])
    colors = PALETTE[:len(stages)]

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (s, d, st) in enumerate(zip(stages, durs, starts)):
        ax.barh(i, d, left=st, color=colors[i], edgecolor="white", height=0.55)
        ax.text(st + d / 2, i, f"{d} дн", ha="center", va="center",
                fontsize=10, color="white", fontweight="bold")
    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels(stages)
    ax.invert_yaxis()
    ax.axvline(AIT_DAYS_TOTAL, ls="--", color=PALETTE[7], lw=1.5,
               label=f"Цикл AIT 1 КА = {AIT_DAYS_TOTAL} дн")
    ax.set_xlabel("Время от старта AIT (дни)")
    ax.set_title(f"Поток AIT одного спутника (Gantt) [{label}]")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"ait_flow_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_production_rate(output_dir, label):
    fig, ax = plt.subplots(figsize=(11, 6))
    for idx, nl in enumerate((1, 2, 3, 4)):
        rate = throughput_sats_per_month(nl)
        m_full = months_to_complete(nl)
        months = np.linspace(0, m_full, 200)
        produced = np.minimum(months * rate, N_SATS)
        ax.plot(months, produced, color=PALETTE[idx], lw=2.5,
                label=f"{nl} лин.: {rate:.1f} КА/мес → {m_full:.1f} мес")
    ax.axhline(N_SATS, ls="--", color=PALETTE[7], lw=1.3,
               label=f"Цель {N_SATS} КА")
    ax.axvline(TARGET_PROGRAM_MONTHS, ls=":", color=PALETTE[0], lw=1.5,
               label=f"Цель {TARGET_PROGRAM_MONTHS} мес")
    ax.set_xlabel("Месяцы от старта серийного производства")
    ax.set_ylabel("Накопленный выпуск КА")
    ax.set_title(f"Темп производства vs число AIT-линий [{label}]")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"production_rate_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_factory_throughput(rates, output_dir, label):
    nls   = list(rates.keys())
    vals  = [rates[n] for n in nls]
    target_rate = N_SATS / TARGET_PROGRAM_MONTHS

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar([str(n) for n in nls], vals,
                  color=[PALETTE[2], PALETTE[3], PALETTE[4], PALETTE[0]],
                  edgecolor="white", width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.01,
                f"{v:.1f}", ha="center", fontsize=11)
    ax.axhline(target_rate, ls="--", color=PALETTE[7], lw=1.5,
               label=f"Целевой темп = {target_rate:.1f} КА/мес ({TARGET_PROGRAM_MONTHS} мес)")
    ax.set_xlabel("Число параллельных AIT-линий")
    ax.set_ylabel("Пропускная способность (КА/мес)")
    ax.set_title(f"Пропускная способность фабрики vs число линий [{label}]")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"factory_throughput_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_deployment_schedule(n_line, output_dir, label):
    rate = throughput_sats_per_month(n_line)
    m_full = months_to_complete(n_line)
    months = np.linspace(0, max(m_full, 40), 300)

    # производственное ограничение
    produced = np.minimum(months * rate, N_SATS)
    # пусковое ограничение: партии по 20 КА, темп 8 пусков/год (~ 0.67 пуска/мес)
    launches_per_month = 8.0 / 12.0
    launched = np.minimum(np.floor(months * launches_per_month) * SATS_PER_LAUNCH, N_SATS)
    # фактически на орбите = min(произведено, запущено)
    on_orbit = np.minimum(produced, launched)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(months, produced / N_SATS * 100, color=PALETTE[3], lw=2.5,
            label="Произведено (ограничение фабрики)")
    ax.plot(months, launched / N_SATS * 100, color=PALETTE[0], lw=2.5, ls="--",
            label="Запущено (8 пусков/год, 20 КА/пуск)")
    ax.plot(months, on_orbit / N_SATS * 100, color=PALETTE[4], lw=3.0, ls="-.",
            label="На орбите (заполнение созвездия)")
    ax.axhline(100, ls=":", color=PALETTE[7], lw=1.3, label="100% созвездие")
    ax.set_xlabel("Месяцы от старта программы")
    ax.set_ylabel("Заполнение созвездия (%)")
    ax.set_title(f"График развёртывания: производство vs пуски [{label}]")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"deployment_schedule_{label}.png"), dpi=150)
    plt.close(fig)


def _save_csv(results, output_dir, label):
    path = os.path.join(output_dir, f"production_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["параметр", "значение", "ед."])
        for s, d in results["ait_stages_days"].items():
            w.writerow([f"Стадия AIT: {s}", d, "дн"])
        w.writerow(["Полный цикл AIT (1 КА)", results["ait_days_total"], "дн"])
        w.writerow(["Такт-время",            f"{results['takt_time_days']:.2f}", "дн/КА"])
        w.writerow(["Число AIT-линий (нужно)", results["lines_needed"], "линий"])
        w.writerow(["Цель программы",        results["target_program_months"], "мес"])
        w.writerow(["Кривая обучения AIT b", results["learning_b_ait"], ""])
        w.writerow([])
        w.writerow(["AIT-линий", "КА/мес", "мес до 300 КА"])
        for n in (1, 2, 3, 4):
            w.writerow([n, f"{results['throughput_per_month'][n]:.2f}",
                        f"{results['months_to_300'][n]:.1f}"])


def print_production_summary(label: str, results: Dict) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  Production & AIT (300 КА) -- {label}")
    print(sep)
    print(f"  Полный цикл AIT (1 КА): {results['ait_days_total']} дн")
    for s, d in results["ait_stages_days"].items():
        print(f"    - {s:<32} {d:>3} дн")
    print(f"  Такт-время:           {results['takt_time_days']:.2f} дн/КА "
          f"(цель {results['target_program_months']} мес)")
    print(f"  AIT-линий (нужно):    {results['lines_needed']}")
    print(f"  {'Линий':<8} {'КА/мес':>10} {'мес до 300':>12}")
    print(f"  {'':-<32}")
    for n in (1, 2, 3, 4):
        print(f"  {n:<8} {results['throughput_per_month'][n]:>10.2f} "
              f"{results['months_to_300'][n]:>12.1f}")
    print(f"  Выбрано {results['lines_needed']} лин. → "
          f"{results['months_to_300_chosen']:.1f} мес до 300 КА")
    print(sep)
