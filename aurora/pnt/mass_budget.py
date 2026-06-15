"""
Satellite Mass and Volume Budget for АВРОРА.

Detailed subsystem mass breakdown for a ~140 кг (малый спутник) LEO-навигации.
Includes dry mass, propellant budget (station-keeping + deorbit), and
structural volume allocations.

Reference: SMAD 3rd ed., Appendix A; ECSS-M-ST-10C (mass properties).
"""

import math
import os
import csv
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Physical constants
G0 = 9.80665   # m/s²

# ── Subsystem mass allocations ────────────────────────────────────────────────
# Each entry: (mass_kg, volume_l, power_w, desc)
SUBSYSTEMS = {
    # Структура и механизмы
    "Структура и панели":          (21.0,  40.0,   0.0, "Несущий корпус, панели, кронштейны (радстойкость 7 лет)"),
    "Система отделения":           ( 2.5,   3.0,   0.0, "Захватный бандаж + пружины"),
    "Механизм раскрытия СБ":       ( 2.0,   2.0,   0.0, "Шарниры и приводы раскрытия"),
    # Энергетика
    "Солнечная батарея (GaAs, 3 м²)": ( 6.0, 2.0,  0.0, "3J GaAs, BOL 30%, 3,0 м²"),
    "АКБ (Li-ion, 160 Вт·ч)":      ( 2.5,   5.5,   0.0, "160 Вт·ч под двухсервисную нагрузку"),
    "ЭПС (кондиционирование)":     ( 4.0,   4.0,  10.0, "PCDU, регуляторы, защита"),
    "Кабельная сеть СБ":           ( 1.5,   0.5,   0.0, "СБ → PCDU"),
    # СУОС
    "Звёздные датчики (x2)":       ( 2.0,   1.5,   8.0, "0,4 кг каждый, 5 угл.с"),
    "ИНС (гироскоп)":              ( 0.8,   0.5,   4.0, "МЭМС/ВОГ, 3 оси"),
    "Магнитометр":                 ( 0.3,   0.2,   0.5, "3-осевой феррозонд"),
    "Маховики (x4)":               ( 4.0,   3.0,   8.0, "4 Н·м·с, резерв"),
    "Вычислитель СУОС":            ( 1.0,   0.8,   5.0, "Встроенный процессор СУОС"),
    # Навигационная ПН (двухсервисная, §65)
    "Генератор сигналов ПНВ (А+Б)": ( 5.0,  4.0,  45.0, "Сервис А L1/L5 + Сервис Б выдел. L"),
    "Антенны навигационные":       ( 4.0,   2.5,   0.0, "RHCP L1/L5 + изо-flux выдел. L (Сервис Б)"),
    "УМ РЧ + диплексер":           ( 4.0,   2.0,  80.0, "Сервис А 5/3 Вт + Сервис Б 30 Вт"),
    "Бортовой GNSS-приёмник":      ( 0.5,   0.3,   5.0, "Определение орбиты"),
    # Часовая архитектура (канон)
    "CSAC (чип-цезий, все КА)":    ( 0.1,   0.1,   0.5, "Microsemi SA.45s, терминал на всех 300 КА"),
    "space-Rb (якорные КА)":       ( 0.5,   0.3,   3.0, "Quantum-18, ~15 якорей (1,8 кг на якоре)"),
    # ISL
    "ISL Ka приёмопередатчик":     ( 3.5,   2.5,  20.0, "Ka 26 ГГц, 2 канала"),
    "Антенны ISL (x2)":            ( 2.0,   1.0,   0.0, "Управляемые Ka-решётки"),
    # ТКС/связь
    "ТКС S-диапазон (радио)":      ( 1.5,   1.0,   8.0, "S-band, приём команд/телеметрия"),
    "Антенна ТКС":                 ( 0.3,   0.2,   0.0, "Всенаправленная S-band"),
    # БК
    "Бортовой компьютер":          ( 1.2,   0.8,  12.0, "Рад-стойкий ЦП + хранилище"),
    "ПО и ПЛИС":                   ( 0.5,   0.3,   3.0, "ПЛИС обработки сигнала"),
    # Терморегулирование
    "Радиаторы":                   ( 2.0,   0.5,   0.0, "Пассивные радиаторы"),
    "ЭВТИ (MLI)":                  ( 1.5,   1.0,   0.0, "Многослойная изоляция"),
    "Нагреватели":                 ( 0.5,   0.2,  15.0, "CSAC + холодные грани"),
    "Тепловые трубы":              ( 0.8,   0.3,   0.0, "Изотермализация панелей"),
    # Двигательная установка
    "Бак рабочего тела":           ( 2.5,   8.0,   0.0, "CFRP, 15 л"),
    "Двигатели (x4)":              ( 1.5,   1.0,   0.0, "4 × 50 мН (ион/холодн. газ)"),
    "Система подачи":              ( 1.0,   0.5,   0.0, "Клапаны, регуляторы, магистрали"),
    # Кабели и прочее
    "Кабельная сеть КА":           ( 4.0,   2.0,   0.0, "Внутренняя проводка"),
    "Крепёж и клеи":               ( 1.0,   0.5,   0.0, "Винты, кронштейны, эпоксид"),
    "Резерв массы (ECSS)":         (10.0,   0.0,   0.0, "Высвобожден малым CSAC; запас на дозревание"),
}

# Propellant budget
PROPELLANT = {
    "station_keeping_dv_ms":  21.0,    # m/s/year × 7 yr = 147 m/s over life
    "deorbit_dv_ms":          214.1,   # to 200 km perigee (from deorbit.py)
    "contingency_pct":        10.0,    # 10% propellant contingency
    "isp_s":                  220.0,   # cold-gas/hydrazine Isp (s)
}

# Design margins per ECSS-M-ST-10C
MASS_MARGINS = {
    "system_margin_pct":    5.0,   # overall system margin
    "subsystem_margin_pct": 3.0,   # per-subsystem maturity margin
}

# Mission design target
AURORA_MASS_TARGET_KG = 140.0
AURORA_VOLUME_LIMIT_L = 120.0   # stacked dispenser envelope


def propellant_mass_kg(
    dry_mass_kg: float,
    total_dv_ms: float,
    isp_s: float,
    contingency_pct: float = 10.0,
) -> Dict:
    """Tsiolkovsky rocket equation for total propellant mass."""
    mp_nominal = dry_mass_kg * (math.exp(total_dv_ms / (isp_s * G0)) - 1)
    mp_with_cont = mp_nominal * (1 + contingency_pct / 100)
    return {
        "dry_mass_kg":       dry_mass_kg,
        "total_dv_ms":       total_dv_ms,
        "isp_s":             isp_s,
        "mp_nominal_kg":     mp_nominal,
        "mp_total_kg":       mp_with_cont,
        "contingency_pct":   contingency_pct,
    }


def run_mass_budget_analysis(
    output_dir: str,
    label: str,
    n_sats: int = 300,
    mission_years: float = 7.0,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    # Sum up subsystem masses and volumes
    mass_by_sub = {name: vals[0] for name, vals in SUBSYSTEMS.items()}
    vol_by_sub  = {name: vals[1] for name, vals in SUBSYSTEMS.items()}
    pow_by_sub  = {name: vals[2] for name, vals in SUBSYSTEMS.items()}

    dry_mass_raw = sum(mass_by_sub.values())
    dry_mass_margin = dry_mass_raw * (1 + MASS_MARGINS["subsystem_margin_pct"] / 100)
    dry_mass_total  = dry_mass_margin * (1 + MASS_MARGINS["system_margin_pct"] / 100)

    # Propellant
    sk_dv = PROPELLANT["station_keeping_dv_ms"] * mission_years
    deo_dv = PROPELLANT["deorbit_dv_ms"]
    total_dv = sk_dv + deo_dv
    prop = propellant_mass_kg(dry_mass_total, total_dv, PROPELLANT["isp_s"],
                               PROPELLANT["contingency_pct"])

    wet_mass = dry_mass_total + prop["mp_total_kg"]
    total_volume = sum(vol_by_sub.values())

    # Group by category for pie chart (русские названия для графиков)
    categories = {
        "Конструкция и механизмы": ["Структура и панели", "Система отделения", "Механизм раскрытия СБ"],
        "СЭП (солн. бат. + АКБ)": ["Солнечная батарея (GaAs, 3 м²)", "АКБ (Li-ion, 160 Вт·ч)", "ЭПС (кондиционирование)", "Кабельная сеть СБ"],
        "СУОС (ADCS)": ["Звёздные датчики (x2)", "ИНС (гироскоп)", "Магнитометр", "Маховики (x4)", "Вычислитель СУОС"],
        "Навигационная ПН (А+Б)": ["Генератор сигналов ПНВ (А+Б)", "Антенны навигационные", "УМ РЧ + диплексер", "Бортовой GNSS-приёмник"],
        "Часы (CSAC + space-Rb)": ["CSAC (чип-цезий, все КА)", "space-Rb (якорные КА)"],
        "ISL (Ka)": ["ISL Ka приёмопередатчик", "Антенны ISL (x2)"],
        "ТМ/КУ + БЦВМ": ["ТКС S-диапазон (радио)", "Антенна ТКС", "Бортовой компьютер", "ПО и ПЛИС"],
        "Терморегулирование": ["Радиаторы", "ЭВТИ (MLI)", "Нагреватели", "Тепловые трубы"],
        "Двиг. установка (сухая)": ["Бак рабочего тела", "Двигатели (x4)", "Система подачи"],
        "Кабели, крепёж, резерв": ["Кабельная сеть КА", "Крепёж и клеи", "Резерв массы (ECSS)"],
    }
    cat_mass = {}
    for cat, subs in categories.items():
        cat_mass[cat] = sum(mass_by_sub.get(s, 0) for s in subs)

    _plot_mass_breakdown(cat_mass, dry_mass_total, wet_mass, prop, output_dir, label)
    _plot_mass_pie(cat_mass, output_dir, label)
    _save_mass_csv(mass_by_sub, vol_by_sub, prop, dry_mass_total, wet_mass, output_dir, label)

    return {
        "dry_mass_raw_kg":    dry_mass_raw,
        "dry_mass_kg":        dry_mass_total,
        "propellant_kg":      prop["mp_total_kg"],
        "wet_mass_kg":        wet_mass,
        "total_volume_l":     total_volume,
        "total_dv_ms":        total_dv,
        "sk_dv_ms":           sk_dv,
        "deorbit_dv_ms":      deo_dv,
        "mass_margin_pct":    (dry_mass_total - dry_mass_raw) / dry_mass_raw * 100,
        "mass_vs_target_kg":  wet_mass - AURORA_MASS_TARGET_KG,
        "cat_mass":           cat_mass,
        "subsystems":         SUBSYSTEMS,
        "n_sats":             n_sats,
        "fleet_mass_t":       wet_mass * n_sats / 1000,
    }


def _plot_mass_breakdown(cat_mass, dry_mass, wet_mass, prop, output_dir, label):
    fig, ax = plt.subplots(figsize=(12, 6))
    names  = list(cat_mass.keys())
    masses = list(cat_mass.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(names)))

    bars = ax.barh(names, masses, color=colors, edgecolor="white", lw=0.8)
    ax.axvline(0, color="black", lw=0.8)
    for bar, v in zip(bars, masses):
        ax.text(v + 0.3, bar.get_y() + bar.get_height()/2,
                f"{v:.1f} кг", va="center", fontsize=8)

    ax.set_xlabel("Масса (кг)")
    ax.set_title(f"АВРОРА — Бюджет массы спутника [{label}]\n"
                 f"Сухая: {dry_mass:.1f} кг  |  "
                 f"Топливо: {prop['mp_total_kg']:.1f} кг  |  "
                 f"Заправл.: {wet_mass:.1f} кг  (цель {AURORA_MASS_TARGET_KG:.0f} кг)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"mass_breakdown_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_mass_pie(cat_mass, output_dir, label):
    fig, ax = plt.subplots(figsize=(10, 7))
    labels = [f"{k}\n{v:.1f} кг" for k, v in cat_mass.items()]
    colors = plt.cm.Set3(np.linspace(0, 1, len(cat_mass)))
    ax.pie(list(cat_mass.values()), labels=labels, colors=colors,
           autopct="%1.0f%%", startangle=90, pctdistance=0.75, textprops={"fontsize": 8})
    ax.set_title(f"АВРОРА — Распределение сухой массы [{label}]")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"mass_pie_{label}.png"), dpi=150)
    plt.close(fig)


def _save_mass_csv(mass_by_sub, vol_by_sub, prop, dry_mass, wet_mass, output_dir, label):
    path = os.path.join(output_dir, f"mass_budget_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subsystem", "mass_kg", "volume_l", "desc"])
        for name, vals in SUBSYSTEMS.items():
            w.writerow([name, vals[0], vals[1], vals[3]])
        w.writerow(["", "", "", ""])
        w.writerow(["DRY MASS (with margins)", f"{dry_mass:.2f}", "", ""])
        w.writerow(["Propellant (nominal+cont)", f"{prop['mp_total_kg']:.2f}", "", ""])
        w.writerow(["WET MASS", f"{wet_mass:.2f}", "", ""])


def print_mass_budget_summary(label: str, result: Dict) -> None:
    sep = "=" * 68
    print(f"\n{sep}")
    print(f"  Satellite Mass Budget -- {label}")
    print(sep)
    print(f"  {'Dry mass (raw)':<40} {result['dry_mass_raw_kg']:>8.1f} kg")
    print(f"  {'Dry mass (+ margins)':<40} {result['dry_mass_kg']:>8.1f} kg")
    print(f"    Subsystem margin: {MASS_MARGINS['subsystem_margin_pct']:.0f}%   "
          f"System margin: {MASS_MARGINS['system_margin_pct']:.0f}%")
    print()
    print(f"  Propellant breakdown:")
    print(f"    Station-keeping:  {result['sk_dv_ms']:.0f} m/s (7 yr × 21 m/s/yr)")
    print(f"    Deorbit:          {result['deorbit_dv_ms']:.1f} m/s (to 200 km perigee)")
    print(f"    Total ΔV:         {result['total_dv_ms']:.0f} m/s  |  Isp = {PROPELLANT['isp_s']:.0f} s")
    print(f"    Propellant mass:  {result['propellant_kg']:.1f} kg  "
          f"(+{PROPELLANT['contingency_pct']:.0f}% contingency)")
    print()
    print(f"  {'WET MASS':<40} {result['wet_mass_kg']:>8.1f} kg  "
          f"({'OK' if result['wet_mass_kg'] <= AURORA_MASS_TARGET_KG else 'EXCEEDS TARGET by '+str(round(result['mass_vs_target_kg'],1))+' kg'})")
    print(f"  Target:                                  {AURORA_MASS_TARGET_KG:.0f} kg")
    print(f"  Total volume:                            {result['total_volume_l']:.0f} L  "
          f"(limit {AURORA_VOLUME_LIMIT_L:.0f} L)")
    print()
    print(f"  Fleet totals ({result['n_sats']} satellites):")
    print(f"    Fleet wet mass:   {result['fleet_mass_t']:.1f} t")
    print(f"    Launches needed:  ~{math.ceil(result['fleet_mass_t'] / 6.5):.0f}  "
          f"(6.5 t payload capacity per launch to 1000 km)")
    print(sep)
