"""
Радиационная среда и радиационная стойкость АВРОРА.

Орбита 1000 км / 75 deg — зона внутреннего пояса протонов Ван Аллена.
Модели: AP8/AE8 (NASA), CREME96 (NRL), экспоненциальное затухание в Al-экране.

Рассчитывает:
  - TID (Total Ionizing Dose) за 7 лет vs толщина экрана
  - SEU rate по типам памяти (SRAM, Flash, MRAM)
  - Влияние на OCXO (частотный уход от радиации)
  - Дозу по подсистемам с оценкой деградации

Ссылки:
  ECSS-E-ST-10-04C (2008) — Space Environment.
  Barth et al. (1999) — Space, Atmospheric and Terrestrial Radiation Environments.
  Messenger & Ash (1992) — The Effects of Radiation on Electronic Systems.
"""

import math, os, csv
from typing import Dict, List
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Параметры орбиты ──────────────────────────────────────────────────────────
ORBIT_ALT_KM   = 1000.0
ORBIT_INC_DEG  = 75.0
MISSION_YEARS  = 7.0
HOURS_PER_YEAR = 8760.0

# ── Фоновая доза на 1000 км без экрана (AP8, солнечный минимум) ───────────────
# Типичное для 1000 km, i=75 deg, оба пояса: ~80 kRad/yr без экрана
TID_UNSHIELDED_KRAD_YR = 80.0   # кРад/год (Si) без экрана

# Постоянная затухания в алюминии (эффективная)
SHIELD_ATTENUATION_LAMBDA_MM = 2.0   # мм Al: глубина для ослабления в e

# ── Флюенс протонов (>10 МэВ) для SEU ────────────────────────────────────────
# На 1000 км, 75 deg: ~5e12 п/см2/год (AP8-MIN)
PROTON_FLUX_P_CM2_YR = 5e12

# ── Типы памяти и их характеристики ──────────────────────────────────────────
MEMORY_TYPES = {
    "SRAM (0.35 мкм)":  {"sigma_cm2": 5e-8, "Eth_MeV": 1.0,  "tid_limit_krad": 30,  "color": "#e17055"},
    "Flash (90 нм)":    {"sigma_cm2": 1e-8, "Eth_MeV": 2.0,  "tid_limit_krad": 50,  "color": "#fdcb6e"},
    "MRAM":             {"sigma_cm2": 5e-9, "Eth_MeV": 5.0,  "tid_limit_krad": 300, "color": "#00b894"},
    "FRAM":             {"sigma_cm2": 1e-9, "Eth_MeV": 10.0, "tid_limit_krad": 500, "color": "#0984e3"},
}

# ── Подсистемы спутника ───────────────────────────────────────────────────────
SUBSYSTEMS = {
    "CSAC (SA.45s)":         {"shield_mm": 7.0, "tid_limit_krad": 20,  "critical": True},
    "space-Rb (Quantum-18)": {"shield_mm": 4.0, "tid_limit_krad": 100, "critical": True},
    "Бортовой компьютер":    {"shield_mm": 4.0, "tid_limit_krad": 100, "critical": True},
    "ADCS":                  {"shield_mm": 3.0, "tid_limit_krad": 50,  "critical": False},
    "Ресивер L1/L5":         {"shield_mm": 3.0, "tid_limit_krad": 50,  "critical": True},
    "ISL трансивер":         {"shield_mm": 2.0, "tid_limit_krad": 30,  "critical": False},
    "АКБ (Li-Ion)":          {"shield_mm": 1.0, "tid_limit_krad": 10,  "critical": False},
    "Солнечная батарея":     {"shield_mm": 0.5, "tid_limit_krad": 5,   "critical": False},
}

# ── OCXO радиационный уход частоты ───────────────────────────────────────────
OCXO_ALPHA_RAD = 1e-10   # (Δf/f) / кРад — типовой коэффициент для SC-cut кварца


def tid_at_shield(shield_mm: float, years: float = MISSION_YEARS) -> float:
    """TID [кРад] при экране shield_mm мм Al за years лет."""
    annual = TID_UNSHIELDED_KRAD_YR * math.exp(-shield_mm / SHIELD_ATTENUATION_LAMBDA_MM)
    return annual * years


def seu_rate(memory_key: str, shield_mm: float) -> float:
    """
    SEU rate [событий/устройство/день].
    Упрощённая модель: flux × sigma × exp(-ослабление).
    """
    m = MEMORY_TYPES[memory_key]
    flux_eff = PROTON_FLUX_P_CM2_YR * math.exp(-shield_mm / (SHIELD_ATTENUATION_LAMBDA_MM * 5))
    # Сечение × поток → события/см2/год; делим на площадь чипа ~1 см2
    rate_yr = flux_eff * m["sigma_cm2"]
    return rate_yr / 365.0   # событий/устройство/день


def ocxo_freq_drift_ppb(shield_mm: float, years: float = MISSION_YEARS) -> float:
    """Частотный уход OCXO от радиации [ppb] за years лет."""
    tid = tid_at_shield(shield_mm, years)
    return OCXO_ALPHA_RAD * tid * 1e9  # ppb


def run_radiation_analysis(output_dir: str, label: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    shield_range = np.linspace(0.5, 10.0, 200)   # мм Al

    # TID vs толщина экрана
    tid_curves = {
        f"{y} лет": [tid_at_shield(s, y) for s in shield_range]
        for y in [1, 3, 7]
    }

    # SEU rate vs экран для каждого типа памяти
    seu_curves = {
        k: [seu_rate(k, s) for s in shield_range]
        for k in MEMORY_TYPES
    }

    # Подсистемы: доза за 7 лет, предел, статус
    subsys_results = {}
    for name, p in SUBSYSTEMS.items():
        d = tid_at_shield(p["shield_mm"])
        margin = p["tid_limit_krad"] / d if d > 0 else 999
        subsys_results[name] = {
            "shield_mm":     p["shield_mm"],
            "tid_7yr_krad":  d,
            "limit_krad":    p["tid_limit_krad"],
            "margin":        margin,
            "status":        "OK" if margin > 2.0 else ("ГРАНИЦА" if margin > 1.0 else "ПРЕВЫШЕНИЕ"),
            "critical":      p["critical"],
        }

    # OCXO уход частоты
    ocxo_drift = {s: ocxo_freq_drift_ppb(s) for s in shield_range}

    _plot_tid_vs_shield(shield_range, tid_curves, output_dir, label)
    _plot_seu_rates(shield_range, seu_curves, output_dir, label)
    _plot_subsystem_doses(subsys_results, output_dir, label)
    _plot_ocxo_drift(shield_range, list(ocxo_drift.values()), output_dir, label)
    _save_csv(subsys_results, output_dir, label)

    return {
        "shield_range":    shield_range.tolist(),
        "tid_curves":      {k: v for k, v in tid_curves.items()},
        "seu_curves":      {k: v for k, v in seu_curves.items()},
        "subsys_results":  subsys_results,
        "ocxo_drift_ppb":  list(ocxo_drift.values()),
    }


def _plot_tid_vs_shield(shield_range, tid_curves, output_dir, label):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#74b9ff", "#0984e3", "#2d3436"]
    for (yr_label, vals), color in zip(tid_curves.items(), colors):
        ax.semilogy(shield_range, vals, color=color, lw=2, label=yr_label)

    # Уровни допустимой дозы типовых компонентов
    ax.axhline(100, ls="--", color="#e17055", lw=1.2, label="100 кРад (Бортовой компьютер)")
    ax.axhline(20,  ls=":",  color="#fdcb6e", lw=1.2, label="20 кРад (CSAC)")
    ax.axhline(5,   ls="-.", color="#6c5ce7", lw=1.0, label="5 кРад (Солнечная батарея)")
    ax.axvline(4.0, ls="--", color="#00b894", lw=1.2, label="4 мм Al (стандарт АВРОРА)")
    ax.set_xlabel("Толщина алюминиевого экрана (мм)")
    ax.set_ylabel("TID (кРад, Si)")
    ax.set_title(f"АВРОРА — Полная ионизирующая доза (TID) vs экран [{label}]")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(shield_range[0], shield_range[-1])
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"radiation_tid_depth_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_seu_rates(shield_range, seu_curves, output_dir, label):
    fig, ax = plt.subplots(figsize=(10, 6))
    for mem_name, vals in seu_curves.items():
        color = MEMORY_TYPES[mem_name]["color"]
        ax.semilogy(shield_range, vals, color=color, lw=2, label=mem_name)

    ax.axhline(1e-3, ls="--", color="#2d3436", lw=1.2, label="1e-3 событий/устр./день (граница)")
    ax.axvline(4.0, ls="--", color="#00b894", lw=1.2, label="4 мм Al")
    ax.set_xlabel("Толщина алюминиевого экрана (мм)")
    ax.set_ylabel("SEU rate (событий/устройство/день)")
    ax.set_title(f"АВРОРА — SEU rate по типам памяти vs экран [{label}]")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"radiation_seu_rate_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_subsystem_doses(subsys_results, output_dir, label):
    names  = list(subsys_results.keys())
    doses  = [subsys_results[n]["tid_7yr_krad"] for n in names]
    limits = [subsys_results[n]["limit_krad"]   for n in names]
    colors = []
    for n in names:
        s = subsys_results[n]["status"]
        colors.append("#00b894" if s == "OK" else ("#fdcb6e" if s == "ГРАНИЦА" else "#e17055"))

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(x - 0.2, doses,  0.35, label="TID за 7 лет (кРад)", color=colors, alpha=0.85, edgecolor="white")
    ax.bar(x + 0.2, limits, 0.35, label="Предел TID (кРад)",  color="#dfe6e9", alpha=0.85, edgecolor="#636e72")
    for bar, dose in zip(bars, doses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{dose:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Доза (кРад, Si)")
    ax.set_title(f"АВРОРА — Радиационная нагрузка подсистем за 7 лет [{label}]")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor="#00b894", label="OK (запас > 2×)"),
                  Patch(facecolor="#fdcb6e", label="Граница (1–2×)"),
                  Patch(facecolor="#e17055", label="Превышение (<1×)")]
    ax.legend(handles=legend_els, loc="upper right", fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"radiation_component_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_ocxo_drift(shield_range, ocxo_drift, output_dir, label):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(shield_range, ocxo_drift, color="#6c5ce7", lw=2)
    ax.axhline(1.0,  ls="--", color="#e17055", lw=1.2, label="1 ppb (допустимый уход)")
    ax.axhline(0.1,  ls=":",  color="#00b894", lw=1.2, label="0.1 ppb (цель проекта)")
    ax.axvline(4.0,  ls="--", color="#0984e3", lw=1.2, label="4 мм Al (АВРОРА)")
    ax.set_xlabel("Толщина экрана (мм Al)")
    ax.set_ylabel("Уход частоты недисципл. кварца (ppb, за 7 лет)")
    ax.set_title(f"АВРОРА — Радиационный уход недисциплинированного кварца [{label}]")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"radiation_ocxo_drift_{label}.png"), dpi=150)
    plt.close(fig)


def _save_csv(subsys_results, output_dir, label):
    path = os.path.join(output_dir, f"radiation_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subsystem", "shield_mm", "tid_7yr_krad", "limit_krad", "margin", "status"])
        for name, r in subsys_results.items():
            w.writerow([name, r["shield_mm"], f"{r['tid_7yr_krad']:.2f}",
                        r["limit_krad"], f"{r['margin']:.2f}", r["status"]])


def print_radiation_summary(label: str, result: Dict) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  Radiation Environment Analysis -- {label}")
    print(sep)
    print(f"  Орбита: {ORBIT_ALT_KM:.0f} км / {ORBIT_INC_DEG:.0f}°  |  Срок: {MISSION_YEARS:.0f} лет")
    print(f"  Фоновый поток: {TID_UNSHIELDED_KRAD_YR:.0f} кРад/год (без экрана)")
    print()
    print(f"  {'Подсистема':<25} {'Экран':>7} {'TID 7л':>10} {'Предел':>8} {'Запас':>7} {'Статус'}")
    print(f"  {'':─<68}")
    for name, r in result["subsys_results"].items():
        print(f"  {name:<25} {r['shield_mm']:>5.1f}мм  {r['tid_7yr_krad']:>8.1f}  "
              f"{r['limit_krad']:>7.0f}  {r['margin']:>6.1f}x  {r['status']}")
    print(sep)
