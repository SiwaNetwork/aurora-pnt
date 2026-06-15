"""
Satellite Power Budget for АВРОРА.

Computes detailed orbital power balance:
  - Solar array generation (BOL/EOL, orientation, eclipse)
  - Subsystem power consumption breakdown
  - Battery charge/discharge cycle per orbit
  - Power margin analysis across mission life

Reference: SMAD Ch.11, Larson & Wertz; ECSS-E-ST-20C (spacecraft electrical design).
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
SOLAR_CONST_W_M2 = 1361.0   # W/m² at 1 AU
R_EARTH_M = 6_371_000.0
MU_EARTH  = 3.986004418e14

# ── Solar array parameters ────────────────────────────────────────────────────
SOLAR_ARRAY = {
    "area_m2":           3.0,     # deployed area
    "cell_eff":          0.286,   # GaAs 3-junction, BOL efficiency
    "packing_factor":    0.85,    # cell packing on panel
    "cover_glass_loss":  0.02,    # UV/ESD darkening
    "temp_coeff":        -0.002,  # %/°C power loss above 28°C
    "operating_temp_c":  60.0,    # on-orbit panel temperature (°C)
    "reference_temp_c":  28.0,    # STC reference temperature
    "degradation_pct_yr": 2.0,   # %/year radiation-induced degradation
    "contingency":        0.05,   # 5% design contingency
}

# ── Subsystem power budget ────────────────────────────────────────────────────
# (all values in Watts; sunlit / eclipse modes)
SUBSYSTEMS = {
    "Нав. ПН Сервис А (L1/L5)": {
        "sunlit_w": 45.0,   "eclipse_w": 45.0,  "duty": 1.0,
        "desc": "Открытый RNSS-сигнал L1/L5, ~5 Вт РЧ (§65)",
    },
    "Нав. ПН Сервис Б (выдел. L)": {
        "sunlit_w": 80.0,   "eclipse_w": 80.0,  "duty": 1.0,
        "desc": "Защищённый сигнал, 30 Вт РЧ, +23 дБ (§65)",
    },
    "ISL приёмопередатчик (Ka)": {
        "sunlit_w": 20.0,   "eclipse_w": 20.0,  "duty": 1.0,
        "desc": "Ka-диапазон 26 ГГц, 2 канала ISL",
    },
    "Атомные часы (CSAC/space-Rb)": {
        "sunlit_w": 6.0,    "eclipse_w": 6.0,   "duty": 1.0,
        "desc": "CSAC (чип-цезий) на всех КА + space-Rb на якорях",
    },
    "СУОС (ориентация и орбита)": {
        "sunlit_w": 10.0,   "eclipse_w": 10.0,  "duty": 1.0,
        "desc": "Звёздные датчики, маховики, магнитометры",
    },
    "Бортовой компьютер": {
        "sunlit_w": 15.0,   "eclipse_w": 15.0,  "duty": 1.0,
        "desc": "Процессор + память + хранилище",
    },
    "ТКС (телеметрия)": {
        "sunlit_w":  8.0,   "eclipse_w":  8.0,  "duty": 1.0,
        "desc": "S-диапазон, приём команд и телеметрия",
    },
    "Терморегулирование (нагрев.)": {
        "sunlit_w":  5.0,   "eclipse_w": 15.0,  "duty": 1.0,
        "desc": "Нагреватели (выше в тени)",
    },
    "Кондиционирование питания": {
        "sunlit_w": 10.0,   "eclipse_w": 10.0,  "duty": 1.0,
        "desc": "Регуляторы, защита, распределение",
    },
    "Двигательная установка": {
        "sunlit_w": 15.0,   "eclipse_w":  0.0,  "duty": 0.3,
        "desc": "Поддержание орбиты (только на свету)",
    },
    "Бортовой GNSS-приёмник": {
        "sunlit_w":  5.0,   "eclipse_w":  5.0,  "duty": 1.0,
        "desc": "Определение орбиты",
    },
}

# ── Battery parameters ────────────────────────────────────────────────────────
BATTERY = {
    "chemistry":         "Li-ion",
    "specific_energy_wh_kg": 200.0,
    "specific_energy_wh_l":  600.0,   # Wh/L
    "dod_limit":         0.80,
    "charge_eff":        0.95,
    "discharge_eff":     0.98,
    "cycles_per_day":    13.3,         # orbits/day at 1000 km
    "design_life_yr":    7.0,
    "cycle_life":        50_000,       # cycles before 20% capacity loss
}

# ── АВРОРА satellite parameters ───────────────────────────────────────────────
AURORA_ALT_M = 1_000_000.0   # 1000 km


def orbital_period_min(altitude_m: float) -> float:
    r = R_EARTH_M + altitude_m
    return 2 * math.pi * math.sqrt(r**3 / MU_EARTH) / 60.0


def eclipse_fraction(altitude_m: float, beta_deg: float = 0.0) -> float:
    r = R_EARTH_M + altitude_m
    sin_rho = R_EARTH_M / r
    rho_max = math.degrees(math.asin(sin_rho))
    if abs(beta_deg) >= rho_max:
        return 0.0
    cos_frac = math.sqrt(sin_rho**2 - math.sin(math.radians(beta_deg))**2) / math.cos(math.radians(beta_deg))
    cos_frac = min(1.0, max(0.0, cos_frac))
    return math.acos(1 - 2 * cos_frac) / (2 * math.pi) if cos_frac <= 1 else 0.0


def solar_array_power(sa: Dict, year: float = 0.0) -> Dict:
    """
    Compute solar array output power at beginning/end of life.

    Args:
        sa:   Solar array parameter dict
        year: Mission year (0 = BOL, 7 = EOL)

    Returns dict with BOL/EOL power and efficiency breakdown.
    """
    area = sa["area_m2"]
    eta_cell = sa["cell_eff"]
    packing  = sa["packing_factor"]
    cg_loss  = sa["cover_glass_loss"]

    # Temperature derating
    dt = sa["operating_temp_c"] - sa["reference_temp_c"]
    temp_factor = 1 + sa["temp_coeff"] * dt

    # Radiation degradation
    deg_factor = (1 - sa["degradation_pct_yr"] / 100) ** year

    # Net efficiency
    eta_net = eta_cell * packing * (1 - cg_loss) * temp_factor * deg_factor

    # Output power
    p_array = SOLAR_CONST_W_M2 * area * eta_net * (1 - sa["contingency"])

    return {
        "year":          year,
        "area_m2":       area,
        "eta_cell":      eta_cell,
        "eta_net":       eta_net,
        "temp_factor":   temp_factor,
        "deg_factor":    deg_factor,
        "power_w":       p_array,
    }


def power_balance(
    altitude_m: float = AURORA_ALT_M,
    beta_deg: float = 0.0,
    mission_year: float = 0.0,
) -> Dict:
    """
    Compute orbital average power balance for a single satellite.

    Returns full power budget with margins.
    """
    period_min = orbital_period_min(altitude_m)
    f_eclipse  = eclipse_fraction(altitude_m, beta_deg)
    f_sunlit   = 1.0 - f_eclipse

    # Solar array power (accounting for off-pointing average ~cos(15°))
    cos_pointing = math.cos(math.radians(15.0))   # average off-nadir angle
    sa_bol = solar_array_power(SOLAR_ARRAY, 0.0)
    sa_eol = solar_array_power(SOLAR_ARRAY, mission_year if mission_year > 0 else 7.0)
    p_sa_bol = sa_bol["power_w"] * cos_pointing
    p_sa_eol = sa_eol["power_w"] * cos_pointing

    # Subsystem consumption
    p_sunlit  = sum(s["sunlit_w"]  * s["duty"] for s in SUBSYSTEMS.values())
    p_eclipse = sum(s["eclipse_w"] * s["duty"] for s in SUBSYSTEMS.values())
    p_avg     = f_sunlit * p_sunlit + f_eclipse * p_eclipse

    # Battery sizing (must cover eclipse energy deficit)
    t_eclipse_h = f_eclipse * period_min / 60.0
    e_eclipse_wh = p_eclipse * t_eclipse_h   # energy needed in eclipse

    batt_capacity_wh = e_eclipse_wh / BATTERY["dod_limit"] / BATTERY["discharge_eff"]
    batt_mass_kg     = batt_capacity_wh / BATTERY["specific_energy_wh_kg"]
    batt_volume_l    = batt_capacity_wh / BATTERY["specific_energy_wh_l"]

    # Charge time (sunlit period must recharge battery)
    t_sunlit_h = f_sunlit * period_min / 60.0
    p_charge_needed_w = (e_eclipse_wh / BATTERY["charge_eff"]) / t_sunlit_h if t_sunlit_h > 0 else 0.0

    # Power available for payload in sunlit period
    p_available_sunlit_bol = p_sa_bol - p_charge_needed_w
    p_available_sunlit_eol = p_sa_eol - p_charge_needed_w

    # Margin (available - consumed)
    margin_bol_w = p_available_sunlit_bol - p_sunlit
    margin_eol_w = p_available_sunlit_eol - p_sunlit

    return {
        "altitude_m":          altitude_m,
        "beta_deg":            beta_deg,
        "mission_year":        mission_year,
        "period_min":          period_min,
        "eclipse_fraction":    f_eclipse,
        "sunlit_fraction":     f_sunlit,
        "eclipse_min":         f_eclipse * period_min,
        "sunlit_min":          f_sunlit * period_min,
        # Solar array
        "sa_bol_w":            p_sa_bol,
        "sa_eol_w":            p_sa_eol,
        "sa_eta_net_bol":      sa_bol["eta_net"],
        "sa_eta_net_eol":      sa_eol["eta_net"],
        # Load
        "p_sunlit_w":          p_sunlit,
        "p_eclipse_w":         p_eclipse,
        "p_avg_w":             p_avg,
        # Battery
        "eclipse_energy_wh":   e_eclipse_wh,
        "battery_capacity_wh": batt_capacity_wh,
        "battery_mass_kg":     batt_mass_kg,
        "battery_volume_l":    batt_volume_l,
        "p_charge_w":          p_charge_needed_w,
        # Margins
        "margin_bol_w":        margin_bol_w,
        "margin_eol_w":        margin_eol_w,
        "margin_bol_pct":      margin_bol_w / p_sunlit * 100 if p_sunlit > 0 else 0,
        "margin_eol_pct":      margin_eol_w / p_sunlit * 100 if p_sunlit > 0 else 0,
        "viable_bol":          margin_bol_w > 0,
        "viable_eol":          margin_eol_w > 0,
        # Subsystem breakdown
        "subsystems":          SUBSYSTEMS,
    }


def run_power_budget_analysis(
    output_dir: str,
    label: str,
    altitude_m: float = AURORA_ALT_M,
    n_sats: int = 180,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    # Main budget (worst-case beta=0°, EOL)
    budget_bol = power_balance(altitude_m, beta_deg=0.0, mission_year=0.0)
    budget_eol = power_balance(altitude_m, beta_deg=0.0, mission_year=7.0)

    # Beta angle sweep
    beta_range = list(range(0, 70, 5))
    budgets_beta = [power_balance(altitude_m, b, 7.0) for b in beta_range]

    # Mission year sweep (degradation)
    year_range = list(range(0, 8))
    budgets_year = [power_balance(altitude_m, 0.0, y) for y in year_range]

    _plot_power_breakdown(budget_eol, output_dir, label)
    _plot_power_vs_beta(beta_range, budgets_beta, output_dir, label)
    _plot_power_vs_year(year_range, budgets_year, output_dir, label)
    _save_power_csv(budget_bol, budget_eol, output_dir, label)

    return {
        "bol":         budget_bol,
        "eol":         budget_eol,
        "altitude_m":  altitude_m,
        "n_sats":      n_sats,
        "beta_range":  beta_range,
        "budgets_beta": budgets_beta,
        "year_range":  year_range,
        "budgets_year": budgets_year,
    }


def _plot_power_breakdown(budget: Dict, output_dir: str, label: str):
    names   = list(SUBSYSTEMS.keys())
    sunlit  = [SUBSYSTEMS[n]["sunlit_w"] * SUBSYSTEMS[n]["duty"] for n in names]
    eclipse = [SUBSYSTEMS[n]["eclipse_w"] * SUBSYSTEMS[n]["duty"] for n in names]
    short_names = [n.split("(")[0].strip()[:22] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    x = np.arange(len(names))
    w = 0.35
    axes[0].bar(x - w/2, sunlit,  w, label="Освещённость", color="#fdcb6e", edgecolor="white")
    axes[0].bar(x + w/2, eclipse, w, label="Затенение", color="#6c5ce7", edgecolor="white")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short_names, rotation=35, ha="right", fontsize=8)
    axes[0].set_ylabel("Мощность (Вт)")
    axes[0].set_title(f"Потребление мощности подсистемами\nОсвещённость: {sum(sunlit):.0f} Вт  |  Затенение: {sum(eclipse):.0f} Вт")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].axhline(0, color="black", lw=0.8)

    # Круговая диаграмма — освещённость
    wedge_colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    wedges, _, autotexts = axes[1].pie(
        sunlit, labels=None, colors=wedge_colors, autopct="%1.0f%%",
        startangle=90, pctdistance=0.75,
    )
    axes[1].legend(wedges, [f"{n}: {v:.0f} Вт" for n, v in zip(short_names, sunlit)],
                   loc="lower left", fontsize=7, bbox_to_anchor=(-0.15, -0.15))
    axes[1].set_title(f"Распределение мощности (освещённость)\nВсего {sum(sunlit):.0f} Вт")

    fig.suptitle(f"АВРОРА — Бюджет мощности спутника [{label}]  (КСЖ, бета=0°)", fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"power_budget_breakdown_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_power_vs_beta(beta_range, budgets, output_dir, label):
    sa_bol = [b["sa_bol_w"] for b in budgets]
    sa_eol = [b["sa_eol_w"] for b in budgets]
    margin_eol = [b["margin_eol_w"] for b in budgets]
    load   = [b["p_sunlit_w"] for b in budgets]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(beta_range, sa_bol,    color="#fdcb6e", lw=2, label="СБ НСЖ")
    axes[0].plot(beta_range, sa_eol,    color="#e17055", lw=2, label="СБ КСЖ (7 лет)")
    axes[0].axhline(load[0], ls="--",  color="#6c5ce7", lw=1.5, label=f"Нагрузка {load[0]:.0f} Вт")
    axes[0].set_xlabel("Бета-угол Солнца (°)")
    axes[0].set_ylabel("Мощность (Вт)")
    axes[0].set_title("Мощность СБ от бета-угла")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    colors = ["#00b894" if m > 0 else "#e17055" for m in margin_eol]
    axes[1].bar(beta_range, margin_eol, width=3, color=colors, edgecolor="white")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xlabel("Бета-угол Солнца (°)")
    axes[1].set_ylabel("Запас мощности КСЖ (Вт)")
    axes[1].set_title("Запас мощности КСЖ от бета-угла")
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle(f"АВРОРА — Мощность от угла Солнца [{label}]")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"power_vs_beta_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_power_vs_year(year_range, budgets, output_dir, label):
    sa_power = [b["sa_eol_w"] for b in budgets]
    margins  = [b["margin_eol_w"] for b in budgets]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(year_range, sa_power, color="#0984e3", lw=2, marker="o", label="SA power (W)")
    ax2 = ax.twinx()
    ax2.bar(year_range, margins, width=0.6, color=["#00b894" if m > 0 else "#e17055" for m in margins],
            alpha=0.7, label="EOL margin (W)")
    ax2.axhline(0, color="red", lw=0.8)
    ax.set_xlabel("Год миссии")
    ax.set_ylabel("Мощность СБ (Вт)", color="#0984e3")
    ax2.set_ylabel("Запас мощности (Вт)", color="#00b894")
    ax.set_title(f"Мощность от года миссии [{label}]  (бета=0°, наихудший случай)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"power_vs_year_{label}.png"), dpi=150)
    plt.close(fig)


def _save_power_csv(bol: Dict, eol: Dict, output_dir: str, label: str):
    path = os.path.join(output_dir, f"power_budget_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["parameter", "BOL", "EOL_7yr"])
        rows = [
            ("solar_array_w",        bol["sa_bol_w"],     eol["sa_eol_w"]),
            ("load_sunlit_w",        bol["p_sunlit_w"],   eol["p_sunlit_w"]),
            ("load_eclipse_w",       bol["p_eclipse_w"],  eol["p_eclipse_w"]),
            ("battery_capacity_wh",  bol["battery_capacity_wh"], eol["battery_capacity_wh"]),
            ("battery_mass_kg",      bol["battery_mass_kg"],     eol["battery_mass_kg"]),
            ("margin_w",             bol["margin_bol_w"], eol["margin_eol_w"]),
            ("margin_pct",           bol["margin_bol_pct"], eol["margin_eol_pct"]),
        ]
        for r in rows:
            w.writerow([r[0], f"{r[1]:.2f}", f"{r[2]:.2f}"])
        w.writerow(["", "", ""])
        w.writerow(["subsystem", "sunlit_w", "eclipse_w"])
        for name, s in SUBSYSTEMS.items():
            w.writerow([name, s["sunlit_w"] * s["duty"], s["eclipse_w"] * s["duty"]])


def print_power_budget_summary(label: str, result: Dict) -> None:
    bol = result["bol"]
    eol = result["eol"]
    sep = "=" * 68
    print(f"\n{sep}")
    print(f"  Satellite Power Budget -- {label}")
    print(sep)
    print(f"  Altitude: {bol['altitude_m']/1000:.0f} km  |  "
          f"Period: {bol['period_min']:.1f} min  |  "
          f"Eclipse: {bol['eclipse_min']:.1f} min ({bol['eclipse_fraction']*100:.1f}%)")
    print()
    print(f"  Solar array ({SOLAR_ARRAY['area_m2']} m², GaAs {SOLAR_ARRAY['cell_eff']*100:.0f}% BOL):")
    print(f"    BOL power:       {bol['sa_bol_w']:.0f} W  (eta_net = {bol['sa_eta_net_bol']*100:.1f}%)")
    print(f"    EOL power (7yr): {eol['sa_eol_w']:.0f} W  (eta_net = {eol['sa_eta_net_eol']*100:.1f}%)")
    print()
    print(f"  Subsystem loads:")
    for name, s in SUBSYSTEMS.items():
        p = s["sunlit_w"] * s["duty"]
        print(f"    {name[:38]:<38} {p:>6.0f} W")
    print(f"    {'─'*48}")
    print(f"    {'TOTAL (sunlit)':<38} {bol['p_sunlit_w']:>6.0f} W")
    print(f"    {'TOTAL (eclipse)':<38} {bol['p_eclipse_w']:>6.0f} W")
    print()
    print(f"  Battery (eclipse={bol['eclipse_min']:.1f} min, load={bol['p_eclipse_w']:.0f} W):")
    print(f"    Energy required: {bol['eclipse_energy_wh']:.1f} Wh")
    print(f"    Capacity (DOD={BATTERY['dod_limit']*100:.0f}%): {bol['battery_capacity_wh']:.1f} Wh")
    print(f"    Mass:            {bol['battery_mass_kg']:.2f} kg  ({BATTERY['chemistry']})")
    print(f"    Volume:          {bol['battery_volume_l']:.2f} L")
    print()
    print(f"  Power margin (beta=0°, worst case):")
    print(f"    BOL: {bol['margin_bol_w']:+.0f} W  ({bol['margin_bol_pct']:+.0f}%)  "
          f"{'[OK]' if bol['viable_bol'] else '[FAIL]'}")
    print(f"    EOL: {eol['margin_eol_w']:+.0f} W  ({eol['margin_eol_pct']:+.0f}%)  "
          f"{'[OK]' if eol['viable_eol'] else '[REVIEW ARRAY SIZING]'}")
    print(sep)
