"""
Релятивистские поправки для АВРОРА.

На орбите LEO 1000 км действуют три основных релятивистских эффекта,
необходимых для наносекундной синхронизации:

1. Гравитационный красный сдвиг (общая теория): часы на орбите идут быстрее
2. Доплеровский сдвиг 2-го порядка (специальная теория): орбитальная скорость
3. Эффект Саньяка: поправка на вращение Земли при распространении сигнала

Ссылки:
  IS-GPS-200 — ICD GPS, Section 20.3.3.3 (relativistic correction).
  Ashby, N. (2003) — Relativity in the Global Positioning System. Living Reviews.
  IERS Conventions (2010) — Chapter 10.
  Petit & Wolf (1994) — Relativistic theory for picosecond time transfer.
"""

import math, os, csv
from typing import Dict, List, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Физические константы ──────────────────────────────────────────────────────
C       = 299_792_458.0        # скорость света (м/с)
GM      = 3.986004418e14       # гравитационный параметр Земли (м³/с²)
R_E     = 6_371_000.0          # средний радиус Земли (м)
OMEGA_E = 7.292115e-5          # угловая скорость вращения Земли (рад/с)
J2      = 1.0826257e-3         # вторая зональная гармоника
C20     = -J2

# ── Параметры орбиты АВРОРА ───────────────────────────────────────────────────
AURORA_ALT_M  = 1_000_000.0
AURORA_INC_D  = 75.0

# ── Параметры орбиты GPS ─────────────────────────────────────────────────────
GPS_ALT_M = 20_200_000.0
GPS_INC_D = 55.0


def orbital_velocity(alt_m: float) -> float:
    """Орбитальная скорость для круговой орбиты (м/с)."""
    return math.sqrt(GM / (R_E + alt_m))


def gravitational_redshift_ppb(alt_m: float) -> float:
    """
    Гравитационный сдвиг частоты: Δf/f = GM/c² × (1/R_E - 1/r_sat).
    Положительный → часы на орбите идут быстрее.
    Возвращает [ppb/сутки → нет, возвращает Δf/f * 1e9 = ppb].
    """
    r = R_E + alt_m
    return (GM / C**2) * (1/R_E - 1/r) * 1e9   # ppb


def second_order_doppler_ppb(alt_m: float) -> float:
    """
    Доплер 2-го порядка (СТО): Δf/f = -v²/(2c²).
    Отрицательный → орбитальное движение замедляет часы.
    """
    v = orbital_velocity(alt_m)
    return -(v**2 / (2 * C**2)) * 1e9   # ppb


def total_clock_correction_ppb(alt_m: float) -> float:
    """Суммарная поправка (гравитация + орбитальная скорость)."""
    return gravitational_redshift_ppb(alt_m) + second_order_doppler_ppb(alt_m)


def clock_correction_ns_per_day(alt_m: float) -> float:
    """Суммарная релятивистская поправка часов [нс/сутки]."""
    delta_f_f = total_clock_correction_ppb(alt_m) * 1e-9
    return delta_f_f * 86400 * 1e9   # нс/сутки


def sagnac_correction_ns(range_m: float, azimuth_deg: float,
                          elevation_deg: float) -> float:
    """
    Поправка Саньяка для сигнала на дальности range_m.
    δt_Sagnac = 2·Ω_E·A / c²
    где A — площадь треугольника (центр Земли, приёмник, спутник),
    проецированная на экватор.

    Упрощение: A ≈ range × R_E × cos(lat) × sin(az) / 2
    Для типового слота: ~200–2000 нс.
    """
    # Проецированная площадь в плоскость экватора
    az_rad = math.radians(azimuth_deg)
    el_rad = math.radians(elevation_deg)
    # Приближение: A ≈ R_E × range × cos(el) × sin(az)
    A_proj = R_E * range_m * math.cos(el_rad) * abs(math.sin(az_rad))
    sagnac_s = 2 * OMEGA_E * A_proj / C**2
    return sagnac_s * 1e9   # нс


def eccentricity_correction_ns(e: float, sqrt_a_m05: float,
                                 eccentric_anomaly_rad: float) -> float:
    """
    Эксцентрическая поправка часов КА (IS-GPS-200):
    Δt_rel = -2 × sqrt(GM) / c² × e × sqrt(a) × sin(E)
    Для АВРОРА: e ≈ 0 → поправка пренебрежимо мала.
    Возвращает нс.
    """
    return -(2 * math.sqrt(GM) / C**2) * e * sqrt_a_m05 * math.sin(eccentric_anomaly_rad) * 1e9


def run_relativistic_analysis(output_dir: str, label: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    alt_range_km = np.linspace(200, 25000, 500)
    alt_range_m  = alt_range_km * 1000

    grav_ppb  = [gravitational_redshift_ppb(a) for a in alt_range_m]
    dop_ppb   = [second_order_doppler_ppb(a)   for a in alt_range_m]
    total_ppb = [total_clock_correction_ppb(a) for a in alt_range_m]
    total_ns_day = [clock_correction_ns_per_day(a) for a in alt_range_m]

    # Саньяк для диапазона дальностей (азимут 90°, угол места 30°)
    range_km = np.linspace(500, 3000, 200)
    sagnac_vals = [sagnac_correction_ns(r*1000, 90.0, 30.0) for r in range_km]

    # Таблица ключевых орбит
    key_orbits = {
        "АВРОРА LEO (1000 км)": AURORA_ALT_M,
        "ISS (400 км)":          400_000.0,
        "GPS MEO (20 200 км)":   GPS_ALT_M,
        "Galileo (23 222 км)":   23_222_000.0,
        "GEO (35 786 км)":       35_786_000.0,
    }

    orbit_table = {}
    for name, alt in key_orbits.items():
        v = orbital_velocity(alt)
        orbit_table[name] = {
            "alt_km":           alt / 1000,
            "v_km_s":           v / 1000,
            "grav_ppb":         gravitational_redshift_ppb(alt),
            "dop_ppb":          second_order_doppler_ppb(alt),
            "total_ppb":        total_clock_correction_ppb(alt),
            "total_ns_day":     clock_correction_ns_per_day(alt),
        }

    _plot_clock_correction(alt_range_km, grav_ppb, dop_ppb, total_ppb, output_dir, label)
    _plot_ns_per_day(alt_range_km, total_ns_day, output_dir, label)
    _plot_sagnac(range_km, sagnac_vals, output_dir, label)
    _plot_orbit_comparison(orbit_table, output_dir, label)
    _save_csv(orbit_table, output_dir, label)

    return {
        "alt_range_km": alt_range_km.tolist(),
        "grav_ppb":     grav_ppb,
        "dop_ppb":      dop_ppb,
        "total_ppb":    total_ppb,
        "total_ns_day": total_ns_day,
        "range_km":     range_km.tolist(),
        "sagnac_ns":    sagnac_vals,
        "orbit_table":  orbit_table,
    }


def _plot_clock_correction(alt_km, grav, dop, total, output_dir, label):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(alt_km, grav,  color="#00b894", lw=2, label="Гравитационный сдвиг (+)")
    ax.plot(alt_km, dop,   color="#e17055", lw=2, label="Доплер 2-го порядка (−)")
    ax.plot(alt_km, total, color="#0984e3", lw=2.5, label="Суммарная поправка")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.axvline(AURORA_ALT_M/1000, color="#6c5ce7", lw=1.5, ls=":", label="АВРОРА 1000 км")
    ax.axvline(GPS_ALT_M/1000,    color="#fdcb6e", lw=1.5, ls=":", label="GPS 20 200 км")
    # Аннотации
    aurora_total = total_clock_correction_ppb(AURORA_ALT_M)
    gps_total    = total_clock_correction_ppb(GPS_ALT_M)
    ax.annotate(f"АВРОРА: {aurora_total:+.2f} ppb",
                xy=(1000, aurora_total), xytext=(3000, aurora_total+0.02),
                arrowprops=dict(arrowstyle="->", color="#6c5ce7"),
                fontsize=9, color="#6c5ce7")
    ax.annotate(f"GPS: {gps_total:+.2f} ppb",
                xy=(20200, gps_total), xytext=(17000, gps_total-0.03),
                arrowprops=dict(arrowstyle="->", color="#fdcb6e"),
                fontsize=9, color="#fdcb6e")
    ax.set_xlabel("Высота орбиты (км)")
    ax.set_ylabel("Относительный сдвиг частоты (ppb = 10⁻⁹)")
    ax.set_title(f"АВРОРА — Релятивистская поправка часов КА [{label}]")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"relativistic_clock_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_ns_per_day(alt_km, ns_day, output_dir, label):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(alt_km, ns_day, color="#0984e3", lw=2)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(AURORA_ALT_M/1000, color="#6c5ce7", lw=1.5, ls=":", label="АВРОРА 1000 км")
    ax.axvline(GPS_ALT_M/1000,    color="#fdcb6e", lw=1.5, ls=":", label="GPS 20 200 км")
    aurora_ns = clock_correction_ns_per_day(AURORA_ALT_M)
    gps_ns    = clock_correction_ns_per_day(GPS_ALT_M)
    ax.annotate(f"{aurora_ns:+.1f} нс/сут", xy=(1000, aurora_ns),
                xytext=(3500, aurora_ns+5000), fontsize=9, color="#6c5ce7",
                arrowprops=dict(arrowstyle="->", color="#6c5ce7"))
    ax.annotate(f"{gps_ns:+.1f} нс/сут", xy=(20200, gps_ns),
                xytext=(16000, gps_ns-5000), fontsize=9, color="#fdcb6e",
                arrowprops=dict(arrowstyle="->", color="#fdcb6e"))
    ax.set_xlabel("Высота орбиты (км)")
    ax.set_ylabel("Поправка часов (нс/сутки)")
    ax.set_title(f"АВРОРА — Суммарная релятивистская поправка [{label}]")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"relativistic_ns_day_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_sagnac(range_km, sagnac_ns, output_dir, label):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range_km, sagnac_ns, color="#e17055", lw=2)
    # Характерные дальности АВРОРА
    ax.axvline(1000, ls=":", color="#0984e3", lw=1.2, label="1000 км (зенит)")
    ax.axvline(2825, ls=":", color="#6c5ce7", lw=1.2, label="2825 км (угол 10°)")
    ax.set_xlabel("Дальность до спутника (км)")
    ax.set_ylabel("Поправка Саньяка (нс)")
    ax.set_title(f"АВРОРА — Эффект Саньяка vs дальность [{label}]  (аз. 90°, ув. 30°)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"relativistic_sagnac_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_orbit_comparison(orbit_table, output_dir, label):
    names = list(orbit_table.keys())
    total = [orbit_table[n]["total_ns_day"] for n in names]
    grav  = [orbit_table[n]["grav_ppb"] * 86.4 for n in names]   # ppb → нс/сут
    dop   = [orbit_table[n]["dop_ppb"]  * 86.4 for n in names]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - 0.25, grav, 0.25, label="Гравитац. (нс/сут)", color="#00b894", alpha=0.85)
    ax.bar(x,        dop,  0.25, label="Доплер 2-го п. (нс/сут)", color="#e17055", alpha=0.85)
    ax.bar(x + 0.25, total,0.25, label="Суммарный (нс/сут)", color="#0984e3", alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("Поправка часов (нс/сутки)")
    ax.set_title(f"АВРОРА — Релятивистские поправки для разных орбит [{label}]")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"relativistic_comparison_{label}.png"), dpi=150)
    plt.close(fig)


def _save_csv(orbit_table, output_dir, label):
    path = os.path.join(output_dir, f"relativistic_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["orbit", "alt_km", "v_km_s", "grav_ppb", "dop_ppb",
                    "total_ppb", "total_ns_day"])
        for name, r in orbit_table.items():
            w.writerow([name, f"{r['alt_km']:.0f}", f"{r['v_km_s']:.3f}",
                        f"{r['grav_ppb']:.4f}", f"{r['dop_ppb']:.4f}",
                        f"{r['total_ppb']:.4f}", f"{r['total_ns_day']:.1f}"])


def print_relativistic_summary(label: str, result: Dict) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  Relativistic Corrections -- {label}")
    print(sep)
    print(f"  {'Орбита':<28} {'Высота':>8} {'V':>8} {'Гравит.':>10} "
          f"{'Доплер':>10} {'Суммарн.':>10} {'нс/сут':>10}")
    print(f"  {'':─<70}")
    for name, r in result["orbit_table"].items():
        print(f"  {name:<28} {r['alt_km']:>7.0f}  {r['v_km_s']:>6.3f}  "
              f"{r['grav_ppb']:>9.4f}  {r['dop_ppb']:>9.4f}  "
              f"{r['total_ppb']:>9.4f}  {r['total_ns_day']:>9.1f}")
    print(sep)
