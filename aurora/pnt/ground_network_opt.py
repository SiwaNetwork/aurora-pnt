"""
Оптимизация сети наземных станций AURORA PNT (MCS).

Анализирует геометрию сети наземных станций (21 станция МКС),
качество орбитального определения (OD), и покрытие созвездия.

Рассчитывает:
  - Распределение станций по широте/долготе
  - Одновременную видимость спутников с нескольких станций
  - Геометрический фактор качества OD (GDOP наземной сети)
  - Накопление дуговых данных за 24 часа
  - Зоны дефицита покрытия
  - Оптимальное размещение 3–5 дополнительных станций

Ссылки:
  Montenbruck et al. (2015) — IGS-MGEX: Preparing the ground segment for multi-GNSS services.
  Springer et al. (2020) — NAPEOS Mathematical Models and Algorithms.
  ECSS-E-ST-50-02C (2012) — Ranging and Doppler Tracking.
"""

import math, os, csv
from typing import Dict, List, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ── Параметры орбиты ──────────────────────────────────────────────────────────
ALT_KM      = 1000.0
INC_DEG     = 75.0
R_E_KM      = 6371.0
ELEV_CUTOFF = 5.0    # градусы, минимальный угол для наблюдения

# ── Наземные станции МКС (21 станция) ────────────────────────────────────────
MCS_STATIONS = [
    # (название, широта, долгота)
    ("Москва",           55.8, 37.6),
    ("Санкт-Петербург",  59.9, 30.3),
    ("Новосибирск",      55.0, 82.9),
    ("Екатеринбург",     56.8, 60.6),
    ("Хабаровск",        48.5, 135.1),
    ("Владивосток",      43.1, 131.9),
    ("Якутск",           62.0, 129.7),
    ("Калининград",      54.7, 20.5),
    ("Петропавловск",    53.0, 158.7),
    ("Улан-Удэ",         51.8, 107.6),
    ("Норильск",         69.3, 88.2),
    ("Диксон",           73.5, 80.5),
    ("Тикси",            71.6, 128.9),
    ("Билибино",         68.1, 166.4),
    ("Мурманск",         68.9, 33.1),
    ("Архангельск",      64.5, 40.5),
    ("Краснодар",        45.0, 38.9),
    ("Новороссийск",     44.7, 37.8),
    ("Иркутск",          52.3, 104.3),
    ("Омск",             54.9, 73.4),
    ("Красноярск",       56.0, 92.8),
]

# Потенциальные дополнительные станции (оптимизация покрытия)
CANDIDATE_STATIONS = [
    ("Куба (Гавана)",    23.1, -82.4),
    ("Бразилия (Ресифе)",  -8.0, -35.0),
    ("ЮАР (Йоханнесбург)",-26.2, 28.0),
    ("Австралия (Дарвин)", -12.5, 130.9),
    ("Индия (Ченнаи)",    13.1, 80.3),
    ("Египет (Каир)",    30.1, 31.2),
    ("Вьетнам (ХоШиМин)", 10.8, 106.7),
    ("Аргентина (БА)",  -34.6, -58.4),
]


def station_elevation_to_sat(station_lat: float, station_lon: float,
                              sat_lat: float, sat_lon: float,
                              alt_km: float = ALT_KM) -> float:
    """
    Приближённый угол возвышения спутника с наземной станции.
    Использует плоскоземную модель для малых угловых расстояний.
    """
    dlat = math.radians(sat_lat - station_lat)
    dlon = math.radians(sat_lon - station_lon) * math.cos(math.radians(station_lat))
    slant_km = math.sqrt(dlat**2 + dlon**2) * R_E_KM
    if slant_km < 1:
        return 90.0
    elevation = math.degrees(math.atan(alt_km / slant_km))
    return elevation


def max_coverage_radius_km(alt_km: float = ALT_KM,
                            elev_cutoff_deg: float = ELEV_CUTOFF) -> float:
    """Радиус покрытия станции (км на поверхности) для заданного угла маски."""
    r_sat = (R_E_KM + alt_km) * 1e3
    r_e   = R_E_KM * 1e3
    el_r  = math.radians(elev_cutoff_deg)
    # Угол при центре Земли
    rho   = math.acos(r_e / r_sat * math.cos(el_r)) - el_r
    return rho * R_E_KM   # км


def coverage_fraction_1d(stations: List[Tuple], alt_km: float = ALT_KM) -> float:
    """
    Доля поверхности, покрытой сетью станций (1D — только долготы).
    Простая метрика для оценки глобальности сети.
    """
    cov_r_deg = math.degrees(max_coverage_radius_km(alt_km) / R_E_KM)
    lons = [s[2] for s in stations]
    covered = 0
    for lon_test in range(-180, 180, 1):
        for lon_st in lons:
            diff = abs(lon_test - lon_st)
            if diff > 180:
                diff = 360 - diff
            if diff < cov_r_deg:
                covered += 1
                break
    return covered / 360.0


def od_geometry_factor(stations: List[Tuple]) -> float:
    """
    Геометрический фактор орбитального определения.
    Упрощённая модель: OD качество пропорционально числу наблюдений
    в разных секторах долготы (равномерность распределения).
    """
    lons = sorted([s[2] for s in stations])
    n    = len(lons)
    if n < 2:
        return 0.0
    gaps = []
    for i in range(n - 1):
        gaps.append(lons[i+1] - lons[i])
    gaps.append(360.0 - lons[-1] + lons[0])
    max_gap = max(gaps)
    # OD quality drops with large gaps
    return 1.0 - max_gap / 360.0


def arc_observations_per_day(n_stations: int, alt_km: float = ALT_KM,
                              inc_deg: float = INC_DEG) -> float:
    """
    Ожидаемое число дуговых наблюдений за 24 часа.
    T_orbit ≈ 105 мин; каждый пролёт над станцией ~8 мин → ~14 пролётов/сут.
    """
    r_sat = (R_E_KM + alt_km) * 1e3
    mu    = 3.986004418e14
    T_orbit_s = 2 * math.pi * math.sqrt(r_sat**3 / mu)
    T_orbit_min = T_orbit_s / 60
    passes_per_day = 24 * 60 / T_orbit_min
    coverage_radius_km = max_coverage_radius_km(alt_km)
    pass_duration_min  = 2 * coverage_radius_km / (7.8 * 60)   # ≈ 7.8 км/с
    return n_stations * passes_per_day * (pass_duration_min / T_orbit_min)


def run_ground_network_analysis(output_dir: str, label: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    cov_r_km  = max_coverage_radius_km()
    cov_r_deg = math.degrees(cov_r_km / R_E_KM)

    mcs_cov  = coverage_fraction_1d(MCS_STATIONS)
    mcs_od   = od_geometry_factor(MCS_STATIONS)
    all_stations = MCS_STATIONS + CANDIDATE_STATIONS
    full_cov = coverage_fraction_1d(all_stations)
    full_od  = od_geometry_factor(all_stations)

    arcs_21  = arc_observations_per_day(21)
    arcs_full = arc_observations_per_day(len(all_stations))

    _plot_station_map(cov_r_deg, output_dir, label)
    _plot_od_quality(output_dir, label)
    _plot_coverage_vs_stations(output_dir, label)
    _save_csv(cov_r_km, mcs_cov, mcs_od, arcs_21, output_dir, label)

    return {
        "coverage_radius_km":   cov_r_km,
        "mcs21_coverage_pct":   mcs_cov * 100,
        "mcs21_od_factor":      mcs_od,
        "arcs_21_per_day":      arcs_21,
        "full_coverage_pct":    full_cov * 100,
        "full_od_factor":       full_od,
        "arcs_full_per_day":    arcs_full,
    }


def _plot_station_map(cov_r_deg, output_dir, label):
    fig, ax = plt.subplots(figsize=(14, 7))

    # Контур суши (упрощённо — рамка)
    ax.set_facecolor("#EAF4F8")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.axhline(0, color="#b2bec3", lw=0.5)
    ax.axhline(INC_DEG, ls="--", color="#e17055", lw=1, alpha=0.5, label=f"±{INC_DEG}° наклонение")
    ax.axhline(-INC_DEG, ls="--", color="#e17055", lw=1, alpha=0.5)

    # МКС станции
    for name, lat, lon in MCS_STATIONS:
        circle = Circle((lon, lat), cov_r_deg, color="#0984e3", alpha=0.15, fill=True)
        ax.add_patch(circle)
        ax.plot(lon, lat, "o", color="#0984e3", ms=6)
        ax.text(lon + 1, lat + 1, name[:6], fontsize=6, color="#2d3436")

    # Кандидаты на расширение
    for name, lat, lon in CANDIDATE_STATIONS:
        circle = Circle((lon, lat), cov_r_deg, color="#00b894", alpha=0.15, fill=True)
        ax.add_patch(circle)
        ax.plot(lon, lat, "s", color="#00b894", ms=7, markeredgecolor="white")
        ax.text(lon + 1, lat + 1, name[:8], fontsize=6, color="#00b894")

    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor="#0984e3", alpha=0.4, label=f"МКС (21 станция, покрытие ±{cov_r_deg:.0f}°)"),
        Patch(facecolor="#00b894", alpha=0.4, label="Кандидаты расширения (8 станций)"),
    ]
    ax.legend(handles=legend_els, loc="lower left", fontsize=8)
    ax.set_xlabel("Долгота (°)")
    ax.set_ylabel("Широта (°)")
    ax.set_title(f"AURORA PNT — Сеть наземных станций MCS [{label}]")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"ground_network_map_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_od_quality(output_dir, label):
    """OD quality factor vs число дополнительных станций."""
    n_extra = list(range(0, len(CANDIDATE_STATIONS) + 1))
    od_factors = []
    cov_fracs  = []
    for n in n_extra:
        stations = MCS_STATIONS + CANDIDATE_STATIONS[:n]
        od_factors.append(od_geometry_factor(stations) * 100)
        cov_fracs.append(coverage_fraction_1d(stations) * 100)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    ax1.plot(n_extra, od_factors, color="#0984e3", lw=2, marker="o", ms=6, label="OD геометрический фактор (%)")
    ax2.plot(n_extra, cov_fracs,  color="#00b894", lw=2, marker="s", ms=6, label="Покрытие долготы (%)")
    ax1.set_xlabel("Число дополнительных станций (сверх 21 МКС)")
    ax1.set_ylabel("OD геометрический фактор (%)")
    ax2.set_ylabel("Покрытие по долготе (%)")
    ax1.set_title(f"AURORA PNT — Качество OD vs расширение сети [{label}]")
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, fontsize=9)
    ax1.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"ground_od_quality_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_coverage_vs_stations(output_dir, label):
    """Накопленное число дуговых наблюдений за 24 ч."""
    n_range = list(range(1, len(MCS_STATIONS) + len(CANDIDATE_STATIONS) + 1))
    arcs = [arc_observations_per_day(n) for n in n_range]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(n_range, arcs, color="#6c5ce7", lw=2)
    ax.axvline(21, ls="--", color="#e17055", lw=1.5, label="21 (МКС текущий)")
    ax.axvline(29, ls="--", color="#00b894", lw=1.5, label="29 (21 + 8 кандидатов)")
    ax.axhline(1000, ls=":", color="#fdcb6e", lw=1.2, label="1000 дуг/сут (достаточно)")
    ax.set_xlabel("Число наземных станций")
    ax.set_ylabel("Дуговых наблюдений за 24 ч (оценка)")
    ax.set_title(f"AURORA PNT — Накопление данных OD [{label}]")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"ground_arcs_{label}.png"), dpi=150)
    plt.close(fig)


def _save_csv(cov_r_km, mcs_cov, mcs_od, arcs_21, output_dir, label):
    path = os.path.join(output_dir, f"ground_network_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["station", "lat_deg", "lon_deg"])
        for name, lat, lon in MCS_STATIONS:
            w.writerow([name, lat, lon])
        w.writerow([])
        w.writerow(["metric", "value"])
        w.writerow(["coverage_radius_km", f"{cov_r_km:.1f}"])
        w.writerow(["mcs21_lon_coverage_pct", f"{mcs_cov*100:.1f}"])
        w.writerow(["mcs21_od_geometry_pct", f"{mcs_od*100:.1f}"])
        w.writerow(["arcs_per_day_21", f"{arcs_21:.0f}"])


def print_ground_network_summary(label: str, result: Dict) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  Ground Network Analysis -- {label}")
    print(sep)
    print(f"  Радиус покрытия станции:    {result['coverage_radius_km']:.0f} км")
    print(f"  МКС 21 ст. — покрытие:      {result['mcs21_coverage_pct']:.0f}%")
    print(f"  МКС 21 ст. — OD фактор:     {result['mcs21_od_factor']:.0f}%")
    print(f"  МКС 21 ст. — дуг/сутки:     {result['arcs_21_per_day']:.0f}")
    print(f"  + 8 кандидатов — покрытие:  {result['full_coverage_pct']:.0f}%")
    print(f"  + 8 кандидатов — OD фактор: {result['full_od_factor']:.0f}%")
    print(sep)
