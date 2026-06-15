"""
Временной анализ DOP и доступности созвездия АВРОРА.

Аналитически (круговая орбита) распространяет упрощённое созвездие
Walker Delta 300/15 (h = 1000 км, i = 75°, T ≈ 105 мин, фазинг F = 1)
на 24 ч с шагом 60 с для набора пользовательских широт. На каждой
эпохе для каждого пользователя определяются спутники выше маски
возвышения (10°) по сферической геометрии, строится матрица геометрии
H (единичные векторы линии визирования + 1 для часов), и вычисляются
GDOP/PDOP/HDOP/VDOP как корни из следов блоков (HᵀH)⁻¹. При числе
видимых спутников < 4 — перерыв обслуживания (outage).

Параметры системы: Walker Delta 300/15, h = 1000 км, i = 75°, L1+L5.
Скорректированный бюджет линии: C/N0 = 52,6 дБ-Гц в зените.

Ссылки:
  Walker (1984) — Satellite Constellations. J. British Interplanetary Society.
  Kaplan & Hegarty (2017) — Understanding GPS/GNSS, 3rd ed. (DOP geometry).
  Montenbruck & Gill (2000) — Satellite Orbits: Models, Methods, Applications.
  Reid et al. (2018) — Broadband LEO Constellations for Navigation. NAVIGATION.
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

# ── Параметры созвездия Walker Delta 300/15/F=1 ─────────────────────────────
N_SAT_TOTAL = 300
N_PLANES = 15
N_PER_PLANE = N_SAT_TOTAL // N_PLANES        # 20 КА на плоскость
PHASING_F = 1
INC_DEG = 75.0                               # наклонение, град
ALT_KM = 1000.0                              # высота, км
R_EARTH_KM = 6371.0
R_ORBIT_KM = R_EARTH_KM + ALT_KM
MU_EARTH = 398600.4418                        # км^3/с^2

ELEV_MASK_DEG = 10.0                          # номинальная маска возвышения
ELEV_MASKS = [5, 10, 15, 20]                  # маски для анализа доступности
PDOP_THRESHOLD = 6.0                          # порог приемлемой геометрии

# Период обращения, с (третий закон Кеплера)
T_ORBIT_S = 2.0 * math.pi * math.sqrt(R_ORBIT_KM ** 3 / MU_EARTH)

SIM_DURATION_S = 24 * 3600
SIM_STEP_S = 60
USER_LATS = [0.0, 30.0, 55.0, 70.0, 80.0]     # пользовательские широты
MOSCOW_LAT = 55.75
MOSCOW_LON = 37.62


def _constellation_eci(t_s: float) -> np.ndarray:
    """ECI-позиции всех 300 КА (км) в момент t (с). Возвращает (N,3)."""
    n = 2.0 * math.pi / T_ORBIT_S              # средняя угловая скорость
    inc = math.radians(INC_DEG)
    pos = np.empty((N_SAT_TOTAL, 3))
    idx = 0
    for p in range(N_PLANES):
        raan = 2.0 * math.pi * p / N_PLANES
        cos_O, sin_O = math.cos(raan), math.sin(raan)
        cos_i, sin_i = math.cos(inc), math.sin(inc)
        for k in range(N_PER_PLANE):
            # фазинг Walker Delta: смещение аномалии между плоскостями
            m0 = 2.0 * math.pi * k / N_PER_PLANE \
                + 2.0 * math.pi * PHASING_F * p / N_SAT_TOTAL
            u = m0 + n * t_s                   # аргумент широты
            xo = R_ORBIT_KM * math.cos(u)
            yo = R_ORBIT_KM * math.sin(u)
            # поворот: плоскость орбиты -> ECI (наклонение, затем RAAN)
            x1 = xo
            y1 = yo * cos_i
            z1 = yo * sin_i
            pos[idx, 0] = x1 * cos_O - y1 * sin_O
            pos[idx, 1] = x1 * sin_O + y1 * cos_O
            pos[idx, 2] = z1
            idx += 1
    return pos


def _user_eci(lat_deg: float, lon_deg: float, t_s: float) -> np.ndarray:
    """ECI-позиция пользователя (км) с учётом вращения Земли."""
    we = 2.0 * math.pi / 86164.0               # звёздная угловая скорость, рад/с
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg) + we * t_s
    return np.array([
        R_EARTH_KM * math.cos(lat) * math.cos(lon),
        R_EARTH_KM * math.cos(lat) * math.sin(lon),
        R_EARTH_KM * math.sin(lat),
    ])


def _enu_basis(user: np.ndarray):
    """Локальный базис ENU (восток, север, верх) в точке пользователя."""
    up = user / np.linalg.norm(user)
    east = np.cross(np.array([0.0, 0.0, 1.0]), up)
    en = np.linalg.norm(east)
    if en < 1e-9:
        east = np.array([1.0, 0.0, 0.0])
    else:
        east = east / en
    north = np.cross(up, east)
    return east, north, up


def _dop_at(user: np.ndarray, sats: np.ndarray, mask_deg: float):
    """Возвращает (n_vis, GDOP, PDOP, HDOP, VDOP) для пользователя."""
    east, north, up = _enu_basis(user)
    los = sats - user                                   # (N,3) вектор на КА
    rng = np.linalg.norm(los, axis=1)
    unit = los / rng[:, None]
    sin_el = unit @ up
    el_deg = np.degrees(np.arcsin(np.clip(sin_el, -1.0, 1.0)))
    vis = el_deg >= mask_deg
    n_vis = int(vis.sum())
    if n_vis < 4:
        return n_vis, None, None, None, None

    u = unit[vis]
    # строки матрицы H в ENU: [-e, -n, -u, 1]
    e_c = u @ east
    n_c = u @ north
    u_c = u @ up
    H = np.column_stack([-e_c, -n_c, -u_c, np.ones(n_vis)])
    try:
        Q = np.linalg.inv(H.T @ H)
    except np.linalg.LinAlgError:
        return n_vis, None, None, None, None
    d = np.diag(Q)
    if np.any(d < 0):
        return n_vis, None, None, None, None
    gdop = math.sqrt(d.sum())
    pdop = math.sqrt(d[0] + d[1] + d[2])
    hdop = math.sqrt(d[0] + d[1])
    vdop = math.sqrt(d[2])
    return n_vis, gdop, pdop, hdop, vdop


def run_dop_temporal_analysis(output_dir: str, label: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    times = np.arange(0, SIM_DURATION_S, SIM_STEP_S)

    # Предрасчёт позиций созвездия на каждой эпохе
    sat_cache = [_constellation_eci(float(t)) for t in times]

    # ── 1) Сводка по широтам (маска 10°) ─────────────────────────────────────
    per_lat = {}
    for lat in USER_LATS:
        pdops, hdops, vdops, nsats = [], [], [], []
        n_outage = 0
        for ti, t in enumerate(times):
            user = _user_eci(lat, 0.0, float(t))
            nv, g, p, h, v = _dop_at(user, sat_cache[ti], ELEV_MASK_DEG)
            nsats.append(nv)
            if p is None or p >= PDOP_THRESHOLD or nv < 4:
                n_outage += 1
                if p is not None:
                    pdops.append(p); hdops.append(h); vdops.append(v)
            else:
                pdops.append(p); hdops.append(h); vdops.append(v)
        pdops = np.array(pdops); nsats = np.array(nsats)
        avail = 100.0 * (1.0 - n_outage / len(times))
        per_lat[lat] = {
            "pdop_mean": float(pdops.mean()),
            "pdop_min":  float(pdops.min()),
            "pdop_max":  float(pdops.max()),
            "avail_pct": avail,
            "nsat_mean": float(nsats.mean()),
            "nsat_min":  int(nsats.min()),
        }
    results = {"per_lat": per_lat, "elev_masks": ELEV_MASKS,
               "period_min": T_ORBIT_S / 60.0}

    # ── 2) Доступность vs маска для нескольких широт ──────────────────────────
    avail_mask = {}
    for lat in USER_LATS:
        row = []
        for m in ELEV_MASKS:
            ok = 0
            for ti, t in enumerate(times):
                user = _user_eci(lat, 0.0, float(t))
                nv, g, p, h, v = _dop_at(user, sat_cache[ti], float(m))
                if p is not None and p < PDOP_THRESHOLD and nv >= 4:
                    ok += 1
            row.append(100.0 * ok / len(times))
        avail_mask[lat] = row
    results["avail_mask"] = avail_mask

    # ── 3) Временные ряды для Москвы (55,75° с.ш.) ───────────────────────────
    t_pd, t_hd, t_vd, t_ns = [], [], [], []
    for ti, t in enumerate(times):
        user = _user_eci(MOSCOW_LAT, MOSCOW_LON, float(t))
        nv, g, p, h, v = _dop_at(user, sat_cache[ti], ELEV_MASK_DEG)
        t_ns.append(nv)
        t_pd.append(p if p is not None else np.nan)
        t_hd.append(h if h is not None else np.nan)
        t_vd.append(v if v is not None else np.nan)
    moscow = {
        "t_h":  times / 3600.0,
        "pdop": np.array(t_pd),
        "hdop": np.array(t_hd),
        "vdop": np.array(t_vd),
        "nsat": np.array(t_ns),
    }
    results["moscow"] = {
        "pdop_mean": float(np.nanmean(moscow["pdop"])),
        "nsat_mean": float(moscow["nsat"].mean()),
        "nsat_min":  int(moscow["nsat"].min()),
    }

    _plot_world_map(sat_cache, times, output_dir, label)
    _plot_avail_vs_mask(avail_mask, output_dir, label)
    _plot_timeseries(moscow, output_dir, label)
    _plot_nsat_hist(moscow, output_dir, label)
    _save_csv(results, output_dir, label)
    return results


def _plot_world_map(sat_cache, times, output_dir, label):
    """Средний PDOP по сетке долгота/широта (подвыборка эпох)."""
    lons = np.arange(-180, 181, 20)
    lats = np.arange(-85, 86, 10)
    sample_idx = np.linspace(0, len(times) - 1, 24).astype(int)
    grid = np.full((lats.size, lons.size), np.nan)
    for i, la in enumerate(lats):
        for j, lo in enumerate(lons):
            vals = []
            for ti in sample_idx:
                user = _user_eci(la, lo, float(times[ti]))
                nv, g, p, h, v = _dop_at(user, sat_cache[ti], ELEV_MASK_DEG)
                if p is not None:
                    vals.append(p)
            if vals:
                grid[i, j] = np.mean(vals)

    fig, ax = plt.subplots(figsize=(13, 6))
    pm = ax.pcolormesh(lons, lats, grid, shading="auto",
                        cmap="viridis_r", vmin=1.0, vmax=4.0)
    cb = fig.colorbar(pm, ax=ax)
    cb.set_label("Средний PDOP")
    ax.axhline(INC_DEG, ls="--", color="#e17055", lw=1.3,
               label=f"Наклонение {INC_DEG:.0f}°")
    ax.axhline(-INC_DEG, ls="--", color="#e17055", lw=1.3)
    ax.set_xlabel("Долгота (°)")
    ax.set_ylabel("Широта (°)")
    ax.set_title(f"Карта среднего PDOP — Walker 300/15 [{label}]")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"dop_world_map_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_avail_vs_mask(avail_mask, output_dir, label):
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (lat, vals) in enumerate(avail_mask.items()):
        ax.plot(ELEV_MASKS, vals, "o-", color=PALETTE[i % len(PALETTE)],
                lw=2.4, ms=8, label=f"φ = {lat:.0f}°")
    ax.axhline(99.5, ls="--", color="#2d3436", lw=1.3,
               label="Цель 99,5%")
    ax.set_xlabel("Угол маски возвышения (°)")
    ax.set_ylabel("Доступность (PDOP<6, N≥4), %")
    ax.set_title(f"Доступность vs маска возвышения [{label}]")
    ax.set_xticks(ELEV_MASKS)
    ax.set_ylim(min(90, ax.get_ylim()[0]), 100.5)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"availability_vs_mask_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_timeseries(moscow, output_dir, label):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(moscow["t_h"], moscow["pdop"], color="#6c5ce7", lw=1.6, label="PDOP")
    ax.plot(moscow["t_h"], moscow["hdop"], color="#0984e3", lw=1.4, label="HDOP")
    ax.plot(moscow["t_h"], moscow["vdop"], color="#e17055", lw=1.4, label="VDOP")
    ax.axhline(PDOP_THRESHOLD, ls="--", color="#2d3436", lw=1.2,
               label=f"Порог PDOP = {PDOP_THRESHOLD:.0f}")
    ax.set_xlabel("Время (ч)")
    ax.set_ylabel("DOP")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.set_title(f"DOP во времени — Москва (55,75° с.ш.) за 24 ч [{label}]")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"dop_timeseries_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_nsat_hist(moscow, output_dir, label):
    ns = moscow["nsat"]
    fig, ax = plt.subplots(figsize=(11, 6))
    bins = np.arange(ns.min() - 0.5, ns.max() + 1.5, 1.0)
    ax.hist(ns, bins=bins, color="#00b894", edgecolor="white", alpha=0.85)
    ax.axvline(ns.mean(), ls="--", color="#2d3436", lw=1.6,
               label=f"Среднее = {ns.mean():.1f}")
    ax.axvline(4, ls=":", color="#e17055", lw=1.6, label="Минимум для PVT (4)")
    ax.set_xlabel("Число видимых спутников (маска 10°)")
    ax.set_ylabel("Число эпох")
    ax.set_title(f"Гистограмма видимых КА — Москва [{label}]")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"nsat_histogram_{label}.png"), dpi=150)
    plt.close(fig)


def _save_csv(results, output_dir, label):
    path = os.path.join(output_dir, f"dop_temporal_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lat_deg", "pdop_mean", "pdop_min", "pdop_max",
                    "avail_pct", "nsat_mean", "nsat_min"])
        for lat, r in results["per_lat"].items():
            w.writerow([f"{lat:.0f}",
                        f"{r['pdop_mean']:.3f}",
                        f"{r['pdop_min']:.3f}",
                        f"{r['pdop_max']:.3f}",
                        f"{r['avail_pct']:.3f}",
                        f"{r['nsat_mean']:.2f}",
                        r['nsat_min']])
        w.writerow([])
        w.writerow(["lat_deg"] + [f"avail_mask{m}deg_pct" for m in results["elev_masks"]])
        for lat, vals in results["avail_mask"].items():
            w.writerow([f"{lat:.0f}"] + [f"{v:.3f}" for v in vals])


def print_dop_temporal_summary(label: str, results: Dict) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  Temporal DOP / Availability -- {label}")
    print(sep)
    print(f"  Созвездие: Walker Delta {N_SAT_TOTAL}/{N_PLANES}, "
          f"h={ALT_KM:.0f} км, i={INC_DEG:.0f}°, T={results['period_min']:.1f} мин")
    print(f"  {'Широта':>7} {'PDOP_ср':>9} {'PDOP_max':>9} "
          f"{'Дост.%':>9} {'N_ср':>7} {'N_min':>6}")
    print(f"  {'-' * 56}")
    for lat, r in results["per_lat"].items():
        print(f"  {lat:>6.0f}° {r['pdop_mean']:>9.2f} {r['pdop_max']:>9.2f} "
              f"{r['avail_pct']:>9.3f} {r['nsat_mean']:>7.1f} {r['nsat_min']:>6d}")
    print(f"  {'-' * 56}")
    m = results["moscow"]
    print(f"  Москва (55,75°): PDOP_ср={m['pdop_mean']:.2f}  "
          f"N_ср={m['nsat_mean']:.1f}  N_min={m['nsat_min']}")
    print("  Проверка: доступность(PDOP<6, маска 10°) > 99,5%; N_ср >= 8")
    print(sep)


if __name__ == "__main__":
    r = run_dop_temporal_analysis("results/dop_temporal", "phase5")
    print_dop_temporal_summary("phase5", r)
