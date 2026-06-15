"""
Сквозная end-to-end PVT-симуляция АВРОРА.

Упрощённая модель «голым numpy» — без scipy / skyfield. Стек:
  1. Пропагация 300 КА: Walker 300/15 (i=75°, h=1000 км), круговые орбиты,
     аналитическое движение по орбите θ = θ0 + n·t, J2-прецессия RAAN.
  2. Геометрия: ECI → ECEF (вращение Земли), пересчёт в локальный ENU,
     отбор спутников выше маски возвышения 10°.
  3. Бюджет ошибок (на псевдодальность):
        σ_eph     = 5  см    (эфемериды + SSR)
        σ_clk     = 2  см    (часы спутника + SSR)
        σ_iono    = 10 см    (двухчастотный остаток)
        σ_tropo   = 5  см    (модель тропосферы NMF + случайный остаток)
        σ_mp      = 10 см    (многолучёвость L1)
        σ_noise   = 15 см    (тепловой шум)
        σ_UERE   ≈ sqrt(5²+2²+10²+5²+10²+15²) ≈ 22 см
  4. PVT: Weighted Least Squares 4×4 на каждой эпохе (x, y, z, dt).
  5. Метрики: RMS_H, RMS_V, CEP95, PDOP, доступность, N_vis.

Ссылки:
  Walker (1984)         — Satellite constellations. J. British Interplan. Soc.
  Vallado (2013)        — Fundamentals of Astrodynamics, 4 ed., §9.7 (J2).
  Misra & Enge (2011)   — GPS: Signals, Measurements, and Performance, §6.
  Kaplan & Hegarty (20)17 — Understanding GPS/GNSS, 3rd ed., §7.
"""

import sys, math, os, csv
from typing import Dict, List, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ── Палитра ───────────────────────────────────────────────────────────────────
COLORS = ["#e17055", "#fdcb6e", "#0984e3", "#00b894", "#6c5ce7", "#74b9ff"]

# ── Физические константы и параметры созвездия ────────────────────────────────
GM      = 3.986004418e14        # м³/с² гравит. параметр Земли
R_E     = 6371000.0             # м, средн. радиус Земли
J2      = 1.08263e-3            # коэффициент сжатия Земли
OMEGA_E = 7.2921159e-5          # рад/с скорость вращения Земли
C_LIGHT = 299792458.0           # м/с

# Walker 300/15
N_SAT   = 300
N_PLANE = 15
N_PER   = N_SAT // N_PLANE      # 20 КА/плоскость
INCL    = math.radians(75.0)
ALT     = 1000_000.0            # м
A       = R_E + ALT             # 7 371 км
F_PHASE = 1                     # phasing parameter

# Параметры симуляции
DURATION_S = 24 * 3600
STEP_S     = 60.0
MASK_DEG   = 10.0
MASK_RAD   = math.radians(MASK_DEG)

# Бюджет ошибок (на псевдодальность, м, 1σ)
SIGMA_EPH   = 0.05
SIGMA_CLK   = 0.02
SIGMA_IONO  = 0.10
SIGMA_TROPO = 0.05
SIGMA_MP    = 0.10
SIGMA_NOISE = 0.15
SIGMA_UERE  = math.sqrt(SIGMA_EPH**2 + SIGMA_CLK**2 + SIGMA_IONO**2 +
                         SIGMA_TROPO**2 + SIGMA_MP**2 + SIGMA_NOISE**2)

# Пользователи
USERS = [
    {"name": "Москва",      "lat":  55.75, "lon":  37.62, "color": "#e17055"},
    {"name": "Норильск",    "lat":  69.35, "lon":  88.20, "color": "#fdcb6e"},
    {"name": "Сингапур",    "lat":   1.35, "lon": 103.82, "color": "#0984e3"},
    {"name": "Кейптаун",    "lat": -33.92, "lon":  18.42, "color": "#00b894"},
]


# ─────────────────────────────────────────────────────────────────────────────
#                       Орбитальная пропагация
# ─────────────────────────────────────────────────────────────────────────────
def _build_constellation() -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Возвращает RAAN0[N], M0[N], n[1] (вс. КА одинаковое), incl[N]."""
    raan0 = np.zeros(N_SAT)
    M0    = np.zeros(N_SAT)
    for p in range(N_PLANE):
        for k in range(N_PER):
            idx = p * N_PER + k
            raan0[idx] = (2.0 * math.pi) * p / N_PLANE
            # фазовый сдвиг между плоскостями: f * Δu / N_plane
            phase_offset = (2.0 * math.pi) * F_PHASE * p / N_SAT
            M0[idx] = (2.0 * math.pi) * k / N_PER + phase_offset
    n = math.sqrt(GM / A**3)                     # рад/с
    return raan0, M0, n, INCL


def _raan_dot_j2(a: float, incl: float) -> float:
    """Скорость дрейфа RAAN из-за J2 (рад/с)."""
    n0 = math.sqrt(GM / a**3)
    return -1.5 * n0 * J2 * (R_E / a) ** 2 * math.cos(incl)


def _sat_eci(t: float, raan0: np.ndarray, M0: np.ndarray,
             n: float, incl: float, raan_dot: float) -> np.ndarray:
    """ECI координаты всех 300 КА в момент t (с). Shape: (N, 3) метры."""
    RAAN = raan0 + raan_dot * t                          # (N,)
    u    = M0   + n * t                                  # arg. of lat (круг. орбита)

    cu, su = np.cos(u),    np.sin(u)
    cO, sO = np.cos(RAAN), np.sin(RAAN)
    ci, si = math.cos(incl), math.sin(incl)

    # Положение в плоскости орбиты: (a·cos u, a·sin u, 0)
    # Поворот: Rz(-Ω) · Rx(-i) · r_pqw
    x = A * (cu * cO - su * ci * sO)
    y = A * (cu * sO + su * ci * cO)
    z = A * (su * si)
    return np.stack([x, y, z], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
#                       Геометрия пользователь/спутник
# ─────────────────────────────────────────────────────────────────────────────
def _user_ecef(lat_deg: float, lon_deg: float, h: float = 0.0) -> np.ndarray:
    """Грубо: сфера радиуса R_E. Достаточно для масштаба задачи."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    r = R_E + h
    return np.array([r * math.cos(lat) * math.cos(lon),
                     r * math.cos(lat) * math.sin(lon),
                     r * math.sin(lat)])


def _eci_to_ecef(pos_eci: np.ndarray, t: float) -> np.ndarray:
    """Поворот ECI → ECEF на угол GMST ≈ OMEGA_E · t."""
    th = OMEGA_E * t
    c, s = math.cos(th), math.sin(th)
    R = np.array([[ c, s, 0.0],
                  [-s, c, 0.0],
                  [0.0, 0.0, 1.0]])
    return pos_eci @ R.T


def _enu_matrix(lat_deg: float, lon_deg: float) -> np.ndarray:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sL, cL = math.sin(lat), math.cos(lat)
    sO, cO = math.sin(lon), math.cos(lon)
    return np.array([[-sO,        cO,         0.0],
                     [-sL * cO,  -sL * sO,    cL],
                     [ cL * cO,   cL * sO,    sL]])


def _visibility_and_los(sats_ecef: np.ndarray, user_ecef: np.ndarray,
                        enu: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Маска видимости (elev>10°) и единичные LOS-векторы (user→sat, ECEF)."""
    diff = sats_ecef - user_ecef                                 # (N, 3)
    rng  = np.linalg.norm(diff, axis=1)
    los  = diff / rng[:, None]
    enu_v = los @ enu.T                                          # (N, 3)
    elev = np.arcsin(enu_v[:, 2])
    visible = elev > MASK_RAD
    return visible, los


# ─────────────────────────────────────────────────────────────────────────────
#                              PVT (WLS)
# ─────────────────────────────────────────────────────────────────────────────
def _wls_pvt_error(los_vis: np.ndarray, enu: np.ndarray, sigma_uere: float,
                   rng_gen: np.random.Generator
                   ) -> Tuple[float, float, float, float]:
    """
    Вычисляет ошибки позиции в локальной ENU-системе через WLS.
    Возвращает (e_horizontal, e_vertical, PDOP, HDOP).
    """
    n = los_vis.shape[0]
    if n < 4:
        return float("nan"), float("nan"), float("nan"), float("nan")

    # Матрица геометрии H: [-los, 1] в ECEF
    H = np.hstack([-los_vis, np.ones((n, 1))])
    try:
        Q = np.linalg.inv(H.T @ H)                               # (4,4)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan"), float("nan"), float("nan")

    # DOP в ECEF — преобразуем 3×3 блок в ENU
    Q_pos_ecef = Q[:3, :3]
    R_e2enu    = enu                                              # ECEF→ENU
    Q_enu      = R_e2enu @ Q_pos_ecef @ R_e2enu.T
    var_enu    = np.diag(Q_enu)                                   # (E,N,U)
    pdop = math.sqrt(max(0.0, np.trace(Q_pos_ecef)))
    hdop = math.sqrt(max(0.0, var_enu[0] + var_enu[1]))
    vdop = math.sqrt(max(0.0, var_enu[2]))

    # Случайная реализация ошибки позиции
    eps = rng_gen.normal(0.0, sigma_uere, size=n)                # шум на каждый КА
    K   = Q @ H.T                                                # (4, n)
    d   = K @ eps                                                # (4,) ECEF dx,dy,dz,dt
    d_enu = enu @ d[:3]
    e_h = math.sqrt(d_enu[0] ** 2 + d_enu[1] ** 2)
    e_v = abs(d_enu[2])
    return e_h, e_v, pdop, hdop


# ─────────────────────────────────────────────────────────────────────────────
#                              Главная функция
# ─────────────────────────────────────────────────────────────────────────────
def run_e2e_pipeline_analysis(output_dir: str, label: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(20260523)

    raan0, M0, n_mean, incl = _build_constellation()
    raan_dot = _raan_dot_j2(A, incl)

    times = np.arange(0.0, DURATION_S + STEP_S * 0.5, STEP_S)
    n_t   = len(times)

    # Заранее переходим в ECI и ECEF — циклы по времени для экономии памяти
    user_results: Dict[str, Dict[str, np.ndarray]] = {}
    for u in USERS:
        user_results[u["name"]] = {
            "t":    times.copy(),
            "e_h":  np.full(n_t, np.nan),
            "e_v":  np.full(n_t, np.nan),
            "pdop": np.full(n_t, np.nan),
            "hdop": np.full(n_t, np.nan),
            "nvis": np.zeros(n_t, dtype=int),
        }
        u["_ecef"] = _user_ecef(u["lat"], u["lon"])
        u["_enu"]  = _enu_matrix(u["lat"], u["lon"])

    for it, t in enumerate(times):
        sats_eci  = _sat_eci(t, raan0, M0, n_mean, incl, raan_dot)
        sats_ecef = _eci_to_ecef(sats_eci, t)
        for u in USERS:
            vis, los = _visibility_and_los(sats_ecef, u["_ecef"], u["_enu"])
            nv = int(vis.sum())
            user_results[u["name"]]["nvis"][it] = nv
            if nv >= 4:
                e_h, e_v, pdop, hdop = _wls_pvt_error(
                    los[vis], u["_enu"], SIGMA_UERE, rng)
                user_results[u["name"]]["e_h"][it]  = e_h
                user_results[u["name"]]["e_v"][it]  = e_v
                user_results[u["name"]]["pdop"][it] = pdop
                user_results[u["name"]]["hdop"][it] = hdop

    # Метрики по пользователям
    summary = []
    for u in USERS:
        d   = user_results[u["name"]]
        ok  = ~np.isnan(d["e_h"])
        avail = float(ok.sum() / n_t * 100.0)
        if ok.sum() > 0:
            rms_h = float(np.sqrt(np.mean(d["e_h"][ok] ** 2))) * 100  # см
            rms_v = float(np.sqrt(np.mean(d["e_v"][ok] ** 2))) * 100
            cep95 = float(np.percentile(d["e_h"][ok], 95)) * 100
            mean_pdop = float(np.nanmean(d["pdop"]))
            mean_nv   = float(np.mean(d["nvis"]))
        else:
            rms_h = rms_v = cep95 = mean_pdop = mean_nv = float("nan")
        summary.append({
            "name":      u["name"],
            "lat":       u["lat"], "lon": u["lon"],
            "avail_pct": avail,
            "mean_pdop": mean_pdop,
            "mean_nvis": mean_nv,
            "rms_h_cm":  rms_h,
            "rms_v_cm":  rms_v,
            "cep95_cm":  cep95,
        })

    results = {
        "users":         user_results,
        "user_summary":  summary,
        "sigma_uere_cm": SIGMA_UERE * 100,
        "n_sat":         N_SAT,
        "n_epochs":      n_t,
        "step_s":        STEP_S,
        "duration_h":    DURATION_S / 3600,
    }

    _plot_position_error(user_results, output_dir, label)
    _plot_pdop(user_results, output_dir, label)
    _plot_nvis(user_results, output_dir, label)
    _plot_cdf(user_results, output_dir, label)
    _save_csv(summary, output_dir, label)
    return results


# ── Графики ───────────────────────────────────────────────────────────────────
def _plot_position_error(user_results, output_dir, label):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    for u, col in zip(USERS, [u["color"] for u in USERS]):
        d = user_results[u["name"]]
        th = d["t"] / 3600.0
        ax1.plot(th, d["e_h"] * 100, color=col, lw=1.2, alpha=0.85, label=u["name"])
        ax2.plot(th, d["e_v"] * 100, color=col, lw=1.2, alpha=0.85, label=u["name"])

    ax1.axhline(50, ls="--", color="#6c5ce7", lw=1.2, label="50 см (цель CEP95)")
    ax1.set_ylabel("Гор. ошибка (см)")
    ax1.set_title(f"АВРОРА E2E PVT — ошибки позиционирования за 24 ч  [{label}]")
    ax1.legend(fontsize=8, ncol=3, loc="upper right")
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 120)

    ax2.axhline(100, ls="--", color="#6c5ce7", lw=1.2, label="100 см (цель верт.)")
    ax2.set_xlabel("Время (часы)")
    ax2.set_ylabel("Верт. ошибка (см)")
    ax2.legend(fontsize=8, ncol=3, loc="upper right")
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 200)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"e2e_position_error_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_pdop(user_results, output_dir, label):
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for u in USERS:
        d = user_results[u["name"]]
        th = d["t"] / 3600.0
        ax.plot(th, d["pdop"], color=u["color"], lw=1.3, alpha=0.85, label=u["name"])
    ax.axhline(2.0, ls="--", color="#00b894", lw=1.3, label="PDOP 2 — хороший")
    ax.axhline(4.0, ls=":",  color="#e17055", lw=1.3, label="PDOP 4 — пороговый")
    ax.set_xlabel("Время (часы)")
    ax.set_ylabel("PDOP")
    ax.set_title(f"АВРОРА E2E PVT — PDOP за 24 ч  [{label}]")
    ax.set_ylim(0, 6)
    ax.legend(fontsize=9, ncol=3, loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"e2e_pdop_timeline_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_nvis(user_results, output_dir, label):
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for u in USERS:
        d = user_results[u["name"]]
        th = d["t"] / 3600.0
        ax.plot(th, d["nvis"], color=u["color"], lw=1.3, alpha=0.85, label=u["name"])
    ax.axhline(4, ls="--", color="#e17055", lw=1.3, label="N_min = 4 (PVT)")
    ax.axhline(8, ls=":",  color="#00b894", lw=1.3, label="N = 8 (целевой минимум)")
    ax.set_xlabel("Время (часы)")
    ax.set_ylabel("N видимых КА (elev > 10°)")
    ax.set_title(f"АВРОРА E2E PVT — число видимых КА  [{label}]")
    ax.legend(fontsize=9, ncol=3, loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"e2e_nsat_visible_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_cdf(user_results, output_dir, label):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for u in USERS:
        d = user_results[u["name"]]
        ok = ~np.isnan(d["e_h"])
        if ok.sum() == 0:
            continue
        eh = np.sort(d["e_h"][ok] * 100)
        cdf = np.arange(1, len(eh) + 1) / len(eh)
        ax.plot(eh, cdf * 100, color=u["color"], lw=2.0, alpha=0.9, label=u["name"])
        # маркер 95 %
        p95 = np.percentile(eh, 95)
        ax.axvline(p95, ls=":", color=u["color"], lw=0.9, alpha=0.5)

    ax.axvline(50.0, ls="--", color="#6c5ce7", lw=1.5, label="0,5 м — цель")
    ax.axhline(95.0, ls=":",  color="#2d3436", lw=1.0, label="95 %")
    ax.set_xlabel("Горизонтальная ошибка (см)")
    ax.set_ylabel("CDF, %")
    ax.set_title(f"АВРОРА E2E PVT — CDF гор. ошибок (4 точки)  [{label}]")
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 102)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"e2e_cdf_comparison_{label}.png"), dpi=150)
    plt.close(fig)


# ── CSV ───────────────────────────────────────────────────────────────────────
def _save_csv(summary, output_dir, label):
    path = os.path.join(output_dir, f"e2e_pipeline_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["user", "lat", "lon", "avail_pct",
                    "mean_PDOP", "mean_Nvis",
                    "RMS_H_cm", "RMS_V_cm", "CEP95_cm"])
        for s in summary:
            w.writerow([s["name"], f"{s['lat']:.2f}", f"{s['lon']:.2f}",
                        f"{s['avail_pct']:.2f}", f"{s['mean_pdop']:.2f}",
                        f"{s['mean_nvis']:.2f}",
                        f"{s['rms_h_cm']:.2f}",   f"{s['rms_v_cm']:.2f}",
                        f"{s['cep95_cm']:.2f}"])


def print_e2e_pipeline_summary(label: str, results: Dict) -> None:
    sep = "=" * 84
    print(f"\n{sep}")
    print(f"  E2E PVT Pipeline -- {label}")
    print(sep)
    print(f"  Созвездие:       Walker {results['n_sat']}/15, i=75°, h=1000 км")
    print(f"  Длительность:    {results['duration_h']:.1f} ч, шаг {results['step_s']:.0f} с, "
          f"{results['n_epochs']} эпох")
    print(f"  σ_UERE:          {results['sigma_uere_cm']:.1f} см "
          f"(eph 5 + clk 2 + iono 10 + tropo 5 + mp 10 + noise 15)")

    print(f"\n  {'Пользователь':<14}{'Avail%':>8}{'PDOP':>7}{'N_vis':>7}"
          f"{'RMS_H(см)':>11}{'RMS_V(см)':>11}{'CEP95(см)':>11}")
    print(f"  {'─' * 70}")
    for s in results["user_summary"]:
        print(f"  {s['name']:<14}{s['avail_pct']:>7.2f}%{s['mean_pdop']:>7.2f}"
              f"{s['mean_nvis']:>7.1f}{s['rms_h_cm']:>11.2f}"
              f"{s['rms_v_cm']:>11.2f}{s['cep95_cm']:>11.2f}")
    print(sep)
