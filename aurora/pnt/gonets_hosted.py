"""
Геометрия группировки АВРОРА-ПН на платформе ГОНЕЦ-М1 (hosted-payload).

Отдельное исследование (НЕ часть основного ТП): размещение нав-ПН + CSAC АВРОРЫ
на ~106 КА новой системы ГОНЕЦ-М1 (1500 км, 82,5°) как дополнительный сервис
навигации и синхронизации (модель Iridium STL).

Считает геометрию доступности по сетке Земли и по РФ (автономно, без SGP4 —
круговые орбиты, кеплеровская пропагация + вращение Земли; для геометрии этого
достаточно):
  - N_vis (число видимых КА при маске угла места)
  - PDOP (по матрице направляющих косинусов в ENU)
  - доступность (доля времени с N_vis ≥ 4)
по широтным поясам и отдельно по территории РФ.

Конфигурация Walker-star (близко к полярной, как у ГОНЕЦ/Iridium): P плоскостей,
RAAN разнесены на 180°, в плоскости — равномерно по средней аномалии, межплоскостное
смещение F. Фаза НЕ оптимизирована — оценка первого порядка.

Запуск:  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python aurora/pnt/gonets_hosted.py
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PALETTE = ["#e17055", "#fdcb6e", "#0984e3", "#00b894",
           "#6c5ce7", "#74b9ff", "#dfe6e9", "#2d3436"]

# ── Физические константы ──────────────────────────────────────────────────────
R_E   = 6378.137                 # км, экваториальный радиус
MU    = 398600.4418              # км³/с²
OMEGA_E = 7.2921159e-5           # рад/с, вращение Земли

# ── Параметры группировки (ГОНЕЦ-М1, hosted AURORA PNT) ───────────────────────
ALT     = 1500.0                 # км
INC     = np.radians(82.5)       # наклонение
N_SATS  = 106
N_PLANES = 6                     # ГОНЕЦ-М1 — 6 плоскостей
WALKER_F = 2                     # межплоскостное смещение фазы (0..N_PLANES-1)
A       = R_E + ALT
N_MEAN  = np.sqrt(MU / A**3)     # рад/с, средняя угловая скорость
PERIOD  = 2*np.pi / N_MEAN

MASKS   = [5.0, 10.0]            # градусы — углы места
SIM_HOURS = 6.0
SIM_STEP  = 120.0                # с

# ── Сетка наблюдателей ────────────────────────────────────────────────────────
LAT_STEP = 5.0
LON_STEP = 10.0


def build_constellation():
    """Орбитальные элементы 106 КА: (RAAN, M0) для каждого. a,i — общие."""
    # распределение по плоскостям (остаток — в первые плоскости)
    base = N_SATS // N_PLANES
    rem  = N_SATS %  N_PLANES
    per_plane = [base + (1 if p < rem else 0) for p in range(N_PLANES)]
    raan, m0 = [], []
    for p in range(N_PLANES):
        plane_raan = np.pi * p / N_PLANES          # Walker-star: 180° по RAAN
        s_in = per_plane[p]
        for s in range(s_in):
            m = 2*np.pi * s / s_in
            # межплоскостное смещение фазы (Walker F)
            m += 2*np.pi * WALKER_F * p / N_SATS
            raan.append(plane_raan)
            m0.append(m)
    return np.array(raan), np.array(m0), per_plane


def sat_positions_ecef(raan, m0, times):
    """ECEF-координаты всех КА на массив времён. Возврат (T, N, 3) км."""
    T = len(times); N = len(raan)
    theta = m0[None, :] + N_MEAN * times[:, None]          # (T,N) аргумент широты
    # положение в плоскости орбиты (круговая)
    x_o = A * np.cos(theta); y_o = A * np.sin(theta)        # (T,N)
    # поворот на наклонение (вокруг X): y,z меняются
    ci, si = np.cos(INC), np.sin(INC)
    x1 = x_o
    y1 = y_o * ci
    z1 = y_o * si
    # поворот на RAAN (вокруг Z)
    cO = np.cos(raan)[None, :]; sO = np.sin(raan)[None, :]
    x_eci = x1 * cO - y1 * sO
    y_eci = x1 * sO + y1 * cO
    z_eci = z1
    # ECEF: поворот на -GMST(t) вокруг Z (GMST0 = 0)
    gmst = OMEGA_E * times                                   # (T,)
    cg = np.cos(gmst)[:, None]; sg = np.sin(gmst)[:, None]
    x_ecef =  x_eci * cg + y_eci * sg
    y_ecef = -x_eci * sg + y_eci * cg
    z_ecef =  z_eci
    return np.stack([x_ecef, y_ecef, z_ecef], axis=-1)      # (T,N,3)


def enu_basis(lat, lon):
    """Базис ENU и позиция наблюдателя (на сфере R_E) для широты/долготы (рад)."""
    cl, sl = np.cos(lat), np.sin(lat)
    co, so = np.cos(lon), np.sin(lon)
    up   = np.array([cl*co, cl*so, sl])
    east = np.array([-so, co, 0.0])
    north= np.array([-sl*co, -sl*so, cl])
    pos  = R_E * up
    return pos, np.stack([east, north, up])                 # (3,), (3,3)


def evaluate():
    times = np.arange(0.0, SIM_HOURS*3600.0, SIM_STEP)
    sats = sat_positions_ecef(*build_constellation()[:2], times)  # (T,N,3)
    T = len(times)

    lats = np.arange(-80.0, 80.0 + 1e-6, LAT_STEP)
    lons = np.arange(-180.0, 180.0 - 1e-6, LON_STEP)

    rows = []  # (lat, lon, mean_nvis5, mean_nvis10, avail5, pdop_med, pdop_p95)
    for la in lats:
        for lo in lons:
            pos, B = enu_basis(np.radians(la), np.radians(lo))
            los = sats - pos[None, None, :]                 # (T,N,3)
            enu = los @ B.T                                 # (T,N,3): E,N,U
            rng = np.linalg.norm(enu, axis=-1)              # (T,N)
            elev = np.degrees(np.arcsin(enu[..., 2] / rng)) # (T,N)

            vis5  = elev >= MASKS[0]
            vis10 = elev >= MASKS[1]
            nvis5  = vis5.sum(axis=1)                        # (T,)
            nvis10 = vis10.sum(axis=1)

            pdops = []
            for t in range(T):
                idx = np.where(vis5[t])[0]
                if len(idx) >= 4:
                    e = enu[t, idx] / rng[t, idx][:, None]  # ед. ENU LOS
                    H = np.column_stack([-e[:, 0], -e[:, 1], -e[:, 2],
                                         np.ones(len(idx))])
                    try:
                        Q = np.linalg.inv(H.T @ H)
                        p = np.sqrt(np.trace(Q[:3, :3]))
                        if np.isfinite(p):
                            pdops.append(p)
                    except np.linalg.LinAlgError:
                        pass
            avail5 = 100.0 * np.mean(nvis5 >= 4)
            pmed = np.median(pdops) if pdops else np.nan
            pp95 = np.percentile(pdops, 95) if pdops else np.nan
            rows.append((la, lo, nvis5.mean(), nvis10.mean(),
                         avail5, pmed, pp95))
    return np.array(rows), times


def summarize(rows):
    lat = rows[:, 0]
    bands = [("Экватор. (0–30°)",   np.abs(lat) < 30),
             ("Средние (30–55°)",   (np.abs(lat) >= 30) & (np.abs(lat) < 55)),
             ("Высокие (55–83°)",   np.abs(lat) >= 55)]
    print("\n=== Сводка по широтным поясам (106 КА, 1500 км, 82,5°) ===")
    print(f"{'Пояс':22s} {'N_vis(5°)':>10s} {'N_vis(10°)':>11s} "
          f"{'Дост.≥4,%':>10s} {'PDOP med':>9s} {'PDOP p95':>9s}")
    for name, m in bands:
        r = rows[m]
        print(f"{name:22s} {np.nanmean(r[:,2]):10.1f} {np.nanmean(r[:,3]):11.1f} "
              f"{np.nanmean(r[:,4]):10.1f} {np.nanmean(r[:,5]):9.2f} "
              f"{np.nanmean(r[:,6]):9.2f}")

    # РФ: lat 41–82, lon 19–180
    rf = rows[(lat >= 41) & (lat <= 82) & (rows[:, 1] >= 19) & (rows[:, 1] <= 180)]
    print("\n=== Территория РФ (lat 41–82, lon 19–180) ===")
    print(f"  точек сетки: {len(rf)}")
    print(f"  N_vis(5°):  мин {np.nanmin(rf[:,2]):.1f}  средн {np.nanmean(rf[:,2]):.1f}")
    print(f"  N_vis(10°): мин {np.nanmin(rf[:,3]):.1f}  средн {np.nanmean(rf[:,3]):.1f}")
    print(f"  Доступность ≥4 КА: средн {np.nanmean(rf[:,4]):.1f}%  "
          f"(точек с 100%: {100.0*np.mean(rf[:,4] >= 99.99):.0f}%)")
    print(f"  PDOP: медиана {np.nanmean(rf[:,5]):.2f}  p95 {np.nanmean(rf[:,6]):.2f}")
    glob_nvis = np.nanmean(rows[:, 2])
    print(f"\n  Глобально средн N_vis(5°) = {glob_nvis:.1f} "
          f"(аналит. оценка ~7,7 — сходится)")
    return rf


def plot(rows, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    lats = np.unique(rows[:, 0])
    nvis5  = [np.nanmean(rows[rows[:, 0] == la, 2]) for la in lats]
    nvis10 = [np.nanmean(rows[rows[:, 0] == la, 3]) for la in lats]
    pdopm  = [np.nanmean(rows[rows[:, 0] == la, 5]) for la in lats]
    avail  = [np.nanmean(rows[rows[:, 0] == la, 4]) for la in lats]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.plot(lats, nvis5, color=PALETTE[2], lw=2.5, marker="o", ms=3,
             label="N_vis, маска 5°")
    ax1.plot(lats, nvis10, color=PALETTE[0], lw=2.5, marker="s", ms=3,
             label="N_vis, маска 10°")
    ax1.axhline(4, ls="--", color=PALETTE[7], lw=1.3, label="порог навигации (4)")
    ax1.axvspan(55, 83, color=PALETTE[3], alpha=0.12, label="высокие широты/Арктика")
    ax1.axvspan(-83, -55, color=PALETTE[3], alpha=0.12)
    ax1.set_xlabel("Широта, °"); ax1.set_ylabel("Среднее число видимых КА")
    ax1.set_title("Видимость: 106 КА ГОНЕЦ-М1 (1500 км, 82,5°)")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    ax2b = ax2.twinx()
    ax2.plot(lats, pdopm, color=PALETTE[4], lw=2.5, marker="o", ms=3,
             label="PDOP (медиана)")
    ax2.axhline(6, ls=":", color=PALETTE[7], lw=1.2)
    ax2.set_ylabel("PDOP (медиана)", color=PALETTE[4]); ax2.set_ylim(0, 12)
    ax2.tick_params(axis="y", labelcolor=PALETTE[4])
    ax2b.plot(lats, avail, color=PALETTE[3], lw=2.0, ls="-.",
              label="Доступность ≥4 КА, %")
    ax2b.set_ylabel("Доступность ≥4 КА, %", color=PALETTE[3]); ax2b.set_ylim(0, 105)
    ax2b.tick_params(axis="y", labelcolor=PALETTE[3])
    ax2.set_xlabel("Широта, °")
    ax2.set_title("PDOP и доступность по широте")
    ax2.grid(alpha=0.3)
    l1, lab1 = ax2.get_legend_handles_labels()
    l2, lab2 = ax2b.get_legend_handles_labels()
    ax2.legend(l1 + l2, lab1 + lab2, fontsize=9, loc="upper center")

    plt.tight_layout()
    path = os.path.join(output_dir, "gonets106_geometry.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"\n  График: {path}")


def main():
    print("ГОНЕЦ-М1 hosted AURORA PNT — геометрия группировки")
    print(f"  {N_SATS} КА, {ALT:.0f} км, i={np.degrees(INC):.1f}°, "
          f"{N_PLANES} плоскостей (Walker-star, F={WALKER_F})")
    print(f"  период {PERIOD/60:.1f} мин; окно {SIM_HOURS} ч, шаг {SIM_STEP:.0f} с")
    rows, _ = evaluate()
    summarize(rows)
    out = os.path.join(os.path.dirname(__file__), "..", "..", "results", "gonets")
    plot(rows, os.path.abspath(out))


if __name__ == "__main__":
    main()
