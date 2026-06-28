"""
Сквозная точность PVT (CEP/верт. 95%) hosted-варианта на ГОНЕЦ-М1 (1500 км).
Часть исследования АВРОРА-ГОНЕЦ-001 (НЕ часть основного ТП).

Методика — как в §45 ТП: точность = геометрия (HDOP/VDOP) × ошибка дальности (UERE).
Геометрия берётся из той же модели, что §3 исследования (reuse gonets_hosted.py);
UERE — из §37.3/§45.2 ТП с поправкой на 1500 км (шум приёмника ×1,5 из-за −3,5 дБ
бюджета линии, §4.1). Коэффициенты CEP сверены с §45.2:
    CEP50 = 0,83·HDOP·UERE,  CEP95 = 1,73·HDOP·UERE,  V95 = 1,96·VDOP·UERE.

Запуск:  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python aurora/pnt/gonets_pvt.py
"""

import sys, os, csv, importlib.util
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PALETTE = ["#e17055", "#fdcb6e", "#0984e3", "#00b894",
           "#6c5ce7", "#74b9ff", "#dfe6e9", "#2d3436"]

# Reuse геометрии из gonets_hosted (минуя пакет aurora.pnt с зависимостью sgp4)
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("gh", os.path.join(_here, "gonets_hosted.py"))
gh = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(gh)

# ── UERE по классам сервиса (м). Компоненты — §37.3 ТП; на 1500 км изменён только
#    шум приёмника (×1,5 из-за −3,5 дБ, §4.1). PPP/PPP-RTK — корректурные режимы
#    (§45.2), малочувствительны к высоте. ──────────────────────────────────────
def uere_code_dual_1500():
    comp = dict(eph_rad=0.45, eph_alongcross=0.15, clk=0.20, iono=0.05,
                tropo=0.35, multipath=0.20, rx_noise=0.15 * 1.50)  # шум ×1,5
    return float(np.sqrt(sum(v**2 for v in comp.values()))), comp

UERE_CLASSES = {
    "Код dual-freq":  uere_code_dual_1500()[0],   # ≈ 0,69 м
    "PPP dual-freq":  0.233,                       # §45.2 (корректурный, ~как АВРОРА)
    "PPP-RTK":        0.131,                       # §45.2
}

# Коэффициенты перевода в CEP/верт. (сверены с §45.2 ТП)
K_CEP50, K_CEP95, K_V95 = 0.83, 1.73, 1.96

# Сетка РФ (как в §3)
RF_LAT = np.arange(41.0, 82.0 + 1e-6, 5.0)
RF_LON = np.arange(19.0, 180.0 + 1e-6, 10.0)
SIM_HOURS, SIM_STEP, MASK = 6.0, 120.0, 5.0


def dop_over_rf():
    """HDOP/VDOP/PDOP по сетке РФ и времени → массивы значений."""
    times = np.arange(0.0, SIM_HOURS * 3600.0, SIM_STEP)
    raan, m0, _ = gh.build_constellation()
    sats = gh.sat_positions_ecef(raan, m0, times)        # (T,N,3)
    H_, V_, P_ = [], [], []
    for la in RF_LAT:
        for lo in RF_LON:
            pos, B = gh.enu_basis(np.radians(la), np.radians(lo))
            enu = (sats - pos[None, None, :]) @ B.T        # (T,N,3)
            rng = np.linalg.norm(enu, axis=-1)
            elev = np.degrees(np.arcsin(enu[..., 2] / rng))
            vis = elev >= MASK
            for t in range(len(times)):
                idx = np.where(vis[t])[0]
                if len(idx) >= 4:
                    e = enu[t, idx] / rng[t, idx][:, None]
                    G = np.column_stack([-e[:, 0], -e[:, 1], -e[:, 2],
                                         np.ones(len(idx))])
                    try:
                        Q = np.linalg.inv(G.T @ G)
                        H_.append(np.sqrt(Q[0, 0] + Q[1, 1]))
                        V_.append(np.sqrt(Q[2, 2]))
                        P_.append(np.sqrt(Q[0, 0] + Q[1, 1] + Q[2, 2]))
                    except np.linalg.LinAlgError:
                        pass
    return np.array(H_), np.array(V_), np.array(P_)


def accuracy(uere, hdop, vdop):
    return dict(cep50=K_CEP50 * hdop * uere,
                cep95=K_CEP95 * hdop * uere,
                v95=K_V95 * vdop * uere)


def main():
    print("Сквозная точность PVT — ГОНЕЦ-М1 hosted (1500 км), над РФ\n")
    H, V, P = dop_over_rf()
    hmed, hp95 = np.median(H), np.percentile(H, 95)
    vmed, vp95 = np.median(V), np.percentile(V, 95)
    pmed, pp95 = np.median(P), np.percentile(P, 95)
    print(f"Геометрия над РФ: HDOP мед {hmed:.2f}/p95 {hp95:.2f}; "
          f"VDOP мед {vmed:.2f}/p95 {vp95:.2f}; PDOP мед {pmed:.2f}/p95 {pp95:.2f}")
    u_code, comp = uere_code_dual_1500()
    print(f"UERE код dual-freq (1500 км) = {u_code:.3f} м "
          f"(шум приёмника {comp['rx_noise']:.3f} vs 0,15 у 1000 км; "
          f"остальное — §37.3). АВРОРА(1000 км) = 0,70 м.\n")

    print(f"{'Класс сервиса':16s} {'UERE,м':>7s} {'CEP50,м':>8s} "
          f"{'CEP95,м':>8s} {'Верт95,м':>9s}  (геометрия: медиана / p95 над РФ)")
    rows = []
    for name, u in UERE_CLASSES.items():
        a_med = accuracy(u, hmed, vmed)
        a_p95 = accuracy(u, hp95, vp95)
        print(f"{name:16s} {u:7.3f} {a_med['cep50']:8.3f} {a_med['cep95']:8.3f} "
              f"{a_med['v95']:9.3f}   (p95: CEP95 {a_p95['cep95']:.2f} м)")
        rows.append((name, u, a_med['cep50'], a_med['cep95'], a_med['v95'],
                     a_p95['cep95']))

    # Комбинированный режим с ГЛОНАСС: совместная геометрия снижает HDOP (грубо ~30%)
    print(f"\nКомбинир. с ГЛОНАСС (геометрия лучше ~30%): "
          f"код dual CEP95 ≈ {K_CEP95*hmed*0.7*u_code:.2f} м над РФ.")

    out = os.path.abspath(os.path.join(_here, "..", "..", "results", "gonets"))
    os.makedirs(out, exist_ok=True)
    _plot(rows, hmed, vmed, out)
    _csv(rows, hmed, hp95, vmed, vp95, pmed, pp95, out)
    print(f"\nГрафик/CSV: {out}")


def _plot(rows, hmed, vmed, out):
    names = [r[0] for r in rows]
    cep95 = [r[3] for r in rows]
    # эталон АВРОРЫ (§45.2): CEP95 для PPP-RTK/PPP/код
    aurora = {"Код dual-freq": 1.0, "PPP dual-freq": 0.448, "PPP-RTK": 0.253}
    aur = [aurora[n] for n in names]
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(11, 6))
    b1 = ax.bar(x - w/2, cep95, w, color=PALETTE[2], edgecolor="white",
                label="ГОНЕЦ-106 hosted (над РФ)")
    b2 = ax.bar(x + w/2, aur, w, color=PALETTE[3], edgecolor="white",
                label="АВРОРА 300 КА (§45.2, эталон)")
    for b in list(b1) + list(b2):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                f"{b.get_height():.2f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("CEP95, м (горизонт.)")
    ax.set_title("Точность PVT над РФ: ГОНЕЦ-106 (1500 км) vs АВРОРА-300\n"
                 f"геометрия HDOP={hmed:.2f}, VDOP={vmed:.2f}; UERE — §37.3/§45.2")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out, "gonets_pvt_accuracy.png"), dpi=150)
    plt.close(fig)


def _csv(rows, hmed, hp95, vmed, vp95, pmed, pp95, out):
    with open(os.path.join(out, "gonets_pvt.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Геометрия над РФ", "медиана", "p95"])
        w.writerow(["HDOP", f"{hmed:.2f}", f"{hp95:.2f}"])
        w.writerow(["VDOP", f"{vmed:.2f}", f"{vp95:.2f}"])
        w.writerow(["PDOP", f"{pmed:.2f}", f"{pp95:.2f}"])
        w.writerow([])
        w.writerow(["Класс сервиса", "UERE,м", "CEP50,м", "CEP95,м",
                    "Верт95,м", "CEP95(геом.p95),м"])
        for r in rows:
            w.writerow([r[0], f"{r[1]:.3f}", f"{r[2]:.3f}", f"{r[3]:.3f}",
                        f"{r[4]:.3f}", f"{r[5]:.3f}"])


if __name__ == "__main__":
    main()
