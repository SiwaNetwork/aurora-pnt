"""
Антенно-сигнальная визуализация АВРОРА:
  1. 3D-диаграммы направленности: Сервис А (изо-flux, ~3,5 дБи) и
     Сервис Б (фазированная решётка, ~8 дБи) — earth-coverage лепестки к надиру.
  2. Созвездие сигнала L1C (данные+пилот, QPSK-подобное) при заданном C/N₀.
  3. Waterfall-спектрограмма канала L1 (TMBOC) с перемежающейся помехой —
     иллюстрация помехозащиты §15.
"""

import sys, os
from typing import Dict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

F_CHIP = 1.023e6
THETA_MAX = 62.0   # угол на лимб Земли с 1000 км (asin(R_E/(R_E+h)))


def _gain_dBi(theta_deg, kind):
    """Изо-flux earth-coverage: усиление растёт к лимбу (компенсация трассы)."""
    th = np.asarray(theta_deg, float)
    if kind == "A":
        g = 2.0 + 3.0 * (np.clip(th, 0, THETA_MAX) / THETA_MAX) ** 2
        roll = np.where(th > THETA_MAX, -1.2 * (th - THETA_MAX), 0.0)
    else:  # B — выше усиление, резче спад
        g = 5.0 + 3.0 * (np.clip(th, 0, THETA_MAX) / THETA_MAX) ** 1.6
        roll = np.where(th > THETA_MAX, -2.2 * (th - THETA_MAX), 0.0)
    return g + roll


def _plot_pattern(ax, kind, title, color):
    th = np.linspace(0, 90, 60)
    ph = np.linspace(0, 2 * np.pi, 80)
    TH, PH = np.meshgrid(th, ph)
    G = _gain_dBi(TH, kind)
    r = np.clip(G - (-6.0), 0, None)        # радиус ∝ (G − floor)
    thr = np.radians(TH)
    # лепесток направлен к надиру (−Z)
    x = r * np.sin(thr) * np.cos(PH)
    y = r * np.sin(thr) * np.sin(PH)
    z = -r * np.cos(thr)
    norm = plt.Normalize(G.min(), G.max())
    fc = plt.cm.plasma(norm(G))
    ax.plot_surface(x, y, z, facecolors=fc, rcount=60, ccount=80,
                    linewidth=0, antialiased=True, shade=False)
    m = max(np.abs([x.min(), x.max(), y.min(), y.max(), z.min(), z.max()]))
    ax.set_xlim(-m, m); ax.set_ylim(-m, m); ax.set_zlim(-m, 1)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.set_axis_off()
    ax.view_init(elev=12, azim=-60)
    ax.set_title(title, color="white", fontsize=11)


def _constellation(ax, cn0_dbhz=45.0):
    n = 3000
    # символьная Es/N0 при 1000 симв/с эквивалентной полосе (иллюстративно)
    esn0_db = cn0_dbhz - 30.0
    sigma = np.sqrt(1.0 / (2.0 * 10 ** (esn0_db / 10.0)))
    di = np.random.choice([-1, 1], n); dq = np.random.choice([-1, 1], n)
    I = di / np.sqrt(2) + np.random.randn(n) * sigma
    Q = dq / np.sqrt(2) + np.random.randn(n) * sigma
    quad = (di > 0).astype(int) * 2 + (dq > 0).astype(int)
    cols = ["#e17055", "#fdcb6e", "#0984e3", "#00b894"]
    ax.scatter(I, Q, s=4, c=[cols[q] for q in quad], alpha=0.4)
    for sx in (-1, 1):
        for sy in (-1, 1):
            ax.scatter([sx / np.sqrt(2)], [sy / np.sqrt(2)], s=110, marker="+",
                       color="white", linewidths=1.6, zorder=5)
    ax.axhline(0, color="#5a6577", lw=0.6); ax.axvline(0, color="#5a6577", lw=0.6)
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6); ax.set_aspect("equal")
    ax.set_facecolor("#0a1020")
    ax.tick_params(colors="#9fb3c8", labelsize=8)
    ax.set_xlabel("I (данные)", color="#9fb3c8", fontsize=9)
    ax.set_ylabel("Q (пилот)", color="#9fb3c8", fontsize=9)
    ax.set_title(f"Созвездие L1C (данные+пилот), C/N₀={cn0_dbhz:.0f} дБ·Гц",
                 color="white", fontsize=11)


def _waterfall(ax):
    f = np.linspace(-12e6, 12e6, 240)
    sinc2 = lambda x: np.sinc(x) ** 2
    boc = lambda fs: 0.5 * (sinc2((f - fs) / F_CHIP) + sinc2((f + fs) / F_CHIP))
    g = (29 / 33) * boc(1 * F_CHIP) + (4 / 33) * boc(6 * F_CHIP)
    base = 10 * np.log10(np.maximum(g / g.max(), 1e-4))   # дБ
    nt = 160
    img = np.tile(base, (nt, 1)) + np.random.randn(nt, len(f)) * 1.2 - 22
    # перемежающаяся CW-помеха на +4 МГц в окне времени
    jbin = np.argmin(np.abs(f - 4e6))
    for ti in range(nt):
        if 55 <= ti <= 105:
            img[ti, jbin - 1:jbin + 2] = 6.0   # яркий стрик помехи
    im = ax.imshow(img, extent=[f[0] / 1e6, f[-1] / 1e6, nt, 0],
                   aspect="auto", cmap="turbo", vmin=-30, vmax=6)
    ax.set_xlabel("Отстройка от L1 (1575,42 МГц), МГц", color="#9fb3c8", fontsize=9)
    ax.set_ylabel("Время →", color="#9fb3c8", fontsize=9)
    ax.tick_params(colors="#9fb3c8", labelsize=8)
    ax.set_title("Waterfall L1 (TMBOC) + перемежающаяся помеха (§15)",
                 color="white", fontsize=11)
    ax.annotate("CW-помеха", (4, 80), color="white", fontsize=8,
                xytext=(7, 50), arrowprops=dict(arrowstyle="->", color="white"))
    return im


def run_antenna_signal(output_dir: str, label: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)

    fig = plt.figure(figsize=(15, 9))
    fig.patch.set_facecolor("#070b14")

    axA = fig.add_subplot(2, 2, 1, projection="3d"); axA.set_facecolor("#070b14")
    _plot_pattern(axA, "A", "Сервис А — изо-flux, ~3,5 дБи (RNSS)", "#0984e3")
    axB = fig.add_subplot(2, 2, 2, projection="3d"); axB.set_facecolor("#070b14")
    _plot_pattern(axB, "B", "Сервис Б — фаз. решётка, ~8 дБи (защищ.)", "#e17055")

    axC = fig.add_subplot(2, 2, 3); _constellation(axC)
    axD = fig.add_subplot(2, 2, 4); im = _waterfall(axD)
    cb = fig.colorbar(im, ax=axD, shrink=0.8, pad=0.02)
    cb.set_label("ПСД, дБ", color="#cfd8e3")
    cb.ax.yaxis.set_tick_params(color="#cfd8e3")
    plt.setp(plt.getp(cb.ax, "yticklabels"), color="#cfd8e3")

    fig.suptitle("АВРОРА — антенны (Сервис А/Б), созвездие сигнала и спектрограмма",
                 color="white", fontsize=13, y=0.98)
    fig.subplots_adjust(left=0.03, right=0.96, top=0.91, bottom=0.06,
                        wspace=0.14, hspace=0.22)
    path = os.path.join(output_dir, f"antenna_signal_{label}.png")
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"image": path,
            "gain_A_nadir": float(_gain_dBi(0, "A")), "gain_A_limb": float(_gain_dBi(THETA_MAX, "A")),
            "gain_B_nadir": float(_gain_dBi(0, "B")), "gain_B_limb": float(_gain_dBi(THETA_MAX, "B"))}


def print_antenna_signal_summary(label: str, r: Dict) -> None:
    print(f"\n  Antenna/signal -- {label}")
    print(f"    Сервис А: надир {r['gain_A_nadir']:.1f} → лимб {r['gain_A_limb']:.1f} дБи (изо-flux)")
    print(f"    Сервис Б: надир {r['gain_B_nadir']:.1f} → лимб {r['gain_B_limb']:.1f} дБи")
    print(f"    Image: {r['image']}")
