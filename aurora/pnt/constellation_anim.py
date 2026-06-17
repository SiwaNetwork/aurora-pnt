"""
Анимация облёта: движение группировки Walker 300/15, бегущие зоны видимости
и N_vis(t) для наземного пользователя. Выход — GIF (PillowWriter, без ffmpeg).

Сцена:
  - верхняя панель: карта мира (процедурная суша), подспутниковые точки 300 КА,
    зоны видимости (footprint) спутников, видимых из Москвы, маркер пользователя;
  - нижняя панель: N_vis(t) с бегущим курсором времени.
"""

import sys, os, math
from typing import Dict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

GM = 3.986004418e14
R_E = 6371e3
ALT = 1000e3
A = R_E + ALT
INCL = math.radians(75.0)
OMEGA_E = 7.2921159e-5
N_PLANE, N_PER = 15, 20
N_SAT = N_PLANE * N_PER
MASK = math.radians(10.0)
N_MEAN = math.sqrt(GM / A**3)
T_ORB = 2 * math.pi / N_MEAN

USER = {"name": "Москва", "lat": 55.75, "lon": 37.62}


def _find_ffmpeg():
    """Путь к ffmpeg: системный, иначе бинарь из imageio-ffmpeg. None — нет."""
    import shutil
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _build():
    raan = np.zeros(N_SAT); u0 = np.zeros(N_SAT)
    for p in range(N_PLANE):
        for k in range(N_PER):
            i = p * N_PER + k
            raan[i] = 2 * math.pi * p / N_PLANE
            u0[i] = 2 * math.pi * k / N_PER + 2 * math.pi * p / N_SAT
    return raan, u0


def _eci(raan, u0, t):
    u = u0 + N_MEAN * t
    x0, y0 = A * np.cos(u), A * np.sin(u)
    # наклон вокруг X
    x1 = x0; y1 = y0 * math.cos(INCL); z1 = y0 * math.sin(INCL)
    # RAAN вокруг Z
    x = x1 * np.cos(raan) - y1 * np.sin(raan)
    y = x1 * np.sin(raan) + y1 * np.cos(raan)
    z = z1
    return np.stack([x, y, z], axis=1)


def _ecef(eci, t):
    th = OMEGA_E * t
    c, s = math.cos(-th), math.sin(-th)
    x = eci[:, 0] * c - eci[:, 1] * s
    y = eci[:, 0] * s + eci[:, 1] * c
    return np.stack([x, y, eci[:, 2]], axis=1)


def _latlon(ecef):
    r = np.linalg.norm(ecef, axis=1)
    lat = np.degrees(np.arcsin(ecef[:, 2] / r))
    lon = np.degrees(np.arctan2(ecef[:, 1], ecef[:, 0]))
    return lat, lon


def _user_ecef():
    la, lo = math.radians(USER["lat"]), math.radians(USER["lon"])
    return R_E * np.array([math.cos(la) * math.cos(lo),
                           math.cos(la) * math.sin(lo), math.sin(la)])


def _visible(ecef, ue):
    los = ecef - ue
    up = ue / np.linalg.norm(ue)
    elev = np.arcsin(np.clip((los @ up) / np.linalg.norm(los, axis=1), -1, 1))
    return elev > MASK


def _landmask(nlat=180, nlon=360):
    la = np.radians(np.linspace(90, -90, nlat))[:, None]
    lo = np.radians(np.linspace(-180, 180, nlon))[None, :]
    land = (np.sin(3 * lo + 1.1) * np.cos(2 * la - 0.4)
            + 0.6 * np.sin(5 * lo - 2.0) * np.cos(4 * la + 0.7)
            + 0.4 * np.cos(2 * lo + 0.5) * np.sin(3 * la))
    return land > 0.45


def run_constellation_anim(output_dir: str, label: str,
                           n_frames: int = 72, fps: int = 12) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    raan, u0 = _build()
    ue = _user_ecef()

    times = np.linspace(0, 1.6 * T_ORB, n_frames)
    nvis_curve = []
    for t in times:
        vis = _visible(_ecef(_eci(raan, u0, t), t), ue)
        nvis_curve.append(int(vis.sum()))
    nvis_curve = np.array(nvis_curve)

    fig = plt.figure(figsize=(13, 8.5))
    fig.patch.set_facecolor("#05080f")
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.28)
    axm = fig.add_subplot(gs[0]); axn = fig.add_subplot(gs[1])

    land = _landmask()
    axm.imshow(land, extent=[-180, 180, -90, 90], origin="upper",
               cmap=matplotlib.colors.ListedColormap(["#0c1c34", "#1b3a26"]),
               aspect="auto", zorder=0)
    for g in range(-60, 61, 30):
        axm.axhline(g, color="white", lw=0.3, alpha=0.12)
    for g in range(-150, 151, 30):
        axm.axvline(g, color="white", lw=0.3, alpha=0.12)
    axm.set_xlim(-180, 180); axm.set_ylim(-90, 90)
    axm.set_facecolor("#05080f")
    axm.tick_params(colors="#9fb3c8", labelsize=8)
    axm.set_xlabel("Долгота, °", color="#9fb3c8", fontsize=9)
    axm.set_ylabel("Широта, °", color="#9fb3c8", fontsize=9)

    sc_all = axm.scatter([], [], s=10, color="#5fd0b4", alpha=0.55, zorder=3)
    sc_vis = axm.scatter([], [], s=42, color="#00ffc8", edgecolors="white",
                         linewidths=0.5, zorder=5)
    foot = [axm.plot([], [], color="#00ffc8", lw=0.8, alpha=0.35, zorder=4)[0]
            for _ in range(40)]
    axm.scatter([USER["lon"]], [USER["lat"]], s=120, marker="*",
                color="#ffd166", edgecolors="#b8860b", linewidths=0.8, zorder=6)
    axm.annotate(USER["name"], (USER["lon"], USER["lat"]),
                 textcoords="offset points", xytext=(8, 6),
                 color="#ffd166", fontsize=9, zorder=6)
    title = axm.set_title("", color="white", fontsize=12)

    axn.plot(times / 60, nvis_curve, color="#5fd0b4", lw=1.6)
    axn.axhline(4, ls="--", color="#e17055", lw=1.0, label="N_min=4")
    axn.set_facecolor("#0a1020"); axn.set_xlim(0, times[-1] / 60)
    axn.set_ylim(0, max(nvis_curve) + 3)
    axn.tick_params(colors="#9fb3c8", labelsize=8)
    axn.set_xlabel("Время, мин", color="#9fb3c8", fontsize=9)
    axn.set_ylabel(f"N_vis ({USER['name']})", color="#9fb3c8", fontsize=9)
    axn.grid(alpha=0.15)
    cursor = axn.axvline(0, color="#ffd166", lw=1.4)
    ntext = axn.text(0.02, 0.85, "", transform=axn.transAxes,
                     color="#ffd166", fontsize=10, fontweight="bold")

    def update(fi):
        t = times[fi]
        ecef = _ecef(_eci(raan, u0, t), t)
        lat, lon = _latlon(ecef)
        vis = _visible(ecef, ue)
        sc_all.set_offsets(np.column_stack([lon, lat]))
        sc_vis.set_offsets(np.column_stack([lon[vis], lat[vis]]))
        # footprints для видимых
        vi = np.where(vis)[0]
        ang = np.linspace(0, 2 * np.pi, 40)
        rho = 19.0  # ~центральный угол зоны (упрощённо, град)
        for j, art in enumerate(foot):
            if j < len(vi):
                clat, clon = lat[vi[j]], lon[vi[j]]
                fl = clat + rho * np.sin(ang)
                fo = clon + rho * np.cos(ang) / max(0.3, math.cos(math.radians(clat)))
                art.set_data(fo, fl)
            else:
                art.set_data([], [])
        title.set_text(f"АВРОРА — облёт группировки 300 КА  |  t = {t/60:5.1f} мин  "
                       f"|  N_vis({USER['name']}) = {int(vis.sum())}")
        cursor.set_xdata([t / 60, t / 60])
        ntext.set_text(f"N_vis = {int(vis.sum())}")
        return [sc_all, sc_vis, title, cursor, ntext] + foot

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)

    ffmpeg = _find_ffmpeg()
    fmt = "gif"
    if ffmpeg:
        matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg
        path = os.path.join(output_dir, f"constellation_flyby_{label}.mp4")
        # yuv420p требует чётных размеров кадра → scale-фильтр подравнивает
        writer = FFMpegWriter(fps=fps, bitrate=3000, codec="libx264",
                              extra_args=["-pix_fmt", "yuv420p",
                                          "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"])
        anim.save(path, writer=writer, dpi=120,
                  savefig_kwargs={"facecolor": fig.get_facecolor()})
        fmt = "mp4"
    else:  # фолбэк — GIF (PillowWriter, без ffmpeg)
        path = os.path.join(output_dir, f"constellation_flyby_{label}.gif")
        anim.save(path, writer=PillowWriter(fps=fps), dpi=90,
                  savefig_kwargs={"facecolor": fig.get_facecolor()})
    plt.close(fig)
    return {"path": path, "format": fmt, "n_frames": n_frames, "fps": fps,
            "nvis_min": int(nvis_curve.min()), "nvis_max": int(nvis_curve.max()),
            "nvis_mean": float(nvis_curve.mean())}


def print_constellation_anim_summary(label: str, r: Dict) -> None:
    print(f"\n  Constellation flyby animation -- {label}")
    print(f"    Кадров: {r['n_frames']} @ {r['fps']} fps  (формат {r['format'].upper()})")
    print(f"    N_vis(Москва): мин {r['nvis_min']} / средн {r['nvis_mean']:.1f} / макс {r['nvis_max']}")
    print(f"    Файл: {r['path']}")
