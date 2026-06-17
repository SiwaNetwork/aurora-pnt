"""
3D-рендер «облёта планеты»: Земля + группировка Walker 300/15 + детальная
3D-модель космического аппарата АВРОРА.

Назначение — «парадная» (hero) визуализация системы: реалистичная Земля с
дневным/ночным терминатором и атмосферной дымкой, наклонённые орбитальные
плоскости группировки и крупный план КА с подписанными элементами
(солнечные панели, навигационная антенна L1/L5/Сервис Б, ISL Ka, звёздный
датчик). Чистый numpy + matplotlib (без внешних текстур/библиотек).

Выходы:
  orbit_render_<label>.png       — композитная сцена (глобус + КА крупно)
  orbit_satellite_<label>.png    — отдельный крупный план КА
"""

import sys, os
from typing import Dict, List
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

R_E   = 6371.0
ALT   = 1000.0
A     = R_E + ALT
INCL  = np.radians(75.0)
N_PLANE = 15
N_PER   = 20

# Направление на Солнце (для терминатора) и направление освещения КА
SUN = np.array([1.0, 0.35, 0.55]); SUN = SUN / np.linalg.norm(SUN)


# ─────────────────────────────────────────────────────────────────────────────
#  Земля
# ─────────────────────────────────────────────────────────────────────────────
def _earth(ax, r=R_E):
    u = np.linspace(0, 2 * np.pi, 90)
    v = np.linspace(0, np.pi, 60)
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones_like(u), np.cos(v))

    # Освещённость (ламберт по направлению на Солнце) → день/ночь
    nx, ny, nz = x / r, y / r, z / r
    lit = nx * SUN[0] + ny * SUN[1] + nz * SUN[2]
    shade = np.clip(0.5 * lit + 0.5, 0.06, 1.0)

    # Псевдо-материки: спокойный шум по сферическим гармоникам (детерминир.)
    lon = np.arctan2(ny, nx); lat = np.arcsin(np.clip(nz, -1, 1))
    land = (np.sin(3 * lon + 1.1) * np.cos(2 * lat - 0.4)
            + 0.6 * np.sin(5 * lon - 2.0) * np.cos(4 * lat + 0.7)
            + 0.4 * np.cos(2 * lon + 0.5) * np.sin(3 * lat))
    is_land = land > 0.45

    ocean = np.array([0.10, 0.28, 0.52])      # глубокий синий
    landc = np.array([0.20, 0.42, 0.24])      # суша (зелёно-коричн.)
    base = np.where(is_land[..., None], landc, ocean)
    rgb = np.clip(base * shade[..., None], 0, 1)
    facecolors = np.concatenate([rgb, np.ones(shade.shape + (1,))], axis=-1)

    ax.plot_surface(x, y, z, rcount=60, ccount=90, facecolors=facecolors,
                    linewidth=0, antialiased=True, shade=False, zorder=1)

    # Атмосферная дымка — полупрозрачная оболочка чуть больше радиуса
    halo = 1.025 * r
    xs = halo * np.outer(np.cos(u), np.sin(v))
    ys = halo * np.outer(np.sin(u), np.sin(v))
    zs = halo * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, rcount=40, ccount=60, color="#5fa8ff",
                    alpha=0.06, linewidth=0, antialiased=True, shade=False, zorder=0)

    # Тонкая сетка параллелей/меридианов
    for lat0 in np.radians([-60, -30, 0, 30, 60]):
        t = np.linspace(0, 2 * np.pi, 120)
        rr = r * np.cos(lat0)
        ax.plot(rr * np.cos(t), rr * np.sin(t), r * np.sin(lat0) * np.ones_like(t),
                color="white", lw=0.4, alpha=0.18, zorder=2)
    for lon0 in np.radians(np.arange(0, 360, 30)):
        t = np.linspace(-np.pi / 2, np.pi / 2, 80)
        ax.plot(r * np.cos(t) * np.cos(lon0), r * np.cos(t) * np.sin(lon0),
                r * np.sin(t), color="white", lw=0.4, alpha=0.12, zorder=2)


def _orbits(ax, highlight=(0, 5, 10)):
    """Орбитальные плоскости Walker (наклон 75°), КА точками."""
    t = np.linspace(0, 2 * np.pi, 200)
    for p in range(N_PLANE):
        raan = 2 * np.pi * p / N_PLANE
        # окружность в плоскости орбиты, затем наклон + поворот RAAN
        xo, yo, zo = A * np.cos(t), A * np.sin(t), np.zeros_like(t)
        # наклон вокруг оси X
        y1 = yo * np.cos(INCL); z1 = yo * np.sin(INCL); x1 = xo
        # поворот RAAN вокруг Z
        x2 = x1 * np.cos(raan) - y1 * np.sin(raan)
        y2 = x1 * np.sin(raan) + y1 * np.cos(raan)
        z2 = z1
        hot = p in highlight
        ax.plot(x2, y2, z2, color="#00d6a4" if hot else "#7fd8c4",
                lw=1.6 if hot else 0.5, alpha=0.95 if hot else 0.30, zorder=3)
        # спутники
        k = np.linspace(0, 2 * np.pi, N_PER, endpoint=False) + raan
        xs, ys = A * np.cos(k), A * np.sin(k)
        ys2 = ys * np.cos(INCL); zs2 = ys * np.sin(INCL); xs2 = xs
        X = xs2 * np.cos(raan) - ys2 * np.sin(raan)
        Y = xs2 * np.sin(raan) + ys2 * np.cos(raan)
        ax.scatter(X, Y, zs2, s=14 if hot else 6,
                   color="#00ffc8" if hot else "#bfe9dd",
                   edgecolors="none", alpha=0.95 if hot else 0.5, zorder=4)


# ─────────────────────────────────────────────────────────────────────────────
#  3D-модель КА из граней (Poly3DCollection) с ламберт-затенением
# ─────────────────────────────────────────────────────────────────────────────
def _box(c, sx, sy, sz):
    cx, cy, cz = c
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    p = [(cx - hx, cy - hy, cz - hz), (cx + hx, cy - hy, cz - hz),
         (cx + hx, cy + hy, cz - hz), (cx - hx, cy + hy, cz - hz),
         (cx - hx, cy - hy, cz + hz), (cx + hx, cy - hy, cz + hz),
         (cx + hx, cy + hy, cz + hz), (cx - hx, cy + hy, cz + hz)]
    faces = [[p[0], p[1], p[2], p[3]], [p[4], p[5], p[6], p[7]],
             [p[0], p[1], p[5], p[4]], [p[2], p[3], p[7], p[6]],
             [p[1], p[2], p[6], p[5]], [p[0], p[3], p[7], p[4]]]
    return faces


def _shade(faces, base_rgb, light=np.array([0.6, 0.5, 0.7])):
    light = light / np.linalg.norm(light)
    base = np.array(base_rgb)
    cols = []
    for f in faces:
        f = np.array(f)
        n = np.cross(f[1] - f[0], f[2] - f[0])
        nn = np.linalg.norm(n)
        b = 0.45 if nn == 0 else 0.45 + 0.55 * abs(float(np.dot(n / nn, light)))
        cols.append(np.clip(base * b, 0, 1))
    return cols


def _satellite(ax, scale=1.0):
    """Детальная 3D-модель КА АВРОРА. +X — надир (на Землю), панели по ±Y."""
    s = scale
    polys, polycols, edges = [], [], []

    # Корпус (золотистый MLI)
    body = _box((0, 0, 0), 1.6 * s, 1.0 * s, 1.0 * s)
    polys += body; polycols += _shade(body, (0.78, 0.62, 0.18))

    # Надирная навигационная панель (тёмная, с антеннами L1/L5/Сервис Б)
    nad = _box((0.85 * s, 0, 0), 0.12 * s, 0.95 * s, 0.95 * s)
    polys += nad; polycols += _shade(nad, (0.12, 0.13, 0.16))

    # Солнечные панели — два крыла по ±Y (тёмно-синие, ячейки)
    cells_lines = []
    for sgn in (+1, -1):
        y0 = sgn * 0.5 * s
        y1 = sgn * 3.3 * s
        wing = [[(-1.3 * s, y0, 0.0), (1.3 * s, y0, 0.0),
                 (1.3 * s, y1, 0.0), (-1.3 * s, y1, 0.0)]]
        polys += wing; polycols += _shade(wing, (0.09, 0.13, 0.32))
        # сетка фотоэлементов
        for fx in np.linspace(-1.3 * s, 1.3 * s, 5):
            cells_lines.append([(fx, y0, 0.001), (fx, y1, 0.001)])
        for fy in np.linspace(y0, y1, 7):
            cells_lines.append([(-1.3 * s, fy, 0.001), (1.3 * s, fy, 0.001)])

    # ISL-антенны Ka (два малых горна по ±Z)
    for sgn in (+1, -1):
        horn = _box((0, 0, sgn * 0.62 * s), 0.4 * s, 0.4 * s, 0.25 * s)
        polys += horn; polycols += _shade(horn, (0.55, 0.57, 0.60))

    # Звёздный датчик (малый блок на −X)
    st = _box((-0.85 * s, 0.25 * s, 0.25 * s), 0.2 * s, 0.25 * s, 0.25 * s)
    polys += st; polycols += _shade(st, (0.20, 0.22, 0.25))

    pc = Poly3DCollection(polys, facecolors=polycols, edgecolors="#2d3436",
                          linewidths=0.3, zorder=10)
    ax.add_collection3d(pc)
    if cells_lines:
        ax.add_collection3d(Line3DCollection(cells_lines, colors="#3a5a9a",
                                             linewidths=0.4, alpha=0.7, zorder=11))

    # Надирная антенна — диск на конце нав-панели
    th = np.linspace(0, 2 * np.pi, 30)
    disk = [list(zip(0.95 * s * np.ones_like(th), 0.32 * s * np.cos(th),
                     0.32 * s * np.sin(th)))]
    ax.add_collection3d(Poly3DCollection(disk, facecolors="#e74c3c",
                                         edgecolors="#7a1f15", linewidths=0.5,
                                         alpha=0.95, zorder=12))


def _set_equal(ax, r):
    ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_zlim(-r, r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.set_axis_off()


def _label_satellite(ax, scale):
    """Выноски с лидер-линиями: точка на КА → подпись в стороне (без наложений)."""
    s = scale
    # (точка на аппарате, точка подписи, текст, цвет)
    items = [
        ((1.0 * s, 0, 0),            (3.1 * s, 1.0 * s, -0.6 * s),
         "Нав. ПН (надир):\nСервис А L1/L5 + Сервис Б", "#e74c3c"),
        ((0, 2.4 * s, 0),            (0, 3.3 * s, 1.7 * s),
         "Солнечные панели\nGaAs, 3 м²", "#5b8def"),
        ((0, 0, 0.62 * s),          (-1.1 * s, -1.9 * s, 2.7 * s),
         "ISL Ka (×2)", "#aeb6bd"),
        ((-0.85 * s, 0.25 * s, 0.25 * s), (-2.7 * s, 0.3 * s, 1.9 * s),
         "Звёздный датчик", "#9fb3c8"),
        ((0, -0.5 * s, -0.5 * s),    (-1.4 * s, -2.6 * s, -1.9 * s),
         "Корпус ≈140 кг\n(золотой MLI)", "#e0b94a"),
    ]
    for (x0, y0, z0), (x1, y1, z1), txt, col in items:
        ax.plot([x0, x1], [y0, y1], [z0, z1], color=col, lw=0.7, alpha=0.7, zorder=18)
        ax.scatter([x0], [y0], [z0], s=10, color=col, zorder=19)
        ax.text(x1, y1, z1, txt, color=col, fontsize=8.5, zorder=20,
                fontweight="bold", ha="center", va="center")


def run_orbit_render(output_dir: str, label: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    # ── Сцена 1: композит (глобус + группировка + КА-инсет) ──
    fig = plt.figure(figsize=(16, 8.2))
    fig.patch.set_facecolor("#05080f")

    axE = fig.add_subplot(1, 2, 1, projection="3d")
    axE.set_facecolor("#05080f")
    _earth(axE); _orbits(axE)
    _set_equal(axE, A * 1.05)
    axE.view_init(elev=22, azim=-58)
    axE.set_title("АВРОРА — группировка Walker 300/15 на 1000 км",
                  color="white", fontsize=13, pad=2)

    axS = fig.add_subplot(1, 2, 2, projection="3d")
    axS.set_facecolor("#05080f")
    _satellite(axS, scale=1.0); _label_satellite(axS, 1.0)
    _set_equal(axS, 3.3)
    axS.view_init(elev=24, azim=-114)
    axS.set_title("Космический аппарат АВРОРА (≈140 кг)",
                  color="white", fontsize=13, pad=2)

    fig.text(0.5, 0.04, "Орбита 1000 км, наклонение 75°, 15 плоскостей × 20 КА; "
             "надир — навигационная ПН (Сервис А L1/L5 + защищённый Сервис Б), "
             "межспутниковые линии Ka.",
             ha="center", color="#9fb3c8", fontsize=9)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.08, wspace=0.02)
    p1 = os.path.join(output_dir, f"orbit_render_{label}.png")
    fig.savefig(p1, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)

    # ── Сцена 2: КА крупным планом (отдельный файл) ──
    fig2 = plt.figure(figsize=(9, 8))
    fig2.patch.set_facecolor("#05080f")
    ax2 = fig2.add_subplot(111, projection="3d")
    ax2.set_facecolor("#05080f")
    _satellite(ax2, scale=1.0); _label_satellite(ax2, 1.0)
    _set_equal(ax2, 3.2)
    ax2.view_init(elev=24, azim=-114)
    ax2.set_title("Космический аппарат АВРОРА — конфигурация",
                  color="white", fontsize=13)
    p2 = os.path.join(output_dir, f"orbit_satellite_{label}.png")
    fig2.savefig(p2, dpi=160, facecolor=fig2.get_facecolor())
    plt.close(fig2)

    return {"scene": p1, "satellite": p2,
            "n_sat": N_PLANE * N_PER, "alt_km": ALT, "incl_deg": 75.0}


def print_orbit_render_summary(label: str, results: Dict) -> None:
    print(f"\n  Orbit render -- {label}")
    print(f"    Группировка: {results['n_sat']} КА, h={results['alt_km']:.0f} км, "
          f"i={results['incl_deg']:.0f}°")
    print(f"    Сцена:    {results['scene']}")
    print(f"    Аппарат:  {results['satellite']}")
