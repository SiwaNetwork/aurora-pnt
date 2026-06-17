"""
Граф межспутниковой mesh-сети (ISL) и иерархия gossip-консенсуса §28.6.

Левая панель — физическая топология: 300 КА (узлы) на орбитах, внутриплоскостные
ISL-кольца и межплоскостные линии, цвет — регион. Правая панель — иерархия
SHIWA TIME-Space: Global → Region (группа плоскостей) → Shard (плоскость Walker,
≤64 узла) → КА.

Светлая тема (для читаемости в ГОСТ-документе и на экране).
"""

import sys, os, math
from typing import Dict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from aurora.pnt.constellation_anim import _build, _eci, A, N_PLANE, N_PER, N_SAT

N_REGION = 3                      # 3 региона × 5 плоскостей
PLANES_PER_REGION = N_PLANE // N_REGION
REGION_COLORS = ["#1a9850", "#2166ac", "#d6604d"]   # зелёный / синий / терракот
REGION_NAMES = ["Регион 1", "Регион 2", "Регион 3"]
BG = "#ffffff"
INK = "#1a2330"          # основной тёмный для текста/осей


def _positions():
    raan, u0 = _build()
    p = _eci(raan, u0, 0.0) / 1000.0  # км
    return p


def run_isl_mesh(output_dir: str, label: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    P = _positions()

    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor(BG)

    # ── Левая панель: 3D топология ──
    ax = fig.add_subplot(1, 2, 1, projection="3d"); ax.set_facecolor(BG)
    # Земля (светло-голубая)
    u = np.linspace(0, 2 * np.pi, 40); v = np.linspace(0, np.pi, 20)
    xe = 6371 * np.outer(np.cos(u), np.sin(v))
    ye = 6371 * np.outer(np.sin(u), np.sin(v))
    ze = 6371 * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xe, ye, ze, color="#cfe3fb", alpha=0.85, linewidth=0, shade=True)

    intra, inter = [], []
    for p in range(N_PLANE):
        idx = [p * N_PER + k for k in range(N_PER)]
        for k in range(N_PER):
            intra.append([P[idx[k]], P[idx[(k + 1) % N_PER]]])
        pn = (p + 1) % N_PLANE
        for k in range(0, N_PER, 2):
            inter.append([P[p * N_PER + k], P[pn * N_PER + k]])
    ax.add_collection3d(Line3DCollection(intra, colors="#7f8c9a", linewidths=0.8, alpha=0.55))
    ax.add_collection3d(Line3DCollection(inter, colors="#aab4c0", linewidths=0.5, alpha=0.35))

    for p in range(N_PLANE):
        idx = [p * N_PER + k for k in range(N_PER)]
        col = REGION_COLORS[p // PLANES_PER_REGION]
        ax.scatter(P[idx, 0], P[idx, 1], P[idx, 2], s=26, color=col,
                   edgecolors="white", linewidths=0.5, depthshade=False, zorder=5)

    # легенда регионов
    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor=REGION_COLORS[i],
                  markeredgecolor="white", markersize=10,
                  label=f"{REGION_NAMES[i]} (5 плоскостей)") for i in range(N_REGION)]
    ax.legend(handles=leg, loc="upper left", fontsize=9, framealpha=0.9,
              facecolor="#f5f7fa", edgecolor="#cfd6df")

    lim = A / 1000 * 1.02
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.set_axis_off()
    ax.view_init(elev=24, azim=-62)
    ax.set_title(f"ISL mesh: {N_SAT} КА, {N_PLANE} плоскостей\n"
                 f"внутриплоск. кольца + межплоск. линии (цвет — регион)",
                 color=INK, fontsize=12, fontweight="bold")

    # ── Правая панель: иерархия gossip §28.6 ──
    axh = fig.add_subplot(1, 2, 2); axh.set_facecolor(BG)
    axh.axis("off"); axh.set_xlim(0, 1); axh.set_ylim(0, 1)

    # Global
    axh.scatter([0.5], [0.92], s=900, color="#f0a500", edgecolors=INK,
                linewidths=1.2, zorder=5)
    axh.text(0.5, 0.92, "GLOBAL", ha="center", va="center", fontsize=11,
             fontweight="bold", color=INK, zorder=6)
    # Regions
    xr = np.linspace(0.18, 0.82, N_REGION)
    for ri, x in enumerate(xr):
        axh.plot([0.5, x], [0.885, 0.72], color="#9aa6b2", lw=1.4, zorder=1)
        axh.scatter([x], [0.70], s=620, color=REGION_COLORS[ri],
                    edgecolors="white", linewidths=1.2, zorder=5)
        axh.text(x, 0.70, f"{REGION_NAMES[ri]}\n(5 пл.)", ha="center", va="center",
                 fontsize=9, color="white", fontweight="bold", zorder=6)
        # Shards (плоскости) этого региона
        xs = np.linspace(x - 0.115, x + 0.115, PLANES_PER_REGION)
        for xx in xs:
            axh.plot([x, xx], [0.675, 0.50], color=REGION_COLORS[ri], lw=1.0, alpha=0.8)
            axh.scatter([xx], [0.48], s=150, color=REGION_COLORS[ri], alpha=0.95,
                        edgecolors="white", linewidths=0.6, zorder=4)
            # КА в шарде (точки)
            yk = np.linspace(0.38, 0.10, 6)
            axh.plot([xx, xx], [0.455, 0.08], color=REGION_COLORS[ri], lw=0.5, alpha=0.5)
            axh.scatter([xx] * len(yk), yk, s=14, color=REGION_COLORS[ri], alpha=0.75,
                        edgecolors="none", zorder=3)

    axh.text(0.5, 0.55, "Shard = плоскость Walker (20 КА ≤ 64 — лимит шарда)",
             ha="center", fontsize=10, color=INK, style="italic", fontweight="bold")
    axh.text(0.5, 0.42, "уровень шардов (15 плоскостей)", ha="center",
             fontsize=9, color="#5a6675")
    axh.text(0.5, 0.025,
             "КА (×20 на шард): CSAC-терминалы, дисциплина по якорям space-Rb; "
             "отбраковка выбросов по MAD, O(N·log N)",
             ha="center", fontsize=9.5, color=INK)
    axh.set_title("Иерархия gossip-консенсуса SHIWA TIME-Space (§28.6.4)\n"
                  "Shard → Region → Global",
                  color=INK, fontsize=12, fontweight="bold")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.04, wspace=0.04)
    path = os.path.join(output_dir, f"isl_mesh_{label}.png")
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)

    return {"image": path, "n_sat": N_SAT, "n_intra": len(intra),
            "n_inter": len(inter), "n_region": N_REGION, "shard_size": N_PER}


def print_isl_mesh_summary(label: str, r: Dict) -> None:
    print(f"\n  ISL mesh -- {label}")
    print(f"    Узлов {r['n_sat']}, внутриплоск. линий {r['n_intra']}, "
          f"межплоск. {r['n_inter']}")
    print(f"    Иерархия: {r['n_region']} региона × шарды по {r['shard_size']} КА")
    print(f"    Image: {r['image']}")
