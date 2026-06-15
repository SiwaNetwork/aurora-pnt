"""
Концептуальные иллюстрации системы АВРОРА.

Генерирует наглядные схемы:
  1. system_overview  — спутник, сигнал, потребители, наземный сегмент
  2. service_scenarios — 4 сценария применения
  3. leo_vs_meo       — сравнение LEO и MEO по уровню сигнала
  4. signal_flow      — поток данных от часов КА до пользователя
"""

import os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, Arc, Wedge, FancyBboxPatch
from matplotlib.lines import Line2D

# ── Палитра ───────────────────────────────────────────────────────────────────
C = {
    "aurora":   "#00b894",
    "aurora2":  "#00cec9",
    "sky":      "#0984e3",
    "earth":    "#2d6a4f",
    "earth2":   "#40916c",
    "atm":      "#74b9ff",
    "atm2":     "#dfe6e9",
    "signal":   "#00b894",
    "signal2":  "#6c5ce7",
    "gps":      "#e17055",
    "galileo":  "#0984e3",
    "user":     "#fdcb6e",
    "ground":   "#b2bec3",
    "mcs":      "#6c5ce7",
    "bg":       "#0d1117",
    "bg2":      "#161b22",
    "text":     "#f0f6fc",
    "subtext":  "#8b949e",
    "white":    "#ffffff",
    "accent":   "#f9826c",
}

SHADOW = [pe.withStroke(linewidth=4, foreground="#00000080")]
GLOW   = [pe.withStroke(linewidth=8, foreground="#00b89430")]


def _earth_arc(ax, cx, cy, r, color_top, color_bot, alpha=1.0):
    """Рисует закрашенный полукруг Земли."""
    theta = np.linspace(0, math.pi, 300)
    xs = cx + r * np.cos(theta)
    ys = cy - r * np.abs(np.sin(theta))
    ax.fill_between(xs, ys, cy - r * 1.05,
                    color=color_top, alpha=alpha, zorder=2)
    ax.fill_between(xs, cy - r * 1.05, cy - r * 1.2,
                    color=color_bot, alpha=alpha * 0.6, zorder=2)


def _satellite(ax, x, y, size=0.045, color=None, zorder=10):
    """Рисует пиктограмму спутника (корпус + 2 панели)."""
    c = color or C["aurora"]
    body = plt.Rectangle((x - size * 0.4, y - size * 0.35),
                          size * 0.8, size * 0.7,
                          color=c, zorder=zorder,
                          linewidth=1.5, edgecolor=C["white"])
    ax.add_patch(body)
    for dx in [-size * 1.1, size * 0.3]:
        panel = plt.Rectangle((x + dx, y - size * 0.1),
                               size * 0.8, size * 0.2,
                               color=C["sky"], zorder=zorder,
                               linewidth=1, edgecolor=C["white"])
        ax.add_patch(panel)
    ax.plot(x, y, "o", color=C["white"], ms=4, zorder=zorder + 1)


def _beam(ax, x1, y1, x2, y2, color, lw=1.8, alpha=0.7,
          style="-", zorder=5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>",
                                color=color, lw=lw,
                                mutation_scale=10),
                alpha=alpha, zorder=zorder)


def _icon_plane(ax, x, y, s=0.028, color=C["user"], zorder=8):
    xs = np.array([-1.0, 0.0, 1.0, 0.6, 0.0, -0.6]) * s
    ys = np.array([ 0.0, 0.5, 0.0,-0.4,-0.2,-0.4]) * s
    ax.fill(x + xs, y + ys, color=color, zorder=zorder, linewidth=0)
    ax.plot(x + xs, y + ys, color=C["white"], lw=0.5, zorder=zorder)


def _icon_car(ax, x, y, s=0.025, color=C["user"], zorder=8):
    body = plt.Rectangle((x - s, y - s * 0.4), s * 2, s * 0.8,
                          color=color, zorder=zorder)
    roof = plt.Rectangle((x - s * 0.55, y + s * 0.4), s * 1.1, s * 0.6,
                          color=color, zorder=zorder)
    ax.add_patch(body); ax.add_patch(roof)
    for wx in [x - s * 0.55, x + s * 0.2]:
        w = plt.Circle((wx, y - s * 0.5), s * 0.28,
                        color=C["bg"], zorder=zorder + 1)
        ax.add_patch(w)


def _icon_ship(ax, x, y, s=0.03, color=C["user"], zorder=8):
    xs = np.array([-1.0, -0.8, 0.8, 1.0]) * s
    ys = np.array([ 0.3, -0.5,-0.5, 0.3]) * s
    ax.fill(x + xs, y + ys, color=color, zorder=zorder)
    ax.plot([x, x], [y + s * 0.3, y + s * 1.3],
            color=C["white"], lw=1.5, zorder=zorder)


def _icon_geodesy(ax, x, y, s=0.025, color=C["aurora"], zorder=8):
    tripod_x = [x - s, x, x + s, x]
    tripod_y = [y - s, y + s, y - s, y - s]
    ax.plot(tripod_x, tripod_y, color=color, lw=2, zorder=zorder)
    ax.plot(x, y + s, "^", color=C["accent"], ms=9, zorder=zorder + 1)


def _label(ax, x, y, text, fontsize=9, color=C["text"], ha="center",
           va="bottom", bold=False, zorder=12):
    ax.text(x, y, text, fontsize=fontsize,
            color=color, ha=ha, va=va, zorder=zorder,
            fontweight="bold" if bold else "normal",
            path_effects=SHADOW)


# ──────────────────────────────────────────────────────────────────────────────
# РИСУНОК 1: Системная концепция
# ──────────────────────────────────────────────────────────────────────────────
def _plot_system_overview(output_dir, label):
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    # ── Звёздное небо ─────────────────────────────────────────────────────────
    rng = np.random.default_rng(7)
    sx = rng.uniform(0, 1, 200); sy = rng.uniform(0.3, 1, 200)
    sb = rng.uniform(0.2, 1.0, 200)
    ax.scatter(sx, sy, s=sb * 1.5, color="white", alpha=sb * 0.5, zorder=1)

    # ── Земля ─────────────────────────────────────────────────────────────────
    earth_cx, earth_cy, earth_r = 0.5, -0.38, 0.62
    theta = np.linspace(0, math.pi, 400)
    ex = earth_cx + earth_r * np.cos(theta)
    ey = earth_cy + earth_r * np.sin(theta)
    ax.fill_between(ex, ey, 0,
                    color=C["earth"], alpha=0.95, zorder=2)
    ax.fill_between(ex, ey, ey + 0.015,
                    color=C["earth2"], alpha=0.4, zorder=3)

    # Атмосфера
    atm_r = earth_r + 0.045
    ax_theta = np.linspace(0, math.pi, 400)
    ax2x = earth_cx + atm_r * np.cos(ax_theta)
    ax2y = earth_cy + atm_r * np.sin(ax_theta)
    ax.fill_between(ex, ey, ax2y, color=C["atm"], alpha=0.12, zorder=3)
    ax.plot(ax2x, ax2y, color=C["atm2"], lw=1.0, alpha=0.4,
            linestyle="--", zorder=3)

    # ── Орбиты (дуги) ─────────────────────────────────────────────────────────
    for r_o, col, lw, alph in [
        (earth_r + 0.19, C["aurora"], 1.8, 0.5),   # АВРОРА LEO 1000 км
        (earth_r + 0.48, C["gps"],    1.0, 0.2),   # GPS MEO (условно)
    ]:
        ot = np.linspace(0.12, math.pi - 0.12, 400)
        ox = earth_cx + r_o * np.cos(ot)
        oy = earth_cy + r_o * np.sin(ot)
        ax.plot(ox, oy, color=col, lw=lw, alpha=alph,
                linestyle="--", zorder=4)

    # Метки орбит
    ax.text(0.91, 0.56, "LEO 1000 км", color=C["aurora"],
            fontsize=8, alpha=0.8, rotation=-18, zorder=4)
    ax.text(0.94, 0.80, "MEO 20 200 км", color=C["gps"],
            fontsize=7, alpha=0.5, rotation=-12, zorder=4)

    # ── Спутники АВРОРА ───────────────────────────────────────────────────────
    sat_pos = [
        (0.500, 0.695),   # центральный (главный)
        (0.230, 0.570),   # левый
        (0.780, 0.610),   # правый
    ]
    for i, (sx2, sy2) in enumerate(sat_pos):
        sz = 0.052 if i == 0 else 0.036
        _satellite(ax, sx2, sy2, size=sz,
                   color=C["aurora"] if i == 0 else C["aurora2"],
                   zorder=10)

    # Надпись у главного спутника
    ax.text(0.500, 0.755, "АВРОРА", color=C["aurora"],
            fontsize=11, fontweight="bold", ha="center", zorder=12,
            path_effects=GLOW)
    ax.text(0.500, 0.733, "L1 1575,42 МГц · L5 1176,45 МГц",
            color=C["aurora2"], fontsize=7.5, ha="center", zorder=12)

    # ISL между спутниками
    for (x1, y1), (x2, y2) in [
        (sat_pos[0], sat_pos[1]),
        (sat_pos[0], sat_pos[2]),
    ]:
        ax.plot([x1, x2], [y1, y2],
                color=C["aurora2"], lw=1.2, alpha=0.5,
                linestyle=":", zorder=6)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + 0.012, "ISL Ka", color=C["aurora2"],
                fontsize=6.5, ha="center", alpha=0.7, zorder=6)

    # ── Потребители на поверхности ────────────────────────────────────────────
    users = [
        # (x,  y,     icon,           label_text,             sublabel,         beam_src)
        (0.14, 0.285, "plane",  "Авиация",        "LPV-200 · < 1 м",    sat_pos[1]),
        (0.35, 0.255, "car",    "Автомобиль",     "< 0,5 м",             sat_pos[0]),
        (0.56, 0.249, "geo",    "Геодезия",       "PPP-RTK · < 1 см",   sat_pos[0]),
        (0.80, 0.260, "ship",   "Морской",        "DGNSS · < 0,5 м",    sat_pos[2]),
    ]

    icon_fns = {
        "plane": _icon_plane,
        "car":   _icon_car,
        "ship":  _icon_ship,
        "geo":   _icon_geodesy,
    }

    for ux, uy, icon, lbl, sub, beam_src in users:
        icon_fns[icon](ax, ux, uy)
        ax.text(ux, uy - 0.055, lbl, color=C["text"],
                fontsize=8.5, ha="center", fontweight="bold",
                zorder=12, path_effects=SHADOW)
        ax.text(ux, uy - 0.075, sub, color=C["aurora"],
                fontsize=7, ha="center", zorder=12)

        # Сигнальный луч от спутника к потребителю (L1+L5)
        bsx, bsy = beam_src
        for col, off in [(C["aurora"], -0.007), (C["signal2"], 0.007)]:
            ax.annotate("",
                        xy=(ux + off, uy + 0.02),
                        xytext=(bsx + off, bsy - 0.025),
                        arrowprops=dict(
                            arrowstyle="-|>",
                            color=col, lw=1.5,
                            mutation_scale=8,
                            connectionstyle="arc3,rad=0.05"),
                        alpha=0.65, zorder=5)

    # Легенда лучей
    ax.plot([], [], color=C["aurora"],  lw=1.5, label="L1 (1575,42 МГц)")
    ax.plot([], [], color=C["signal2"], lw=1.5, label="L5 (1176,45 МГц)")
    ax.plot([], [], color=C["aurora2"], lw=1.2, ls=":",
            label="ISL Ka-диапазон (26 ГГц)")

    # ── Наземная станция МКС ──────────────────────────────────────────────────
    mcs_x, mcs_y = 0.67, 0.262
    ax.plot([mcs_x - 0.012, mcs_x, mcs_x + 0.012],
            [mcs_y - 0.02, mcs_y + 0.03, mcs_y - 0.02],
            color=C["mcs"], lw=2.5, zorder=8)
    ax.plot([mcs_x - 0.025, mcs_x + 0.025], [mcs_y - 0.02, mcs_y - 0.02],
            color=C["mcs"], lw=2.5, zorder=8)
    ax.plot(mcs_x - 0.015, mcs_y + 0.01,
            marker=(3, 0, -30), ms=10, color=C["mcs"], zorder=9)
    ax.text(mcs_x, mcs_y - 0.045, "МКС АВРОРА\n21 станция",
            color=C["mcs"], fontsize=7.5, ha="center",
            fontweight="bold", zorder=12, path_effects=SHADOW)

    # Аплинк МКС → спутник
    ax.annotate("",
                xy=(sat_pos[0][0] - 0.03, sat_pos[0][1] - 0.02),
                xytext=(mcs_x, mcs_y + 0.04),
                arrowprops=dict(arrowstyle="-|>",
                                color=C["mcs"], lw=1.3,
                                mutation_scale=8,
                                linestyle="dashed"),
                alpha=0.7, zorder=5)
    ax.text(0.60, 0.43, "ТМ/ТК\nS-диап.", color=C["mcs"],
            fontsize=6.5, ha="center", alpha=0.8, zorder=6)

    # ── RSN станция ───────────────────────────────────────────────────────────
    rsn_x, rsn_y = 0.455, 0.258
    ax.plot(rsn_x, rsn_y, "D", color=C["aurora"], ms=7, zorder=9,
            markeredgecolor=C["white"], markeredgewidth=0.8)
    ax.text(rsn_x, rsn_y - 0.04, "RSN\n(PPP-RTK)", color=C["aurora"],
            fontsize=6.5, ha="center", zorder=12)

    # RSN → ЦУП стрелка (оптика)
    ax.annotate("",
                xy=(mcs_x - 0.03, mcs_y),
                xytext=(rsn_x + 0.01, rsn_y),
                arrowprops=dict(arrowstyle="<->",
                                color=C["aurora"], lw=1.0,
                                mutation_scale=7,
                                linestyle="dotted"),
                alpha=0.6, zorder=5)

    # ── Заголовок и подпись ───────────────────────────────────────────────────
    ax.text(0.5, 0.965,
            "АВРОРА — Системная концепция",
            color=C["white"], fontsize=16, fontweight="bold",
            ha="center", va="top", zorder=15,
            path_effects=[pe.withStroke(linewidth=6,
                                        foreground=C["aurora"] + "40")])
    ax.text(0.5, 0.940,
            "Walker Delta 300/15/1 · h = 1000 км · i = 75° · "
            "L1+L5 · PPP-RTK · ISL · LPT",
            color=C["subtext"], fontsize=8.5, ha="center",
            va="top", zorder=15)

    leg = ax.legend(loc="upper left", fontsize=8,
                    framealpha=0.25, edgecolor=C["aurora"],
                    labelcolor=C["text"],
                    facecolor=C["bg2"])
    plt.setp(leg.get_texts(), color=C["text"])

    plt.tight_layout(pad=0.3)
    path = os.path.join(output_dir, f"concept_system_overview_{label}.png")
    fig.savefig(path, dpi=180, facecolor=C["bg"],
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# РИСУНОК 2: Сценарии применения (4 панели)
# ──────────────────────────────────────────────────────────────────────────────
def _plot_service_scenarios(output_dir, label):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor(C["bg"])
    plt.subplots_adjust(hspace=0.08, wspace=0.05,
                        left=0.02, right=0.98,
                        top=0.93, bottom=0.04)

    scenarios = [
        {
            "title":   "Авиация — захода на посадку (LPV-200)",
            "acc":     "< 1 м вертикально",
            "ttff":    "Горячий старт: 1,5 с",
            "service": "АВРОРА PPP + СДКМ",
            "color":   "#0984e3",
            "icon":    "plane",
            "details": [
                "CAT-I ILS эквивалент",
                "TIR < 10⁻⁷/ч",
                "TTPR < 6 с",
                "N_vis ≥ 36 (PDOP < 1,8)",
                "Доп.: SBAS (СДКМ)",
            ],
            "sat_h":   0.73,
        },
        {
            "title":   "Высокоточная геодезия (PPP-RTK)",
            "acc":     "0,5–1 см (H-68%)",
            "ttff":    "Сходимость: 5 с",
            "service": "АВРОРА PPP-RTK + RSN",
            "color":   C["aurora"],
            "icon":    "geo",
            "details": [
                "RSN шаг 300 км",
                "Задержка E2E: 70 мс",
                "SSR поправки L1+L5",
                "Точность: 0,5 + 0,03·d см",
                "Совм. с ГЛОНАСС",
            ],
            "sat_h":   0.73,
        },
        {
            "title":   "Автомобиль / БПЛА (Lane-level)",
            "acc":     "< 0,5 м (H-95%)",
            "ttff":    "Тёплый старт: 5 с",
            "service": "АВРОРА PPP / PPP-RTK",
            "color":   "#fdcb6e",
            "icon":    "car",
            "details": [
                "Полосная точность",
                "Устойчивость в туннеле",
                "CRPA-антенна (MP ↓)",
                "TESLA MAC аутентиф.",
                "32 канала, BW 18 Гц",
            ],
            "sat_h":   0.73,
        },
        {
            "title":   "Синхронизация / LPT-сервис",
            "acc":     "< 5 нс (UTC(SU))",
            "ttff":    "Время удержания: 72 ч",
            "service": "АВРОРА LPT · ISL-сетка",
            "color":   C["mcs"],
            "icon":    "tower",
            "details": [
                "Cs-стандарт: 0,01 нс/6ч",
                "6 ISL-хопов: 2,58 нс",
                "OCXO без ISL: 7 нс/6ч",
                "Протокол: PTP IEEE 1588",
                "Замена GPS-синхрон.",
            ],
            "sat_h":   0.73,
        },
    ]

    for idx, (ax, sc) in enumerate(zip(axes.flat, scenarios)):
        ax.set_facecolor(C["bg2"])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")
        col = sc["color"]

        # Рамка
        rect = FancyBboxPatch((0.01, 0.01), 0.98, 0.98,
                              boxstyle="round,pad=0.01",
                              linewidth=2, edgecolor=col,
                              facecolor="none", zorder=10)
        ax.add_patch(rect)

        # Цветная полоска заголовка
        ax.add_patch(plt.Rectangle((0.01, 0.87), 0.98, 0.12,
                                   color=col, alpha=0.25, zorder=1))
        ax.text(0.5, 0.932, sc["title"], color=C["white"],
                fontsize=10.5, fontweight="bold", ha="center",
                va="center", zorder=11)

        # ── Миниатюра сцены ───────────────────────────────────────────────
        scene_cx = 0.5
        # Земля
        tet = np.linspace(0, math.pi, 200)
        ex = scene_cx + 0.42 * np.cos(tet)
        ey = 0.24 + 0.22 * np.sin(tet) * 0.5
        ax.fill_between(ex, 0.0, ey, color=C["earth"], alpha=0.7, zorder=2)

        # Атмосфера
        aex = scene_cx + 0.44 * np.cos(tet)
        aey = 0.24 + 0.235 * np.sin(tet) * 0.5
        ax.fill_between(ex, ey, aey, color=C["atm"], alpha=0.12, zorder=2)

        # Орбитальная дуга АВРОРА
        sat_y = sc["sat_h"]
        orb_r = 0.36
        ot = np.linspace(0.25, math.pi - 0.25, 200)
        ox = scene_cx + orb_r * np.cos(ot)
        oy2 = sat_y - 0.06 + orb_r * 0.18 * np.sin(ot)
        ax.plot(ox, oy2, color=col, lw=1.2, alpha=0.3,
                linestyle="--", zorder=3)

        # Спутник
        sat_x = scene_cx
        _satellite(ax, sat_x, sat_y, size=0.045, color=col, zorder=9)
        ax.text(sat_x, sat_y + 0.07, "АВРОРА", color=col,
                fontsize=7.5, ha="center", fontweight="bold",
                zorder=10, path_effects=SHADOW)

        # Потребитель
        user_x = scene_cx + 0.18
        user_y = 0.32
        if sc["icon"] == "plane":
            _icon_plane(ax, user_x, user_y + 0.06, s=0.042, color=col)
        elif sc["icon"] == "geo":
            _icon_geodesy(ax, user_x - 0.04, user_y, s=0.035, color=col)
        elif sc["icon"] == "car":
            _icon_car(ax, user_x, user_y, s=0.038, color=col)
        elif sc["icon"] == "tower":
            ax.plot([user_x, user_x], [user_y - 0.04, user_y + 0.06],
                    color=col, lw=3, zorder=8)
            ax.plot([user_x - 0.04, user_x + 0.04],
                    [user_y + 0.04, user_y + 0.04],
                    color=col, lw=2, zorder=8)
            ax.plot(user_x + 0.015, user_y + 0.055, "o",
                    ms=7, color=col, zorder=9)
            ax.text(user_x, user_y - 0.07, "Базовая\nстанция",
                    color=C["text"], fontsize=6.5, ha="center")

        # Сигнальные лучи
        for off, beam_col in [(-0.012, col), (0.012, C["signal2"])]:
            ax.annotate("",
                        xy=(user_x + off, user_y + 0.05),
                        xytext=(sat_x + off, sat_y - 0.03),
                        arrowprops=dict(arrowstyle="-|>",
                                        color=beam_col, lw=1.6,
                                        mutation_scale=8),
                        alpha=0.7, zorder=5)

        # Зона покрытия (конус)
        cone_w = 0.18
        cone_xs = [sat_x - cone_w, sat_x, sat_x + cone_w,
                   sat_x + 0.02, sat_x - 0.02]
        cone_ys = [0.26, sat_y - 0.03, 0.26,
                   sat_y - 0.03, sat_y - 0.03]
        ax.fill(cone_xs, cone_ys, color=col, alpha=0.07, zorder=3)

        # ── Метрики ───────────────────────────────────────────────────────
        metrics_y = 0.59
        ax.text(0.5, metrics_y + 0.04, sc["acc"],
                color=col, fontsize=12, ha="center",
                fontweight="bold", zorder=10,
                path_effects=[pe.withStroke(linewidth=4,
                                            foreground=C["bg2"])])
        ax.text(0.5, metrics_y - 0.01, sc["ttff"],
                color=C["subtext"], fontsize=8.5, ha="center", zorder=10)
        ax.text(0.5, metrics_y - 0.06, sc["service"],
                color=C["aurora2"], fontsize=8, ha="center",
                fontweight="bold", zorder=10)

        # Разделитель
        ax.plot([0.06, 0.94], [metrics_y - 0.11, metrics_y - 0.11],
                color=col, lw=0.8, alpha=0.4)

        # ── Детальный список ──────────────────────────────────────────────
        for di, detail in enumerate(sc["details"]):
            dy = metrics_y - 0.17 - di * 0.07
            ax.text(0.12, dy, "▸", color=col,
                    fontsize=8, va="center", zorder=10)
            ax.text(0.18, dy, detail, color=C["text"],
                    fontsize=8, va="center", zorder=10)

    fig.text(0.5, 0.975, "АВРОРА — Сценарии применения",
             color=C["white"], fontsize=15, fontweight="bold",
             ha="center", va="top",
             path_effects=[pe.withStroke(linewidth=6,
                                         foreground=C["aurora"] + "40")])

    path = os.path.join(output_dir, f"concept_service_scenarios_{label}.png")
    fig.savefig(path, dpi=180, facecolor=C["bg"],
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# РИСУНОК 3: LEO vs MEO — визуальное сравнение
# ──────────────────────────────────────────────────────────────────────────────
def _plot_leo_vs_meo(output_dir, label):
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    rng = np.random.default_rng(42)
    sx = rng.uniform(0, 1, 180); sy = rng.uniform(0.22, 1, 180)
    sb = rng.uniform(0.2, 1, 180)
    ax.scatter(sx, sy, s=sb * 1.5, color="white", alpha=sb * 0.4, zorder=1)

    # ── Земля ─────────────────────────────────────────────────────────────────
    for cx, half in [(0.27, "left"), (0.73, "right")]:
        tet = np.linspace(0, math.pi, 300)
        ex = cx + 0.26 * np.cos(tet)
        ey_top = 0.195 + 0.09 * np.sin(tet)
        ax.fill_between(ex, 0, ey_top,
                        color=C["earth"], alpha=0.9, zorder=2)
        aex2 = cx + 0.275 * np.cos(tet)
        aey2 = 0.195 + 0.096 * np.sin(tet)
        ax.fill_between(ex, ey_top, aey2,
                        color=C["atm"], alpha=0.18, zorder=2)

    # Разделитель
    ax.plot([0.5, 0.5], [0.02, 0.97],
            color=C["subtext"], lw=1.5, alpha=0.3,
            linestyle="--", zorder=3)
    ax.text(0.5, 0.985, "vs", color=C["subtext"],
            fontsize=14, ha="center", va="top", alpha=0.6)

    systems = [
        {
            "cx": 0.27, "sat_y": 0.54,
            "label": "АВРОРА LEO",
            "sublabel": "h = 1000 км",
            "color": C["aurora"],
            "sat_size": 0.05,
            "cone_w": 0.22,
            "signal_lw": 3.5,
            "power_label": "EIRP: −107 дБВт",
            "power_color": C["aurora"],
            "advantages": [
                ("Уровень сигнала", "+23 дБ vs GPS", C["aurora"]),
                ("FSPL (1000 км)", "−159 дБ", C["aurora"]),
                ("Время прол.", "~11 мин", C["aurora"]),
                ("Доплер", "±38,6 кГц / 38,6 Гц/с", "#fdcb6e"),
                ("TTFF (PPP-RTK)", "5 с", C["aurora"]),
                ("Задержка E2E", "3,3 мс (↑LOS)", C["aurora"]),
                ("Геометрия", "PDOP < 1,8 (36 спутн.)", C["aurora"]),
            ],
        },
        {
            "cx": 0.73, "sat_y": 0.88,
            "label": "GPS MEO",
            "sublabel": "h = 20 200 км",
            "color": C["gps"],
            "sat_size": 0.04,
            "cone_w": 0.24,
            "signal_lw": 1.0,
            "power_label": "EIRP: −158 дБВт",
            "power_color": C["gps"],
            "advantages": [
                ("Уровень сигнала", "−158 дБВт (ref.)", C["gps"]),
                ("FSPL (20 200 км)", "−182 дБ", "#e17055"),
                ("Время прол.", "~4 ч", C["gps"]),
                ("Доплер", "±4,9 кГц / ~0,9 Гц/с", C["gps"]),
                ("TTFF (PPP)", "1 500 с (25 мин)", "#e17055"),
                ("Задержка E2E", "67 мс (сигн. LOS)", "#e17055"),
                ("Геометрия", "PDOP ~1,5 (8–12 спутн.)", C["gps"]),
            ],
        },
    ]

    for sc in systems:
        cx = sc["cx"]
        sat_y = sc["sat_y"]
        col = sc["color"]

        _satellite(ax, cx, sat_y, size=sc["sat_size"], color=col, zorder=10)

        # Конус покрытия
        cw = sc["cone_w"]
        cone_xs = [cx - cw, cx, cx + cw,
                   cx + 0.01, cx - 0.01]
        cone_ys = [0.22, sat_y - 0.04, 0.22,
                   sat_y - 0.04, sat_y - 0.04]
        ax.fill(cone_xs, cone_ys, color=col, alpha=0.10, zorder=3)
        ax.plot([cx - cw, cx, cx + cw],
                [0.22, sat_y - 0.04, 0.22],
                color=col, lw=1.0, alpha=0.5, linestyle=":", zorder=4)

        # Пучок сигнала (визуализация мощности через толщину)
        for ox in np.linspace(-0.012, 0.012, 5):
            alph = 0.8 - abs(ox) * 30
            ax.plot([cx + ox, cx + ox * 3],
                    [sat_y - 0.04, 0.26],
                    color=col, lw=sc["signal_lw"],
                    alpha=max(0, alph) * 0.55, zorder=5)

        # Метка спутника
        ax.text(cx, sat_y + 0.065, sc["label"],
                color=col, fontsize=13, fontweight="bold",
                ha="center", zorder=12,
                path_effects=[pe.withStroke(linewidth=5,
                                            foreground=C["bg"])])
        ax.text(cx, sat_y + 0.04, sc["sublabel"],
                color=C["subtext"], fontsize=9, ha="center", zorder=12)
        ax.text(cx, sat_y - 0.075, sc["power_label"],
                color=sc["power_color"], fontsize=9, ha="center",
                fontweight="bold", zorder=12)

        # Потребитель (пешеход / смартфон)
        user_x, user_y = cx + 0.03, 0.235
        ax.plot(user_x, user_y + 0.015, "o",
                ms=9, color=col, zorder=9,
                markeredgecolor=C["white"], markeredgewidth=1)
        ax.plot([user_x, user_x], [user_y - 0.02, user_y + 0.015],
                color=C["white"], lw=1.5, zorder=8)
        ax.plot([user_x - 0.015, user_x + 0.015],
                [user_y - 0.005, user_y - 0.005],
                color=C["white"], lw=1.5, zorder=8)

        # Таблица характеристик
        tbl_y = 0.76
        for i, (param, val, vc) in enumerate(sc["advantages"]):
            dy = tbl_y - i * 0.072
            ax.text(cx - 0.18, dy, param + ":",
                    color=C["subtext"], fontsize=8.5,
                    ha="left", va="center", zorder=10)
            ax.text(cx + 0.18, dy, val,
                    color=vc, fontsize=8.5, fontweight="bold",
                    ha="right", va="center", zorder=10)
            ax.plot([cx - 0.18, cx + 0.18], [dy - 0.025, dy - 0.025],
                    color=col, lw=0.4, alpha=0.3)

    # ── Сравнительная аннотация в центре ──────────────────────────────────────
    for cy2, txt, col2 in [
        (0.58, "+23 дБ", C["aurora"]),
        (0.52, "× 200 по", C["white"]),
        (0.47, "подавлению", C["white"]),
    ]:
        ax.text(0.5, cy2, txt, color=col2,
                fontsize=13 if col2 == C["aurora"] else 10,
                ha="center", fontweight="bold" if col2 == C["aurora"] else "normal",
                zorder=12, path_effects=SHADOW)

    ax.annotate("", xy=(0.42, 0.55), xytext=(0.36, 0.55),
                arrowprops=dict(arrowstyle="<-",
                                color=C["aurora"], lw=2.0,
                                mutation_scale=12), zorder=10)
    ax.annotate("", xy=(0.58, 0.55), xytext=(0.64, 0.55),
                arrowprops=dict(arrowstyle="<-",
                                color=C["gps"], lw=2.0,
                                mutation_scale=12), zorder=10)

    # Заголовок
    ax.text(0.5, 0.972,
            "АВРОРА LEO vs GPS MEO — Сравнение уровня сигнала и характеристик",
            color=C["white"], fontsize=14, fontweight="bold",
            ha="center", va="top", zorder=15,
            path_effects=[pe.withStroke(linewidth=6,
                                        foreground=C["aurora"] + "40")])

    path = os.path.join(output_dir, f"concept_leo_vs_meo_{label}.png")
    fig.savefig(path, dpi=180, facecolor=C["bg"],
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# РИСУНОК 4: Поток данных от часов КА до навигационного решения
# ──────────────────────────────────────────────────────────────────────────────
def _plot_signal_flow(output_dir, label):
    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    stages = [
        {
            "x": 0.06, "label": "Cs/Rb\nстандарт", "sub": "< 0,01 нс",
            "color": C["mcs"], "icon": "clock",
        },
        {
            "x": 0.21, "label": "Навигационный\nпередатчик", "sub": "L1+L5\n5+3 Вт",
            "color": C["aurora"], "icon": "sat",
        },
        {
            "x": 0.38, "label": "L1/L5\nсигнал", "sub": "1000 км\nFSPL −159 дБ",
            "color": C["signal2"], "icon": "wave",
        },
        {
            "x": 0.55, "label": "Приёмник\nAURORA", "sub": "C/N₀ ≥ 45 дБ·Гц\n238 каналов",
            "color": C["user"], "icon": "recv",
        },
        {
            "x": 0.72, "label": "Навигационный\nфильтр", "sub": "Kalman / PPP\nL1+L5 IF",
            "color": C["aurora2"], "icon": "filter",
        },
        {
            "x": 0.89, "label": "PNT решение", "sub": "< 0,5 м\n< 5 нс UTC",
            "color": C["aurora"], "icon": "fix",
        },
    ]

    box_y = 0.52
    box_h = 0.26
    box_w = 0.12

    for i, st in enumerate(stages):
        x = st["x"]
        col = st["color"]

        # Блок
        rect = FancyBboxPatch((x - box_w / 2, box_y - box_h / 2),
                              box_w, box_h,
                              boxstyle="round,pad=0.01",
                              linewidth=2, edgecolor=col,
                              facecolor=C["bg2"], zorder=5)
        ax.add_patch(rect)

        # Цветная подложка верха
        ax.add_patch(FancyBboxPatch(
            (x - box_w / 2, box_y + box_h / 2 - 0.08),
            box_w, 0.08,
            boxstyle="round,pad=0.005",
            linewidth=0, facecolor=col, alpha=0.3, zorder=6))

        # Иконка
        icon_y = box_y + box_h / 2 - 0.04
        icons_map = {
            "clock":  "Cs",
            "sat":    "TX",
            "wave":   "~L1",
            "recv":   "RX",
            "filter": "KF",
            "fix":    "PNT",
        }
        ax.text(x, icon_y, icons_map.get(st["icon"], "•"),
                color=C["white"], fontsize=14, ha="center",
                va="center", zorder=7)

        # Текст
        ax.text(x, box_y + 0.015, st["label"],
                color=C["text"], fontsize=8.5, ha="center",
                va="center", fontweight="bold", zorder=7,
                linespacing=1.4)
        ax.text(x, box_y - 0.09, st["sub"],
                color=col, fontsize=7.5, ha="center",
                va="center", zorder=7, linespacing=1.4)

        # Стрелка к следующему
        if i < len(stages) - 1:
            nx = stages[i + 1]["x"]
            ncol = stages[i + 1]["color"]
            # Градиент: цвет текущего → следующего
            ax.annotate("",
                        xy=(nx - box_w / 2 - 0.005, box_y),
                        xytext=(x + box_w / 2 + 0.005, box_y),
                        arrowprops=dict(
                            arrowstyle="-|>",
                            color=col, lw=2.0,
                            mutation_scale=12),
                        zorder=8)

        # Задержка/параметр под стрелкой
        delays = [
            "ADEV\n< 10⁻¹³/с",
            "ANAV\n5000 бит",
            "ΔT = 3,3 мс\nTESLA MAC",
            "PLL BW\n1,4–20 Гц",
            "IF комб.\nσ_iono < 2 мм",
            "",
        ]
        if delays[i]:
            ax.text(x, box_y - 0.20, delays[i],
                    color=C["subtext"], fontsize=7, ha="center",
                    va="top", zorder=7, linespacing=1.3)

    # Нижняя полоса — цепочка синхронизации
    chain_y = 0.14
    ax.text(0.5, chain_y, "Цепочка синхронизации UTC(SU) → КА → Пользователь",
            color=C["mcs"], fontsize=9, ha="center",
            fontweight="bold", zorder=10)

    chain_items = [
        (0.06, "UTC(SU)\n± 5 нс"),
        (0.20, "Cs КА\n± 0,01 нс/6ч"),
        (0.36, "ISL хоп\n± 0,43 нс"),
        (0.52, "× 6 ISL\n± 2,58 нс"),
        (0.68, "Троп./Ион.\n± 0,5 см"),
        (0.84, "Пользов.\n< 5 нс"),
    ]
    for cx2, ctxt in chain_items:
        ax.text(cx2, chain_y - 0.06, ctxt,
                color=C["subtext"], fontsize=7.5, ha="center",
                va="top", linespacing=1.3)
    for i in range(len(chain_items) - 1):
        x1 = chain_items[i][0] + 0.06
        x2 = chain_items[i + 1][0] - 0.06
        ax.annotate("", xy=(x2, chain_y - 0.055),
                    xytext=(x1, chain_y - 0.055),
                    arrowprops=dict(arrowstyle="-|>",
                                    color=C["mcs"], lw=1.0,
                                    mutation_scale=7),
                    alpha=0.6, zorder=7)

    ax.text(0.5, 0.955,
            "АВРОРА — Поток сигнала и данных: от атомных часов до навигационного решения",
            color=C["white"], fontsize=13, fontweight="bold",
            ha="center", va="top", zorder=15,
            path_effects=[pe.withStroke(linewidth=6,
                                        foreground=C["aurora"] + "40")])

    path = os.path.join(output_dir, f"concept_signal_flow_{label}.png")
    fig.savefig(path, dpi=180, facecolor=C["bg"],
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────────────────────────────────────
def run_system_concept(output_dir: str, label: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    print(f"  Generating system concept figures -> {output_dir}")
    _plot_system_overview(output_dir, label)
    _plot_service_scenarios(output_dir, label)
    _plot_leo_vs_meo(output_dir, label)
    _plot_signal_flow(output_dir, label)
    return {
        "figures": [
            f"concept_system_overview_{label}.png",
            f"concept_service_scenarios_{label}.png",
            f"concept_leo_vs_meo_{label}.png",
            f"concept_signal_flow_{label}.png",
        ]
    }


if __name__ == "__main__":
    run_system_concept("results/system_concept", "phase4")
