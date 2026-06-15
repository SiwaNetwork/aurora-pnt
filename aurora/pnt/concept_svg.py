"""
Генерация SVG-иллюстраций системы АВРОРА.

Производит чистые векторные схемы, пригодные для публикации:
  1. system_overview  — спутник, сигнал, сегменты, потребители
  2. service_scenarios — 4 сценария применения
  3. leo_vs_meo        — сравнение LEO и MEO
  4. signal_flow       — поток данных от часов до PNT-решения

Выходной формат: .svg (открывается в браузере/Inkscape) + .png (через cairosvg, если установлен).
"""

import os, math, random

# ── Попытка конвертации в PNG ─────────────────────────────────────────────────
try:
    import cairosvg
    HAS_CAIRO = True
except ImportError:
    HAS_CAIRO = False

try:
    from PIL import Image
    import io
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ── Утилиты ───────────────────────────────────────────────────────────────────
def _stars(n=160, w=1400, h=800, seed=42):
    return ""  # светлая тема (ГОСТ-печать): звёздный фон убран
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        x = rng.uniform(0, w)
        y = rng.uniform(0, h * 0.78)
        r = rng.uniform(0.4, 1.8)
        op = rng.uniform(0.3, 0.95)
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="white" opacity="{op:.2f}"/>')
    return "\n".join(out)


def _arc_path(cx, cy, r, a1_deg, a2_deg):
    """SVG arc path from a1 to a2 (degrees, CCW from right)."""
    a1 = math.radians(a1_deg)
    a2 = math.radians(a2_deg)
    x1, y1 = cx + r * math.cos(a1), cy - r * math.sin(a1)
    x2, y2 = cx + r * math.cos(a2), cy - r * math.sin(a2)
    large = 1 if abs(a2_deg - a1_deg) > 180 else 0
    return f"M {x1:.1f} {y1:.1f} A {r} {r} 0 {large} 0 {x2:.1f} {y2:.1f}"


def _sat_icon(cx, cy, size=28, color="#00b894", angle=0, glow=True):
    """SVG спутник: корпус + 2 солнечные панели + антенна."""
    bw, bh = size * 0.55, size * 0.7
    pw, ph = size * 0.9, size * 0.22
    gf = 'filter="url(#glow_green)"' if glow else ""
    g = f'<g transform="translate({cx},{cy}) rotate({angle})" {gf}>'
    # Корпус
    g += f'<rect x="{-bw/2:.1f}" y="{-bh/2:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="3" fill="{color}" stroke="white" stroke-width="1.2"/>'
    # Панели
    for sx in [-(bw/2 + pw + 2), bw/2 + 2]:
        g += f'<rect x="{sx:.1f}" y="{-ph/2:.1f}" width="{pw:.1f}" height="{ph:.1f}" rx="2" fill="#0984e3" stroke="white" stroke-width="0.8"/>'
        # Сетка панели
        for k in range(1, 3):
            lx = sx + pw * k / 3
            g += f'<line x1="{lx:.1f}" y1="{-ph/2:.1f}" x2="{lx:.1f}" y2="{ph/2:.1f}" stroke="white" stroke-width="0.4" opacity="0.5"/>'
    # Антенна (тарелка)
    g += f'<line x1="0" y1="{-bh/2:.1f}" x2="0" y2="{-bh/2-size*0.35:.1f}" stroke="white" stroke-width="1.5"/>'
    g += f'<ellipse cx="0" cy="{-bh/2-size*0.38:.1f}" rx="{size*0.22:.1f}" ry="{size*0.1:.1f}" fill="none" stroke="{color}" stroke-width="1.5"/>'
    # Огонёк
    g += f'<circle cx="0" cy="0" r="3" fill="white" opacity="0.9"/>'
    g += "</g>"
    return g


def _beam_cone(sx, sy, ux, uy, w_top=6, color="#00b894", opacity=0.18, id_suffix=""):
    """Заполненный конус луча от спутника к потребителю."""
    dx, dy = ux - sx, uy - sy
    length = math.hypot(dx, dy)
    if length == 0:
        return ""
    # Перпендикуляр
    px, py = -dy / length, dx / length
    hw = w_top / 2
    # Низ конуса — широкий (у потребителя)
    w_bot = 30
    # Вершина
    pts_top = f"{sx + px*hw:.1f},{sy + py*hw:.1f} {sx - px*hw:.1f},{sy - py*hw:.1f}"
    pts_bot = f"{ux - px*w_bot:.1f},{uy - py*w_bot:.1f} {ux + px*w_bot:.1f},{uy + py*w_bot:.1f}"
    return (f'<polygon points="{pts_top} {pts_bot}" fill="{color}" opacity="{opacity}" />'
            f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ux:.1f}" y2="{uy:.1f}" '
            f'stroke="{color}" stroke-width="1.8" opacity="0.75" stroke-dasharray="6,3"/>')


def _user_plane(cx, cy, size=22, color="#fdcb6e"):
    s = size
    body = f"M{cx},{cy-s*0.5} L{cx+s*0.15},{cy} L{cx},{cy+s*0.15} L{cx-s*0.15},{cy} Z"
    lw = f"M{cx-s*0.6},{cy+s*0.1} L{cx+s*0.6},{cy+s*0.1} L{cx+s*0.4},{cy+s*0.3} L{cx-s*0.4},{cy+s*0.3} Z"
    tail = f"M{cx-s*0.18},{cy+s*0.1} L{cx},{cy+s*0.45} L{cx+s*0.18},{cy+s*0.1}"
    return (f'<g filter="url(#glow_user)">'
            f'<path d="{lw}" fill="{color}" opacity="0.85"/>'
            f'<path d="{body}" fill="{color}"/>'
            f'<path d="{tail}" fill="{color}" opacity="0.7"/>'
            f'</g>')


def _user_car(cx, cy, size=22, color="#fdcb6e"):
    bw, bh = size * 1.6, size * 0.6
    rw, rh = size * 1.0, size * 0.55
    wr = size * 0.22
    s = (f'<g filter="url(#glow_user)">'
         f'<rect x="{cx-bw/2:.1f}" y="{cy-bh/2:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="5" fill="{color}"/>'
         f'<rect x="{cx-rw/2:.1f}" y="{cy-bh/2-rh:.1f}" width="{rw:.1f}" height="{rh+4:.1f}" rx="6" fill="{color}" opacity="0.85"/>'
         f'<circle cx="{cx-bw*0.3:.1f}" cy="{cy+bh/2:.1f}" r="{wr:.1f}" fill="#1a1a2e" stroke="white" stroke-width="1"/>'
         f'<circle cx="{cx+bw*0.3:.1f}" cy="{cy+bh/2:.1f}" r="{wr:.1f}" fill="#1a1a2e" stroke="white" stroke-width="1"/>'
         f'<rect x="{cx-rw/2+4:.1f}" y="{cy-bh/2-rh+3:.1f}" width="{rw-8:.1f}" height="{rh-6:.1f}" rx="4" fill="#0d1117" opacity="0.5"/>'
         f'</g>')
    return s


def _user_ship(cx, cy, size=24, color="#74b9ff"):
    hw = size * 0.9
    pts = (f"{cx-hw:.1f},{cy-size*0.1:.1f} "
           f"{cx-hw*1.1:.1f},{cy+size*0.5:.1f} "
           f"{cx+hw*1.1:.1f},{cy+size*0.5:.1f} "
           f"{cx+hw:.1f},{cy-size*0.1:.1f}")
    mast = (f'<line x1="{cx:.1f}" y1="{cy-size*0.1:.1f}" '
            f'x2="{cx:.1f}" y2="{cy-size*0.85:.1f}" stroke="white" stroke-width="2"/>'
            f'<line x1="{cx:.1f}" y1="{cy-size*0.7:.1f}" '
            f'x2="{cx+size*0.45:.1f}" y2="{cy-size*0.45:.1f}" stroke="white" stroke-width="1.2"/>')
    return (f'<g filter="url(#glow_user)">'
            f'<polygon points="{pts}" fill="{color}"/>'
            f'{mast}</g>')


def _user_geodesy(cx, cy, size=22, color="#00b894"):
    # Треногу
    legs = (f"M{cx},{cy-size*0.6} L{cx-size*0.6},{cy+size*0.4} "
            f"M{cx},{cy-size*0.6} L{cx+size*0.6},{cy+size*0.4} "
            f"M{cx},{cy-size*0.6} L{cx},{cy+size*0.45} "
            f"M{cx-size*0.2},{cy+size*0.4} L{cx+size*0.2},{cy+size*0.4}")
    return (f'<g filter="url(#glow_user)">'
            f'<path d="{legs}" stroke="{color}" stroke-width="2.5" fill="none" stroke-linecap="round"/>'
            f'<rect x="{cx-size*0.22:.1f}" y="{cy-size*0.9:.1f}" '
            f'width="{size*0.44:.1f}" height="{size*0.35:.1f}" rx="3" fill="{color}"/>'
            f'<circle cx="{cx:.1f}" cy="{cy-size*0.74:.1f}" r="{size*0.1:.1f}" fill="white"/>'
            f'</g>')


def _mcs_dish(cx, cy, size=30, color="#a29bfe"):
    pole = (f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{cx:.1f}" y2="{cy-size*0.8:.1f}" '
            f'stroke="{color}" stroke-width="3" stroke-linecap="round"/>')
    base = (f'<line x1="{cx-size*0.4:.1f}" y1="{cy:.1f}" x2="{cx+size*0.4:.1f}" y2="{cy:.1f}" '
            f'stroke="{color}" stroke-width="3" stroke-linecap="round"/>')
    dish = (f'<path d="M{cx-size*0.5:.1f},{cy-size*0.55:.1f} '
            f'Q{cx:.1f},{cy-size*1.2:.1f} {cx+size*0.5:.1f},{cy-size*0.55:.1f}" '
            f'stroke="{color}" stroke-width="2.5" fill="none" filter="url(#glow_mcs)"/>')
    return f'<g>{pole}{base}{dish}</g>'


def _esc(t: str) -> str:
    """Escape XML special characters for SVG text content."""
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


import re as _re

def _sanitize_svg(svg: str) -> str:
    """Fix unescaped < and > inside SVG <text> element content."""
    def fix_text_content(m):
        tag_open = m.group(1)   # everything up to first >
        content  = m.group(2)   # between > and </text>
        # escape bare < and > but preserve already-escaped entities
        content = content.replace("&lt;", "\x00LT\x00").replace("&gt;", "\x00GT\x00")
        content = content.replace("&amp;", "\x00AMP\x00")
        content = content.replace("&", "&amp;")                       # сначала bare '&'
        content = content.replace("<", "&lt;").replace(">", "&gt;")   # затем '<','>'
        content = content.replace("\x00LT\x00", "&lt;").replace("\x00GT\x00", "&gt;")
        content = content.replace("\x00AMP\x00", "&amp;")
        return f"{tag_open}>{content}</text>"
    # Match <text ...>CONTENT</text>
    svg = _re.sub(r'(<text[^>]*)>(.*?)</text>', fix_text_content, svg, flags=_re.DOTALL)
    return svg


def _label(x, y, text, size=13, color="white", anchor="middle", bold=False, dy=0, opacity=1.0):
    text = _esc(text)
    fw = "bold" if bold else "normal"
    return (f'<text x="{x:.1f}" y="{y+dy:.1f}" text-anchor="{anchor}" '
            f'font-family="Arial,sans-serif" font-size="{size}" font-weight="{fw}" '
            f'fill="{color}" opacity="{opacity}">{text}</text>')


def _defs(extra=""):
    return f"""<defs>
  <!-- Фон космос -->
  <linearGradient id="space_bg" x1="0" y1="0" x2="0" y2="1" gradientUnits="objectBoundingBox">
    <stop offset="0%" stop-color="#f6f8fb"/>
    <stop offset="100%" stop-color="#e7eef6"/>
  </linearGradient>
  <!-- Земля -->
  <radialGradient id="earth_grad" cx="42%" cy="38%" r="60%">
    <stop offset="0%" stop-color="#2d9d78"/>
    <stop offset="45%" stop-color="#1b6b4a"/>
    <stop offset="100%" stop-color="#0a2e1f"/>
  </radialGradient>
  <!-- Атмосфера -->
  <radialGradient id="atm_grad" cx="50%" cy="50%" r="50%">
    <stop offset="80%" stop-color="#74b9ff" stop-opacity="0"/>
    <stop offset="100%" stop-color="#74b9ff" stop-opacity="0.18"/>
  </radialGradient>
  <!-- Свечение сигнала L1 -->
  <filter id="glow_green" x="-50%" y="-50%" width="200%" height="200%">
    <feGaussianBlur in="SourceGraphic" stdDeviation="4" result="blur"/>
    <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
  <!-- Свечение потребителя -->
  <filter id="glow_user" x="-60%" y="-60%" width="220%" height="220%">
    <feGaussianBlur in="SourceGraphic" stdDeviation="3" result="blur"/>
    <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
  <!-- Свечение МКС -->
  <filter id="glow_mcs" x="-60%" y="-60%" width="220%" height="220%">
    <feGaussianBlur in="SourceGraphic" stdDeviation="5" result="blur"/>
    <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
  <!-- Свечение текста -->
  <filter id="text_glow">
    <feGaussianBlur in="SourceGraphic" stdDeviation="3" result="blur"/>
    <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
  <!-- Маска Земли (полукруг) -->
  <clipPath id="earth_clip">
    <rect x="0" y="0" width="1400" height="800"/>
  </clipPath>
  {extra}
</defs>"""


# ──────────────────────────────────────────────────────────────────────────────
# РИСУНОК 1: Системный обзор
# ──────────────────────────────────────────────────────────────────────────────
def _svg_system_overview():
    W, H = 1400, 820
    ECX, ECY, ER = 700, 1020, 620   # центр и радиус Земли (частично за краем)

    parts = [f'<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg">']
    parts.append(_defs())

    # Фон
    parts.append(f'<rect width="{W}" height="{H}" fill="url(#space_bg)"/>')

    # Звёзды
    parts.append(_stars(170, W, H * 0.82))

    # Земля
    parts.append(f'<circle cx="{ECX}" cy="{ECY}" r="{ER}" fill="url(#earth_grad)"/>')
    # Атмосферное свечение
    for i in range(4):
        rr = ER + 8 + i * 7
        op = 0.10 - i * 0.02
        parts.append(f'<circle cx="{ECX}" cy="{ECY}" r="{rr}" '
                     f'fill="none" stroke="#74b9ff" stroke-width="4" opacity="{op:.2f}"/>')

    # Береговые линии (упрощённые пятна суши)
    lands = [
        (640, 430, 90, 55, 15),   # Европа/Россия
        (530, 480, 60, 35, -5),   # Африка сев
        (800, 420, 55, 40, 10),   # Азия
        (420, 460, 45, 30,  5),
    ]
    for lx, ly, lw, lh, rot in lands:
        parts.append(f'<ellipse cx="{lx}" cy="{ly}" rx="{lw}" ry="{lh}" '
                     f'fill="#1b7a50" opacity="0.55" transform="rotate({rot},{lx},{ly})"/>')

    # Орбита АВРОРА (LEO 1000 км) — дуга
    orb_r = ER - 180   # визуально
    op_path = _arc_path(ECX, ECY, orb_r, 25, 155)
    parts.append(f'<path d="{op_path}" fill="none" stroke="#00b894" '
                 f'stroke-width="1.4" stroke-dasharray="10,6" opacity="0.45"/>')

    # Метка орбиты
    parts.append(_label(980, 285, "LEO 1000 км", 13, "#00b894", opacity=0.75))

    # ── Спутники ──────────────────────────────────────────────────────────────
    sats = [
        (700, 198, 34, 0,    "#00b894", True),   # главный — центр
        (310, 350, 26, -15,  "#00cec9", False),  # левый
        (1070, 320, 26,  15, "#00cec9", False),  # правый
    ]
    for sx, sy, sz, ang, col, is_main in sats:
        parts.append(_sat_icon(sx, sy, sz, col, ang, glow=True))

    # Заголовок у главного спутника
    parts.append(f'<text x="700" y="148" text-anchor="middle" '
                 f'font-family="Arial,sans-serif" font-size="18" font-weight="bold" '
                 f'fill="#00b894" filter="url(#text_glow)">АВРОРА</text>')
    parts.append(_label(700, 168, "L1 · 1575,42 МГц  |  L5 · 1176,45 МГц",
                        11, "#00cec9", opacity=0.85))

    # ISL между спутниками
    for (x1, y1), (x2, y2) in [((700, 198), (310, 350)), ((700, 198), (1070, 320))]:
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                     f'stroke="#6c5ce7" stroke-width="1.5" stroke-dasharray="5,4" opacity="0.55"/>')
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        parts.append(_label(mx, my - 10, "ISL Ka", 10, "#a29bfe", opacity=0.8))

    # ── Потребители + лучи ────────────────────────────────────────────────────
    users = [
        # (ux, uy,  icon,     lbl,                sub,                  L1_src,         L5_src,        l1col,      l5col)
        (155, 510, "plane",  "Авиация",          "LPV-200 · < 1 м",   (310, 350),     (700, 198),    "#00b894", "#6c5ce7"),
        (430, 568, "car",    "Автомобиль / БПЛА","< 0,5 м (H-95%)",   (700, 198),     (700, 198),    "#00b894", "#6c5ce7"),
        (730, 575, "geo",    "Геодезия PPP-RTK", "< 1 см · 5 с",      (700, 198),     (700, 198),    "#00b894", "#6c5ce7"),
        (1060, 545,"ship",   "Морской флот",     "DGNSS · < 0,5 м",   (1070, 320),    (700, 198),    "#00b894", "#6c5ce7"),
    ]

    icon_fn = {
        "plane": _user_plane,
        "car":   _user_car,
        "ship":  _user_ship,
        "geo":   _user_geodesy,
    }

    for ux, uy, icon, lbl, sub, l1src, l5src, l1c, l5c in users:
        # Лучи
        parts.append(_beam_cone(l1src[0], l1src[1] + 20, ux, uy - 30, 5, l1c, 0.14))
        parts.append(f'<line x1="{l5src[0]}" y1="{l5src[1]+20}" x2="{ux+8}" y2="{uy-25}" '
                     f'stroke="{l5c}" stroke-width="1.4" opacity="0.5" stroke-dasharray="4,3"/>')
        # Иконка
        parts.append(icon_fn[icon](ux, uy))
        # Подписи
        parts.append(_label(ux, uy + 38, lbl, 13, "white", bold=True))
        parts.append(_label(ux, uy + 55, sub, 11, "#00b894", opacity=0.9))

    # ── Наземная станция МКС ──────────────────────────────────────────────────
    parts.append(_mcs_dish(875, 558, 36, "#a29bfe"))
    parts.append(_label(875, 600, "МКС АВРОРА", 12, "#a29bfe", bold=True))
    parts.append(_label(875, 616, "21 станция", 10, "#a29bfe", opacity=0.75))
    # Аплинк
    parts.append(f'<line x1="875" y1="522" x2="712" y2="220" '
                 f'stroke="#a29bfe" stroke-width="1.5" stroke-dasharray="6,4" opacity="0.6"/>')
    parts.append(_label(804, 365, "S-диап. ТМ/ТК", 10, "#a29bfe", opacity=0.7))

    # RSN-точка
    parts.append(f'<circle cx="595" cy="572" r="7" fill="#00b894" stroke="white" stroke-width="1.5"/>')
    parts.append(_label(595, 592, "RSN", 10, "#00b894"))
    parts.append(_label(595, 606, "PPP-RTK", 9, "#00b894", opacity=0.75))

    # ── Легенда ───────────────────────────────────────────────────────────────
    lx, ly = 30, 36
    legend_items = [
        ("#00b894", "L1 (1575,42 МГц) — основной навигационный"),
        ("#6c5ce7", "L5 (1176,45 МГц) — двухчастотная коррекция"),
        ("#a29bfe", "ISL Ka-диапазон (26 ГГц) — межспутниковая связь"),
        ("#a29bfe", "S-диапазон — телеметрия / командование (МКС)"),
    ]
    parts.append(f'<rect x="{lx-8}" y="{ly-18}" width="360" height="{len(legend_items)*22+14}" '
                 f'rx="8" fill="#0d1117" opacity="0.7"/>')
    for i, (col, txt) in enumerate(legend_items):
        y2 = ly + i * 22
        dash = "5,3" if "ISL" in txt or "S-диап" in txt else "none"
        parts.append(f'<line x1="{lx}" y1="{y2}" x2="{lx+28}" y2="{y2}" '
                     f'stroke="{col}" stroke-width="2.5" stroke-dasharray="{dash}"/>')
        parts.append(_label(lx + 35, y2 + 4, txt, 11, "#c9d1d9", anchor="start"))

    # Заголовок документа
    parts.append(f'<text x="{W//2}" y="28" text-anchor="middle" '
                 f'font-family="Arial,sans-serif" font-size="22" font-weight="bold" '
                 f'fill="white" filter="url(#text_glow)">'
                 f'АВРОРА — Системная концепция</text>')
    parts.append(_label(W // 2, 52,
                        "Walker Delta 300/15/1 · h = 1000 км · i = 75°  |  L1 + L5  |  PPP-RTK · ISL · LPT",
                        12, "#8b949e"))

    parts.append("</svg>")
    return "\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# РИСУНОК 2: Сценарии применения (4 панели)
# ──────────────────────────────────────────────────────────────────────────────
def _svg_service_scenarios():
    W, H = 1400, 820

    scenarios = [
        {
            "title": "Авиация — заход на посадку (LPV-200)",
            "acc":   "< 1 м вертикально",
            "serv":  "АВРОРА PPP + СДКМ",
            "ttff":  "Горячий старт: 1,5 с",
            "color": "#0984e3",
            "icon":  "plane",
            "details": ["TIR < 1e-7 /ч", "TTPR < 6 с", "N_vis ≈ 14", "RAIM активен", "Совм. DO-229"],
        },
        {
            "title": "Геодезия / точн. земледелие (PPP-RTK)",
            "acc":   "0,5–1 см (H-68%)",
            "serv":  "АВРОРА PPP-RTK + RSN 300 км",
            "ttff":  "Сходимость: 5 с",
            "color": "#00b894",
            "icon":  "geo",
            "details": ["RSN шаг 300 км", "E2E задержка 70 мс", "IF-комб. L1+L5", "TESLA MAC защита", "ГЛОНАСС совм."],
        },
        {
            "title": "Автомобиль / БПЛА (lane-level)",
            "acc":   "< 0,5 м (H-95%)",
            "serv":  "АВРОРА PPP / PPP-RTK",
            "ttff":  "Тёплый старт: 5 с",
            "color": "#fdcb6e",
            "icon":  "car",
            "details": ["Полосная точность", "Устойч. +23 дБ vs ГЛОНАСС", "CRPA-антенна", "32 канала", "BW 18 Гц"],
        },
        {
            "title": "Синхронизация / LPT-сервис",
            "acc":   "< 5 нс UTC(SU)",
            "serv":  "АВРОРА LPT · ISL-сетка",
            "ttff":  "Удержание: 72 ч автономно",
            "color": "#a29bfe",
            "icon":  "tower",
            "details": ["CSAC: 2,7 нс/6ч", "6 ISL-хопов: 2,58 нс", "PTP IEEE 1588", "Импортозамещение GPS-синхр.", "UTC(SU) < 5 нс"],
        },
    ]

    parts = [f'<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg">']
    parts.append(_defs())
    parts.append(f'<rect width="{W}" height="{H}" fill="url(#space_bg)"/>')

    # Заголовок
    parts.append(f'<text x="{W//2}" y="38" text-anchor="middle" '
                 f'font-family="Arial,sans-serif" font-size="22" font-weight="bold" '
                 f'fill="white" filter="url(#text_glow)">АВРОРА — Сценарии применения</text>')

    pw, ph = 660, 355
    positions = [(30, 58), (710, 58), (30, 432), (710, 432)]

    icon_fn = {
        "plane": lambda cx, cy: _user_plane(cx, cy, 32, "#fdcb6e"),
        "geo":   lambda cx, cy: _user_geodesy(cx, cy, 30, "#00b894"),
        "car":   lambda cx, cy: _user_car(cx, cy, 30, "#fdcb6e"),
        "tower": lambda cx, cy: _mcs_dish(cx, cy, 32, "#a29bfe"),
    }

    for sc, (px, py) in zip(scenarios, positions):
        col = sc["color"]
        # Панель
        parts.append(f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="12" '
                     f'fill="#0d1117" stroke="{col}" stroke-width="2" opacity="0.95"/>')
        # Заголовочная полоса
        parts.append(f'<rect x="{px}" y="{py}" width="{pw}" height="48" rx="12" '
                     f'fill="{col}" opacity="0.22"/>')
        parts.append(f'<rect x="{px}" y="{py+36}" width="{pw}" height="12" '
                     f'fill="{col}" opacity="0.22"/>')
        parts.append(f'<text x="{px+pw//2}" y="{py+30}" text-anchor="middle" '
                     f'font-family="Arial,sans-serif" font-size="14.5" font-weight="bold" '
                     f'fill="white">{sc["title"]}</text>')

        # Мини-сцена: Земля + спутник + пользователь + луч
        scx = px + 170   # центр мини-сцены
        er_mini = 160
        ey_mini = py + ph + 30
        # Земля мини
        parts.append(f'<circle cx="{scx}" cy="{ey_mini}" r="{er_mini}" '
                     f'fill="url(#earth_grad)" clip-path="url(#panel_clip_{px}_{py})"/>')
        parts.append(f'<clipPath id="panel_clip_{px}_{py}">'
                     f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}"/>'
                     f'</clipPath>')
        # Атмосфера
        parts.append(f'<circle cx="{scx}" cy="{ey_mini}" r="{er_mini+12}" '
                     f'fill="none" stroke="#74b9ff" stroke-width="5" opacity="0.12" '
                     f'clip-path="url(#panel_clip_{px}_{py})"/>')

        # Спутник мини
        sat_mx, sat_my = scx - 5, py + 110
        parts.append(_sat_icon(sat_mx, sat_my, 22, col, 0))

        # Пользователь мини
        user_mx = scx + 55
        user_my = ey_mini - er_mini + 45
        parts.append(icon_fn[sc["icon"]](user_mx, user_my))

        # Луч
        parts.append(_beam_cone(sat_mx, sat_my + 18, user_mx, user_my - 20, 4, col, 0.15))
        parts.append(f'<line x1="{sat_mx+5}" y1="{sat_my+18}" '
                     f'x2="{user_mx+8}" y2="{user_my-18}" '
                     f'stroke="{col}" stroke-width="2" opacity="0.7" stroke-dasharray="5,3" '
                     f'clip-path="url(#panel_clip_{px}_{py})"/>')

        # Метрики
        mx = px + 360
        parts.append(f'<text x="{mx}" y="{py+90}" text-anchor="middle" '
                     f'font-family="Arial,sans-serif" font-size="26" font-weight="bold" '
                     f'fill="{col}" filter="url(#text_glow)">{sc["acc"]}</text>')
        parts.append(_label(mx, py + 118, sc["ttff"], 12, "#8b949e"))
        parts.append(f'<text x="{mx}" y="{py+142}" text-anchor="middle" '
                     f'font-family="Arial,sans-serif" font-size="13" font-weight="bold" '
                     f'fill="{col}">{sc["serv"]}</text>')

        # Разделитель
        parts.append(f'<line x1="{mx-130}" y1="{py+155}" x2="{mx+130}" y2="{py+155}" '
                     f'stroke="{col}" stroke-width="0.8" opacity="0.35"/>')

        # Детали
        for i, det in enumerate(sc["details"]):
            dy2 = py + 175 + i * 34
            parts.append(f'<circle cx="{mx-125}" cy="{dy2-4}" r="4" fill="{col}" opacity="0.8"/>')
            parts.append(_label(mx - 115, dy2, det, 12, "#c9d1d9", anchor="start"))

    parts.append("</svg>")
    return "\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# РИСУНОК 3: LEO vs MEO
# ──────────────────────────────────────────────────────────────────────────────
def _svg_leo_vs_meo():
    W, H = 1400, 820

    parts = [f'<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg">']
    parts.append(_defs())
    parts.append(f'<rect width="{W}" height="{H}" fill="url(#space_bg)"/>')
    parts.append(_stars(160, W, H * 0.85))

    # Разделитель
    parts.append(f'<line x1="{W//2}" y1="55" x2="{W//2}" y2="{H-20}" '
                 f'stroke="#30363d" stroke-width="2" stroke-dasharray="8,5"/>')
    parts.append(_label(W // 2, 36, "vs", 18, "#8b949e"))

    systems = [
        {
            "cx": 340, "col": "#00b894", "label": "АВРОРА LEO",
            "sub": "h = 1000 км", "sat_y": 220, "sat_sz": 38,
            "earth_r": 290, "earth_cy": 950,
            "beam_w": 18, "beam_op": 0.25,
            "params": [
                ("Уровень сигнала (EIRP)",  "−107 дБВт", "#00b894"),
                ("Потери в пространстве",   "−159 дБ (1000 км)", "#00b894"),
                ("Усиление vs ГЛОНАСС",      "+23 дБ  (×200 по мощн.)", "#00b894"),
                ("Время пролёта",           "~11 мин", "#00b894"),
                ("Макс. Доплер L1",         "±38,6 кГц / 38,6 Гц/с", "#fdcb6e"),
                ("TTFF (PPP-RTK)",           "5 с", "#00b894"),
                ("Задержка сигнала LOS",    "3,3 мс", "#00b894"),
                ("Геометрия (PDOP p95)",    "1,7 комб. · N_vis ≈ 14", "#00b894"),
            ],
        },
        {
            "cx": 1060, "col": "#e17055", "label": "ГЛОНАСС MEO",
            "sub": "h = 19 100 км", "sat_y": 135, "sat_sz": 30,
            "earth_r": 290, "earth_cy": 950,
            "beam_w": 8, "beam_op": 0.12,
            "params": [
                ("Уровень сигнала (EIRP)",  "−158 дБВт", "#e17055"),
                ("Потери в пространстве",   "−182 дБ (19 100 км)", "#e17055"),
                ("Усиление vs ГЛОНАСС",      "0 дБ (референс)", "#8b949e"),
                ("Время пролёта",           "~4 ч", "#e17055"),
                ("Макс. Доплер L1",         "±4,9 кГц / ~0,9 Гц/с", "#e17055"),
                ("TTFF (PPP классич.)",      "1 500 с (25 мин)", "#d63031"),
                ("Задержка сигнала LOS",    "67 мс", "#d63031"),
                ("Геометрия (PDOP p95)",    "~1,5  (N_vis ≈ 8–12)", "#e17055"),
            ],
        },
    ]

    for sc in systems:
        cx, col = sc["cx"], sc["col"]
        er, ecy = sc["earth_r"], sc["earth_cy"]

        # Земля
        parts.append(f'<circle cx="{cx}" cy="{ecy}" r="{er}" fill="url(#earth_grad)"/>')
        for i in range(3):
            rr = er + 7 + i * 7
            op = 0.10 - i * 0.02
            parts.append(f'<circle cx="{cx}" cy="{ecy}" r="{rr}" '
                         f'fill="none" stroke="#74b9ff" stroke-width="4" opacity="{op:.2f}"/>')

        # Поверхность Земли — горизонт
        hor_y = ecy - er
        parts.append(f'<line x1="{cx-er}" y1="{hor_y}" x2="{cx+er}" y2="{hor_y}" '
                     f'stroke="#2d9d78" stroke-width="0" opacity="0"/>')

        # Конус покрытия
        user_x, user_y = cx + 60, hor_y + 18
        sat_x, sat_y = cx, sc["sat_y"]
        bw = sc["beam_w"]
        parts.append(_beam_cone(sat_x, sat_y + sc["sat_sz"] // 2,
                                user_x, user_y, bw, col, sc["beam_op"]))

        # Ширина луча (угол покрытия)
        half_cov = 140 if sc["label"].startswith("АВРОРА") else 200
        sz2 = sc["sat_sz"] // 2
        parts.append(f'<line x1="{sat_x}" y1="{sat_y + sz2}" '
                     f'x2="{cx-half_cov}" y2="{hor_y+5}" '
                     f'stroke="{col}" stroke-width="1" opacity="0.25" stroke-dasharray="4,3"/>')
        parts.append(f'<line x1="{sat_x}" y1="{sat_y + sz2}" '
                     f'x2="{cx+half_cov}" y2="{hor_y+5}" '
                     f'stroke="{col}" stroke-width="1" opacity="0.25" stroke-dasharray="4,3"/>')

        # Спутник
        parts.append(_sat_icon(sat_x, sat_y, sc["sat_sz"], col))

        # Метки спутника
        sat_lbl_y = sat_y - sc["sat_sz"] - 22
        sat_sub_y = sat_y - sc["sat_sz"] - 4
        parts.append(f'<text x="{sat_x}" y="{sat_lbl_y}" '
                     f'text-anchor="middle" font-family="Arial,sans-serif" '
                     f'font-size="20" font-weight="bold" fill="{col}" '
                     f'filter="url(#text_glow)">{sc["label"]}</text>')
        parts.append(_label(sat_x, sat_sub_y, sc["sub"], 13, "#8b949e"))

        # Высотная стрелка (без маркеров)
        parts.append(f'<line x1="{sat_x+90}" y1="{hor_y}" '
                     f'x2="{sat_x+90}" y2="{sat_y}" '
                     f'stroke="{col}" stroke-width="1.2" opacity="0.5"/>'
                     f'<circle cx="{sat_x+90}" cy="{hor_y}" r="3" fill="{col}" opacity="0.6"/>'
                     f'<circle cx="{sat_x+90}" cy="{sat_y}" r="3" fill="{col}" opacity="0.6"/>')
        alt_lbl = "1 000 км" if sc["label"].startswith("AUR") else "20 200 км"
        parts.append(_label(sat_x + 115, (hor_y + sat_y) // 2, alt_lbl, 11, col, opacity=0.75))

        # Потребитель
        parts.append(_user_geodesy(user_x, user_y, 20, col))

        # Таблица параметров
        tbl_x = cx - 175 if sc["label"].startswith("AUR") else cx - 155
        tbl_y = 450
        row_h = 42
        parts.append(f'<rect x="{tbl_x-10}" y="{tbl_y-30}" width="350" '
                     f'height="{len(sc["params"])*row_h + 20}" rx="8" '
                     f'fill="#0d1117" opacity="0.7"/>')
        for i, (param, val, vc) in enumerate(sc["params"]):
            ry = tbl_y + i * row_h
            parts.append(_label(tbl_x, ry, param + ":", 11, "#8b949e", anchor="start"))
            parts.append(f'<text x="{tbl_x+330}" y="{ry}" text-anchor="end" '
                         f'font-family="Arial,sans-serif" font-size="12" font-weight="bold" '
                         f'fill="{vc}">{val}</text>')
            if i < len(sc["params"]) - 1:
                parts.append(f'<line x1="{tbl_x-5}" y1="{ry+12}" x2="{tbl_x+335}" '
                             f'y2="{ry+12}" stroke="{col}" stroke-width="0.4" opacity="0.2"/>')

    # Центральная аннотация
    mx = W // 2
    parts.append(f'<text x="{mx}" y="390" text-anchor="middle" '
                 f'font-family="Arial,sans-serif" font-size="28" font-weight="bold" '
                 f'fill="#00b894" filter="url(#text_glow)">+23 дБ</text>')
    parts.append(_label(mx, 418, "сигнал сильнее", 13, "white"))
    parts.append(_label(mx, 440, "× 200 по", 13, "white"))
    parts.append(_label(mx, 460, "устойчивости к", 13, "white"))
    parts.append(_label(mx, 480, "подавлению", 13, "white"))
    parts.append(f'<line x1="{mx-60}" y1="405" x2="{mx-80}" y2="405" '
                 f'stroke="#00b894" stroke-width="2"/>'
                 f'<polygon points="{mx-80},402 {mx-80},408 {mx-92},405" fill="#00b894"/>')
    parts.append(f'<line x1="{mx+60}" y1="405" x2="{mx+80}" y2="405" '
                 f'stroke="#e17055" stroke-width="2"/>'
                 f'<polygon points="{mx+80},402 {mx+80},408 {mx+92},405" fill="#e17055"/>')

    # Заголовок
    parts.append(f'<text x="{W//2}" y="36" text-anchor="middle" '
                 f'font-family="Arial,sans-serif" font-size="21" font-weight="bold" '
                 f'fill="white" filter="url(#text_glow)">'
                 f'АВРОРА LEO vs ГЛОНАСС MEO — Преимущество сигнала и характеристики</text>')

    parts.append("</svg>")
    return "\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# РИСУНОК 4: Поток сигнала
# ──────────────────────────────────────────────────────────────────────────────
def _svg_signal_flow():
    W, H = 1400, 560

    stages = [
        {
            "x": 110, "label": "Атомный\nстандарт", "sub": "H-мазер → CSAC",
            "color": "#a29bfe", "tag": "T0",
            "desc": "ADEV < 1e-13 /с (мазер)",
        },
        {
            "x": 330, "label": "ANAV\nпередатчик", "sub": "L1 5Вт · L5 3Вт",
            "color": "#00b894", "tag": "TX",
            "desc": "5000 бит/10с · 500 бит/с",
        },
        {
            "x": 560, "label": "Радиосигнал\nL1 + L5", "sub": "FSPL −159 дБ",
            "color": "#6c5ce7", "tag": "RF",
            "desc": "1000 км · 3,3 мс",
        },
        {
            "x": 790, "label": "Приёмник\nАВРОРА", "sub": "≥ 45 дБ·Гц",
            "color": "#fdcb6e", "tag": "RX",
            "desc": "238 каналов · BW 1,4 Гц",
        },
        {
            "x": 1020, "label": "Kalman-\nфильтр", "sub": "IF L1+L5",
            "color": "#00cec9", "tag": "KF",
            "desc": "PPP / PPP-RTK · SSR",
        },
        {
            "x": 1250, "label": "PNT\nрешение", "sub": "< 0,5 м · < 5 нс",
            "color": "#00b894", "tag": "OUT",
            "desc": "UTC(SU) · H-95%",
        },
    ]

    parts = [f'<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg">']
    parts.append(_defs())
    parts.append(f'<rect width="{W}" height="{H}" fill="url(#space_bg)"/>')

    # Заголовок
    parts.append(f'<text x="{W//2}" y="38" text-anchor="middle" '
                 f'font-family="Arial,sans-serif" font-size="19" font-weight="bold" '
                 f'fill="white">АВРОРА — Поток сигнала и данных</text>')
    parts.append(_label(W // 2, 60, "от атомных часов КА до навигационного решения потребителя", 12, "#8b949e"))

    bw, bh = 155, 120
    cy = 200

    for i, st in enumerate(stages):
        x, col = st["x"], st["color"]

        # Блок
        parts.append(f'<rect x="{x-bw//2}" y="{cy-bh//2}" width="{bw}" height="{bh}" rx="10" '
                     f'fill="#0d1117" stroke="{col}" stroke-width="2"/>')
        # Цветная верхняя полоса
        parts.append(f'<rect x="{x-bw//2}" y="{cy-bh//2}" width="{bw}" height="36" rx="10" '
                     f'fill="{col}" opacity="0.25"/>')
        parts.append(f'<rect x="{x-bw//2}" y="{cy-bh//2+26}" width="{bw}" height="10" '
                     f'fill="{col}" opacity="0.25"/>')

        # Тег
        parts.append(f'<text x="{x}" y="{cy-bh//2+22}" text-anchor="middle" '
                     f'font-family="Arial,sans-serif" font-size="15" font-weight="bold" '
                     f'fill="{col}">{st["tag"]}</text>')

        # Метка
        lines = st["label"].split("\n")
        for j, line in enumerate(lines):
            parts.append(f'<text x="{x}" y="{cy+j*20-10}" text-anchor="middle" '
                         f'font-family="Arial,sans-serif" font-size="13" font-weight="bold" '
                         f'fill="white">{line}</text>')

        # Подзаголовок
        parts.append(_label(x, cy + 42, st["sub"], 11, col))

        # Описание под блоком
        parts.append(_label(x, cy + bh // 2 + 22, st["desc"], 10, "#8b949e"))

        # Стрелка к следующему
        if i < len(stages) - 1:
            nx = stages[i + 1]["x"]
            ncol = stages[i + 1]["color"]
            ax1, ax2 = x + bw // 2 + 4, nx - bw // 2 - 4
            parts.append(f'<defs><marker id="arr_{i}" markerWidth="8" markerHeight="8" '
                         f'refX="6" refY="3" orient="auto">'
                         f'<path d="M0,0 L0,6 L8,3 z" fill="{col}"/></marker></defs>')
            parts.append(f'<line x1="{ax1}" y1="{cy}" x2="{ax2}" y2="{cy}" '
                         f'stroke="{col}" stroke-width="2.2" '
                         f'marker-end="url(#arr_{i})"/>')

    # Нижняя цепочка синхронизации
    chain_y = 400
    parts.append(f'<rect x="30" y="{chain_y-18}" width="{W-60}" height="130" rx="10" '
                 f'fill="#0d1117" opacity="0.6"/>')
    parts.append(f'<text x="{W//2}" y="{chain_y+2}" text-anchor="middle" '
                 f'font-family="Arial,sans-serif" font-size="13" font-weight="bold" '
                 f'fill="#a29bfe">Цепочка синхронизации UTC(SU) → КА → Пользователь</text>')

    chain = [
        (110,  "UTC(SU)", "± 5 нс"),
        (310,  "Cs КА",   "± 0,01 нс/6ч"),
        (510,  "ISL ×1",  "± 0,43 нс/хоп"),
        (710,  "ISL ×6",  "± 2,58 нс"),
        (920,  "Тропосф.", "± ~6 см"),
        (1120, "Приёмник", "< 5 нс итого"),
        (1300, "UTC пользов.", "< 5 нс"),
    ]
    for i, (cx2, lbl, val) in enumerate(chain):
        col2 = "#a29bfe" if i == 0 else ("#00b894" if i == len(chain) - 1 else "#555577")
        parts.append(_label(cx2, chain_y + 32, lbl, 11, "white", bold=(i == len(chain) - 1)))
        parts.append(_label(cx2, chain_y + 52, val, 10, col2))
        if i < len(chain) - 1:
            nx2 = chain[i + 1][0]
            parts.append(f'<line x1="{cx2+50}" y1="{chain_y+38}" '
                         f'x2="{nx2-50}" y2="{chain_y+38}" '
                         f'stroke="#a29bfe" stroke-width="1.2" opacity="0.45" '
                         f'stroke-dasharray="5,3"/>')

    parts.append("</svg>")
    return "\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────────────────────────────────────
def _svg_time_dissemination():
    """Двухуровневая синхронизация: LPT (космос) + SHIWA TIME mesh (земля)."""
    W, H = 1400, 820
    parts = [f'<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg">']
    parts.append(_defs())
    parts.append(f'<rect width="{W}" height="{H}" fill="url(#space_bg)"/>')
    parts.append(f'<text x="{W//2}" y="40" text-anchor="middle" '
                 f'font-family="Arial,sans-serif" font-size="20" font-weight="bold" '
                 f'fill="white">АВРОРА + SHIWA TIME — двухуровневая синхронизация времени</text>')

    parts.append(f'<defs><marker id="amk" markerWidth="9" markerHeight="9" refX="7" refY="3" '
                 f'orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#566072"/></marker></defs>')

    def box(x, y, w, h, col):
        return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" '
                f'fill="#0d1117" stroke="{col}" stroke-width="2"/>'
                f'<rect x="{x}" y="{y}" width="{w}" height="28" rx="10" fill="{col}" opacity="0.22"/>'
                f'<rect x="{x}" y="{y+18}" width="{w}" height="10" fill="{col}" opacity="0.22"/>')

    # Разделитель сегментов
    seg_y = 472
    parts.append(f'<line x1="40" y1="{seg_y}" x2="{W-40}" y2="{seg_y}" '
                 f'stroke="#30363d" stroke-width="1.5" stroke-dasharray="10,6"/>')
    parts.append(_label(70, seg_y-12, "Космический сегмент · LPT (радиоканал)", 13, "#00b894", anchor="start", bold=True))
    parts.append(_label(70, seg_y+24, "Наземный сегмент · SHIWA TIME (P2P-mesh)", 13, "#6c5ce7", anchor="start", bold=True))

    # UTC(SU)
    ux0, uy0 = 95, 110
    parts.append(box(ux0, uy0, 200, 90, "#0984e3"))
    parts.append(_label(ux0+100, uy0+22, "UTC(SU)", 14, "white", bold=True))
    parts.append(_label(ux0+100, uy0+48, "H-мазер МКС", 12, "#0984e3"))
    parts.append(_label(ux0+100, uy0+72, "эталон шкалы", 10, "#8b949e"))
    parts.append(f'<line x1="{ux0+200}" y1="{uy0+45}" x2="612" y2="135" '
                 f'stroke="#0984e3" stroke-width="2.2" marker-end="url(#amk)"/>')
    parts.append(_label(445, 110, "загрузка шкалы (ТМ/ТК)", 10, "#8b949e"))

    # Спутник АВРОРА
    sat_x, sat_y = 700, 160
    parts.append(_sat_icon(sat_x, sat_y, 36, "#00b894", 0))
    parts.append(_label(sat_x, sat_y-52, "АВРОРА КА", 14, "#00b894", bold=True))
    parts.append(_label(sat_x, sat_y+60, "Cs/Rb · бортовая шкала (§8)", 11, "#00b894"))

    # Луч LPT
    ref_x, ref_y = 700, 388
    parts.append(_beam_cone(sat_x, sat_y+22, ref_x, ref_y-34, 8, "#00b894", 0.15))
    parts.append(f'<line x1="{sat_x}" y1="{sat_y+22}" x2="{ref_x}" y2="{ref_y-36}" '
                 f'stroke="#00b894" stroke-width="2" stroke-dasharray="6,4" opacity="0.8"/>')
    parts.append(_label(sat_x+135, 285, "LPT · одностороннее", 11, "#00b894"))
    parts.append(_label(sat_x+135, 303, "время, < 10 нс", 11, "#00b894"))

    # Опорный узел SHIWA
    parts.append(box(ref_x-135, ref_y-28, 270, 86, "#6c5ce7"))
    parts.append(_label(ref_x, ref_y-4, "Опорный узел SHIWA", 13, "white", bold=True))
    parts.append(_label(ref_x, ref_y+20, "LPT-приёмник → источник mesh", 10, "#6c5ce7"))
    parts.append(_label(ref_x, ref_y+42, "UTC < 10 нс", 11, "#8b949e"))

    # Потребители
    consumers = [
        (330,  "ЦОД / LAN",      "профиль A · ±5 нс",   "итого ≈ 7–11 нс", "#0984e3"),
        (700,  "Оператор / WAN", "профиль B · ±50 нс",  "итого ≈ 50 нс",   "#00b894"),
        (1070, "Мобильный / LTE","профиль C · ±200 нс", "итого ≈ 200 нс",  "#fdcb6e"),
    ]
    cons_y = 660
    xs = []
    for cx, title, prof, total, col in consumers:
        parts.append(box(cx-120, cons_y-30, 240, 92, col))
        parts.append(_label(cx, cons_y-8, title, 13, "white", bold=True))
        parts.append(_label(cx, cons_y+16, prof, 11, col))
        parts.append(_label(cx, cons_y+40, total, 11, "#8b949e"))
        xs.append(cx)
        parts.append(f'<line x1="{ref_x}" y1="{ref_y+58}" x2="{cx}" y2="{cons_y-30}" '
                     f'stroke="{col}" stroke-width="2" opacity="0.7" marker-end="url(#amk)"/>')
    for a in range(len(xs)):
        for b in range(a+1, len(xs)):
            parts.append(f'<line x1="{xs[a]}" y1="{cons_y+64}" x2="{xs[b]}" y2="{cons_y+64}" '
                         f'stroke="#6c5ce7" stroke-width="1.2" stroke-dasharray="5,4" opacity="0.5"/>')
    parts.append(_label(W//2, cons_y+92, "P2P-mesh (libp2p · «счастливый пакет» · без Grandmaster)", 11, "#6c5ce7"))

    parts.append("</svg>")
    return "\n".join(parts)


def _lighten(svg: str) -> str:
    """Пост-процессор: тёмная 'космическая' палитра → светлая (для печати по ГОСТ).

    Меняет только значения атрибутов (fill/stroke/filter), текст не трогает.
    Фон space_bg переопределён в _defs; звёзды убраны в _stars.
    """
    reps = [
        # тёмные карточки/панели → белые
        ('fill="#0d1117"', 'fill="#ffffff"'),
        ('fill="#1a1a2e"', 'fill="#2c3e50"'),
        # текст и обводки «white» → тёмные / серые (на белом фоне)
        ('fill="white"', 'fill="#1a2433"'),
        ('stroke="white"', 'stroke="#5a6678"'),
        # убрать «свечения» — на белом дают грязный ореол
        ('filter="url(#text_glow)"', ''),
        ('filter="url(#glow_green)"', ''),
        ('filter="url(#glow_user)"', ''),
        ('filter="url(#glow_mcs)"', ''),
        # низкоконтрастные акценты → темнее (читаемость на белом)
        ('fill="#fdcb6e"', 'fill="#b8860b"'),   # жёлтый → тёмное золото
        ('fill="#c9d1d9"', 'fill="#2c3e50"'),   # светло-серый текст → тёмный
        ('fill="#8b949e"', 'fill="#566072"'),   # серый → темнее
        ('fill="#00cec9"', 'fill="#0a8f8a"'),   # циан → тёмный бирюз.
        ('stroke="#30363d"', 'stroke="#aeb8c4"'),
        ('stroke="#555577"', 'stroke="#9aa3b5"'),
    ]
    for a, b in reps:
        svg = svg.replace(a, b)
    return svg


def run_concept_svg(output_dir: str, label: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    figures = []

    generators = [
        ("system_overview",   _svg_system_overview),
        ("service_scenarios", _svg_service_scenarios),
        ("leo_vs_meo",        _svg_leo_vs_meo),
        ("signal_flow",       _svg_signal_flow),
        ("time_dissemination", _svg_time_dissemination),
    ]

    for name, fn in generators:
        svg_data = _sanitize_svg(_lighten(fn()))
        svg_path = os.path.join(output_dir, f"concept_{name}_{label}.svg")
        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(svg_data)

        png_path = svg_path.replace(".svg", ".png")
        if HAS_CAIRO:
            cairosvg.svg2png(url=svg_path, write_to=png_path, scale=1.5)
            print(f"  PNG: {png_path}")
            figures.append(f"concept_{name}_{label}.png")
        else:
            print(f"  SVG: {svg_path}  (cairosvg not found — open in browser)")
            figures.append(f"concept_{name}_{label}.svg")

    return {"figures": figures, "has_png": HAS_CAIRO}


if __name__ == "__main__":
    run_concept_svg("results/system_concept", "phase4")
