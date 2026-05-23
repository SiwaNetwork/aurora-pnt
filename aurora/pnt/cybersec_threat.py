"""
Модель угроз и киберустойчивость AURORA PNT (STRIDE / PASTA).

Сегменты, рассматриваемые в модели:
  1. SIS (Signal-In-Space)        — спуфинг, meaconing, replay, jamming.
  2. Бортовой (Spacecraft)        — компрометация ключей TESLA через side-channel,
                                    наводки на бортовые часы, отказ программного
                                    обеспечения навигационного процессора.
  3. Наземный (MCS, Mission Control) — атаки на сервер OD/SSR, цепочка поставок ПО,
                                    физический доступ к станциям.
  4. TT&C (Telemetry, Tracking & Command) — повреждение/инжекция команд,
                                    MitM, повтор криптокоманд.
  5. Пользовательский сегмент    — приёмники-приманки, side-channel на ключи в
                                    клиентских модулях, фальсифицированное ПО.

Для каждой угрозы:
    P  — вероятность реализации (0..1),
    I  — воздействие в шкале 0..10,
    m  — эффективность контрмеры (0..1),
    R  = P × I × (1 - m)  — остаточный риск.

Ссылки:
  ISO/IEC 27005:2022 — Risk management for information security.
  NIST SP 800-30 r1  — Guide for Conducting Risk Assessments.
  ECSS-E-ST-40C      — Software engineering, secure coding for space.
  CCSDS 350.0-G-3    — Security architecture for space data systems.
  Humphreys (2013)   — Detection strategy for cryptographic GNSS anti-spoofing.
"""

import sys, math, os, csv
from typing import Dict, List, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ── Палитра ───────────────────────────────────────────────────────────────────
COLORS = ["#e17055", "#fdcb6e", "#0984e3", "#00b894", "#6c5ce7", "#74b9ff"]

# Цвета для сегментов
SEG_COLORS = {
    "SIS":  "#e17055",
    "Bort": "#fdcb6e",
    "MCS":  "#0984e3",
    "TTC":  "#00b894",
    "User": "#6c5ce7",
}

# ── Угрозы (id, сегмент, категория, описание, P, I, m, контрмера) ────────────
# m — эффективность применяемой контрмеры (доля снижения P×I)
ATTACKS: List[Tuple[str, str, str, str, float, float, float, str]] = [
    # ── SIS сегмент ───────────────────────────────────────────────────
    ("T01", "SIS",  "Спуфинг",
     "Generated GPS-style спуфинг на сигнал L1 BOC(1,1)",      0.50, 8.0, 0.85, "CM01"),
    ("T02", "SIS",  "Глушение",
     "Узкополосный CW-jam в полосе L5 (10 МГц)",                0.70, 5.0, 0.70, "CM02"),
    ("T03", "SIS",  "Спуфинг",
     "Meaconing/replay со сдвигом задержки 0,2–3 мс",           0.40, 7.0, 0.80, "CM01"),
    ("T04", "SIS",  "Глушение",
     "Широкополосный barrage jam (L1+L5, 30 МГц)",              0.45, 7.0, 0.65, "CM02"),
    ("T05", "SIS",  "Перехват",
     "Криптоанализ публичного канала E1-B (низкий)",            0.15, 4.0, 0.50, "CM03"),

    # ── Бортовой сегмент ──────────────────────────────────────────────
    ("T06", "Bort", "Криптовзлом",
     "Извлечение ключей TESLA через DPA side-channel",          0.10, 9.0, 0.80, "CM04"),
    ("T07", "Bort", "RF-воздействие",
     "RF-инжекция в шину часов USO бортового приёмника",        0.05, 9.0, 0.60, "CM05"),
    ("T08", "Bort", "ПО",
     "Эксплойт в навигационном процессоре (SEU-trigger)",       0.15, 8.0, 0.55, "CM06"),
    ("T09", "Bort", "Аппаратное",
     "Отказ FPGA-кодогенератора (single-event latch-up)",       0.08, 7.0, 0.70, "CM07"),

    # ── Наземный сегмент (MCS) ────────────────────────────────────────
    ("T10", "MCS",  "Цепочка поставок",
     "Backdoor в стороннем ПО SSR-расчёта",                      0.20, 10.0, 0.55, "CM08"),
    ("T11", "MCS",  "Сеть",
     "DDoS на сервер OD (Orbit Determination)",                  0.50, 5.0, 0.70, "CM09"),
    ("T12", "MCS",  "Привилегии",
     "Эскалация привилегий на операторской станции",             0.20, 9.0, 0.65, "CM10"),
    ("T13", "MCS",  "Физика",
     "Физический доступ к опорной MCS-станции (insider)",        0.10, 8.0, 0.75, "CM11"),

    # ── TT&C сегмент ──────────────────────────────────────────────────
    ("T14", "TTC",  "Инжекция",
     "Повреждение TC-команд (TT&C MitM)",                        0.15, 9.0, 0.85, "CM12"),
    ("T15", "TTC",  "Повтор",
     "Replay криптокоманд TT&C без счётчика",                    0.25, 7.0, 0.85, "CM12"),
    ("T16", "TTC",  "Перехват",
     "Перехват TM-данных (телеметрии) — пассивный",              0.40, 3.0, 0.60, "CM13"),

    # ── Пользовательский сегмент ──────────────────────────────────────
    ("T17", "User", "Спуфинг",
     "Spoofing-атака на пользовательский приёмник (наземный)",   0.55, 6.0, 0.80, "CM01"),
    ("T18", "User", "Side-channel",
     "Извлечение клиентских ключей через power-trace",           0.20, 6.0, 0.60, "CM14"),
    ("T19", "User", "Подделка",
     "Поддельные приёмники с предзагруженным trust-anchor",      0.30, 7.0, 0.55, "CM15"),
    ("T20", "User", "ПО",
     "Уязвимость в драйвере декодера TESLA на устройстве",       0.25, 7.0, 0.65, "CM15"),
]

# ── Защитные меры (CMxx) ─────────────────────────────────────────────────────
COUNTERMEASURES: Dict[str, Tuple[str, str, str, float]] = {
    "CM01": ("TESLA MAC",                     "HMAC-SHA256, ротация ключа 30 с",        "Спуфинг",         0.85),
    "CM02": ("Антенный array-nulling",         "8-эл. CRPA, подавление помехи -30 дБ",  "Глушение",        0.70),
    "CM03": ("Шифрование сервисного канала",   "AES-256 на E6/L6-подобной полосе",       "Перехват",        0.55),
    "CM04": ("HSM на борту",                   "Защ. от DPA, TRNG, секц. изоляция",     "Криптовзлом",     0.80),
    "CM05": ("Экранирование USO + фильтры",    "EMI-экран, LNA с notch-filter",          "RF-воздействие",  0.60),
    "CM06": ("ECC память + watchdog",          "SECDED, periodic memory scrub",         "ПО",              0.55),
    "CM07": ("Rad-hard FPGA + TMR",            "Triple modular redundancy логики",      "Аппаратное",      0.70),
    "CM08": ("Code-signing + SBOM-аудит",      "Sigstore, SBOM CycloneDX, reproducible","Цепочка поставок",0.55),
    "CM09": ("Anti-DDoS + WAF",                "Cloudflare/Качество сервиса, rate-limit","Сеть",            0.70),
    "CM10": ("RBAC + MFA + zero-trust",        "Аппаратные ключи FIDO2, журналирование","Привилегии",      0.65),
    "CM11": ("Физ. безопасность станций",       "СКУД, видеонаблюдение, охрана",         "Физика",          0.75),
    "CM12": ("Auth+counter TT&C",              "CCSDS SDLS, monotonic counter",         "Инжекция/Повтор", 0.85),
    "CM13": ("Шифрование TM",                  "AES-GCM по линии вниз",                 "Перехват",        0.60),
    "CM14": ("Защищённое хранилище ключей",     "TPM 2.0 / Secure Element на устройстве","Side-channel",    0.60),
    "CM15": ("Attestation + secure boot",      "Remote attestation, signed firmware",   "Подделка/ПО",     0.60),
}


def run_cybersec_analysis(output_dir: str, label: str) -> Dict:
    """Запускает анализ модели угроз AURORA PNT."""
    os.makedirs(output_dir, exist_ok=True)

    threats = []
    for (tid, seg, cat, desc, P, I, m, cm) in ATTACKS:
        raw = P * I
        res = P * I * (1.0 - m)
        threats.append({
            "id": tid, "segment": seg, "category": cat, "description": desc,
            "P": P, "I": I, "raw": raw, "m": m, "residual": res, "cm": cm,
        })

    total_raw      = sum(t["raw"]      for t in threats)
    total_residual = sum(t["residual"] for t in threats)
    risk_reduction = (1.0 - total_residual / total_raw) * 100.0 if total_raw > 0 else 0.0

    # Группировка по сегментам
    by_seg = {}
    for t in threats:
        s = t["segment"]
        if s not in by_seg:
            by_seg[s] = {"raw": 0.0, "residual": 0.0, "n": 0}
        by_seg[s]["raw"]      += t["raw"]
        by_seg[s]["residual"] += t["residual"]
        by_seg[s]["n"]        += 1

    # Топ-5 остаточных рисков
    top5 = sorted(threats, key=lambda x: -x["residual"])[:5]

    results = {
        "threats":        threats,
        "by_segment":     by_seg,
        "total_raw":      total_raw,
        "total_residual": total_residual,
        "risk_reduction": risk_reduction,
        "top5_residual":  top5,
        "n_threats":      len(threats),
        "n_cms":          len(COUNTERMEASURES),
    }

    _plot_threat_tree(threats, output_dir, label)
    _plot_risk_matrix(threats, output_dir, label)
    _plot_countermeasures(output_dir, label)
    _plot_residual_risk(threats, output_dir, label)
    _save_csv(threats, output_dir, label)
    return results


# ── 1. Дерево угроз ──────────────────────────────────────────────────────────
def _plot_threat_tree(threats, output_dir, label):
    """Иерархическая диаграмма угроз: ствол AURORA → 5 сегментов → угрозы."""
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Корневой узел
    root_x, root_y = 50, 92
    root = FancyBboxPatch((root_x - 11, root_y - 3), 22, 6,
                          boxstyle="round,pad=0.3",
                          facecolor="#2d3436", edgecolor="white", linewidth=2)
    ax.add_patch(root)
    ax.text(root_x, root_y, "AURORA PNT", color="white",
            ha="center", va="center", fontsize=12, fontweight="bold")

    # 5 сегментов
    seg_names = {
        "SIS":  "Сигнал-в-эфире\n(SIS)",
        "Bort": "Бортовой\nсегмент",
        "MCS":  "Наземный\n(MCS)",
        "TTC":  "Сеть TT&C",
        "User": "Пользовательский\nсегмент",
    }
    seg_keys = ["SIS", "Bort", "MCS", "TTC", "User"]
    seg_xs   = [10, 30, 50, 70, 90]
    seg_y    = 72

    for sx, sk in zip(seg_xs, seg_keys):
        # связь от корня
        arrow = FancyArrowPatch((root_x, root_y - 3), (sx, seg_y + 3.5),
                                 arrowstyle="-", color="#636e72", lw=1.4,
                                 connectionstyle="arc3,rad=0.0")
        ax.add_patch(arrow)
        # узел сегмента
        seg = FancyBboxPatch((sx - 8, seg_y - 3.5), 16, 7,
                              boxstyle="round,pad=0.25",
                              facecolor=SEG_COLORS[sk], edgecolor="white", linewidth=1.5)
        ax.add_patch(seg)
        ax.text(sx, seg_y, seg_names[sk], color="white",
                ha="center", va="center", fontsize=9, fontweight="bold")

        # угрозы данного сегмента
        ts = [t for t in threats if t["segment"] == sk]
        # располагаем угрозы вертикально под сегментом
        ys = np.linspace(58, 58 - (len(ts) - 1) * 7, len(ts)) if len(ts) > 0 else []
        for ty, t in zip(ys, ts):
            ax.plot([sx, sx], [seg_y - 3.5, ty + 2], color="#b2bec3", lw=0.9)
            lab = f"{t['id']}\n{t['category']}"
            box = FancyBboxPatch((sx - 6.5, ty - 2.2), 13, 4.4,
                                  boxstyle="round,pad=0.18",
                                  facecolor="white", edgecolor=SEG_COLORS[sk], linewidth=1.1)
            ax.add_patch(box)
            ax.text(sx, ty, lab, color="#2d3436",
                    ha="center", va="center", fontsize=7)
            # ярлык остаточного риска
            ax.text(sx + 5.6, ty - 1.7, f"R={t['residual']:.2f}",
                    fontsize=6, color="#e17055", ha="left")

    ax.set_title(f"Дерево угроз AURORA PNT  [{label}]   "
                 f"({len(threats)} угроз, 5 сегментов)", fontsize=12, pad=10)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"cyber_threat_tree_{label}.png"), dpi=150)
    plt.close(fig)


# ── 2. Матрица рисков (P × I heatmap) ────────────────────────────────────────
def _plot_risk_matrix(threats, output_dir, label):
    fig, ax = plt.subplots(figsize=(11, 7))

    # Heatmap фон: 0..1 (P) x 0..10 (I)
    P_grid = np.linspace(0.0, 1.0, 80)
    I_grid = np.linspace(0.0, 10.0, 80)
    PP, II = np.meshgrid(P_grid, I_grid)
    risk = PP * II
    im = ax.imshow(risk, extent=[0, 1, 0, 10], origin="lower",
                   aspect="auto", cmap="RdYlGn_r", alpha=0.55)
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Сырой риск  P × I", fontsize=9)

    # Уровни риска
    ax.contour(PP, II, risk, levels=[2, 4, 6], colors="#2d3436",
               linewidths=0.6, alpha=0.5, linestyles=":")

    # Угрозы
    for t in threats:
        c = SEG_COLORS[t["segment"]]
        size = 100 + 40 * t["I"]
        ax.scatter(t["P"], t["I"], s=size, c=c, edgecolor="white",
                   linewidth=1.2, alpha=0.85, zorder=4)
        ax.annotate(t["id"], (t["P"], t["I"]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=7, color="#2d3436", fontweight="bold", zorder=5)

    # Легенда по сегментам
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=col, markeredgecolor="white",
                       markersize=10, label=seg)
               for seg, col in SEG_COLORS.items()]
    ax.legend(handles=handles, loc="upper left", fontsize=8, title="Сегмент")

    ax.set_xlabel("Вероятность P")
    ax.set_ylabel("Воздействие I")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 10)
    ax.set_title(f"Матрица рисков AURORA PNT (Probability × Impact)  [{label}]")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"cyber_risk_matrix_{label}.png"), dpi=150)
    plt.close(fig)


# ── 3. Контрмеры — bar эффективности ─────────────────────────────────────────
def _plot_countermeasures(output_dir, label):
    cm_ids   = list(COUNTERMEASURES.keys())
    names    = [f"{cid}: {COUNTERMEASURES[cid][0]}" for cid in cm_ids]
    effs     = [COUNTERMEASURES[cid][3] * 100.0    for cid in cm_ids]
    targets  = [COUNTERMEASURES[cid][2]            for cid in cm_ids]

    palette = ["#e17055", "#fdcb6e", "#0984e3", "#00b894", "#6c5ce7", "#74b9ff",
               "#fab1a0", "#ffeaa7", "#81ecec", "#55efc4", "#a29bfe", "#dfe6e9",
               "#d63031", "#e84393", "#00cec9"]
    colors = palette[:len(cm_ids)]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(names, effs, color=colors, edgecolor="white", height=0.62)
    for bar, e, tgt in zip(bars, effs, targets):
        ax.text(bar.get_width() + 0.6, bar.get_y() + bar.get_height() / 2,
                f"{e:.0f}%  → {tgt}", va="center", fontsize=8)

    ax.axvline(70, ls="--", color="#00b894", lw=1.3, label="70 % — целевое")
    ax.axvline(85, ls=":",  color="#6c5ce7", lw=1.3, label="85 % — высокая защита")
    ax.set_xlim(0, 105)
    ax.set_xlabel("Эффективность снижения риска, %")
    ax.set_title(f"Защитные меры AURORA PNT — эффективность  [{label}]")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"cyber_countermeasures_{label}.png"), dpi=150)
    plt.close(fig)


# ── 4. Остаточный риск (до / после) ──────────────────────────────────────────
def _plot_residual_risk(threats, output_dir, label):
    ids   = [t["id"]       for t in threats]
    raw   = np.array([t["raw"]      for t in threats])
    resd  = np.array([t["residual"] for t in threats])
    miti  = raw - resd
    colors_seg = [SEG_COLORS[t["segment"]] for t in threats]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Левый: стек-бар по каждой угрозе
    x = np.arange(len(ids))
    ax1.bar(x, resd, color=colors_seg, edgecolor="white", label="Остаточный риск")
    ax1.bar(x, miti, bottom=resd, color="#dfe6e9", edgecolor="white",
            alpha=0.85, label="Снято мерой")
    ax1.set_xticks(x)
    ax1.set_xticklabels(ids, rotation=45, fontsize=7)
    ax1.set_ylabel("Риск R = P × I")
    ax1.set_title(f"Снижение риска: до и после мер  [{label}]")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # Правый: суммарный риск по сегментам (до vs после)
    seg_keys = ["SIS", "Bort", "MCS", "TTC", "User"]
    seg_raw  = [sum(t["raw"]      for t in threats if t["segment"] == k) for k in seg_keys]
    seg_res  = [sum(t["residual"] for t in threats if t["segment"] == k) for k in seg_keys]
    bx = np.arange(len(seg_keys))
    w = 0.36
    ax2.bar(bx - w/2, seg_raw, w, color="#e17055", edgecolor="white", label="До мер")
    ax2.bar(bx + w/2, seg_res, w, color="#00b894", edgecolor="white", label="После мер")
    for i, (r, s) in enumerate(zip(seg_raw, seg_res)):
        red = (1 - s / r) * 100 if r > 0 else 0
        ax2.text(i, max(r, s) + 0.6, f"-{red:.0f}%", ha="center",
                  fontsize=9, color="#6c5ce7", fontweight="bold")
    ax2.set_xticks(bx)
    ax2.set_xticklabels(seg_keys)
    ax2.set_ylabel("Суммарный риск сегмента")
    ax2.set_title(f"Снижение риска по сегментам  [{label}]")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"cyber_residual_risk_{label}.png"), dpi=150)
    plt.close(fig)


# ── CSV ───────────────────────────────────────────────────────────────────────
def _save_csv(threats, output_dir, label):
    path = os.path.join(output_dir, f"cybersec_threats_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "segment", "category", "description",
                    "P", "I", "raw_PxI", "m_effectiveness",
                    "residual_R", "countermeasure_id", "countermeasure_name"])
        for t in threats:
            cm_name = COUNTERMEASURES[t["cm"]][0] if t["cm"] in COUNTERMEASURES else ""
            w.writerow([t["id"], t["segment"], t["category"], t["description"],
                        f"{t['P']:.2f}", f"{t['I']:.1f}", f"{t['raw']:.2f}",
                        f"{t['m']:.2f}", f"{t['residual']:.3f}",
                        t["cm"], cm_name])


def print_cybersec_summary(label: str, results: Dict) -> None:
    sep = "=" * 78
    print(f"\n{sep}")
    print(f"  Cybersecurity Threat Model -- {label}")
    print(sep)
    print(f"  Угроз всего:          {results['n_threats']}")
    print(f"  Контрмер:             {results['n_cms']}")
    print(f"  Сырой риск (P×I сум.):  {results['total_raw']:8.2f}")
    print(f"  Остаточный риск:        {results['total_residual']:8.2f}")
    print(f"  Снижение риска:         {results['risk_reduction']:8.1f}%")

    print(f"\n  {'Сегмент':<8} {'#угр':>5} {'Raw':>10} {'Resid':>10} {'-%':>8}")
    print(f"  {'─' * 50}")
    for seg, d in results["by_segment"].items():
        red = (1 - d["residual"] / d["raw"]) * 100 if d["raw"] > 0 else 0
        print(f"  {seg:<8} {d['n']:>5d} {d['raw']:>10.2f} "
              f"{d['residual']:>10.3f} {red:>7.1f}%")

    print(f"\n  Топ-5 остаточных рисков:")
    print(f"  {'ID':<6}{'Сегмент':<8}{'Категория':<18}{'R':>8}")
    print(f"  {'─' * 48}")
    for t in results["top5_residual"]:
        print(f"  {t['id']:<6}{t['segment']:<8}{t['category']:<18}{t['residual']:>8.3f}")
    print(sep)
