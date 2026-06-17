"""
Сравнительный анализ АВРОРА с аналогами.

Категории:
  1. Традиционные MEO GNSS: ГЛОНАСС, GPS Block III, Galileo, BeiDou-3
  2. Существующие / строящиеся LEO PNT:
       - Xona Space Systems PULSAR (США, коммерческий)
       - Satelles STL (на базе Iridium, 66 КА, 780 км)
       - ESA Moonlight / Galileo 2nd Gen LEO layer (ЕС, проект)
       - BeiDou-3 LEO augmentation (Китай, экспериментальный)
       - SpaceX Starlink PNT (потенциальный, FCC заявка)
       - Amazon Kuiper PNT (концептуальный)

Рассчитывает и визуализирует:
  - Сравнительную таблицу орбит, сигналов, точности, статусов
  - Spider-chart (лепестковая) по 6 ключевым метрикам
  - Временну́ю шкалу IOC/FOC по всем системам
  - URE / UERE budget comparison (bar chart)
  - PPP convergence time comparison
  - Покрытие vs Число КА (scatter)

Источники:
  GLONASS ICD (2016) v5.1; IS-GPS-200L (2020); Galileo OS-SIS-ICD 2.0 (2021);
  BDS-SIS-ICD-B1C (2017); Xona Space Systems (2023 white paper);
  Satelles STL ICD (2022); ESA Moonlight Phase A (2022);
  SpaceX FCC Filing 2020 (PNT); ITU-R M.2092-0 (2015).
"""

import math, os, csv
from typing import Dict, List
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches


# ─────────────────────────────────────────────────────────────────────────────
#  Справочные данные систем
# ─────────────────────────────────────────────────────────────────────────────

MEO_SYSTEMS: Dict[str, Dict] = {
    "GPS Block III": {
        "orbit_km": 20200, "inc_deg": 55.0, "n_sats": 31,
        "n_planes": 6, "signals": "L1 C/A, L1C, L2C, L5",
        "ure_m": 0.218, "uere_l1_m": 4.023, "uere_dual_m": 0.461,
        "cep_combined_m": 1.8,   # типовая гор. CEP50 (одиночная система, dual)
        "pdop_p95_global": 2.5, "pdop_p95_russia": 2.8,
        "ppp_conv_min": 25.0, "time_acc_ns": 20,
        "anti_spoof": "P(Y)-code, MNAV", "auth": "Нет (публичный)",
        "status": "Операционный", "ioc_year": 1993, "foc_year": 1995,
        "operator": "US DoD / Space Force", "cost_b_usd": 5.5,
        "color": "#e17055",
    },
    "ГЛОНАСС-K2": {
        "orbit_km": 19100, "inc_deg": 64.8, "n_sats": 24,
        "n_planes": 3, "signals": "L1OF/L2OF (FDMA), L1OC/L2OC/L3OC (CDMA)",
        "ure_m": 0.35, "uere_l1_m": 3.5, "uere_dual_m": 0.52,
        "cep_combined_m": 2.5,   # типовая гор. CEP50
        "pdop_p95_global": 3.0, "pdop_p95_russia": 2.0,
        "ppp_conv_min": 30.0, "time_acc_ns": 10,
        "anti_spoof": "GOVSS (засекр.)", "auth": "Нет (публичный)",
        "status": "Операционный/развитие", "ioc_year": 1993, "foc_year": 1995,
        "operator": "Роскосмос / ВКС России", "cost_b_usd": 2.5,
        "color": "#6c5ce7",
    },
    "Galileo (FOC)": {
        "orbit_km": 23222, "inc_deg": 56.0, "n_sats": 28,
        "n_planes": 3, "signals": "E1, E5a, E5b, E6",
        "ure_m": 0.163, "uere_l1_m": 2.524, "uere_dual_m": 0.363,
        "cep_combined_m": 1.5,   # типовая гор. CEP50
        "pdop_p95_global": 2.3, "pdop_p95_russia": 2.5,
        "ppp_conv_min": 22.0, "time_acc_ns": 15,
        "anti_spoof": "OSNMA (открытый, 2023)", "auth": "OSNMA",
        "status": "Операционный", "ioc_year": 2016, "foc_year": 2019,
        "operator": "ESA / EU GSA", "cost_b_usd": 10.0,
        "color": "#0984e3",
    },
    "BeiDou-3": {
        "orbit_km": 21528, "inc_deg": 55.0, "n_sats": 30,
        "n_planes": 3, "signals": "B1C, B2a, B3I, B2b (PPP-B2b)",
        "ure_m": 0.20, "uere_l1_m": 3.0, "uere_dual_m": 0.42,
        "pdop_p95_global": 2.4, "pdop_p95_russia": 2.6,
        "ppp_conv_min": 20.0, "time_acc_ns": 10,
        "anti_spoof": "Засекречен", "auth": "Нет",
        "status": "Операционный (2020)", "ioc_year": 2018, "foc_year": 2020,
        "operator": "CNSA / Китай", "cost_b_usd": 8.0,
        "color": "#fdcb6e",
    },
}

LEO_SYSTEMS: Dict[str, Dict] = {
    "АВРОРА\n(этот проект)": {
        "orbit_km": 1000, "inc_deg": 75.0, "n_sats": 300,
        "n_planes": 15, "signals": "Сервис А: L1/L5 (GPS-совм.); Сервис Б: L-полоса AURORA",
        # Канон §8/§65 (только проверенные числа): URE 0,45 м; UERE одночаст. иона-огранич.;
        # UERE dual 0,7 м (до 1,0 м у терминального КА). По одной псевдодальности —
        # сопоставимо с ГЛОНАСС, уступает GPS/Galileo; преимущество в геометрии комбинир. режима.
        "ure_m": 0.45, "uere_l1_m": 2.5, "uere_dual_m": 0.7,
        "cep_combined_m": 1.0,   # комбинир. LEO+ГЛОНАСС, CEP ≈1 м (§8.6) — геометрия
        "pdop_p95_global": 5.04, "pdop_p95_russia": 1.67,  # автоном. / комбинир. с ГЛОНАСС
        "ppp_conv_min": 1.0, "time_acc_ns": 5,
        "anti_spoof": "TESLA MAC (Стрибог)", "auth": "TESLA MAC (открытый)",
        "signal_power_dbw": -107, "ioc_year": 2029, "foc_year": 2033,  # Rx Сервис Б, §65
        "operator": "ШИВА НЕТВОРК (РФ)", "cost_b_usd": 2.8,
        "status": "Проект", "color": "#00b894",
    },
    "Xona PULSAR\n(США)": {
        "orbit_km": 700, "inc_deg": 60.0, "n_sats": 300,
        "n_planes": 12, "signals": "L1, L5 (совместим с GPS/Galileo)",
        "ure_m": 0.10, "uere_l1_m": 0.25, "uere_dual_m": 0.22,
        "cep_combined_m": 0.3,   # суб-метровая (быстрый PPP)
        "pdop_p95_global": 1.8, "pdop_p95_russia": 2.5,
        "ppp_conv_min": 0.2, "time_acc_ns": 3,
        "anti_spoof": "Navigation Message Authentication", "auth": "NMA",
        "signal_power_dbw": -108, "ioc_year": 2027, "foc_year": 2030,
        "operator": "Xona Space Systems (США)", "cost_b_usd": 3.0,
        "status": "Демо-спутники (2023)", "color": "#e17055",
    },
    "Satelles STL\n(Iridium, 66 КА)": {
        "orbit_km": 780, "inc_deg": 86.4, "n_sats": 66,
        "n_planes": 6, "signals": "L-band (~1626 МГц, Iridium)",
        "ure_m": 50.0, "uere_l1_m": 100.0, "uere_dual_m": 100.0,
        "cep_combined_m": 50.0,   # служба времени/грубое позиц.
        "pdop_p95_global": None, "pdop_p95_russia": None,
        "ppp_conv_min": None, "time_acc_ns": 100,
        "anti_spoof": "Засекречен (закрытый)", "auth": "Засекречен",
        "signal_power_dbw": -130, "ioc_year": 2016, "foc_year": 2019,
        "operator": "Satelles Inc. / Iridium (США)", "cost_b_usd": 0.5,
        "status": "Операционный (PNT-as-a-service)", "color": "#fdcb6e",
    },
    "ESA Moonlight\n(Galileo LEO layer)": {
        "orbit_km": 1200, "inc_deg": 63.4, "n_sats": 6,
        "n_planes": 3, "signals": "E1, E5a (augmentation)",
        "ure_m": 0.20, "uere_l1_m": 0.50, "uere_dual_m": 0.35,
        "pdop_p95_global": None, "pdop_p95_russia": None,
        "ppp_conv_min": 5.0, "time_acc_ns": 10,
        "anti_spoof": "OSNMA (Galileo)", "auth": "OSNMA",
        "signal_power_dbw": -115, "ioc_year": 2030, "foc_year": 2035,
        "operator": "ESA / EU", "cost_b_usd": 1.5,
        "status": "Фаза A (исследование)", "color": "#74b9ff",
    },
    "BeiDou-3 LEO\n(КНР, экспер.)": {
        "orbit_km": 1000, "inc_deg": 55.0, "n_sats": 12,
        "n_planes": 2, "signals": "B1C, B2a (LEO augment.)",
        "ure_m": 0.30, "uere_l1_m": 0.70, "uere_dual_m": 0.45,
        "pdop_p95_global": None, "pdop_p95_russia": None,
        "ppp_conv_min": 5.0, "time_acc_ns": 20,
        "anti_spoof": "Засекречен", "auth": "Нет",
        "signal_power_dbw": -118, "ioc_year": 2025, "foc_year": 2028,
        "operator": "CNSA / Китай", "cost_b_usd": 2.0,
        "status": "Экспериментальный", "color": "#a29bfe"},
    "SpaceX Starlink\n(PNT потенциал)": {
        "orbit_km": 550, "inc_deg": 53.0, "n_sats": 5000,
        "n_planes": 72, "signals": "Ku/Ka (широкополосный, PNT не запущен)",
        "ure_m": None, "uere_l1_m": None, "uere_dual_m": None,
        "pdop_p95_global": None, "pdop_p95_russia": None,
        "ppp_conv_min": None, "time_acc_ns": None,
        "anti_spoof": "Не определён", "auth": "Не определён",
        "signal_power_dbw": -100, "ioc_year": None, "foc_year": None,
        "operator": "SpaceX (США)", "cost_b_usd": 10.0,
        "status": "FCC заявка (PNT), нет публ. сигналов", "color": "#636e72"},
}


# ─────────────────────────────────────────────────────────────────────────────
#  Китайские LEO-инициативы (по открытым публикациям, статус на 2026)
#  Источники: PMC10301026 (CentiSpace); geely.com / SpaceNews (Geespace);
#  en.wikipedia.org/Qianfan, spacenews.com (Qianfan/Guowang); ITU filings 2020.
#  ВНИМАНИЕ: ТТХ ориентировочные; точные значения у операторов закрыты.
# ─────────────────────────────────────────────────────────────────────────────
CHINA_LEO: List[Dict] = [
    {
        "name": "CentiSpace\n(未来导航)",
        "operator": "Beijing Future Navigation + CETC-29",
        "orbit_km": "≈975",  "inc": "55°/30° (накл.+поляр.)",
        "in_orbit": "неск. эксп. КА (с 2018)", "plan": "≈160",
        "role": "Целевое навиг.-усиление (прямой аналог АВРОРА)",
        "note": "цель <10 см, PPP ≈1 мин; CCST-подавление самопомех (приём GNSS + передача в той же полосе)",
        "color": "#d63031",
    },
    {
        "name": "Geespace\n(GEESATCOM)",
        "operator": "Geely / Geespace",
        "orbit_km": "600",   "inc": "50°",
        "in_orbit": "64 (Фаза I)", "plan": "72 → ~240",
        "role": "Связь/IoT + высокоточное позиц. (доп.)",
        "note": "см-уровень через наземное усиление; для автономного транспорта",
        "color": "#e17055",
    },
    {
        "name": "Qianfan / 千帆\n(G60)",
        "operator": "SSST / Spacesail",
        "orbit_km": "≈1160", "inc": "—",
        "in_orbit": "≈180 (05.2026)", "plan": "≈14 000–15 000 (2030)",
        "role": "Широкополосный интернет; PNT-потенциал",
        "note": "аналог Starlink PNT — навигация не основная функция",
        "color": "#fdcb6e",
    },
    {
        "name": "Guowang / 国网",
        "operator": "China SatNet (гос.)",
        "orbit_km": "<500 / 600–1145", "inc": "—",
        "in_orbit": "неск. десятков", "plan": "ITU 12 992; 400 к 2027",
        "role": "Широкополосный интернет (гос.); PNT-потенциал",
        "note": "мегагруппировка; навигация не основная функция",
        "color": "#a29bfe",
    },
]


def run_competitor_analysis(output_dir: str, label: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    _plot_constellation_scatter(output_dir, label)
    _plot_spider_chart(output_dir, label)
    _plot_ure_comparison(output_dir, label)
    _plot_ppp_convergence_comparison(output_dir, label)
    _plot_timeline(output_dir, label)
    _plot_glonass_deep(output_dir, label)
    _plot_china_leo(output_dir, label)
    _save_csv(output_dir, label)

    return {
        "meo_systems": list(MEO_SYSTEMS.keys()),
        "leo_systems": list(LEO_SYSTEMS.keys()),
        "china_leo": [c["name"].replace("\n", " ") for c in CHINA_LEO],
    }


def _plot_constellation_scatter(output_dir, label):
    """Scatter: число КА vs высота орбиты — позиционирование всех систем."""
    fig, ax = plt.subplots(figsize=(13, 7))

    # Смещения подписей (pt) — разводим слипающиеся точки
    # (Xona/AURORA на ~1000 км; кластер MEO на ~20 000 км)
    _OFF = {
        "АВРОРА":   (12, -17, "left"),
        "Xona":     (-12, 13, "right"),
        "Starlink": (12, 6,   "left"),
        "Satelles": (12, 4,   "left"),
        "BeiDou-3": (12, 4,   "left"),
        "Moonlight":(12, 4,   "left"),
        "GPS":      (10, 15,  "left"),
        "Galileo":  (10, -17, "left"),
        "BeiDou":   (10, 1,   "left"),
        "ГЛОНАСС":  (-10,-17, "right"),
    }
    def _off(name):
        for k, v in _OFF.items():
            if k in name:
                return v
        return (8, 3, "left")

    for name, s in MEO_SYSTEMS.items():
        ax.scatter(s["orbit_km"], s["n_sats"], s=180, color=s["color"],
                   marker="o", zorder=5, edgecolors="white", linewidths=1.5)
        dx, dy, ha = _off(name)
        ax.annotate(name, (s["orbit_km"], s["n_sats"]),
                    textcoords="offset points", xytext=(dx, dy), fontsize=8, ha=ha)

    for name, s in LEO_SYSTEMS.items():
        if s["n_sats"]:
            marker = "^" if "АВРОРА" in name else "s"
            size = 260 if "АВРОРА" in name else 160
            ax.scatter(s["orbit_km"], s["n_sats"], s=size, color=s["color"],
                       marker=marker, zorder=5, edgecolors="white", linewidths=1.5)
            n = name.replace("\n", " ")
            dx, dy, ha = _off(n)
            ax.annotate(n, (s["orbit_km"], s["n_sats"]),
                        textcoords="offset points", xytext=(dx, dy), fontsize=8, ha=ha)

    ax.axvspan(500, 2000, alpha=0.05, color="#00b894", label="Зона LEO PNT (500–2000 км)")
    ax.axvspan(18000, 26000, alpha=0.05, color="#e17055", label="Зона MEO GNSS (18–26 тыс. км)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Высота орбиты (км)")
    ax.set_ylabel("Число спутников")
    ax.set_title(f"АВРОРА — Позиционирование на рынке GNSS/LEO PNT [{label}]")

    meo_patch = mpatches.Patch(color="#e17055", alpha=0.7, label="MEO GNSS")
    leo_patch = mpatches.Patch(color="#00b894", alpha=0.7, label="LEO PNT")
    ax.legend(handles=[meo_patch, leo_patch], fontsize=9)
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"comp_constellation_scatter_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_spider_chart(output_dir, label):
    """Лепестковая диаграмма (spider/radar) — сравнение по 6 метрикам."""
    categories = ["Точность дальности\n(1/UERE)", "Точность позиц.\n(CEP, комбин.)",
                  "PPP\nсходимость", "Покрытие\nАрктика",
                  "Стойкость к\nпомехам", "Целостность\n(RAIM)", "Временна́я\nточность"]
    N = len(categories)

    def score(sys_name, sys_data, is_leo=False):
        # Нормированные оценки 0–10
        uere = sys_data.get("uere_dual_m", 1.0) or 1.0
        accuracy = min(10, 10 * (0.5 / uere))   # точность ОДНОЙ псевдодальности (АВРОРА ~7,1)
        cep = sys_data.get("cep_combined_m", 5.0) or 5.0
        positioning = min(10, 9.0 / cep)        # ИТОГОВАЯ гор. точность, рабочий режим (АВРОРА комбин. ~1 м → 9)

        conv = sys_data.get("ppp_conv_min", 30.0) or 30.0
        convergence = min(10, 10 * (5.0 / conv))  # АВРОРА ~10, GPS ~1.7

        inc = sys_data.get("inc_deg", 55.0)
        arctic = min(10, inc / 7.5)  # ГЛОНАСС 64.8° → 8.6; АВРОРА 75° → 10

        sig_pow = sys_data.get("signal_power_dbw", -130)
        anti_jam = min(10, (sig_pow + 107) * 0.5 + 5) if sig_pow else 3.0  # АВРОРА -107 → 10, GPS -130 → 3.5

        n_sats = sys_data.get("n_sats", 24)
        integrity = min(10, math.log(n_sats, 3))  # больше спутников → лучше RAIM

        t_ns = sys_data.get("time_acc_ns", 30) or 30
        timing = min(10, 10 * (5.0 / t_ns))  # АВРОРА 5 нс → 10; GPS 20 нс → 2.5

        return [accuracy, positioning, convergence, arctic, anti_jam, integrity, timing]

    systems_to_plot = {
        "АВРОРА": (LEO_SYSTEMS["АВРОРА\n(этот проект)"], "#00b894", "-"),
        "Xona PULSAR": (LEO_SYSTEMS["Xona PULSAR\n(США)"], "#e17055", "--"),
        "ГЛОНАСС-K2": (MEO_SYSTEMS["ГЛОНАСС-K2"], "#6c5ce7", "-."),
        "GPS Block III": (MEO_SYSTEMS["GPS Block III"], "#e17055", ":"),
        "Galileo": (MEO_SYSTEMS["Galileo (FOC)"], "#0984e3", "--"),
        "Satelles STL": (LEO_SYSTEMS["Satelles STL\n(Iridium, 66 КА)"], "#fdcb6e", ":"),
    }

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    for sys_name, (sys_data, color, ls) in systems_to_plot.items():
        values = score(sys_name, sys_data)
        values += values[:1]
        lw = 2.8 if "АВРОРА" in sys_name else 1.8
        ax.plot(angles, values, color=color, linewidth=lw, linestyle=ls,
                marker="o", markersize=4, label=sys_name.replace("\n", " "))
        ax.fill(angles, values, color=color, alpha=0.06)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.tick_params(axis="x", pad=14)           # отодвинуть подписи категорий от края
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=7)
    ax.set_title(f"АВРОРА — Spider-диаграмма сравнения систем [{label}]",
                 pad=24, fontsize=12)
    # легенда снизу в 3 колонки — не перекрывает диаграмму и не обрезается
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3,
              fontsize=9, frameon=True)
    fig.subplots_adjust(left=0.12, right=0.88, top=0.86, bottom=0.16)
    fig.savefig(os.path.join(output_dir, f"comp_spider_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_ure_comparison(output_dir, label):
    """Сравнение URE и UERE dual-freq по всем системам."""
    systems = {}
    for n, s in MEO_SYSTEMS.items():
        systems[n.replace("\n", " ")] = s
    for n, s in LEO_SYSTEMS.items():
        if s.get("ure_m") is not None:
            systems[n.replace("\n", " ")] = s

    names  = list(systems.keys())
    ures   = [systems[n].get("ure_m", 0) or 0 for n in names]
    ueres  = [systems[n].get("uere_dual_m", 0) or 0 for n in names]
    colors = [systems[n].get("color", "#b2bec3") for n in names]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(14, 6))
    b1 = ax.bar(x - 0.2, ures,  0.35, label="URE (м)", color=colors, alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + 0.2, ueres, 0.35, label="UERE dual-freq (м)", color=colors, alpha=0.5,
                edgecolor=colors, linewidth=2)
    for bar, v in zip(b1, ures):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", fontsize=7, fontweight="bold")
    for bar, v in zip(b2, ueres):
        if v:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([n.split("(")[0].strip().replace(" ", "\n") for n in names],
                        fontsize=8, rotation=0)
    ax.set_ylabel("Ошибка дальности 1σ (м)")
    ax.set_title(f"АВРОРА — Сравнение URE и UERE: все системы [{label}]")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"comp_ure_all_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_ppp_convergence_comparison(output_dir, label):
    """Время сходимости PPP по системам (bar)."""
    conv_data = {
        "GPS Block III": (25.0, "#e17055"),
        "ГЛОНАСС-K2":   (30.0, "#6c5ce7"),
        "Galileo":       (22.0, "#0984e3"),
        "BeiDou-3":      (20.0, "#fdcb6e"),
        "GPS+ГЛОНАСС":   (18.0, "#b2bec3"),
        "BDS LEO aug.":  (5.0,  "#a29bfe"),
        "ESA Moonlight": (5.0,  "#74b9ff"),
        "Satelles STL":  (None, "#dfe6e9"),
        "Xona PULSAR":   (0.2,  "#e17055"),
        "АВРОРА":    (1.0,  "#00b894"),
    }
    names  = [n for n, (t, c) in conv_data.items() if t is not None]
    times  = [conv_data[n][0] for n in names]
    colors = [conv_data[n][1] for n in names]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.barh(names, times, color=colors, edgecolor="white", height=0.6)
    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f"{t:.1f} мин", va="center", fontsize=9,
                fontweight="bold" if t < 5 else "normal")
    ax.axvline(5.0,  ls="--", color="#00b894", lw=1.5, label="5 мин (цель LEO PNT)")
    ax.axvline(20.0, ls=":",  color="#e17055", lw=1.5, label="20 мин (типов. MEO PPP)")
    ax.set_xlabel("Время сходимости PPP (мин, σ_H < 10 см)")
    ax.set_title(f"АВРОРА — Время сходимости PPP: все системы [{label}]")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"comp_ppp_convergence_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_timeline(output_dir, label):
    """Временна́я шкала IOC/FOC всех систем LEO PNT."""
    timeline = {
        "Satelles STL":         (2016, 2019, "#fdcb6e"),
        "BeiDou-3 LEO (экспер.)": (2023, 2028, "#a29bfe"),
        "Xona PULSAR":          (2027, 2030, "#e17055"),
        "ESA Moonlight":        (2030, 2035, "#74b9ff"),
        "АВРОРА":           (2029, 2033, "#00b894"),
        "SpaceX Starlink PNT":  (None, None, "#636e72"),
    }

    fig, ax = plt.subplots(figsize=(12, 5))
    y_pos = list(range(len(timeline)))
    current_year = 2026

    for i, (name, (ioc, foc, color)) in enumerate(timeline.items()):
        if ioc and foc:
            ax.barh(i, foc - ioc, left=ioc, height=0.4, color=color,
                    alpha=0.8, edgecolor="white")
            ax.text(ioc + 0.1, i, f"IOC {ioc}", va="center", fontsize=8)
            ax.text(foc + 0.1, i, f"FOC {foc}", va="center", fontsize=8)
        elif not ioc:
            ax.text(2035, i, "Дата не определена", va="center", fontsize=8,
                    color="#636e72", style="italic")

    ax.axvline(current_year, ls="--", color="#2d3436", lw=2, label=f"Сегодня ({current_year})")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(list(timeline.keys()), fontsize=9)
    ax.set_xlabel("Год")
    ax.set_title(f"АВРОРА — Временна́я шкала LEO PNT систем [{label}]")
    ax.legend(fontsize=9)
    ax.set_xlim(2014, 2040)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"comp_timeline_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_glonass_deep(output_dir, label):
    """Детальное сравнение АВРОРА vs ГЛОНАСС по ключевым параметрам."""
    metrics = {
        "URE (м)":                  (0.35,  0.45,   True),   # (ГЛОНАСС, АВРОРА, меньше=лучше)
        "UERE dual (м)":            (0.52,  0.70,   True),   # по псевдодальн. сопоставимо/уступает
        "PDOP p95 (Россия)":        (2.0,   1.67,   True),   # комбинир. LEO+ГЛОНАСС
        "PPP конверг. (мин)":       (30.0,  1.0,    True),
        "N_vis (среднее)":          (6.0,   14.0,   False),
        "Сигнал Б +дБ (vs MEO)":    (0.0,   23.0,   False),  # Сервис Б, выделенная полоса §65
        "Вр. точность (нс)":        (10.0,  5.0,    True),
        "Срок орб. жизни (лет)":    (7.0,   7.0,    False),
        "Покрытие Арктики":         (9.0,   10.0,   False),
        "Аутентификация":           (0.0,   10.0,   False),
    }

    names_m = list(metrics.keys())
    glo_vals = [v[0] for v in metrics.values()]
    aur_vals = [v[1] for v in metrics.values()]
    less_is_better = [v[2] for v in metrics.values()]

    # Нормализовать к 0–10 (больше = лучше АВРОРА)
    def normalize(g, a, lib):
        mx = max(g, a)
        if mx == 0:
            return 5.0, 5.0
        if lib:  # меньше = лучше
            gn = 10 * (1 - g / (mx * 1.2))
            an = 10 * (1 - a / (mx * 1.2))
        else:
            gn = 10 * g / (mx * 1.1)
            an = 10 * a / (mx * 1.1)
        return max(0, gn), max(0, an)

    glo_norm = []
    aur_norm = []
    for g, a, lib in zip(glo_vals, aur_vals, less_is_better):
        gn, an = normalize(g, a, lib)
        glo_norm.append(gn)
        aur_norm.append(an)

    x = np.arange(len(names_m))
    fig, (ax_bar, ax_raw) = plt.subplots(1, 2, figsize=(16, 7))

    w = 0.35
    b1 = ax_bar.bar(x - w/2, glo_norm, w, label="ГЛОНАСС-K2", color="#6c5ce7",
                    alpha=0.85, edgecolor="white")
    b2 = ax_bar.bar(x + w/2, aur_norm, w, label="АВРОРА", color="#00b894",
                    alpha=0.85, edgecolor="white")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(names_m, rotation=28, ha="right", fontsize=8)
    ax_bar.set_ylabel("Нормированный балл (10 = лучший)")
    ax_bar.set_title(f"АВРОРА vs ГЛОНАСС-K2 — нормир. баллы [{label}]")
    ax_bar.legend(fontsize=9)
    ax_bar.grid(axis="y", alpha=0.3)
    ax_bar.set_ylim(0, 12)

    # Правый график — абсолютные значения
    table_data = []
    for nm, gv, av, lib in zip(names_m, glo_vals, aur_vals, less_is_better):
        better = "АВРОРА" if (av < gv and lib) or (av > gv and not lib) else ("Равно" if av == gv else "ГЛОНАСС")
        table_data.append([nm, f"{gv}", f"{av}", better])

    ax_raw.axis("off")
    col_labels = ["Метрика", "ГЛОНАСС-K2", "АВРОРА", "Победитель"]
    tbl = ax_raw.table(cellText=table_data, colLabels=col_labels,
                       cellLoc="center", loc="center",
                       colWidths=[0.38, 0.18, 0.18, 0.18])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2d3436")
            cell.set_text_props(color="white", fontweight="bold")
        elif col == 3:
            val = table_data[row-1][3] if row > 0 else ""
            if val == "АВРОРА":
                cell.set_facecolor("#d4f5e9")
            elif val == "ГЛОНАСС":
                cell.set_facecolor("#e8e0f5")
    ax_raw.set_title(f"Сравнение АВРОРА vs ГЛОНАСС-K2 [{label}]", fontsize=10, pad=15)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"comp_aurora_vs_glonass_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_china_leo(output_dir, label):
    """Обзор китайских LEO-инициатив (таблица-рисунок, по открытым источникам)."""
    fig, ax = plt.subplots(figsize=(15, 6.0))
    ax.axis("off")

    col_labels = ["Система", "Оператор", "Орбита, км",
                  "КА на орбите / план", "Назначение и PNT-роль"]
    rows, row_colors = [], []
    for c in CHINA_LEO:
        rows.append([
            c["name"].replace("\n", " "),
            c["operator"],
            f"{c['orbit_km']}\n({c['inc']})",
            f"{c['in_orbit']}\n/ {c['plan']}",
            f"{c['role']}.\n{c['note']}",
        ])
        row_colors.append(c["color"])

    tbl = ax.table(cellText=rows, colLabels=col_labels, cellLoc="left",
                   loc="center", colWidths=[0.13, 0.18, 0.14, 0.17, 0.38])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 2.6)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_text_props(va="center")
        if row == 0:
            cell.set_facecolor("#2d3436")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            if col == 0:
                cell.set_facecolor(row_colors[row - 1])
                cell.set_text_props(color="white", fontweight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#f5f6fa")

    ax.set_title(
        f"Китайские LEO-инициативы с PNT-функцией — обзор по открытым источникам "
        f"(статус на 2026) [{label}]", fontsize=11, pad=16)
    fig.text(0.5, 0.02,
             "Прямой аналог концепции АВРОРА — только CentiSpace (целевое навиг.-усиление). "
             "Geespace — связь+позиционирование; Qianfan/Guowang — широкополосные мегагруппировки "
             "с PNT-потенциалом. ТТХ ориентировочные (данные операторов закрыты).",
             ha="center", fontsize=7.5, style="italic", color="#636e72", wrap=True)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.10)
    fig.savefig(os.path.join(output_dir, f"comp_china_leo_{label}.png"), dpi=150)
    plt.close(fig)


def _save_csv(output_dir, label):
    path = os.path.join(output_dir, f"competitor_analysis_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "type", "orbit_km", "inc_deg", "n_sats",
                    "ure_m", "uere_dual_m", "pdop_p95_global", "ppp_conv_min",
                    "time_acc_ns", "ioc_year", "foc_year", "status", "operator"])
        for name, s in MEO_SYSTEMS.items():
            w.writerow([name, "MEO", s["orbit_km"], s["inc_deg"], s["n_sats"],
                        s["ure_m"], s["uere_dual_m"], s["pdop_p95_global"],
                        s["ppp_conv_min"], s["time_acc_ns"],
                        s["ioc_year"], s["foc_year"], s["status"], s["operator"]])
        for name, s in LEO_SYSTEMS.items():
            nm = name.replace("\n", " ")
            w.writerow([nm, "LEO", s["orbit_km"], s["inc_deg"], s["n_sats"],
                        s.get("ure_m"), s.get("uere_dual_m"), s.get("pdop_p95_global"),
                        s.get("ppp_conv_min"), s.get("time_acc_ns"),
                        s.get("ioc_year"), s.get("foc_year"), s["status"], s["operator"]])


def print_competitor_summary(label: str, result: Dict) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  Competitor Analysis -- {label}")
    print(sep)
    print(f"\n  MEO GNSS ({len(result['meo_systems'])} систем):")
    for n, s in MEO_SYSTEMS.items():
        print(f"    {n:<20} URE={s['ure_m']:.3f}м  UERE_DF={s['uere_dual_m']:.3f}м  "
              f"PPP={s['ppp_conv_min']:.0f}мин  [{s['status']}]")
    print(f"\n  LEO PNT ({len(result['leo_systems'])} систем):")
    for n, s in LEO_SYSTEMS.items():
        nm = n.replace("\n", " ")
        ure  = f"{s['ure_m']:.3f}м" if s.get("ure_m") else "—"
        conv = f"{s['ppp_conv_min']:.1f}мин" if s.get("ppp_conv_min") else "—"
        print(f"    {nm:<35} URE={ure:<8}  PPP={conv:<10}  [{s['status']}]")
    print(sep)
