"""
Assisted-GNSS (A-GPS / A-GNSS) Server Analysis for AURORA PNT.

Модель серверного ассистирования: какие данные сервер передаёт приёмнику по
мобильному/интернет-каналу, чтобы сократить холодный TTFF (Time-To-First-Fix)
со ~149 с до < 15 с.

Без ассистирования холодный старт тратит время на:
  - слепой поиск сигнала по всему доплеровскому/кодовому пространству;
  - сбор эфемерид (и альманаха) по медленному навигационному каналу (250 бит/с).

Сервер A-GNSS передаёт по быстрому каналу (LTE/интернет): приближённые время и
координаты (от соты), список видимых КА с прогнозом доплера, действующие
эфемериды и альманах. Это сужает поиск и убирает ожидание скачивания, оставляя
лишь захват + измерение + решение.

References:
  3GPP TS 36.355 (LPP — LTE Positioning Protocol), A-GNSS assistance data.
  Kaplan & Hegarty, "Understanding GPS/GNSS", 3rd ed., Ch. acquisition/TTFF.
"""

import os
import csv
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Параметры сигнала и приёмника ─────────────────────────────────────────────
NAV_PAYLOAD_BPS   = 250.0     # полезная скорость навигационного сообщения, бит/с
LTE_BPS           = 1_000_000 # ассистирующий канал (мобильный), бит/с

# Объёмы данных навигационного сообщения (см. §20)
EPHEM_BITS_PER_SV = 527       # эфемерида одного КА
ALMANAC_BITS      = 25_920    # полный альманах (180 КА)
REFPOS_TIME_BITS  = 200       # приближённые время + координаты от соты

# ── Время захвата (с) ─────────────────────────────────────────────────────────
# Холодный: слепой поиск по всему доплеру ±42 кГц (§42) и неизвестной видимости,
# с накоплением для чувствительности. A-GNSS: доплер сужен прогнозом до ±0.5 кГц,
# код — приближённым временем, известны видимые PRN → захват на порядок быстрее.
T_ACQ_COLD_S      = 33.0
T_ACQ_AGPS_S      = 3.0
N_VIS             = 14         # видимых КА (с ассистированием известны)
T_MEAS_FIX_S      = 4.0        # измерение псевдодальностей + первое решение


def ttff_breakdown() -> Dict:
    """TTFF по составляющим: холодный (без ассистирования) и A-GNSS."""
    # Холодный старт: слепой поиск + сбор данных по нав-каналу (250 бит/с) + решение
    t_data_cold = (4 * EPHEM_BITS_PER_SV + ALMANAC_BITS) / NAV_PAYLOAD_BPS
    cold = {
        "acq_s":   T_ACQ_COLD_S,
        "data_s":  t_data_cold,
        "fix_s":   T_MEAS_FIX_S,
        "total_s": T_ACQ_COLD_S + t_data_cold + T_MEAS_FIX_S,
    }

    # A-GNSS: известны видимые КА и доплер → узкий поиск; данные по LTE
    t_acq_agps = T_ACQ_AGPS_S
    assist_bits = 8 * EPHEM_BITS_PER_SV + ALMANAC_BITS + REFPOS_TIME_BITS
    t_data_agps = assist_bits / LTE_BPS
    agps = {
        "acq_s":   t_acq_agps,
        "data_s":  t_data_agps,
        "fix_s":   T_MEAS_FIX_S,
        "total_s": t_acq_agps + t_data_agps + T_MEAS_FIX_S,
        "assist_kbytes": assist_bits / 8 / 1024,
    }
    return {"cold": cold, "agps": agps,
            "speedup": cold["total_s"] / agps["total_s"]}


def run_agps_server_analysis(output_dir: str, label: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    res = ttff_breakdown()
    c, a = res["cold"], res["agps"]

    # ── График: разбивка TTFF ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    cats = ["Захват", "Скачивание данных", "Измерение + решение"]
    cold_vals = [c["acq_s"], c["data_s"], c["fix_s"]]
    agps_vals = [a["acq_s"], a["data_s"], a["fix_s"]]
    colors = ["#0984e3", "#e17055", "#00b894"]

    bottoms_c = 0.0
    bottoms_a = 0.0
    for k, (cv, av, col) in enumerate(zip(cold_vals, agps_vals, colors)):
        ax.bar("Холодный\n(без A-GNSS)", cv, bottom=bottoms_c, color=col,
               edgecolor="white", label=cats[k])
        ax.bar("A-GNSS\n(с сервером)", av, bottom=bottoms_a, color=col,
               edgecolor="white")
        bottoms_c += cv
        bottoms_a += av

    ax.text(0, c["total_s"] + 3, f"{c['total_s']:.0f} с", ha="center",
            fontweight="bold", fontsize=12)
    ax.text(1, a["total_s"] + 3, f"{a['total_s']:.1f} с", ha="center",
            fontweight="bold", fontsize=12, color="#00b894")
    ax.axhline(15, ls="--", color="#636e72", lw=1.2)
    ax.text(1.35, 16, "цель < 15 с", color="#636e72", fontsize=9)
    ax.set_ylabel("TTFF (с)")
    ax.set_title(f"Сокращение TTFF ассистирующим сервером (A-GNSS) [{label}]")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"agps_ttff_{label}.png"), dpi=150)
    plt.close(fig)

    # ── CSV ──────────────────────────────────────────────────────────────────
    with open(os.path.join(output_dir, f"agps_server_{label}.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mode", "acq_s", "data_s", "fix_s", "total_s"])
        w.writerow(["cold", f"{c['acq_s']:.1f}", f"{c['data_s']:.1f}",
                    f"{c['fix_s']:.1f}", f"{c['total_s']:.1f}"])
        w.writerow(["agps", f"{a['acq_s']:.2f}", f"{a['data_s']:.3f}",
                    f"{a['fix_s']:.1f}", f"{a['total_s']:.1f}"])

    print(f"  A-GNSS сервер -- {label}")
    print(f"    Холодный TTFF:  {c['total_s']:.0f} с "
          f"(захват {c['acq_s']:.0f} + данные {c['data_s']:.0f} + решение {c['fix_s']:.0f})")
    print(f"    A-GNSS TTFF:    {a['total_s']:.1f} с "
          f"(захват {a['acq_s']:.1f} + данные {a['data_s']:.2f} + решение {a['fix_s']:.0f})")
    print(f"    Ускорение:      {res['speedup']:.0f}x; объём ассистирования {a['assist_kbytes']:.1f} КБ")
    return res


if __name__ == "__main__":
    run_agps_server_analysis("results/acquisition", "phase4")
