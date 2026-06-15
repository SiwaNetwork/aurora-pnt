"""
Двухсервисная архитектура сигналов АВРОРА: открытый (А) и защищённый (Б).

Одна группировка (1000 км) формирует ДВА независимых сервиса с разными полосами и
регуляторными режимами по плотности потока мощности (ПФП):

  Сервис А — открытый, RNSS-совместимый (L1/L5, наложение на полосу GPS/Galileo):
    ПФП ограничена маской МСЭ-R (−121,5 дБВт/м²), поэтому преимущество по мощности
    регуляторно ограничено ≈ +10 дБ. Назначение: совместимость с GNSS-приёмниками,
    комбинированный режим с ГЛОНАСС, массовый рынок.

  Сервис Б — защищённый, в ВЫДЕЛЕННОЙ полосе АВРОРА (национальная координация
    ГКРЧ/МСЭ, первичный статус): в своей полосе допустима бо́льшая ПФП, что даёт
    +23 дБ (×200) — как у Xona (+20 дБ) и Iridium STL (+30 дБ), но БЕЗ снижения
    орбиты. Назначение: помехозащита, работа в помещениях, аутентифицированный
    суверенный PNT для критических потребителей (приёмники АВРОРА).

Уникальность: конкуренты дают ЛИБО мощный сигнал в своей полосе (Xona/Iridium),
ЛИБО GNSS-совместимость (MEO) — АВРОРА даёт ОБА с одной группировки.

References:
  Iridium STL (Satelles): +30 дБ, 780 км, MSS-полоса; Xona PULSAR: +20 дБ (×100),
  ~550 км, собственная L/C-полоса; §10.2 (линк-бюджет), §43 (ПФП/МСЭ).
"""

import os
import csv
import math
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Геометрия и эталон ────────────────────────────────────────────────────────
FSPL_ZENITH_DB = 156.4         # потери трассы L1, зенит, 1000 км (§10.2)
N0_DBM_HZ      = -172.0        # спектр. плотность шума ПРМ (§10.2)
MEO_RX_DBM     = -130.0        # принимаемая мощность MEO-ГНСС (ГЛОНАСС, §5.4)
ITU_PFD_LIMIT  = -121.5        # маска ПФП МСЭ-R в RNSS-полосе, дБВт/м²
PFD_GEOM_DB    = 131.0         # 10·log10(4πR²), R=1000 км

# ── Два сервиса: задаём передатчик и антенну (рациональный выбор) ──────────────
# EIRP[дБВт] = P_tx[дБВт] + G_ant[дБи] − потери фидера
SERVICES = {
    "А (открытый, RNSS)": {
        "p_tx_w": 5.0, "g_ant_dbi": 3.5, "feed_db": 1.5,
        "band": "L1/L5 (RNSS, совм. с GPS)", "regime": "маска МСЭ −121,5",
        "col": "#0984e3",
    },
    "Б (защищённый, своя полоса)": {
        "p_tx_w": 30.0, "g_ant_dbi": 8.0, "feed_db": 1.5,
        "band": "выделенная L-полоса АВРОРА", "regime": "первичный статус",
        "col": "#0652DD",
    },
}

# Конкуренты (опубликованные значения, см. References)
COMPETITORS = {
    "GPS/ГЛОНАСС (MEO)": 0.0,
    "АВРОРА А": None,        # вычисляется
    "Xona PULSAR": 20.0,
    "АВРОРА Б": None,        # вычисляется
    "Iridium STL": 30.0,
}


def _eirp_dbw(p_tx_w: float, g_ant_dbi: float, feed_db: float) -> float:
    p_tx_dbw = 10.0 * math.log10(p_tx_w)
    return p_tx_dbw + g_ant_dbi - feed_db


def compute() -> Dict:
    out = {}
    for name, s in SERVICES.items():
        eirp = _eirp_dbw(s["p_tx_w"], s["g_ant_dbi"], s["feed_db"])
        rx_dbm = eirp + 30.0 - FSPL_ZENITH_DB - 0.5 - 1.5    # дБВт→дБм, атмо, потери ПРМ
        pfd = eirp - PFD_GEOM_DB
        adv = rx_dbm - MEO_RX_DBM
        cn0 = rx_dbm - N0_DBM_HZ
        jam_x = 10 ** (adv / 10.0)
        out[name] = {"eirp_dbw": eirp, "rx_dbm": rx_dbm, "pfd": pfd,
                     "adv_db": adv, "cn0": cn0, "jam_x": jam_x,
                     "itu_ok": pfd <= ITU_PFD_LIMIT, **s}
    return out


def run_dual_service_analysis(output_dir: str, label: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    res = compute()
    adv_a = res["А (открытый, RNSS)"]["adv_db"]
    adv_b = res["Б (защищённый, своя полоса)"]["adv_db"]

    comp = dict(COMPETITORS)
    comp["АВРОРА А"] = adv_a
    comp["АВРОРА Б"] = adv_b

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.3))

    # ── Панель 1: преимущество по сигналу vs GPS (дБ) — АВРОРА и конкуренты ───
    names = list(comp.keys())
    vals = [comp[n] for n in names]
    cols = []
    for n in names:
        if n == "АВРОРА Б":   cols.append("#0652DD")
        elif n == "АВРОРА А": cols.append("#0984e3")
        elif "MEO" in n:      cols.append("#b2bec3")
        else:                 cols.append("#e17055")
    bars = ax1.bar(names, vals, color=cols, edgecolor="white")
    for rect, v in zip(bars, vals):
        ax1.text(rect.get_x() + rect.get_width() / 2, v + 0.4,
                 f"+{v:.0f} дБ", ha="center", fontweight="bold", fontsize=9)
    ax1.set_ylabel("Преимущество по сигналу над GPS, дБ")
    ax1.set_title("Сила сигнала: АВРОРА (2 сервиса) vs конкуренты")
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=18, ha="right", fontsize=8)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim(0, 34)

    # ── Панель 2: ПФП и маска МСЭ — А комплаентен, Б в своей полосе ───────────
    snames = list(res.keys())
    pfds = [res[n]["pfd"] for n in snames]
    pcols = [res[n]["col"] for n in snames]
    bars2 = ax2.bar([s.split(" ")[0] for s in snames], pfds, color=pcols,
                    edgecolor="white", width=0.5)
    for rect, n in zip(bars2, snames):
        r = res[n]
        ax2.text(rect.get_x() + rect.get_width() / 2, r["pfd"] + 0.5,
                 f"{r['pfd']:.0f}\n{r['regime']}", ha="center", fontsize=7)
    ax2.axhline(ITU_PFD_LIMIT, ls="--", color="#d63031", lw=1.5)
    ax2.text(1.4, ITU_PFD_LIMIT + 0.5, "маска МСЭ RNSS −121,5", color="#d63031",
             fontsize=8, ha="right")
    ax2.set_ylabel("ПФП у поверхности, дБВт/м²")
    ax2.set_title("ПФП: А под маской RNSS · Б в выделенной полосе")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(f"АВРОРА — двухсервисная архитектура сигналов (А + Б) [{label}]",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(output_dir, f"dual_service_{label}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(output_dir, f"dual_service_{label}.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["service", "EIRP_dBW", "Rx_dBm", "PFD_dBWm2",
                    "advantage_dB", "CN0_dBHz", "jam_margin_x", "ITU_RNSS_ok"])
        for n, r in res.items():
            w.writerow([n, f"{r['eirp_dbw']:.1f}", f"{r['rx_dbm']:.1f}",
                        f"{r['pfd']:.1f}", f"{r['adv_db']:.1f}",
                        f"{r['cn0']:.1f}", f"{r['jam_x']:.0f}", r["itu_ok"]])

    for n, r in res.items():
        print(f"  Сервис {n}: EIRP {r['eirp_dbw']:.1f} дБВт | Rx {r['rx_dbm']:.0f} дБм | "
              f"ПФП {r['pfd']:.0f} | +{r['adv_db']:.0f} дБ (×{r['jam_x']:.0f}) | "
              f"C/N₀ {r['cn0']:.0f} дБ·Гц | МСЭ-RNSS: {'OK' if r['itu_ok'] else 'своя полоса'}")
    return res


if __name__ == "__main__":
    run_dual_service_analysis("results/dual_service", "phase4")
