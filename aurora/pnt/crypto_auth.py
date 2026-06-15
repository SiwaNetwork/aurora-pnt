"""
Криптографическая защита навигационного сообщения АВРОРА на ГОСТ-алгоритмах.

Суверенная система использует российскую криптографию вместо зарубежной (NIST):

  - односторонняя цепочка ключей TESLA      → ГОСТ Р 34.11-2012 «Стрибог» (256 бит);
  - имитовставка (MAC) сообщения            → HMAC-Стрибог (RFC 7836), усечённый тег;
  - подпись корня доверия / навигационного  → ГОСТ Р 34.10-2012 (ЭП на эллиптич.
    сообщения (аналог Galileo OSNMA)            кривых, 256-бит уровень стойкости);
  - шифрование/имитозащита команд TT&C      → ГОСТ Р 34.12-2015 «Кузнечик»
    и обёртка ключей                            в режиме MGM по ГОСТ Р 34.13-2015.

Модель оценивает накладные расходы аутентификации (бит/с и доля скорости
навигационного сообщения) для разных длин усечённого тега и интервалов раскрытия
ключа, а также уровни стойкости суверенной крипто-связки в сравнении с NIST.

References:
  ГОСТ Р 34.10-2012, ГОСТ Р 34.11-2012, ГОСТ Р 34.12-2015, ГОСТ Р 34.13-2015;
  RFC 7836 (HMAC_GOSTR3411_2012); RFC 9058 (MGM); Galileo OSNMA SIS ICD (2023).
"""

import os
import csv
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Параметры навигационного сообщения ────────────────────────────────────────
NAV_BPS = 250.0          # полезная скорость нав-сообщения, бит/с (§7)
KEY_BITS = 256           # длина ключа цепочки «Стрибог-256», бит
TAGS = [80, 96, 128]     # варианты длины усечённого тега MAC, бит
INTERVALS = [10, 30]     # интервалы раскрытия ключа TESLA, с

# ── Крипто-связка: ГОСТ vs NIST, уровни стойкости (бит) ────────────────────────
# Формат: функция → (ГОСТ алгоритм, бит стойкости, NIST аналог, бит стойкости)
SUITE = [
    ("Хэш-цепочка (прообраз)", "Стрибог-256",       256, "SHA-256",     256),
    ("Коллизионная стойкость", "Стрибог-256",       128, "SHA-256",     128),
    ("Имитовставка MAC",       "HMAC-Стрибог/128",  128, "HMAC-SHA256", 128),
    ("Подпись (ЭП)",           "ГОСТ Р 34.10-256",  128, "ECDSA P-256", 128),
    ("Шифр команд TT&C",       "Кузнечик-256",      256, "AES-256",     256),
]


def overhead_bps(tag_bits: int, interval_s: float) -> Dict:
    """Накладные расходы аутентификации: тег каждого интервала + раскрытие ключа."""
    mac_bps = tag_bits / interval_s
    key_bps = KEY_BITS / interval_s
    total = mac_bps + key_bps
    return {
        "tag_bits": tag_bits, "interval_s": interval_s,
        "mac_bps": mac_bps, "key_bps": key_bps, "total_bps": total,
        "pct_nav": 100.0 * total / NAV_BPS,
        "forge_prob_log2": -tag_bits,    # вероятность подделки тега 2^-tag
    }


def compute() -> Dict:
    rows = [overhead_bps(t, i) for i in INTERVALS for t in TAGS]
    return {"rows": rows, "suite": SUITE}


def run_crypto_auth_analysis(output_dir: str, label: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    res = compute()
    rows = res["rows"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.3))

    # ── Панель 1: оверхед бит/с по (тег × интервал) ──────────────────────────
    labels = [f"{r['tag_bits']}б/{int(r['interval_s'])}с" for r in rows]
    totals = [r["total_bps"] for r in rows]
    cols = ["#0984e3" if r["interval_s"] == 30 else "#74b9ff" for r in rows]
    bars = ax1.bar(labels, totals, color=cols, edgecolor="white")
    for rect, r in zip(bars, rows):
        ax1.text(rect.get_x() + rect.get_width() / 2, r["total_bps"] + 0.4,
                 f"{r['total_bps']:.1f}\n({r['pct_nav']:.1f}%)",
                 ha="center", fontsize=8)
    ax1.set_ylabel("Накладные расходы аутентификации, бит/с")
    ax1.set_xlabel("длина тега MAC / интервал раскрытия ключа")
    ax1.set_title("Оверхед TESLA на HMAC-Стрибог\n(тег + раскрытие ключа 256 бит)")
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim(0, max(totals) * 1.25)

    # ── Панель 2: уровни стойкости ГОСТ vs NIST ──────────────────────────────
    suite = res["suite"]
    y = range(len(suite))
    names = [s[0] for s in suite]
    gost_bits = [s[2] for s in suite]
    nist_bits = [s[4] for s in suite]
    bw = 0.38
    ax2.barh([i + bw / 2 for i in y], gost_bits, height=bw,
             color="#0652DD", label="ГОСТ (РФ)")
    ax2.barh([i - bw / 2 for i in y], nist_bits, height=bw,
             color="#b2bec3", label="NIST (для сравнения)")
    for i, s in enumerate(suite):
        ax2.text(s[2] + 4, i + bw / 2, f"{s[1]} ({s[2]})", va="center", fontsize=8)
    ax2.set_yticks(list(y))
    ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlabel("Уровень стойкости, бит")
    ax2.set_title("Криптопримитивы защиты сообщения: ГОСТ ↔ NIST")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.set_xlim(0, 300)
    ax2.grid(axis="x", alpha=0.3)

    fig.suptitle(f"АВРОРА — криптозащита навигационного сообщения (ГОСТ) [{label}]",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(output_dir, f"crypto_auth_{label}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── CSV ──────────────────────────────────────────────────────────────────
    with open(os.path.join(output_dir, f"crypto_auth_{label}.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tag_bits", "interval_s", "mac_bps", "key_bps",
                    "total_bps", "pct_nav", "forge_prob_log2"])
        for r in rows:
            w.writerow([r["tag_bits"], int(r["interval_s"]),
                        f"{r['mac_bps']:.2f}", f"{r['key_bps']:.2f}",
                        f"{r['total_bps']:.2f}", f"{r['pct_nav']:.2f}",
                        r["forge_prob_log2"]])

    base = next(r for r in rows if r["tag_bits"] == 128 and r["interval_s"] == 30)
    print(f"  Криптозащита сообщения (ГОСТ) -- {label}")
    print("    Связка: Стрибог-256 (цепочка) + HMAC-Стрибог (MAC) + "
          "ГОСТ Р 34.10-2012 (ЭП) + Кузнечик (TT&C)")
    print(f"    Базовый профиль (тег 128 б, интервал 30 с): "
          f"MAC {base['mac_bps']:.2f} + ключ {base['key_bps']:.2f} = "
          f"{base['total_bps']:.2f} бит/с ({base['pct_nav']:.1f}% нав-канала)")
    print("    Вероятность подделки тега: 2^-128; подделка цепочки: 2^-256")
    return res


if __name__ == "__main__":
    run_crypto_auth_analysis("results/crypto_auth", "phase4")
