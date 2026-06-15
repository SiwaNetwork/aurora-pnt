"""
ITU/МСЭ координация частот АВРОРА.

Анализирует:
- Мощностная спектральная плотность (PSD) сигналов на L1/L5
- Коэффициент спектрального разделения (SSC) — АВРОРА vs GPS/Galileo/ГЛОНАСС
- Частотный план RNSS-полос (ITU-R M.1787)
- Маска внеполосных излучений (OOB emission mask)

Ссылки:
  ITU-R M.1787-3 (2015) — Description and characteristics of RNSS.
  ITU-R M.2092-0 (2015) — Technical characteristics of APNT systems.
  WRC-19 Resolution 610 — Spectrum for RNSS.
  Betz, J.W. (2016) — Engineering Satellite-Based Navigation and Timing, Ch. 3.
  IS-GPS-200L (2020) — GPS Interface Specification.
"""

import math, os, csv
from typing import Dict, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

F_L1 = 1575.42e6
F_L5 = 1176.45e6
C    = 299792458.0

# ── Описание сигналов ─────────────────────────────────────────────────────────
SIGNALS = {
    "GPS L1 C/A\n(BPSK-1)": {
        "freq": F_L1, "chip_rate": 1.023e6,
        "boc_f": 0.0, "modulation": "BPSK",
        "power_dbw": -158.5, "color": "#e17055", "ls": "-",
    },
    "GPS L1C pilot\n(TMBOC-6,1)": {
        "freq": F_L1, "chip_rate": 1.023e6,
        "boc_f": 6 * 1.023e6, "modulation": "BOC",
        "power_dbw": -158.5, "color": "#fab1a0", "ls": "--",
    },
    "Galileo E1 OS\n(BOC-1,1)": {
        "freq": F_L1, "chip_rate": 1.023e6,
        "boc_f": 1.023e6, "modulation": "BOC",
        "power_dbw": -160.0, "color": "#0984e3", "ls": "-.",
    },
    "АВРОРА L1\n(TMBOC-6,1,4/33)": {
        "freq": F_L1, "chip_rate": 1.023e6,
        "boc_f": 6 * 1.023e6, "modulation": "BOC",
        "power_dbw": -107.0, "color": "#00b894", "ls": "-",
    },
    "GPS L5\n(BPSK-10)": {
        "freq": F_L5, "chip_rate": 10.23e6,
        "boc_f": 0.0, "modulation": "BPSK",
        "power_dbw": -157.9, "color": "#fdcb6e", "ls": "-",
    },
    "Galileo E5a\n(AltBOC)": {
        "freq": F_L5, "chip_rate": 10.23e6,
        "boc_f": 0.0, "modulation": "BPSK",
        "power_dbw": -158.0, "color": "#74b9ff", "ls": "--",
    },
    "АВРОРА L5\n(BPSK-10)": {
        "freq": F_L5, "chip_rate": 10.23e6,
        "boc_f": 0.0, "modulation": "BPSK",
        "power_dbw": -107.0, "color": "#00b894", "ls": "-.",
    },
}

# ── RNSS-полосы (ITU-R M.1787) ────────────────────────────────────────────────
RNSS_BANDS = {
    "L1 (1559–1610 МГц)": (1559e6, 1610e6, F_L1),
    "L5 (1164–1215 МГц)": (1164e6, 1215e6, F_L5),
}


def _psd_bpsk(f: np.ndarray, fc: float, chip_rate: float) -> np.ndarray:
    x = (f - fc) / chip_rate
    return np.sinc(x) ** 2 / chip_rate


def _psd_boc(f: np.ndarray, fc: float, chip_rate: float, boc_freq: float) -> np.ndarray:
    x = (f - fc) / chip_rate
    sinc_p = np.sinc(x)
    bf = (f - fc) / (2 * boc_freq + 1e-30)
    boc_p = np.where(np.abs(bf) < 1e-9, 1.0,
                     np.sin(math.pi * bf) / (math.pi * bf + 1e-30))
    return (sinc_p * boc_p) ** 2 / chip_rate


def _get_psd(sig: Dict, f: np.ndarray) -> np.ndarray:
    if sig["modulation"] == "BPSK":
        return _psd_bpsk(f, 0.0, sig["chip_rate"])
    else:
        return _psd_boc(f, 0.0, sig["chip_rate"], sig["boc_f"])


def spectral_separation_coeff(sig1: Dict, sig2: Dict,
                               bw_hz: float = 30e6) -> float:
    """
    Коэффициент спектрального разделения SSC (дБ/Гц).
    SSC = ∫ G₁(f)·G₂(f) df  (нормированные PSD).
    """
    if abs(sig1["freq"] - sig2["freq"]) > 50e6:
        return -200.0
    f = np.linspace(-bw_hz / 2, bw_hz / 2, 6001)
    df = f[1] - f[0]
    g1 = _get_psd(sig1, f)
    g2 = _get_psd(sig2, f)
    g1 /= np.sum(g1) * df + 1e-30
    g2 /= np.sum(g2) * df + 1e-30
    ssc = float(np.sum(g1 * g2) * df)
    return 10 * math.log10(max(ssc, 1e-30))


def run_itu_coordination_analysis(output_dir: str, label: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    names = list(SIGNALS.keys())
    n = len(names)
    ssc_mat = np.full((n, n), -200.0)
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            ssc_mat[i, j] = spectral_separation_coeff(SIGNALS[n1], SIGNALS[n2])

    results = {
        "signal_names": names,
        "ssc_matrix":   ssc_mat,
        "aurora_l1_power_dbw": SIGNALS["АВРОРА L1\n(TMBOC-6,1,4/33)"]["power_dbw"],
        "aurora_l5_power_dbw": SIGNALS["АВРОРА L5\n(BPSK-10)"]["power_dbw"],
    }

    _plot_psd(output_dir, label)
    _plot_ssc_matrix(results, output_dir, label)
    _plot_band_plan(output_dir, label)
    _plot_oob_mask(output_dir, label)
    _save_csv(results, output_dir, label)
    return results


def _plot_psd(output_dir, label):
    bw = 32e6
    f = np.linspace(-bw / 2, bw / 2, 4001)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    for name, sig in SIGNALS.items():
        psd = _get_psd(sig, f)
        psd_db = 10 * np.log10(psd / max(psd.max(), 1e-30) + 1e-12)
        ax = ax1 if sig["freq"] == F_L1 else ax2
        ax.plot(f / 1e6, psd_db, color=sig["color"], lw=2,
                ls=sig["ls"], label=name.replace("\n", " "))

    for ax, title, band_mhz in [
        (ax1, f"L1 = 1575,42 МГц [{label}]", 10.23),
        (ax2, f"L5 = 1176,45 МГц [{label}]", 10.23),
    ]:
        ax.set_xlim(-16, 16)
        ax.set_ylim(-45, 5)
        ax.axvspan(-band_mhz, band_mhz, alpha=0.06, color="#00b894")
        ax.set_xlabel("Смещение от центральной частоты (МГц)")
        ax.set_ylabel("Нормированная PSD (дБ)")
        ax.set_title(f"PSD сигналов — {title}")
        ax.legend(fontsize=8, loc="lower center")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"itu_psd_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_ssc_matrix(results, output_dir, label):
    mat = results["ssc_matrix"].copy()
    mat[mat < -100] = np.nan
    names_short = [n.split("\n")[0] for n in results["signal_names"]]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mat, cmap="RdYlGn_r", aspect="auto", vmin=-80, vmax=-40)
    ax.set_xticks(range(len(names_short)))
    ax.set_yticks(range(len(names_short)))
    ax.set_xticklabels(names_short, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(names_short, fontsize=8)
    for i in range(len(names_short)):
        for j in range(len(names_short)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:.0f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, label="SSC (дБ/Гц)")
    ax.set_title(f"Коэффициент спектрального разделения (SSC) [{label}]")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"itu_ssc_matrix_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_band_plan(output_dir, label):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    l1_entries = [
        ("GPS L1 C/A (BPSK-1)",       F_L1, 2.046e6,  "#e17055", 0),
        ("GPS L1C pilot (TMBOC)",      F_L1, 12.276e6, "#fab1a0", 1),
        ("Galileo E1 OS (BOC-1,1)",    F_L1, 8.184e6,  "#0984e3", 2),
        ("ГЛОНАСС L1OC (BOC-1,1)",    F_L1, 4.092e6,  "#6c5ce7", 3),
        ("АВРОРА L1 (TMBOC-6,1,4/33)",F_L1, 12.276e6, "#00b894", 4),
    ]
    l5_entries = [
        ("GPS L5 I+Q (BPSK-10)",       F_L5, 20.46e6,  "#e17055", 0),
        ("Galileo E5a (BPSK-10)",      F_L5, 20.46e6,  "#0984e3", 1),
        ("BeiDou B2a (BPSK-10)",       F_L5, 20.46e6,  "#fdcb6e", 2),
        ("АВРОРА L5 (BPSK-10)",        F_L5, 20.46e6,  "#00b894", 3),
    ]

    for ax, entries, band_lo, band_label in [
        (axes[0], l1_entries, 1559e6, "L1 (1559–1610 МГц)"),
        (axes[1], l5_entries, 1164e6, "L5 (1164–1215 МГц)"),
    ]:
        for name, fc, bw, color, row in entries:
            f_lo = (fc - bw / 2 - band_lo) / 1e6
            ax.barh(row, bw / 1e6, left=f_lo, height=0.7,
                    color=color, alpha=0.75, edgecolor="white")
            ax.text(f_lo + bw / 2e6, row, name, ha="center", va="center",
                    fontsize=7.5, fontweight="bold")
        ax.axvline((F_L1 if "L1" in band_label else F_L5) / 1e6 - band_lo / 1e6,
                   ls="--", color="#2d3436", lw=1.2)
        ax.set_yticks([])
        ax.set_xlabel("Частота (МГц от нижней границы защищённой полосы)")
        ax.set_title(f"Частотный план {band_label} [{label}]")
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"itu_band_plan_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_oob_mask(output_dir, label):
    f_mhz = np.concatenate([
        np.linspace(-50, -6.138, 200),
        np.linspace(-6.138, 6.138, 800),
        np.linspace(6.138, 50, 200),
    ])
    f_hz = f_mhz * 1e6

    # TMBOC PSD нормированная
    cr = 1.023e6
    bf = 6.138e6
    psd = _psd_boc(f_hz, 0, cr, bf)
    psd_db = 10 * np.log10(psd / max(psd.max(), 1e-30) + 1e-12)

    # Маска ITU-R M.1787 (упрощённая)
    itu_mask = np.where(np.abs(f_mhz) <= 10.23, -60.0, -80.0)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(f_mhz, psd_db, color="#00b894", lw=2.5, label="АВРОРА L1 TMBOC (PSD)")
    ax.step(f_mhz, itu_mask, color="#d63031", lw=2.0, ls="--",
            label="Предел ITU-R M.1787 (OOB)")
    ax.fill_between(f_mhz, psd_db, -100,
                    where=(psd_db > itu_mask + 5), alpha=0.1, color="#00b894")
    ax.axvspan(-10.23, 10.23, alpha=0.05, color="#00b894", label="Рабочая полоса ±10,23 МГц")
    ax.set_xlabel("Смещение от центра L1 (МГц)")
    ax.set_ylabel("Нормированная PSD (дБс)")
    ax.set_ylim(-100, 5)
    ax.set_title(f"АВРОРА — маска внеполосных излучений OOB [{label}]")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"itu_oob_mask_{label}.png"), dpi=150)
    plt.close(fig)


def _save_csv(results, output_dir, label):
    path = os.path.join(output_dir, f"itu_coordination_{label}.csv")
    names = results["signal_names"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["signal_i", "signal_j", "ssc_db_hz"])
        for i, n1 in enumerate(names):
            for j, n2 in enumerate(names):
                if results["ssc_matrix"][i, j] > -100 and i != j:
                    w.writerow([n1.replace("\n", " "), n2.replace("\n", " "),
                                 f"{results['ssc_matrix'][i,j]:.2f}"])


def print_itu_summary(label: str, results: Dict) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  ITU Coordination Analysis -- {label}")
    print(sep)
    names = results["signal_names"]
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            ssc = results["ssc_matrix"][i, j]
            if ssc > -100 and i < j:
                print(f"  SSC({n1.split(chr(10))[0]:22s}, {n2.split(chr(10))[0]:22s}) = {ssc:.1f} дБ/Гц")
    print(sep)
