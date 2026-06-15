"""
Бюджет целостности АВРОРА — диаграмма Стэнфорда, защита vs аварийные
пределы, доступность LPV-200 и CAT-I, параметры ISM.

Вычисляет:
- Синтетические пары (ошибка позиции PE, уровень защиты PL), N = 20000
- Классификацию по диаграмме Стэнфорда: Норма / Система недоступна /
  Вводящая в заблуждение информация (MI) / Опасная (HMI)
- Доступность сервисов LPV-200 (VAL=35 м) и CAT-I (VAL=15 м)
- Географическое распределение доступности LPV-200 (сетка lon/lat)
- Таблицу параметров ISM (Integrity Support Message)

Ссылки:
  RTCA DO-229E (2016) — MOPS for GPS/SBAS Airborne Equipment (Stanford diagram).
  Blanch et al. (2015) — Baseline Advanced RAIM User Algorithm. NAVIGATION.
  ICAO Annex 10, Vol. I — LPV-200 / CAT-I requirements.
  Walter & Enge (1995) — Weighted RAIM for Precision Approach. ION GPS.
  GSA (2021) — Galileo OS SDD; EU ARAIM ISM definition.
"""

import math, os, csv, sys
from typing import Dict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Гарантируем UTF-8 вывод на Windows (cp1251 не кодирует σ, ↳ и т.п.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PALETTE = ["#e17055", "#fdcb6e", "#0984e3", "#00b894",
           "#6c5ce7", "#74b9ff", "#dfe6e9", "#2d3436"]

# ── Параметры системы АВРОРА ─────────────────────────────────────────────────
N_SAMPLES = 20000
SIGMA_V   = 1.5        # м, σ вертикальной ошибки позиции
SIGMA_H   = 1.0        # м, σ горизонтальной ошибки позиции
K_PL      = 5.33       # множитель уровня защиты
PL_MARGIN = 1.5        # м, дополнительный запас в PL

VAL_LPV200 = 35.0      # м
HAL_LPV200 = 40.0      # м
VAL_CATI   = 15.0      # м
HAL_CATI   = 40.0      # м

# Параметры ISM (Integrity Support Message)
ISM = {
    "b_nom (м)":     0.75,
    "σ_URA (м)":     0.50,
    "P_sat":         1.0e-5,
    "P_const":       1.0e-4,
    "R_irr (1/ч)":   1.0e-8,
}


def _generate_samples(seed: int):
    rng = np.random.default_rng(seed)
    # σ варьируется (геометрия) → σ_v эффективный
    sig_v = SIGMA_V * rng.uniform(0.85, 1.20, N_SAMPLES)
    sig_h = SIGMA_H * rng.uniform(0.85, 1.20, N_SAMPLES)

    pe_v = np.abs(rng.normal(0.0, sig_v))
    pe_h = np.abs(rng.normal(0.0, sig_h))

    vpl = K_PL * sig_v + PL_MARGIN + rng.uniform(0.0, 2.0, N_SAMPLES)
    hpl = K_PL * sig_h + PL_MARGIN + rng.uniform(0.0, 2.0, N_SAMPLES)
    return pe_v, pe_h, vpl, hpl


def _classify(pe, pl, al):
    """Категории диаграммы Стэнфорда (по вертикали)."""
    cat = np.empty(len(pe), dtype="<U24")
    nominal   = (pe <= pl) & (pl <= al)
    unavail   = (pl > al) & (pe <= pl)
    misleading = (pe > al) & (pe <= pl)               # PL>=PE но PE>AL
    hazardous = pe > pl
    cat[nominal]    = "Норма"
    cat[unavail]    = "Система недоступна"
    cat[misleading] = "MI (заблуждение)"
    cat[hazardous]  = "HMI (опасная)"
    return cat


def run_integrity_analysis(output_dir: str, label: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    pe_v, pe_h, vpl, hpl = _generate_samples(seed=777)
    cat = _classify(pe_v, vpl, VAL_LPV200)

    counts = {
        "Норма":              int(np.sum(cat == "Норма")),
        "Система недоступна":  int(np.sum(cat == "Система недоступна")),
        "MI (заблуждение)":    int(np.sum(cat == "MI (заблуждение)")),
        "HMI (опасная)":       int(np.sum(cat == "HMI (опасная)")),
    }

    avail_lpv200 = float(np.mean((vpl < VAL_LPV200) & (hpl < HAL_LPV200)))
    avail_cati   = float(np.mean((vpl < VAL_CATI)   & (hpl < HAL_CATI)))

    # Географическая сетка доступности LPV-200 (синтетическая вариация)
    lons = np.linspace(-180, 180, 73)
    lats = np.linspace(-80, 80, 41)
    LON, LAT = np.meshgrid(lons, lats)
    rng = np.random.default_rng(42)
    base = 99.6 + 0.35 * np.cos(np.radians(LAT * 2.0)) \
        + 0.15 * np.sin(np.radians(LON))
    geo_avail = np.clip(base + rng.normal(0.0, 0.05, LON.shape),
                        99.0, 99.99)

    results = {
        "category_counts":      counts,
        "n_samples":            N_SAMPLES,
        "availability_lpv200":  avail_lpv200,
        "availability_cati":    avail_cati,
        "VAL_LPV200": VAL_LPV200, "HAL_LPV200": HAL_LPV200,
        "VAL_CATI":   VAL_CATI,   "HAL_CATI":   HAL_CATI,
        "vpl_lt_val_fraction":  float(np.mean(vpl < VAL_LPV200)),
        "ism_params":           dict(ISM),
        "_pe_v": pe_v, "_pl_v": vpl, "_cat": cat,
        "_hpl": hpl,
        "_geo": (lons, lats, geo_avail),
        "geo_avail_mean":       float(np.mean(geo_avail)),
        "geo_avail_min":        float(np.min(geo_avail)),
        "geo_avail_max":        float(np.max(geo_avail)),
    }

    _plot_stanford(results, output_dir, label)
    _plot_pl_vs_al(results, output_dir, label)
    _plot_availability_map(results, output_dir, label)
    _plot_ism_parameters(results, output_dir, label)
    _save_csv(results, output_dir, label)

    # очистим тяжёлые массивы из возвращаемого словаря
    for k in ("_pe_v", "_pl_v", "_cat", "_hpl", "_geo"):
        results.pop(k, None)
    return results


def _plot_stanford(results, output_dir, label):
    pe = results["_pe_v"]
    pl = results["_pl_v"]
    cat = results["_cat"]
    al = results["VAL_LPV200"]

    fig, ax = plt.subplots(figsize=(11, 7))
    color_map = {
        "Норма":             "#00b894",
        "Система недоступна": "#fdcb6e",
        "MI (заблуждение)":  "#0984e3",
        "HMI (опасная)":     "#e17055",
    }
    for name, c in color_map.items():
        m = cat == name
        if np.any(m):
            ax.scatter(pe[m], pl[m], s=6, c=c, alpha=0.35,
                       label=f"{name} (n={int(np.sum(m))})")

    lim = max(al * 1.6, pl.max() * 1.05)
    ax.plot([0, lim], [0, lim], "--", color="#2d3436", lw=1.3,
            label="PL = PE (диагональ)")
    ax.axvline(al, ls=":", color="#e17055", lw=1.5, label=f"VAL = {al:.0f} м")
    ax.axhline(al, ls=":", color="#6c5ce7", lw=1.5)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Ошибка позиции PE (м, вертик.)")
    ax.set_ylabel("Уровень защиты PL (м, вертик.)")
    ax.set_title(f"Диаграмма Стэнфорда — целостность АВРОРА [{label}]")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"stanford_diagram_{label}.png"),
                dpi=150)
    plt.close(fig)


def _plot_pl_vs_al(results, output_dir, label):
    vpl = results["_pl_v"]
    hpl = results["_hpl"]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(vpl, bins=60, color="#0984e3", alpha=0.6,
            label="VPL (вертик.)", edgecolor="white")
    ax.hist(hpl, bins=60, color="#00b894", alpha=0.6,
            label="HPL (гориз.)", edgecolor="white")
    ax.axvline(results["VAL_LPV200"], ls="--", color="#e17055", lw=1.8,
               label=f"VAL LPV-200 = {results['VAL_LPV200']:.0f} м")
    ax.axvline(results["VAL_CATI"], ls="-.", color="#6c5ce7", lw=1.8,
               label=f"VAL CAT-I = {results['VAL_CATI']:.0f} м")
    ax.axvline(results["HAL_LPV200"], ls=":", color="#fdcb6e", lw=1.8,
               label=f"HAL = {results['HAL_LPV200']:.0f} м")
    ax.set_xlabel("Уровень защиты (м)")
    ax.set_ylabel("Частота (число выборок)")
    ax.set_title(f"Распределение VPL/HPL vs аварийные пределы [{label}]")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"pl_vs_al_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_availability_map(results, output_dir, label):
    lons, lats, geo = results["_geo"]
    fig, ax = plt.subplots(figsize=(13, 6))
    pcm = ax.pcolormesh(lons, lats, geo, cmap="YlGnBu",
                        shading="auto", vmin=99.0, vmax=100.0)
    cb = fig.colorbar(pcm, ax=ax)
    cb.set_label("Доступность LPV-200 (%)")
    ax.contour(lons, lats, geo, levels=[99.5, 99.9],
               colors="#2d3436", linewidths=0.8, alpha=0.6)
    ax.set_xlabel("Долгота (°)")
    ax.set_ylabel("Широта (°)")
    ax.set_title(f"Карта доступности LPV-200 (среднее "
                 f"{results['geo_avail_mean']:.3f}%) [{label}]")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir,
                f"availability_lpv200_map_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_ism_parameters(results, output_dir, label):
    ism = results["ism_params"]
    names = list(ism.keys())
    vals  = list(ism.values())
    fig, ax = plt.subplots(figsize=(11, 6))
    y = np.arange(len(names))
    colors = ["#e17055", "#fdcb6e", "#0984e3", "#00b894", "#6c5ce7"]
    # лог-шкала из-за смешения вероятностей и метров
    bars = ax.barh(y, [abs(v) for v in vals], color=colors,
                   edgecolor="white", height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xscale("log")
    for bar, v in zip(bars, vals):
        ax.text(abs(v) * 1.2, bar.get_y() + bar.get_height() / 2,
                f"{v:g}", va="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("Значение параметра (лог. шкала)")
    ax.set_title(f"Параметры ISM (Integrity Support Message) [{label}]")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"ism_parameters_{label}.png"),
                dpi=150)
    plt.close(fig)


def _save_csv(results, output_dir, label):
    path = os.path.join(output_dir, f"integrity_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["parameter", "value", "unit"])
        w.writerow(["n_samples", results["n_samples"], "samples"])
        for name, c in results["category_counts"].items():
            w.writerow([f"count_{name}", c, "samples"])
        w.writerow(["availability_lpv200",
                    f"{results['availability_lpv200']:.5f}", "fraction"])
        w.writerow(["availability_cati",
                    f"{results['availability_cati']:.5f}", "fraction"])
        w.writerow(["vpl_lt_val_fraction",
                    f"{results['vpl_lt_val_fraction']:.5f}", "fraction"])
        w.writerow(["VAL_LPV200", f"{results['VAL_LPV200']:.1f}", "m"])
        w.writerow(["HAL_LPV200", f"{results['HAL_LPV200']:.1f}", "m"])
        w.writerow(["VAL_CATI",   f"{results['VAL_CATI']:.1f}",   "m"])
        w.writerow(["HAL_CATI",   f"{results['HAL_CATI']:.1f}",   "m"])
        w.writerow(["geo_avail_mean", f"{results['geo_avail_mean']:.4f}", "%"])
        w.writerow(["geo_avail_min",  f"{results['geo_avail_min']:.4f}",  "%"])
        w.writerow(["geo_avail_max",  f"{results['geo_avail_max']:.4f}",  "%"])
        for name, v in results["ism_params"].items():
            w.writerow([f"ISM_{name}", f"{v:g}", ""])


def print_integrity_summary(label: str, results: Dict) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  Integrity Budget Analysis -- {label}")
    print(sep)
    print(f"  Выборок: {results['n_samples']}")
    print(f"  {'':-<64}")
    print("  Категории диаграммы Стэнфорда:")
    for name, c in results["category_counts"].items():
        pct = 100.0 * c / results["n_samples"]
        print(f"    {name:<24} {c:>8}  ({pct:6.3f}%)")
    print(f"  {'':-<64}")
    av_l = results["availability_lpv200"] * 100.0
    av_c = results["availability_cati"] * 100.0
    l_ok = "OK" if av_l >= 99.0 else "FAIL"
    print(f"  Доступность LPV-200: {av_l:7.3f}%  [{l_ok}]")
    print(f"  Доступность CAT-I:   {av_c:7.3f}%")
    print(f"  VPL < VAL (LPV-200): "
          f"{results['vpl_lt_val_fraction']*100:.3f}%")
    print(f"  Карта LPV-200: ср. {results['geo_avail_mean']:.3f}%  "
          f"(мин {results['geo_avail_min']:.3f}%, "
          f"макс {results['geo_avail_max']:.3f}%)")
    print(f"  {'':-<64}")
    print("  Параметры ISM:")
    for name, v in results["ism_params"].items():
        print(f"    {name:<16} = {v:g}")
    print(sep)


if __name__ == "__main__":
    r = run_integrity_analysis("results/integrity", "phase5")
    print_integrity_summary("phase5", r)
