"""
Тропосферная задержка и коррекция для АВРОРА.

Тропосфера вносит задержку в распространение радиосигнала:
  - Гидростатическая часть (ZHD): ~2.3 м в зените, хорошо предсказуема
  - Влажная часть (ZWD): 0.05–0.4 м, переменна, труднее моделировать

Модели:
  - Саастамойнена (Saastamoinen, 1972) — стандарт IGS
  - Модифицированная Hopfield — упрощённая
  - Функции отображения Niell (NMF) и BoehмPaper (VMF3-коэффициенты)
  - Коррекция GPT2 (Global Pressure and Temperature v2)

Ссылки:
  Saastamoinen, J. (1972) — Atmospheric correction for troposphere.
  Niell, A.E. (1996) — Global mapping functions for atmosphere.
  Boehm et al. (2006) — Global Mapping Function (GMF).
  IERS Conventions (2010) — Chapter 9.
"""

import math, os, csv
from typing import Dict, List, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Константы ────────────────────────────────────────────────────────────────
G   = 9.80665      # ускорение свободного падения (м/с²)
Rd  = 287.053      # газовая постоянная сухого воздуха (Дж/(кг·К))
Rv  = 461.495      # газовая постоянная водяного пара

# ── Стандартная атмосфера (ISA, ИКАО) ────────────────────────────────────────
P0_PA  = 101325.0   # давление на уровне моря (Па)
T0_K   = 288.15     # температура на уровне моря (К)
LAPSE  = 0.0065     # градиент температуры (К/м)

# ── Модели ───────────────────────────────────────────────────────────────────

def isa_pressure(height_m: float) -> float:
    """Давление по стандартной атмосфере на высоте h [Па]."""
    return P0_PA * (1 - LAPSE * height_m / T0_K) ** (G / (Rd * LAPSE))

def isa_temperature(height_m: float) -> float:
    return T0_K - LAPSE * height_m

def saturation_pressure_pa(T_K: float) -> float:
    """Давление насыщенного водяного пара по формуле Магнуса [Па]."""
    T_C = T_K - 273.15
    return 611.2 * math.exp(17.67 * T_C / (T_C + 243.5))

def water_vapor_pressure_pa(T_K: float, rh_pct: float = 50.0) -> float:
    """Парциальное давление водяного пара [Па]."""
    return (rh_pct / 100.0) * saturation_pressure_pa(T_K)


# ── Модель Саастамойнена (зенитные задержки) ──────────────────────────────────

def zhd_saastamoinen(P_pa: float, lat_deg: float, H_m: float) -> float:
    """
    Зенитная гидростатическая задержка [м] (Saastamoinen 1972).
    ZHD = 0.0022768 × P / (1 - 0.00266·cos2φ - 0.00028·H[km])
    """
    P_hPa = P_pa / 100.0
    lat_r = math.radians(lat_deg)
    denom = 1 - 0.00266 * math.cos(2 * lat_r) - 0.00028 * (H_m / 1000.0)
    return 0.0022768 * P_hPa / denom

def zwd_saastamoinen(T_K: float, e_pa: float) -> float:
    """
    Зенитная влажная задержка [м] (Saastamoinen 1972).
    ZWD = 0.0022768 × (1255/T + 0.05) × e[hPa]
    """
    e_hPa = e_pa / 100.0
    return 0.0022768 * (1255.0 / T_K + 0.05) * e_hPa

def ztd_saastamoinen(lat_deg: float, H_m: float, rh_pct: float = 50.0) -> Dict:
    """Полная зенитная задержка по Саастамойнену для заданной широты и высоты."""
    P = isa_pressure(H_m)
    T = isa_temperature(H_m)
    e = water_vapor_pressure_pa(T, rh_pct)
    zhd = zhd_saastamoinen(P, lat_deg, H_m)
    zwd = zwd_saastamoinen(T, e)
    return {"ZHD_m": zhd, "ZWD_m": zwd, "ZTD_m": zhd + zwd,
            "P_hPa": P/100, "T_K": T, "e_hPa": e/100}

# ── Функции отображения (Mapping Functions) ───────────────────────────────────

def mf_niell_hydrostatic(elevation_deg: float, lat_deg: float,
                           doy: int = 180) -> float:
    """
    Функция отображения Niell (NMF) — гидростатическая составляющая.
    Аппроксимация по таблицам Niell (1996).
    mh(ε) = 1 / sin(ε + 0.00143/(tan(ε) + 0.0445))
    """
    el_r = math.radians(max(elevation_deg, 1.0))
    mh = 1.0 / (math.sin(el_r) + 0.00143 / (math.tan(el_r) + 0.0445))
    # Широтная коррекция (упрощённая)
    lat_factor = 1.0 + 0.002 * (1 - math.cos(math.radians(lat_deg)))
    return mh * lat_factor

def mf_niell_wet(elevation_deg: float) -> float:
    """Функция отображения Niell — влажная составляющая."""
    el_r = math.radians(max(elevation_deg, 1.0))
    return 1.0 / (math.sin(el_r) + 0.00035 / (math.tan(el_r) + 0.017))

def slant_delay(lat_deg: float, elevation_deg: float, H_m: float = 0.0,
                rh_pct: float = 50.0) -> Dict:
    """Наклонная тропосферная задержка [м] для данного угла места."""
    z = ztd_saastamoinen(lat_deg, H_m, rh_pct)
    mh = mf_niell_hydrostatic(elevation_deg, lat_deg)
    mw = mf_niell_wet(elevation_deg)
    std = z["ZHD_m"] * mh + z["ZWD_m"] * mw
    return {"STD_m": std, "ZHD_m": z["ZHD_m"], "ZWD_m": z["ZWD_m"],
            "mh": mh, "mw": mw}

# ── Остаточная ошибка после коррекции ────────────────────────────────────────

CORRECTION_MODELS = {
    "Без коррекции":    1.00,   # остаток = 100% ZTD
    "Саастамойнен":     0.10,   # 10% остаток
    "GPT2":             0.05,   # 5% остаток
    "ECMWF-числовой":   0.02,   # 2% остаток
    "Двойная диф.":     0.01,   # 1% (RTK)
}

# ── Запуск анализа ────────────────────────────────────────────────────────────

def run_troposphere_analysis(output_dir: str, label: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    lats    = np.linspace(-80, 80, 100)
    elev_r  = np.linspace(5, 90, 200)

    # ZTD по широтам (стандартная атмосфера, RH=50%)
    ztd_vals = [ztd_saastamoinen(lat, 0.0)["ZTD_m"] for lat in lats]
    zhd_vals = [ztd_saastamoinen(lat, 0.0)["ZHD_m"] for lat in lats]
    zwd_vals = [ztd_saastamoinen(lat, 0.0)["ZWD_m"] for lat in lats]

    # Функции отображения
    mh_curve = [mf_niell_hydrostatic(el, 55.0) for el in elev_r]
    mw_curve = [mf_niell_wet(el)               for el in elev_r]

    # Наклонная задержка по углу места (широта 55°)
    std_vals = [slant_delay(55.0, el)["STD_m"] for el in elev_r]

    # Остаточная ошибка по моделям
    # ZTD @ 55° lat ~ 2.35 м; переводим в вклад UERE (cos(30°) ≈ 0.5)
    ztd_ref  = ztd_saastamoinen(55.0, 0.0)["ZTD_m"]
    residuals = {m: ztd_ref * f * 0.5 for m, f in CORRECTION_MODELS.items()}

    # Сезонная вариация ZTD на широте 55° (север)
    doy_range = np.linspace(1, 365, 200)
    # Простая синусоида: амплитуда ±10% от ZWD
    ztd_seasonal = [ztd_saastamoinen(55.0, 0.0)["ZTD_m"] +
                    ztd_saastamoinen(55.0, 0.0)["ZWD_m"] *
                    0.4 * math.sin(2*math.pi*(d-91)/365) for d in doy_range]

    # UERE вклад тропосферы
    elev_for_uere = np.linspace(5, 90, 50)
    uere_no_corr  = [slant_delay(55.0, el)["STD_m"] for el in elev_for_uere]
    uere_saas     = [slant_delay(55.0, el)["STD_m"] * 0.10 for el in elev_for_uere]
    uere_gpt2     = [slant_delay(55.0, el)["STD_m"] * 0.05 for el in elev_for_uere]

    _plot_ztd_vs_lat(lats, zhd_vals, zwd_vals, ztd_vals, output_dir, label)
    _plot_mapping_function(elev_r, mh_curve, mw_curve, std_vals, output_dir, label)
    _plot_seasonal(doy_range, ztd_seasonal, output_dir, label)
    _plot_model_compare(residuals, output_dir, label)
    _plot_uere(elev_for_uere, uere_no_corr, uere_saas, uere_gpt2, output_dir, label)
    _save_csv(lats, zhd_vals, zwd_vals, ztd_vals, output_dir, label)

    return {
        "lats": lats.tolist(), "ZTD_m": ztd_vals,
        "ZHD_m": zhd_vals, "ZWD_m": zwd_vals,
        "elev_range": elev_r.tolist(),
        "mapping_hydrostatic": mh_curve, "mapping_wet": mw_curve,
        "slant_delay_m": std_vals,
        "residuals": residuals,
        "seasonal_ztd": ztd_seasonal,
    }


def _plot_ztd_vs_lat(lats, zhd, zwd, ztd, output_dir, label):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(lats, ztd, color="#0984e3", lw=2.5, label="ZTD (суммарная)")
    ax.plot(lats, zhd, color="#00b894", lw=2,   label="ZHD (гидростатическая)")
    ax.plot(lats, zwd, color="#e17055", lw=2,   label="ZWD (влажная)")
    ax.axvline(55,  ls=":", color="#6c5ce7", lw=1.2, label="55° (Москва)")
    ax.axvline(70,  ls=":", color="#fdcb6e", lw=1.2, label="70° (Арктика)")
    ax.set_xlabel("Широта (°)")
    ax.set_ylabel("Зенитная задержка (м)")
    ax.set_title(f"АВРОРА — Тропосферная зенитная задержка vs широта [{label}]")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"tropo_ztd_vs_lat_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_mapping_function(elev, mh, mw, std, output_dir, label):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(elev, mh, color="#0984e3", lw=2, label="NMF гидростатическая")
    axes[0].plot(elev, mw, color="#e17055", lw=2, label="NMF влажная")
    axes[0].axhline(1.0, ls="--", color="gray", lw=0.8)
    axes[0].set_xlabel("Угол места (°)")
    axes[0].set_ylabel("Функция отображения")
    axes[0].set_title("Функции отображения Niell (NMF)")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)
    axes[0].set_xlim(5, 90)

    axes[1].semilogy(elev, std, color="#6c5ce7", lw=2)
    axes[1].axhline(0.5, ls="--", color="#e17055", lw=1.2, label="0.5 м")
    axes[1].axhline(0.1, ls=":",  color="#00b894", lw=1.2, label="0.1 м")
    axes[1].axvline(10, ls=":", color="#fdcb6e", lw=1.2, label="10° (маска)")
    axes[1].set_xlabel("Угол места (°)")
    axes[1].set_ylabel("Наклонная задержка STD (м)")
    axes[1].set_title("Наклонная задержка vs угол места (шир. 55°)")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].set_xlim(5, 90)
    fig.suptitle(f"АВРОРА — Функция отображения и наклонная задержка [{label}]")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"tropo_mapping_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_seasonal(doy, ztd_seasonal, output_dir, label):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(doy, ztd_seasonal, color="#0984e3", lw=2)
    ax.axhline(np.mean(ztd_seasonal), ls="--", color="#e17055", lw=1.2, label="Среднее")
    ax.set_xlabel("День года")
    ax.set_ylabel("ZTD (м)")
    ax.set_title(f"АВРОРА — Сезонная вариация ZTD (шир. 55°) [{label}]")
    ax.set_xticks([1, 90, 180, 270, 365])
    ax.set_xticklabels(["1 янв", "1 апр", "1 июл", "1 окт", "31 дек"])
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"tropo_seasonal_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_model_compare(residuals, output_dir, label):
    names  = list(residuals.keys())
    values = list(residuals.values())
    colors = ["#e17055","#fdcb6e","#74b9ff","#00b894","#0984e3"]
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(names, values, color=colors, edgecolor="white", alpha=0.85)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                f"{v*100:.1f} см", ha="center", fontsize=9)
    ax.axhline(0.01, ls="--", color="#2d3436", lw=1.2, label="1 см (цель PPP)")
    ax.set_ylabel("Остаточная ошибка в UERE (м, 1σ)")
    ax.set_title(f"АВРОРА — Остаток тропосферной ошибки по моделям [{label}]")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=18, ha="right", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"tropo_model_compare_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_uere(elev, no_corr, saas, gpt2, output_dir, label):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(elev, no_corr, color="#e17055", lw=2, label="Без коррекции")
    ax.semilogy(elev, saas,    color="#fdcb6e", lw=2, label="Саастамойнен (10%)")
    ax.semilogy(elev, gpt2,    color="#00b894", lw=2, label="GPT2 / ECMWF (5%)")
    ax.axhline(0.5, ls="--", color="#0984e3", lw=1.2, label="0.5 м UERE цель")
    ax.axvline(10, ls=":",  color="#6c5ce7", lw=1.2, label="10° (маска АВРОРА)")
    ax.set_xlabel("Угол места (°)")
    ax.set_ylabel("Вклад тропосферы в UERE (м)")
    ax.set_title(f"АВРОРА — Тропосферный вклад в UERE [{label}]")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"tropo_uere_{label}.png"), dpi=150)
    plt.close(fig)


def _save_csv(lats, zhd, zwd, ztd, output_dir, label):
    path = os.path.join(output_dir, f"troposphere_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lat_deg", "ZHD_m", "ZWD_m", "ZTD_m"])
        for lat, h, w_, t in zip(lats[::5], zhd[::5], zwd[::5], ztd[::5]):
            w.writerow([f"{lat:.1f}", f"{h:.4f}", f"{w_:.4f}", f"{t:.4f}"])


def print_troposphere_summary(label: str, result: Dict) -> None:
    sep = "=" * 68
    print(f"\n{sep}")
    print(f"  Troposphere Analysis -- {label}")
    print(sep)
    for lat_target in [0, 30, 55, 70]:
        idx = int((lat_target + 80) / 160 * len(result["lats"]))
        idx = min(idx, len(result["ZTD_m"]) - 1)
        z = result["ZTD_m"][idx]
        h = result["ZHD_m"][idx]
        w = result["ZWD_m"][idx]
        print(f"  Широта {lat_target:>3}°:  ZTD={z:.3f} м  (ZHD={h:.3f} + ZWD={w:.3f})")
    print()
    print(f"  Остаточные ошибки в UERE (ε=30°):")
    for model, res in result["residuals"].items():
        print(f"    {model:<25}: {res*100:.1f} см")
    print(sep)
