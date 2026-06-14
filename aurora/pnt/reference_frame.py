"""
Геодезическая система отсчёта AURORA PNT и параметры вращения Земли (EOP).

Пространственный датум суверенной системы — ПЗ-90.11 (как у ГЛОНАСС), совмещённый
с ITRF2008 на уровне миллиметров. Модель:

  1. 7-параметрическое преобразование Гельмерта ПЗ-90.11 ↔ ITRF2008 (↔ WGS-84):
     по официальным данным это практически только сдвиг ~3 мм, без значимых
     вращения и масштаба — т.е. координатная основа AURORA совместима с ГЛОНАСС,
     GPS и Galileo на сантиметровом уровне без потери точности на трансформации.

  2. Бюджет параметров вращения Земли (EOP). Для перехода ECI↔ECEF в реальном
     времени нужны прогнозируемые UT1−UTC и координаты полюса (xp, yp). Ошибка их
     прогноза разворачивает земную систему и даёт позиционную ошибку:
        δr ≈ R_⊕ · ω_⊕ · δ(UT1)        (от ошибки UT1)
        δr ≈ R_⊕ · δθ_pole             (от ошибки координат полюса)
     Это ограничивает допустимую задержку обновления EOP наземным сегментом.

Параметры ПЗ-90.11 → ITRF2008: ΔX=+0,003, ΔY=+0,001, ΔZ=+0,001 м (УНООСА/ИКГ,
ГЛОНАСС ИКД ред. 5.1). EOP — служба IERS / государственная служба времени.
"""

import os
import csv
import math
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Физические константы ──────────────────────────────────────────────────────
R_E = 6_371_000.0          # средний радиус Земли, м
OMEGA_E = 7.2921150e-5     # угловая скорость вращения Земли, рад/с
MAS2RAD = math.pi / (180.0 * 3600.0 * 1000.0)   # миллисекунда дуги → рад

# ── ПЗ-90.11 → ITRF2008 (7 параметров Гельмерта) ──────────────────────────────
# Сдвиги (м); вращения (mas); масштаб (10^-9). Практически только сдвиг.
PZ9011_TO_ITRF2008 = {
    "dX": 0.003, "dY": 0.001, "dZ": 0.001,      # м
    "wx": 0.0,   "wy": 0.0,   "wz": 0.0,        # mas (незначимы)
    "scale_ppb": 0.0,
}


def helmert(xyz: List[float], p: Dict) -> List[float]:
    """7-параметрическое преобразование Гельмерта (малые углы)."""
    x, y, z = xyz
    wx, wy, wz = p["wx"] * MAS2RAD, p["wy"] * MAS2RAD, p["wz"] * MAS2RAD
    s = 1.0 + p["scale_ppb"] * 1e-9
    xo = p["dX"] + s * (x - wz * y + wy * z)
    yo = p["dY"] + s * (wz * x + y - wx * z)
    zo = p["dZ"] + s * (-wy * x + wx * y + z)
    return [xo, yo, zo]


def eop_position_error(ut1_err_ms: float, pole_err_mas: float) -> Dict:
    """Позиционная ошибка от ошибок прогноза EOP (на поверхности Земли)."""
    err_ut1 = R_E * OMEGA_E * (ut1_err_ms * 1e-3)     # м
    err_pole = R_E * (pole_err_mas * MAS2RAD)          # м
    return {"ut1_m": err_ut1, "pole_m": err_pole,
            "total_m": math.hypot(err_ut1, err_pole)}


def compute() -> Dict:
    # Контрольная точка (≈ поверхность Земли, экватор) в ПЗ-90.11
    pt_pz = [2_845_455.0, 2_160_954.0, 5_265_993.0]   # м, ECEF
    pt_itrf = helmert(pt_pz, PZ9011_TO_ITRF2008)
    dr = math.sqrt(sum((a - b) ** 2 for a, b in zip(pt_pz, pt_itrf)))

    # EOP-бюджет: ошибка позиции vs задержка прогноза UT1
    ut1_grid = [0.05, 0.1, 0.2, 0.5, 1.0]   # мс ошибки прогноза UT1
    pole_fixed = 3.0                          # mas, типовая ошибка прогноза полюса
    eop_rows = [{"ut1_ms": u, **eop_position_error(u, pole_fixed)} for u in ut1_grid]

    return {"pt_pz": pt_pz, "pt_itrf": pt_itrf, "datum_shift_m": dr,
            "transform": PZ9011_TO_ITRF2008, "eop_rows": eop_rows,
            "pole_err_mas": pole_fixed}


def run_reference_frame_analysis(output_dir: str, label: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    res = compute()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))

    # ── Панель 1: компоненты сдвига датума ПЗ-90.11 → ITRF2008 (мм) ───────────
    comps = ["ΔX", "ΔY", "ΔZ"]
    vals_mm = [res["transform"]["dX"] * 1000, res["transform"]["dY"] * 1000,
               res["transform"]["dZ"] * 1000]
    ax1.bar(comps, vals_mm, color=["#0984e3", "#00b894", "#e17055"],
            edgecolor="white")
    for i, v in enumerate(vals_mm):
        ax1.text(i, v + 0.05, f"{v:.0f} мм", ha="center", fontweight="bold")
    ax1.axhline(0, color="#2d3436", lw=0.8)
    ax1.set_ylabel("Сдвиг компоненты, мм")
    ax1.set_title("Датум ПЗ-90.11 → ITRF2008\n(только сдвиг, |Δr| ≈ %.0f мм; "
                  "вращение/масштаб ≈ 0)" % (res["datum_shift_m"] * 1000))
    ax1.set_ylim(0, 5)
    ax1.grid(axis="y", alpha=0.3)

    # ── Панель 2: позиционная ошибка от прогноза EOP ─────────────────────────
    u = [r["ut1_ms"] for r in res["eop_rows"]]
    tot = [r["total_m"] for r in res["eop_rows"]]
    ut1c = [r["ut1_m"] for r in res["eop_rows"]]
    ax2.plot(u, tot, "o-", color="#0652DD", lw=2, label="суммарно (UT1+полюс)")
    ax2.plot(u, ut1c, "s--", color="#74b9ff", lw=1.5, label="вклад UT1")
    ax2.axhline(res["eop_rows"][0]["pole_m"], ls=":", color="#e17055",
                label=f"вклад полюса ({res['pole_err_mas']:.0f} mas)")
    ax2.axhline(0.10, ls="--", color="#636e72", lw=1)
    ax2.text(0.5, 0.11, "целевой бюджет 0,1 м", color="#636e72", fontsize=8)
    for x, y in zip(u, tot):
        ax2.text(x, y + 0.01, f"{y:.2f}", ha="center", fontsize=8)
    ax2.set_xlabel("Ошибка прогноза UT1−UTC, мс")
    ax2.set_ylabel("Позиционная ошибка, м")
    ax2.set_title("Бюджет параметров вращения Земли (EOP)")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(alpha=0.3)

    fig.suptitle(f"AURORA PNT — геодезическая система отсчёта (ПЗ-90.11) и EOP [{label}]",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(output_dir, f"reference_frame_{label}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(output_dir, f"reference_frame_{label}.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ut1_err_ms", "err_ut1_m", "err_pole_m", "err_total_m"])
        for r in res["eop_rows"]:
            w.writerow([r["ut1_ms"], f"{r['ut1_m']:.3f}",
                        f"{r['pole_m']:.3f}", f"{r['total_m']:.3f}"])

    print(f"  Геодезическая основа -- {label}")
    print(f"    Датум: ПЗ-90.11; сдвиг к ITRF2008 |Δr| = {res['datum_shift_m']*1000:.1f} мм "
          f"(вращение/масштаб ≈ 0) → совместимость с ГЛОНАСС/GPS/Galileo на см-уровне")
    base = res["eop_rows"][1]   # 0.1 ms
    print(f"    EOP: при ошибке UT1 0,1 мс и полюса 3 mas → "
          f"позиц. ошибка {base['total_m']:.2f} м (UT1 {base['ut1_m']:.2f} + полюс {base['pole_m']:.2f})")
    return res


if __name__ == "__main__":
    run_reference_frame_analysis("results/reference_frame", "phase4")
