"""
Eclipse / Earth Shadow Analysis for AURORA PNT.

Computes when LEO satellites enter Earth's umbra/penumbra, the fraction of
each orbit spent in shadow, and the resulting implications for:
  - Solar power generation (zero during eclipse)
  - Battery sizing (must cover eclipse duration)
  - OCXO thermal stability (temperature swings affect clock performance)
  - Eclipse-induced orbit perturbations (solar radiation pressure drops to zero)

References:
  Vallado, "Fundamentals of Astrodynamics and Applications", 4th ed., Ch. 3.
  Wertz, "Space Mission Analysis and Design" (SMAD), Ch. 11.
"""

import math
import os
import csv
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Physical constants
MU_EARTH   = 3.986004418e14   # m^3/s^2
R_EARTH_M  = 6_371_000.0      # m
R_SUN_M    = 6.957e8          # m
AU_M       = 1.496e11         # m (1 AU)

# Solar power parameters (typical 28% efficient GaAs cells)
SOLAR_CELL_EFF  = 0.28
ARRAY_PACKING   = 0.85        # packing factor
SOLAR_IRRAD     = 1361.0      # W/m^2 (solar constant at 1 AU)
AURORA_ARRAY_M2 = 3.0         # solar array area (m^2) per satellite
AURORA_POWER_W  = 300.0       # satellite peak power need (W)

# Battery parameters (Li-ion, 200 Wh/kg)
BATTERY_SPECIFIC_WH_KG = 200.0
BATTERY_DOD = 0.80            # depth of discharge (80%)
AURORA_BATTERY_TARGET_MIN = 30  # target: cover 30 min of max eclipse


def orbital_period_s(altitude_m: float) -> float:
    """Circular orbital period (seconds)."""
    r = R_EARTH_M + altitude_m
    return 2 * math.pi * math.sqrt(r**3 / MU_EARTH)


def eclipse_fraction(altitude_m: float) -> float:
    """
    Fraction of circular orbit spent in Earth's umbra.

    Uses the geometric cylindrical shadow model (conservative — ignores penumbra
    and solar parallax; accurate to ~1% for LEO).
    """
    r_orb = R_EARTH_M + altitude_m
    # Half-angle of shadow cone
    sin_rho = R_EARTH_M / r_orb
    rho = math.asin(sin_rho)
    # Fraction of orbit in shadow = rho/pi (for circular orbit)
    return rho / math.pi


def eclipse_fraction_exact(altitude_m: float, inclination_deg: float,
                            sun_beta_deg: float) -> float:
    """
    More accurate eclipse fraction accounting for Sun-orbit plane angle (beta angle).

    beta = angle between Sun vector and orbital plane.
    When |beta| > rho_max, satellite never enters shadow (always in sunlight).

    Returns:
        fraction [0.0, ~0.4] of orbit in umbra.
    """
    r_orb = R_EARTH_M + altitude_m
    sin_rho = R_EARTH_M / r_orb
    rho_max_deg = math.degrees(math.asin(sin_rho))

    beta_abs = abs(sun_beta_deg)

    if beta_abs >= rho_max_deg:
        return 0.0   # sun-synchronous or polar — always in sunlight

    # Fraction of orbit in shadow
    cos_frac = math.sqrt(sin_rho**2 - math.sin(math.radians(beta_abs))**2) / math.cos(math.radians(beta_abs))
    cos_frac = min(1.0, max(0.0, cos_frac))
    return math.acos(1 - 2 * cos_frac) / (2 * math.pi) if cos_frac <= 1 else 0.0


def eclipse_duration_min(altitude_m: float, beta_deg: float = 0.0) -> float:
    """
    Eclipse duration (minutes) per orbit at given beta angle.

    beta=0: worst case (Sun in orbital plane, maximum eclipse).
    """
    frac = eclipse_fraction_exact(altitude_m, 0.0, beta_deg)
    period_s = orbital_period_s(altitude_m)
    return frac * period_s / 60.0


def max_eclipse_duration_min(altitude_m: float) -> float:
    """Worst-case eclipse duration (beta=0°, Sun in orbital plane)."""
    frac     = eclipse_fraction(altitude_m)
    period_s = orbital_period_s(altitude_m)
    return frac * period_s / 60.0


def solar_power_w(array_area_m2: float = AURORA_ARRAY_M2,
                  eta: float = SOLAR_CELL_EFF,
                  packing: float = ARRAY_PACKING,
                  degradation_per_year: float = 0.02,
                  mission_years: float = 7.0) -> float:
    """
    Solar array output power at end-of-life (W).
    Degradation: GaAs ~2%/year due to radiation at 1000 km.
    """
    bol_power = SOLAR_IRRAD * array_area_m2 * eta * packing
    eol_factor = (1 - degradation_per_year) ** mission_years
    return bol_power * eol_factor


def battery_sizing(eclipse_min: float,
                   power_eclipse_w: float = AURORA_POWER_W,
                   dod: float = BATTERY_DOD,
                   specific_wh_kg: float = BATTERY_SPECIFIC_WH_KG) -> Dict:
    """
    Size battery to cover one eclipse period.

    Args:
        eclipse_min:    Maximum eclipse duration (minutes)
        power_eclipse_w: Power consumed during eclipse (W) — typically full
        dod:            Allowed depth of discharge [0–1]
        specific_wh_kg: Battery energy density (Wh/kg)

    Returns:
        dict with required capacity, mass, and volume.
    """
    energy_req_wh = power_eclipse_w * eclipse_min / 60.0   # Wh
    capacity_wh   = energy_req_wh / dod                    # actual capacity with DOD margin
    mass_kg       = capacity_wh / specific_wh_kg
    # Li-ion typical: ~650 Wh/L
    volume_l      = capacity_wh / 650.0

    return {
        "eclipse_min":       eclipse_min,
        "power_eclipse_w":   power_eclipse_w,
        "energy_req_wh":     energy_req_wh,
        "capacity_wh":       capacity_wh,
        "battery_mass_kg":   mass_kg,
        "battery_volume_l":  volume_l,
        "dod":               dod,
    }


def ocxo_temperature_swing(eclipse_min: float,
                            sun_temp_k: float = 320.0,
                            shadow_temp_k: float = 240.0) -> Dict:
    """
    Estimate OCXO frequency deviation due to temperature cycling in eclipse.

    OCXO temperature coefficient: ~1e-10/K for high-quality units.
    Temperature swings between sunlit side and shadow side of orbit.
    """
    temp_coeff_per_k = 1e-10      # fractional frequency change per Kelvin
    delta_t_k        = sun_temp_k - shadow_temp_k
    df_f             = temp_coeff_per_k * delta_t_k   # fractional freq change
    df_ppb           = df_f * 1e9                     # in ppb
    # Convert to timing error over eclipse duration
    timing_err_ns    = df_ppb * 1e-9 * 3e8 * eclipse_min * 60 * 1e9 / 3e8
    # Simpler: timing error = df/f * T_eclipse * 1e9 ns
    timing_err_ns2   = df_f * eclipse_min * 60 * 1e9

    return {
        "sun_temp_k":          sun_temp_k,
        "shadow_temp_k":       shadow_temp_k,
        "delta_t_k":           delta_t_k,
        "temp_coeff_per_k":    temp_coeff_per_k,
        "df_f":                df_f,
        "df_ppb":              df_ppb,
        "timing_error_ns":     timing_err_ns2,
        "eclipse_min":         eclipse_min,
    }


def run_eclipse_analysis(
    output_dir: str,
    label: str,
    altitude_m: float = 1_000_000.0,
    n_sats: int = 180,
) -> Dict:
    """Run full eclipse analysis and produce plots + CSV."""
    os.makedirs(output_dir, exist_ok=True)

    alt_km = altitude_m / 1000

    # ── Key values at AURORA altitude ─────────────────────────────────────────
    period_min    = orbital_period_s(altitude_m) / 60.0
    frac_worst    = eclipse_fraction(altitude_m)           # beta=0
    eclipse_min   = max_eclipse_duration_min(altitude_m)   # worst case
    orbits_per_day = 1440.0 / period_min

    # Beta angle sweep: 0° (worst) to rho_max (never in eclipse)
    sin_rho     = R_EARTH_M / (R_EARTH_M + altitude_m)
    rho_max_deg = math.degrees(math.asin(sin_rho))
    beta_range  = list(range(0, int(rho_max_deg) + 5, 1))
    eclipse_by_beta = [eclipse_duration_min(altitude_m, b) for b in beta_range]

    # ── Solar power ───────────────────────────────────────────────────────────
    bol_power = solar_power_w()
    eol_power = solar_power_w(mission_years=7.0)
    power_margin_w = bol_power - AURORA_POWER_W
    power_margin_eol = eol_power - AURORA_POWER_W

    # ── Battery sizing ────────────────────────────────────────────────────────
    batt = battery_sizing(eclipse_min, power_eclipse_w=AURORA_POWER_W)

    # ── OCXO thermal effect ───────────────────────────────────────────────────
    ocxo = ocxo_temperature_swing(eclipse_min)

    # ── Altitude sweep ────────────────────────────────────────────────────────
    alt_range_km = list(range(400, 1501, 50))
    eclipse_by_alt  = [max_eclipse_duration_min(a * 1000) for a in alt_range_km]
    period_by_alt   = [orbital_period_s(a * 1000) / 60 for a in alt_range_km]
    fraction_by_alt = [eclipse_fraction(a * 1000) * 100 for a in alt_range_km]

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_eclipse_vs_beta(beta_range, eclipse_by_beta, rho_max_deg, output_dir, label, alt_km)
    _plot_eclipse_vs_altitude(alt_range_km, eclipse_by_alt, fraction_by_alt, period_by_alt,
                               output_dir, label)
    _save_eclipse_csv(alt_range_km, eclipse_by_alt, fraction_by_alt, output_dir, label)

    return {
        "altitude_m":          altitude_m,
        "alt_km":              alt_km,
        "n_sats":              n_sats,
        "period_min":          period_min,
        "orbits_per_day":      orbits_per_day,
        "eclipse_fraction_worst": frac_worst,
        "eclipse_min_worst":   eclipse_min,
        "eclipse_min_per_day": eclipse_min * orbits_per_day,
        "rho_max_deg":         rho_max_deg,
        "solar_bol_w":         bol_power,
        "solar_eol_w":         eol_power,
        "aurora_power_w":      AURORA_POWER_W,
        "power_margin_bol_w":  power_margin_w,
        "power_margin_eol_w":  power_margin_eol,
        "battery":             batt,
        "ocxo_thermal":        ocxo,
        "beta_range":          beta_range,
        "eclipse_by_beta":     eclipse_by_beta,
    }


def _plot_eclipse_vs_beta(beta_range, eclipse_by_beta, rho_max_deg,
                           output_dir, label, alt_km):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(beta_range, eclipse_by_beta, color="#6c5ce7", lw=2)
    ax.fill_between(beta_range, eclipse_by_beta, alpha=0.2, color="#6c5ce7")
    ax.axvline(rho_max_deg, ls="--", color="#00b894", lw=1.5,
               label=f"Граница без затенения: {rho_max_deg:.1f}°")
    ax.axhline(35.0, ls=":", color="#e17055", lw=1.2, label="Типовой расчёт АКБ ~35 мин")
    ax.set_xlabel("Бета-угол Солнца (градусы)")
    ax.set_ylabel("Длительность затенения за виток (мин)")
    ax.set_title(f"AURORA — Длительность затенения от угла Солнца [{label}]\n"
                 f"(высота = {alt_km:.0f} км)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"eclipse_vs_beta_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_eclipse_vs_altitude(alt_range_km, eclipse_by_alt, fraction_by_alt,
                               period_by_alt, output_dir, label):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(alt_range_km, eclipse_by_alt, color="#e17055", lw=2, label="Затенение (мин)")
    axes[0].plot(alt_range_km, period_by_alt, color="#0984e3", lw=2, ls="--", label="Период орбиты (мин)")
    axes[0].axvline(1000, ls=":", color="#6c5ce7", lw=1.2, label="AURORA 1000 км")
    axes[0].set_xlabel("Высота орбиты (км)")
    axes[0].set_ylabel("Длительность (мин)")
    axes[0].set_title("Затенение и период орбиты от высоты")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(alt_range_km, fraction_by_alt, color="#00b894", lw=2)
    axes[1].axvline(1000, ls=":", color="#6c5ce7", lw=1.2)
    axes[1].axhline(35, ls="--", color="#e17055", lw=1.2, label="Орбита GPS ~35%")
    axes[1].set_xlabel("Высота орбиты (км)")
    axes[1].set_ylabel("Доля затенения (% витка)")
    axes[1].set_title("Доля затенения от высоты орбиты")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"AURORA PNT — Анализ затенения [{label}]")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"eclipse_vs_altitude_{label}.png"), dpi=150)
    plt.close(fig)


def _save_eclipse_csv(alt_range_km, eclipse_by_alt, fraction_by_alt, output_dir, label):
    path = os.path.join(output_dir, f"eclipse_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["altitude_km", "eclipse_min_worst", "eclipse_fraction_pct",
                    "period_min", "orbits_per_day"])
        for alt, ecl, frac in zip(alt_range_km, eclipse_by_alt, fraction_by_alt):
            period = orbital_period_s(alt * 1000) / 60
            w.writerow([alt, f"{ecl:.1f}", f"{frac:.1f}", f"{period:.2f}",
                        f"{1440/period:.2f}"])


def print_eclipse_summary(label: str, result: Dict) -> None:
    sep = "=" * 66

    batt  = result["battery"]
    ocxo  = result["ocxo_thermal"]

    print(f"\n{sep}")
    print(f"  Eclipse / Earth Shadow Analysis -- {label}")
    print(sep)
    print(f"  Altitude:        {result['alt_km']:.0f} km")
    print(f"  Orbital period:  {result['period_min']:.1f} min  "
          f"({result['orbits_per_day']:.1f} orbits/day)")
    print()
    print(f"  Eclipse (worst case, beta=0 / Sun in orbital plane):")
    print(f"    Duration:      {result['eclipse_min_worst']:.1f} min  "
          f"({result['eclipse_fraction_worst']*100:.1f}% of orbit)")
    print(f"    Per day:       {result['eclipse_min_per_day']:.0f} min/day "
          f"({result['eclipse_min_per_day']/60:.1f} h/day in shadow)")
    print(f"    No-eclipse if: |beta| > {result['rho_max_deg']:.1f}°")
    print()
    print(f"  Solar power:")
    print(f"    Array area:    {AURORA_ARRAY_M2:.1f} m^2  "
          f"(eta={SOLAR_CELL_EFF*100:.0f}%, packing={ARRAY_PACKING*100:.0f}%)")
    print(f"    BOL output:    {result['solar_bol_w']:.0f} W")
    print(f"    EOL output:    {result['solar_eol_w']:.0f} W  (7 yr, 2%/yr degradation)")
    print(f"    Satellite need:{result['aurora_power_w']:.0f} W")
    print(f"    BOL margin:    {result['power_margin_bol_w']:+.0f} W  "
          f"{'[OK]' if result['power_margin_bol_w'] > 0 else '[SHORTFALL]'}")
    print(f"    EOL margin:    {result['power_margin_eol_w']:+.0f} W  "
          f"{'[OK]' if result['power_margin_eol_w'] > 0 else '[SHORTFALL - review array sizing]'}")
    print()
    print(f"  Battery sizing (cover {result['eclipse_min_worst']:.1f} min @ {AURORA_POWER_W:.0f} W):")
    print(f"    Energy needed: {batt['energy_req_wh']:.0f} Wh")
    print(f"    Capacity (DOD={batt['dod']*100:.0f}%): {batt['capacity_wh']:.0f} Wh")
    print(f"    Battery mass:  {batt['battery_mass_kg']:.1f} kg  "
          f"({BATTERY_SPECIFIC_WH_KG:.0f} Wh/kg Li-ion)")
    print(f"    Battery volume:{batt['battery_volume_l']:.2f} L")
    print()
    print(f"  OCXO thermal effect (eclipse entry/exit temperature cycling):")
    print(f"    Sun side:      {ocxo['sun_temp_k']:.0f} K")
    print(f"    Shadow side:   {ocxo['shadow_temp_k']:.0f} K")
    print(f"    Delta T:       {ocxo['delta_t_k']:.0f} K")
    print(f"    OCXO temp coeff: {ocxo['temp_coeff_per_k']:.0e} /K")
    print(f"    Freq shift:    {ocxo['df_ppb']:.3f} ppb")
    print(f"    Timing error:  {ocxo['timing_error_ns']:.1f} ns  (over {ocxo['eclipse_min']:.1f} min eclipse)")
    print(f"    => Thermal compensation or oven stabilization required")
    print(sep)
