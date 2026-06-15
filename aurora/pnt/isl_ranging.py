"""
ISL Ranging Module for АВРОРА.

Autonomous orbit determination via inter-satellite link distance measurements.
Models ephemeris holdover quality, ISL ranging geometry, and positioning impact.

Key concept (borrowed from Xona PULSAR): crosslink ranging measurements allow
satellites to maintain accurate ephemerides without continuous ground contact.
"""

import math
import os
import csv
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Physical constants
MU_EARTH = 3.986004418e14   # m^3/s^2
R_EARTH  = 6_378_137.0      # m WGS84
C_LIGHT  = 299_792_458.0    # m/s

# ISL ranging noise (1-sigma)
ISL_NOISE_CODE_M   = 0.30   # code-based ranging
ISL_NOISE_PHASE_M  = 0.01   # carrier-phase ranging

# Ephemeris error growth (conservative, based on orbital mechanics drift)
EPH_INITIAL_M           = 0.05   # after fresh MCS upload
EPH_GROWTH_NO_ISL_M_HR  = 0.50   # without ISL OD
EPH_GROWTH_WITH_ISL_M_HR = 0.02  # with ISL OD active

# ISL OD convergence
OD_CONVERGENCE_MIN  = 30    # minutes to initial OD solution
OD_UPDATE_MIN       = 10    # update interval once converged


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def orbital_period_s(altitude_m: float) -> float:
    r = R_EARTH + altitude_m
    return 2 * math.pi * math.sqrt(r**3 / MU_EARTH)


def isl_geometry(
    n_planes: int,
    n_sats_per_plane: int,
    altitude_m: float,
    inclination_deg: float,
    max_range_km: float = 5000.0,
) -> Dict:
    """
    Compute ISL geometry statistics for a Walker-Delta constellation.
    Returns link counts, range distributions, and observability metrics.
    """
    r_orb = R_EARTH + altitude_m

    # In-plane adjacent range (chord)
    theta_in  = 2 * math.pi / n_sats_per_plane
    d_inplane = 2 * r_orb * math.sin(theta_in / 2)

    # Cross-plane adjacent range (equatorial region, RAAN spacing)
    raan_step_rad = 2 * math.pi / n_planes
    d_cross_eq    = r_orb * math.sqrt(2 * (1 - math.cos(raan_step_rad)))

    # Radio horizon range at this altitude
    horizon_m = math.sqrt((r_orb**2) - (R_EARTH**2))

    # Effective ISL links per satellite
    # In-plane: always 2 (±1 neighbor in same plane)
    # Cross-plane: depends on geometry — typically 2 (one each to adjacent planes)
    links_inplane    = 2
    links_crossplane = 2
    links_total_per_sat = links_inplane + links_crossplane

    total_links = n_planes * n_sats_per_plane * links_total_per_sat // 2

    # Observability: for autonomous OD each satellite has 6 state unknowns (r, v)
    # Each ISL range provides 1 scalar observable -> need >= 6 independent ranges
    # With 4 links per sat, geometry is sufficient for full OD
    obs_per_sat = links_total_per_sat  # direct links
    od_overdetermination = obs_per_sat / 6.0  # >1 = overdetermined

    # ISL-OD GDOP analog (geometric dilution for orbit determination)
    # Approximated from uniform 4-link geometry
    isl_od_gdop = 1.0 / math.sqrt(links_total_per_sat / 3.0)

    period_min = orbital_period_s(altitude_m) / 60.0

    return {
        "n_satellites":           n_planes * n_sats_per_plane,
        "n_planes":               n_planes,
        "n_sats_per_plane":       n_sats_per_plane,
        "altitude_km":            altitude_m / 1000,
        "inclination_deg":        inclination_deg,
        "orbital_period_min":     period_min,
        "orbit_radius_km":        r_orb / 1000,
        "d_inplane_km":           d_inplane / 1000,
        "d_crossplane_eq_km":     d_cross_eq / 1000,
        "horizon_range_km":       horizon_m / 1000,
        "max_isl_range_km":       min(max_range_km, horizon_m / 1000),
        "links_inplane_per_sat":  links_inplane,
        "links_cross_per_sat":    links_crossplane,
        "links_per_sat":          links_total_per_sat,
        "total_isl_links":        total_links,
        "obs_per_sat":            obs_per_sat,
        "od_overdetermination":   od_overdetermination,
        "isl_od_gdop":            isl_od_gdop,
    }


# ---------------------------------------------------------------------------
# Ranging noise and OD accuracy
# ---------------------------------------------------------------------------

def isl_od_accuracy_m(
    n_links: int,
    noise_m: float = ISL_NOISE_CODE_M,
    gdop: float = 1.0,
) -> float:
    """Position-domain OD accuracy from ISL ranging (RSS over n_links)."""
    return gdop * noise_m / math.sqrt(n_links)


def ephemeris_error_m(
    holdover_h: float,
    with_isl_od: bool = True,
    od_convergence_h: float = OD_CONVERGENCE_MIN / 60.0,
) -> float:
    """
    Ephemeris 1-sigma error as function of time since last MCS contact.

    Without ISL OD: grows linearly at 0.5 m/hr.
    With ISL OD: after convergence, grows at 0.02 m/hr (ISL keeps sats calibrated).
    """
    if not with_isl_od:
        return EPH_INITIAL_M + EPH_GROWTH_NO_ISL_M_HR * holdover_h

    # ISL OD converges after ~30 min; before that, still using MCS-uploaded ephem
    if holdover_h < od_convergence_h:
        return EPH_INITIAL_M + EPH_GROWTH_NO_ISL_M_HR * holdover_h
    else:
        # OD locks in; only slow drift remains
        eph_at_conv = EPH_INITIAL_M + EPH_GROWTH_NO_ISL_M_HR * od_convergence_h
        return eph_at_conv + EPH_GROWTH_WITH_ISL_M_HR * (holdover_h - od_convergence_h)


def holdover_until_threshold_h(
    threshold_m: float,
    with_isl_od: bool,
    od_convergence_h: float = OD_CONVERGENCE_MIN / 60.0,
) -> float:
    """
    Time (hours) until ephemeris error exceeds threshold_m.
    Returns math.inf if threshold is never reached (ISL OD case with very large threshold).
    """
    if not with_isl_od:
        if EPH_GROWTH_NO_ISL_M_HR == 0:
            return math.inf
        return max(0.0, (threshold_m - EPH_INITIAL_M) / EPH_GROWTH_NO_ISL_M_HR)

    # With ISL OD
    eph_at_conv = EPH_INITIAL_M + EPH_GROWTH_NO_ISL_M_HR * od_convergence_h
    if eph_at_conv >= threshold_m:
        # Threshold hit before OD converges
        return max(0.0, (threshold_m - EPH_INITIAL_M) / EPH_GROWTH_NO_ISL_M_HR)
    # After convergence
    if EPH_GROWTH_WITH_ISL_M_HR == 0:
        return math.inf
    return od_convergence_h + (threshold_m - eph_at_conv) / EPH_GROWTH_WITH_ISL_M_HR


# ---------------------------------------------------------------------------
# UERE impact
# ---------------------------------------------------------------------------

def uere_with_eph_error(
    eph_error_m: float,
    base_uere_m: float = 3.10,
    eph_base_m: float = 0.50,
) -> float:
    """
    Total UERE RSS with current ephemeris error replacing the baseline ephemeris term.
    base_uere_m is the nominal UERE including eph_base_m as the ephemeris contribution.
    """
    # Remove nominal eph term, add current eph error
    uere_sq = base_uere_m**2 - eph_base_m**2 + eph_error_m**2
    return math.sqrt(max(uere_sq, base_uere_m**2 * 0.01))


def cep_from_uere(uere_m: float, pdop: float = 5.0) -> float:
    """Approximate CEP (50% circular error probable) from UERE and PDOP."""
    sigma_pos = pdop * uere_m
    return 0.59 * sigma_pos  # CEP ~ 0.59 * sigma for 2D Gaussian


# ---------------------------------------------------------------------------
# MCS contact schedule analysis
# ---------------------------------------------------------------------------

def mcs_contact_analysis(
    n_planes: int,
    n_sats_per_plane: int,
    altitude_m: float,
    n_stations: int = 21,
    station_lat_deg: float = 55.0,
    min_el_deg: float = 10.0,
) -> Dict:
    """
    Estimate maximum gap between MCS contacts for a satellite.
    Uses simplified geometric model.
    """
    r_orb   = R_EARTH + altitude_m
    period_h = orbital_period_s(altitude_m) / 3600.0

    # Ground station footprint half-angle
    rho = math.asin(R_EARTH / r_orb)
    el_rad = math.radians(min_el_deg)
    lambda_max = math.pi/2 - el_rad - rho  # max nadir angle for visibility

    # Footprint radius on ground
    fp_radius_deg = math.degrees(lambda_max)

    # Estimated coverage fraction with n_stations distributed globally
    # Very rough: each station covers a cap of area ~ 2*pi*(1-cos(lambda_max)) steradians
    sphere_frac = (1 - math.cos(lambda_max)) / 2
    total_coverage = min(1.0, n_stations * sphere_frac * 2)  # factor 2 for Russia-centric

    # Max contact gap: time between passes for a single station
    # At 1000 km, period ~105 min, passes every ~12 passes/day for mid-lat station
    passes_per_day_per_station = 24.0 / period_h * sphere_frac * 2 * math.pi
    passes_total = passes_per_day_per_station * n_stations

    # Max gap without any contact
    if passes_total > 0:
        max_gap_h = 24.0 / passes_total * n_planes * n_sats_per_plane
        # Each sat sees ~passes_total/n_sats contacts per day
        contacts_per_sat_per_day = passes_total / (n_planes * n_sats_per_plane)
        avg_gap_h = 24.0 / max(contacts_per_sat_per_day, 0.1)
    else:
        max_gap_h = 24.0
        contacts_per_sat_per_day = 0.0
        avg_gap_h = 24.0

    return {
        "n_stations":                   n_stations,
        "footprint_radius_deg":         fp_radius_deg,
        "coverage_fraction":            total_coverage,
        "passes_per_day_total":         passes_total,
        "contacts_per_sat_per_day":     contacts_per_sat_per_day,
        "avg_gap_h":                    avg_gap_h,
        "max_gap_h":                    min(max_gap_h, 24.0),
        "period_h":                     period_h,
    }


# ---------------------------------------------------------------------------
# Full analysis runner
# ---------------------------------------------------------------------------

def run_isl_ranging_analysis(
    output_dir: str,
    label: str,
    n_planes: int = 15,
    n_sats_per_plane: int = 20,
    altitude_m: float = 1_000_000,
    inclination_deg: float = 75.0,
    n_mcs_stations: int = 21,
    ranging_mode: str = "code",   # "code" or "phase"
    base_uere_m: float = 3.10,
    pdop: float = 5.15,
) -> Dict:

    os.makedirs(output_dir, exist_ok=True)

    noise_m = ISL_NOISE_CODE_M if ranging_mode == "code" else ISL_NOISE_PHASE_M

    # 1. ISL geometry
    geom = isl_geometry(n_planes, n_sats_per_plane, altitude_m, inclination_deg)

    # 2. OD accuracy from ISL
    od_acc_m = isl_od_accuracy_m(geom["links_per_sat"], noise_m, geom["isl_od_gdop"])

    # 3. MCS contact analysis
    mcs = mcs_contact_analysis(n_planes, n_sats_per_plane, altitude_m, n_mcs_stations)

    # 4. Ephemeris holdover curves
    holdover_hours = [h / 10.0 for h in range(0, 481)]  # 0..48 hours

    eph_no_isl  = [ephemeris_error_m(h, with_isl_od=False) for h in holdover_hours]
    eph_with_isl = [ephemeris_error_m(h, with_isl_od=True) for h in holdover_hours]

    # 5. UERE and CEP degradation curves
    uere_no_isl   = [uere_with_eph_error(e, base_uere_m) for e in eph_no_isl]
    uere_with_isl_ = [uere_with_eph_error(e, base_uere_m) for e in eph_with_isl]
    cep_no_isl    = [cep_from_uere(u, pdop) for u in uere_no_isl]
    cep_with_isl  = [cep_from_uere(u, pdop) for u in uere_with_isl_]

    # 6. Holdover thresholds
    thresholds_m = [0.5, 1.0, 2.0, 5.0, 10.0]
    holdover_table = []
    for th in thresholds_m:
        h_no  = holdover_until_threshold_h(th, with_isl_od=False)
        h_isl = holdover_until_threshold_h(th, with_isl_od=True)
        holdover_table.append({
            "threshold_m": th,
            "holdover_no_isl_h":   h_no,
            "holdover_with_isl_h": h_isl,
            "gain_factor":         h_isl / h_no if h_no > 0 else float("inf"),
        })

    # 7. Generate plots
    _plot_ephemeris_holdover(
        holdover_hours, eph_no_isl, eph_with_isl, output_dir, label
    )
    _plot_cep_degradation(
        holdover_hours, cep_no_isl, cep_with_isl, output_dir, label
    )
    _plot_isl_geometry(geom, output_dir, label)

    # 8. Save CSVs
    _save_holdover_csv(holdover_hours, eph_no_isl, eph_with_isl,
                       uere_no_isl, uere_with_isl_, cep_no_isl, cep_with_isl,
                       output_dir, label)
    _save_summary_csv(geom, mcs, od_acc_m, holdover_table, output_dir, label)

    result = {
        "geometry":        geom,
        "mcs":             mcs,
        "od_accuracy_m":   od_acc_m,
        "ranging_mode":    ranging_mode,
        "noise_m":         noise_m,
        "holdover_table":  holdover_table,
        "base_uere_m":     base_uere_m,
        "pdop":            pdop,
        "cep_nominal_m":   cep_from_uere(base_uere_m, pdop),
        "cep_48h_no_isl":  cep_no_isl[-1],
        "cep_48h_isl":     cep_with_isl[-1],
    }
    return result


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_ephemeris_holdover(
    hours: List[float],
    eph_no: List[float],
    eph_isl: List[float],
    output_dir: str,
    label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(hours, eph_no,  color="#e74c3c", lw=2,  label="Без ISL-OD")
    ax.semilogy(hours, eph_isl, color="#00b894", lw=2,  label="С ISL-OD (АВРОРА)")
    ax.axhline(0.5,  ls="--", color="#7f8c8d", lw=0.8, alpha=0.8, label="0,5 м")
    ax.axhline(5.0,  ls="--", color="#e67e22", lw=0.8, alpha=0.8, label="5 м")
    ax.axvline(OD_CONVERGENCE_MIN / 60.0, ls=":", color="#6c5ce7", lw=1.2, label="Сходимость OD")
    ax.set_xlabel("Время удержания (ч с последнего сеанса MCS)")
    ax.set_ylabel("Ошибка эфемерид 1σ (м)")
    ax.set_title(f"АВРОРА — Качество удержания эфемерид [{label}]")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(0, max(hours))
    plt.tight_layout()
    path = os.path.join(output_dir, f"isl_eph_holdover_{label}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_cep_degradation(
    hours: List[float],
    cep_no: List[float],
    cep_isl: List[float],
    output_dir: str,
    label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hours, cep_no,  color="#e74c3c", lw=2, label="Без ISL-OD")
    ax.plot(hours, cep_isl, color="#00b894", lw=2, label="С ISL-OD (АВРОРА)")
    ax.axhline(10.0, ls="--", color="#7f8c8d", lw=0.8, alpha=0.8, label="CEP 10 м")
    ax.axhline(50.0, ls="--", color="#e67e22", lw=0.8, alpha=0.8, label="CEP 50 м")
    ax.set_xlabel("Время удержания (ч с последнего сеанса MCS)")
    ax.set_ylabel("CEP 50% (м)")
    ax.set_title(f"АВРОРА — Деградация CEP при удержании без MCS [{label}]")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(hours))
    ax.set_ylim(0, min(300, max(cep_no) * 1.1))
    plt.tight_layout()
    path = os.path.join(output_dir, f"isl_cep_degradation_{label}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_isl_geometry(geom: Dict, output_dir: str, label: str) -> None:
    """Bar chart of key ISL geometry parameters."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Left: link distances
    ax = axes[0]
    dist_labels = ["Внутри-\nплоскостной\nISL (км)", "Меж-\nплоскостной\nISL (км)", "Дальность\nдо горизонта\n(км)"]
    dist_vals   = [
        geom["d_inplane_km"],
        geom["d_crossplane_eq_km"],
        geom["horizon_range_km"],
    ]
    colors = ["#0984e3", "#6c5ce7", "#00b894"]
    bars = ax.bar(dist_labels, dist_vals, color=colors, edgecolor="white", width=0.55)
    ax.bar_label(bars, fmt="%.0f км", padding=3, fontsize=9)
    ax.set_ylabel("Расстояние (км)")
    ax.set_title("Дальности ISL-линий")
    ax.set_ylim(0, max(dist_vals) * 1.2)
    ax.grid(axis="y", alpha=0.3)

    # Right: OD metrics
    ax2 = axes[1]
    od_labels = ["Линий/КА", "Переопределён-\nность OD", "ISL-OD\nGDOP x10"]
    od_vals   = [
        geom["links_per_sat"],
        geom["od_overdetermination"],
        geom["isl_od_gdop"] * 10,
    ]
    colors2 = ["#e17055", "#fdcb6e", "#55efc4"]
    bars2 = ax2.bar(od_labels, od_vals, color=colors2, edgecolor="white", width=0.55)
    ax2.bar_label(bars2, fmt="%.2f", padding=3, fontsize=9)
    ax2.set_title("Геометрия ISL-OD")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(f"АВРОРА — Геометрия ISL-измерений — {label}", fontsize=12, y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, f"isl_geometry_{label}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def _save_holdover_csv(
    hours, eph_no, eph_isl, uere_no, uere_isl, cep_no, cep_isl,
    output_dir, label
) -> None:
    path = os.path.join(output_dir, f"isl_holdover_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "holdover_h", "eph_no_isl_m", "eph_with_isl_m",
            "uere_no_isl_m", "uere_with_isl_m",
            "cep_no_isl_m", "cep_with_isl_m",
        ])
        for row in zip(hours, eph_no, eph_isl, uere_no, uere_isl, cep_no, cep_isl):
            w.writerow([f"{v:.4f}" for v in row])


def _save_summary_csv(geom, mcs, od_acc_m, holdover_table, output_dir, label) -> None:
    path = os.path.join(output_dir, f"isl_summary_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["parameter", "value", "unit"])
        for k, v in geom.items():
            w.writerow([k, f"{v:.3f}" if isinstance(v, float) else v, ""])
        w.writerow(["od_accuracy_code_ranging", f"{od_acc_m:.4f}", "m"])
        w.writerow(["avg_mcs_gap_h", f"{mcs['avg_gap_h']:.2f}", "h"])
        w.writerow(["max_mcs_gap_h", f"{mcs['max_gap_h']:.2f}", "h"])
        w.writerow(["contacts_per_sat_day", f"{mcs['contacts_per_sat_per_day']:.1f}", ""])
        w.writerow(["", "", ""])
        w.writerow(["threshold_m", "holdover_no_isl_h", "holdover_with_isl_h", "gain_factor"])
        for row in holdover_table:
            w.writerow([
                row["threshold_m"],
                f"{row['holdover_no_isl_h']:.2f}",
                f"{row['holdover_with_isl_h']:.1f}" if row["holdover_with_isl_h"] < 1e6 else "inf",
                f"{row['gain_factor']:.1f}" if row["gain_factor"] < 1e6 else "inf",
            ])


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def print_isl_summary(label: str, result: Dict) -> None:
    g = result["geometry"]
    m = result["mcs"]
    sep = "=" * 68

    print(f"\n{sep}")
    print(f"  ISL Ranging & Autonomous OD -- {label}")
    print(sep)
    print(f"  Constellation:  {g['n_planes']}x{g['n_sats_per_plane']} = {g['n_satellites']} sats "
          f"@ {g['altitude_km']:.0f} km, {g['inclination_deg']:.0f} deg")
    print(f"  Orbital period: {g['orbital_period_min']:.1f} min")
    print()
    print(f"  ISL Geometry:")
    print(f"    In-plane link range:    {g['d_inplane_km']:.1f} km")
    print(f"    Cross-plane link range: {g['d_crossplane_eq_km']:.1f} km")
    print(f"    Radio horizon:          {g['horizon_range_km']:.1f} km")
    print(f"    Links per satellite:    {g['links_per_sat']}")
    print(f"    Total ISL links:        {g['total_isl_links']}")
    print()
    print(f"  Autonomous OD (ISL Ranging):")
    print(f"    Ranging mode:           {result['ranging_mode']} ({result['noise_m']:.2f} m noise)")
    print(f"    OD overdetermination:   {g['od_overdetermination']:.2f}x  (>1 = well-constrained)")
    print(f"    OD GDOP:                {g['isl_od_gdop']:.3f}")
    print(f"    OD position accuracy:   {result['od_accuracy_m']*100:.1f} cm")
    print()
    print(f"  MCS Contact ({m['n_stations']} stations):")
    print(f"    Contacts per sat/day:   {m['contacts_per_sat_per_day']:.1f}")
    print(f"    Average gap:            {m['avg_gap_h']:.1f} h")
    print(f"    Max gap (worst case):   {m['max_gap_h']:.1f} h")
    print()
    print(f"  Ephemeris Holdover (time until error exceeds threshold):")
    print(f"  {'Threshold':>10}  {'Without ISL OD':>14}  {'With ISL OD':>14}  {'Gain':>8}")
    print("  " + "-" * 52)
    for row in result["holdover_table"]:
        h_no  = f"{row['holdover_no_isl_h']:.1f} h"
        h_isl = ">>48 h" if row["holdover_with_isl_h"] > 100 else f"{row['holdover_with_isl_h']:.1f} h"
        gain  = ">>100x" if row["gain_factor"] > 100 else f"{row['gain_factor']:.1f}x"
        print(f"  {row['threshold_m']:>8.1f} m  {h_no:>14}  {h_isl:>14}  {gain:>8}")
    print()
    print(f"  CEP impact (PDOP={result['pdop']:.2f}):")
    print(f"    Nominal CEP (fresh ephem):     {result['cep_nominal_m']:.2f} m")
    print(f"    CEP at 48h WITHOUT ISL OD:     {result['cep_48h_no_isl']:.1f} m")
    print(f"    CEP at 48h WITH ISL OD:        {result['cep_48h_isl']:.1f} m")
    print(sep)
