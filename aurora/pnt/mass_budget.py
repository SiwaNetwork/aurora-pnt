"""
Satellite Mass and Volume Budget for AURORA PNT.

Detailed subsystem mass breakdown for a ~150 kg LEO navigation satellite.
Includes dry mass, propellant budget (station-keeping + deorbit), and
structural volume allocations.

Reference: SMAD 3rd ed., Appendix A; ECSS-M-ST-10C (mass properties).
"""

import math
import os
import csv
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Physical constants
G0 = 9.80665   # m/s²

# ── Subsystem mass allocations ────────────────────────────────────────────────
# Each entry: (mass_kg, volume_l, power_w, desc)
SUBSYSTEMS = {
    # Structure & Mechanisms
    "Structure & panels":          (18.0,  40.0,   0.0, "Primary structure, panels, brackets"),
    "Separation system":           ( 2.5,   3.0,   0.0, "Clamp band + ADCS springs"),
    "Solar array mechanism":       ( 2.0,   2.0,   0.0, "Deployment hinges + drives"),
    # Power
    "Solar array (GaAs, 3 m²)":   ( 6.0,   2.0,   0.0, "3J GaAs, BOL 28%, 3.0 m² deployed"),
    "Battery (Li-ion)":            ( 3.5,   5.5,   0.0, "54.5 Wh/0.27 kg — sized to eclipse"),
    "EPS (power conditioning)":    ( 4.0,   4.0,  10.0, "PCDU, regulators, protection"),
    "Solar array harness":         ( 1.5,   0.5,   0.0, "Array-to-PCDU wiring"),
    # AOCS
    "Star trackers (x2)":          ( 2.0,   1.5,   8.0, "0.4 kg each, 5 arcsec accuracy"),
    "IMU (gyroscope)":             ( 0.8,   0.5,   4.0, "MEMS/FOG, 3-axis"),
    "Magnetometer":                ( 0.3,   0.2,   0.5, "3-axis fluxgate"),
    "Reaction wheels (x4)":        ( 4.0,   3.0,   8.0, "4 Nms, 4 wheels for redundancy"),
    "AOCS computer":               ( 1.0,   0.8,   5.0, "Embedded AOCS processor"),
    # Navigation payload
    "PNT signal generator":        ( 5.0,   4.0,  40.0, "L1+L5 BPSK signal generator"),
    "Navigation antennas (x2)":    ( 3.0,   2.0,   0.0, "L1/L5 nadir patch array"),
    "RF amplifiers + diplexer":    ( 2.0,   1.5,  40.0, "20 W PA x2 per signal"),
    "On-board GNSS receiver":      ( 0.5,   0.3,   5.0, "GPS/GLONASS orbit determination"),
    # Atomic clock
    "Cs frequency standard":       ( 0.7,   0.5,   4.0, "1 per plane anchor (15 total)"),
    "Rb frequency standard":       ( 0.4,   0.3,   3.0, "3 per plane relay (45 total)"),
    "OCXO oscillator":             ( 0.2,   0.1,   2.5, "16 per plane terminal (240 total)"),
    # ISL
    "ISL Ka-band transceiver":     ( 3.5,   2.5,  30.0, "Ka 26 GHz, 10 W TX, 4 links"),
    "ISL antennas (x4)":           ( 2.0,   1.0,   0.0, "Steerable Ka phased arrays"),
    # TT&C
    "S-band TT&C radio":           ( 1.5,   1.0,  12.0, "2 W TX, 2 kbps uplink"),
    "TT&C antenna":                ( 0.3,   0.2,   0.0, "Omnidirectional S-band"),
    # OBC
    "On-board computer":           ( 1.2,   0.8,  12.0, "Main CPU + 64 GB storage"),
    "Software & FPGA":             ( 0.5,   0.3,   3.0, "Navigation signal processing FPGA"),
    # Thermal
    "Radiators":                   ( 2.0,   0.5,   0.0, "Passive radiator panels"),
    "MLI blankets":                ( 1.5,   1.0,   0.0, "Multi-layer insulation"),
    "Thermal heaters":             ( 0.5,   0.2,  25.0, "OCXO + cold-face heaters"),
    "Heat pipes":                  ( 0.8,   0.3,   0.0, "Isothermal panel coupling"),
    # Propulsion
    "Propellant tank":             ( 2.5,   8.0,   0.0, "CFRP overwrap, 15 L"),
    "Thruster (x4 cold-gas/ion)":  ( 1.5,   1.0,   0.0, "4 x 50 mN thrusters"),
    "Feed system + valves":        ( 1.0,   0.5,   0.0, "Latch valves, regulators, lines"),
    # Harness & misc
    "Satellite harness":           ( 4.0,   2.0,   0.0, "All internal cabling"),
    "Fasteners & adhesives":       ( 1.0,   0.5,   0.0, "Screws, brackets, epoxy"),
}

# Propellant budget
PROPELLANT = {
    "station_keeping_dv_ms":  21.0,    # m/s/year × 7 yr = 147 m/s over life
    "deorbit_dv_ms":          214.1,   # to 200 km perigee (from deorbit.py)
    "contingency_pct":        10.0,    # 10% propellant contingency
    "isp_s":                  220.0,   # cold-gas/hydrazine Isp (s)
}

# Design margins per ECSS-M-ST-10C
MASS_MARGINS = {
    "system_margin_pct":    5.0,   # overall system margin
    "subsystem_margin_pct": 3.0,   # per-subsystem maturity margin
}

# Mission design target
AURORA_MASS_TARGET_KG = 150.0
AURORA_VOLUME_LIMIT_L = 120.0   # stacked dispenser envelope


def propellant_mass_kg(
    dry_mass_kg: float,
    total_dv_ms: float,
    isp_s: float,
    contingency_pct: float = 10.0,
) -> Dict:
    """Tsiolkovsky rocket equation for total propellant mass."""
    mp_nominal = dry_mass_kg * (math.exp(total_dv_ms / (isp_s * G0)) - 1)
    mp_with_cont = mp_nominal * (1 + contingency_pct / 100)
    return {
        "dry_mass_kg":       dry_mass_kg,
        "total_dv_ms":       total_dv_ms,
        "isp_s":             isp_s,
        "mp_nominal_kg":     mp_nominal,
        "mp_total_kg":       mp_with_cont,
        "contingency_pct":   contingency_pct,
    }


def run_mass_budget_analysis(
    output_dir: str,
    label: str,
    n_sats: int = 180,
    mission_years: float = 7.0,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    # Sum up subsystem masses and volumes
    mass_by_sub = {name: vals[0] for name, vals in SUBSYSTEMS.items()}
    vol_by_sub  = {name: vals[1] for name, vals in SUBSYSTEMS.items()}
    pow_by_sub  = {name: vals[2] for name, vals in SUBSYSTEMS.items()}

    dry_mass_raw = sum(mass_by_sub.values())
    dry_mass_margin = dry_mass_raw * (1 + MASS_MARGINS["subsystem_margin_pct"] / 100)
    dry_mass_total  = dry_mass_margin * (1 + MASS_MARGINS["system_margin_pct"] / 100)

    # Propellant
    sk_dv = PROPELLANT["station_keeping_dv_ms"] * mission_years
    deo_dv = PROPELLANT["deorbit_dv_ms"]
    total_dv = sk_dv + deo_dv
    prop = propellant_mass_kg(dry_mass_total, total_dv, PROPELLANT["isp_s"],
                               PROPELLANT["contingency_pct"])

    wet_mass = dry_mass_total + prop["mp_total_kg"]
    total_volume = sum(vol_by_sub.values())

    # Group by category for pie chart
    categories = {
        "Structure & Mechanisms": ["Structure & panels", "Separation system", "Solar array mechanism"],
        "Power (EPS + SA + Battery)": ["Solar array (GaAs, 3 m²)", "Battery (Li-ion)", "EPS (power conditioning)", "Solar array harness"],
        "AOCS": ["Star trackers (x2)", "IMU (gyroscope)", "Magnetometer", "Reaction wheels (x4)", "AOCS computer"],
        "Navigation payload": ["PNT signal generator", "Navigation antennas (x2)", "RF amplifiers + diplexer", "On-board GNSS receiver"],
        "Atomic clocks": ["Cs frequency standard", "Rb frequency standard", "OCXO oscillator"],
        "ISL": ["ISL Ka-band transceiver", "ISL antennas (x4)"],
        "TT&C + OBC": ["S-band TT&C radio", "TT&C antenna", "On-board computer", "Software & FPGA"],
        "Thermal": ["Radiators", "MLI blankets", "Thermal heaters", "Heat pipes"],
        "Propulsion (dry)": ["Propellant tank", "Thruster (x4 cold-gas/ion)", "Feed system + valves"],
        "Harness & misc": ["Satellite harness", "Fasteners & adhesives"],
    }
    cat_mass = {}
    for cat, subs in categories.items():
        cat_mass[cat] = sum(mass_by_sub.get(s, 0) for s in subs)

    _plot_mass_breakdown(cat_mass, dry_mass_total, wet_mass, prop, output_dir, label)
    _plot_mass_pie(cat_mass, output_dir, label)
    _save_mass_csv(mass_by_sub, vol_by_sub, prop, dry_mass_total, wet_mass, output_dir, label)

    return {
        "dry_mass_raw_kg":    dry_mass_raw,
        "dry_mass_kg":        dry_mass_total,
        "propellant_kg":      prop["mp_total_kg"],
        "wet_mass_kg":        wet_mass,
        "total_volume_l":     total_volume,
        "total_dv_ms":        total_dv,
        "sk_dv_ms":           sk_dv,
        "deorbit_dv_ms":      deo_dv,
        "mass_margin_pct":    (dry_mass_total - dry_mass_raw) / dry_mass_raw * 100,
        "mass_vs_target_kg":  wet_mass - AURORA_MASS_TARGET_KG,
        "cat_mass":           cat_mass,
        "subsystems":         SUBSYSTEMS,
        "n_sats":             n_sats,
        "fleet_mass_t":       wet_mass * n_sats / 1000,
    }


def _plot_mass_breakdown(cat_mass, dry_mass, wet_mass, prop, output_dir, label):
    fig, ax = plt.subplots(figsize=(12, 6))
    names  = list(cat_mass.keys())
    masses = list(cat_mass.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(names)))

    bars = ax.barh(names, masses, color=colors, edgecolor="white", lw=0.8)
    ax.axvline(0, color="black", lw=0.8)
    for bar, v in zip(bars, masses):
        ax.text(v + 0.3, bar.get_y() + bar.get_height()/2,
                f"{v:.1f} kg", va="center", fontsize=8)

    ax.set_xlabel("Масса (кг)")
    ax.set_title(f"AURORA — Бюджет массы спутника [{label}]\n"
                 f"Сухая: {dry_mass:.1f} кг  |  "
                 f"Топливо: {prop['mp_total_kg']:.1f} кг  |  "
                 f"Заправл.: {wet_mass:.1f} кг  (цель {AURORA_MASS_TARGET_KG:.0f} кг)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"mass_breakdown_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_mass_pie(cat_mass, output_dir, label):
    fig, ax = plt.subplots(figsize=(10, 7))
    labels = [f"{k}\n{v:.1f} kg" for k, v in cat_mass.items()]
    colors = plt.cm.Set3(np.linspace(0, 1, len(cat_mass)))
    ax.pie(list(cat_mass.values()), labels=labels, colors=colors,
           autopct="%1.0f%%", startangle=90, pctdistance=0.75, textprops={"fontsize": 8})
    ax.set_title(f"AURORA — Распределение сухой массы [{label}]")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"mass_pie_{label}.png"), dpi=150)
    plt.close(fig)


def _save_mass_csv(mass_by_sub, vol_by_sub, prop, dry_mass, wet_mass, output_dir, label):
    path = os.path.join(output_dir, f"mass_budget_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subsystem", "mass_kg", "volume_l", "desc"])
        for name, vals in SUBSYSTEMS.items():
            w.writerow([name, vals[0], vals[1], vals[3]])
        w.writerow(["", "", "", ""])
        w.writerow(["DRY MASS (with margins)", f"{dry_mass:.2f}", "", ""])
        w.writerow(["Propellant (nominal+cont)", f"{prop['mp_total_kg']:.2f}", "", ""])
        w.writerow(["WET MASS", f"{wet_mass:.2f}", "", ""])


def print_mass_budget_summary(label: str, result: Dict) -> None:
    sep = "=" * 68
    print(f"\n{sep}")
    print(f"  Satellite Mass Budget -- {label}")
    print(sep)
    print(f"  {'Dry mass (raw)':<40} {result['dry_mass_raw_kg']:>8.1f} kg")
    print(f"  {'Dry mass (+ margins)':<40} {result['dry_mass_kg']:>8.1f} kg")
    print(f"    Subsystem margin: {MASS_MARGINS['subsystem_margin_pct']:.0f}%   "
          f"System margin: {MASS_MARGINS['system_margin_pct']:.0f}%")
    print()
    print(f"  Propellant breakdown:")
    print(f"    Station-keeping:  {result['sk_dv_ms']:.0f} m/s (7 yr × 21 m/s/yr)")
    print(f"    Deorbit:          {result['deorbit_dv_ms']:.1f} m/s (to 200 km perigee)")
    print(f"    Total ΔV:         {result['total_dv_ms']:.0f} m/s  |  Isp = {PROPELLANT['isp_s']:.0f} s")
    print(f"    Propellant mass:  {result['propellant_kg']:.1f} kg  "
          f"(+{PROPELLANT['contingency_pct']:.0f}% contingency)")
    print()
    print(f"  {'WET MASS':<40} {result['wet_mass_kg']:>8.1f} kg  "
          f"({'OK' if result['wet_mass_kg'] <= AURORA_MASS_TARGET_KG else 'EXCEEDS TARGET by '+str(round(result['mass_vs_target_kg'],1))+' kg'})")
    print(f"  Target:                                  {AURORA_MASS_TARGET_KG:.0f} kg")
    print(f"  Total volume:                            {result['total_volume_l']:.0f} L  "
          f"(limit {AURORA_VOLUME_LIMIT_L:.0f} L)")
    print()
    print(f"  Fleet totals ({result['n_sats']} satellites):")
    print(f"    Fleet wet mass:   {result['fleet_mass_t']:.1f} t")
    print(f"    Launches needed:  ~{math.ceil(result['fleet_mass_t'] / 6.5):.0f}  "
          f"(6.5 t payload capacity per launch to 1000 km)")
    print(sep)
