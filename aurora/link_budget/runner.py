"""
Link budget simulation runner.

For each time step, computes link budget for every visible GS–satellite pair
using positions from the PNT simulation TLE data.
"""

import math

import numpy as np
from astropy import units as astro_units
from astropy.time import Time
from tqdm import tqdm

from aurora.main import calculate_link_params, setup_ground_stations
from aurora.network_state.gsl_attachment.gsl_attachment_strategies import *  # noqa: F401,F403
from aurora.tles.read_tles import read_tles
from aurora.topology import distance_tools
from aurora.topology.satellite.satellite import Satellite
from aurora.topology.topology import ConstellationData
from aurora.network_state.helpers import _build_topologies, _compute_isls

from .models import compute_link_budget, max_leo_velocity_m_s, BANDS


def run_link_budget_sim(
    config: dict,
    tle_path: str,
    freq_band: str = "L1",
    tx_power_dbw: float = 16.0,
    tx_gain_dbi: float = 14.0,
    rx_gain_dbi: float = 3.0,
    noise_temp_k: float = 290.0,
    noise_figure_db: float = 2.0,
    bandwidth_hz: float = 10.23e6,
) -> tuple[list[dict], list[str]]:
    """
    For each timestep and each GS, compute link budget to the nearest satellite.
    Returns (snapshots, gs_names).

    Each snapshot:
        time_h, gs_name, sat_id, distance_km, elevation_deg,
        rx_power_dbw, cn0_dbhz, snr_db, doppler_hz, link_margin_db
    """
    freq_hz = BANDS.get(freq_band, BANDS["L1"])
    altitude_m = config["satellite"]["altitude_m"]
    v_max = max_leo_velocity_m_s(altitude_m)

    parsed = read_tles(tle_path)
    satellites = [
        Satellite(id=i, ephem_obj_manual=ep, ephem_obj_direct=ep)
        for i, ep in enumerate(parsed["satellites"])
    ]
    ground_stations = setup_ground_stations(config)
    max_gsl, max_isl = calculate_link_params(config)

    constellation_data = ConstellationData(
        orbits=parsed["n_orbits"],
        sats_per_orbit=parsed["n_sats_per_orbit"],
        epoch=parsed["epoch"],
        max_gsl_length_m=max_gsl,
        max_isl_length_m=max_isl,
        satellites=satellites,
    )

    gs_names = [gs.name for gs in ground_stations]
    min_el = config.get("pnt", {}).get("min_elevation_deg", 10.0)

    sim_cfg = config["simulation"]
    end_ns = int(sim_cfg["end_time_hours"] * 3600 * 1e9)
    step_ns = int(sim_cfg["time_step_minutes"] * 60 * 1e9)
    offset_ns = int(sim_cfg.get("offset_ns", 0))

    epoch_t = Time(parsed["epoch"], scale="tdb")
    snapshots = []

    time_steps = list(range(offset_ns, end_ns, step_ns))
    for t_ns in tqdm(time_steps, desc="Link budget"):
        t_abs = epoch_t + t_ns * astro_units.ns
        time_str = str(t_abs.strftime("%Y/%m/%d %H:%M:%S.%f")[:-3])
        epoch_str = parsed["epoch"]

        for gs in ground_stations:
            best_dist = float("inf")
            best_sat = None

            for sat in satellites:
                try:
                    dist = distance_tools.distance_m_ground_station_to_satellite(
                        gs, sat, epoch_str, time_str
                    )
                except Exception:
                    continue
                if dist <= max_gsl and dist < best_dist:
                    best_dist = dist
                    best_sat = sat

            if best_sat is None:
                continue

            # Estimate elevation angle: arcsin(h / d) approximation
            # More accurate: use geometry from ECEF positions
            el_deg = _elevation_deg(best_dist, altitude_m)

            # Doppler: use max orbital velocity projected (conservative estimate)
            # A proper implementation would need velocity vector; use ±v_max as bound
            # For mean estimate, use fraction based on elevation
            v_radial = v_max * math.cos(math.radians(el_deg))

            budget = compute_link_budget(
                distance_m=best_dist,
                elevation_deg=el_deg,
                freq_hz=freq_hz,
                tx_power_dbw=tx_power_dbw,
                tx_gain_dbi=tx_gain_dbi,
                rx_gain_dbi=rx_gain_dbi,
                noise_temp_k=noise_temp_k,
                noise_figure_db=noise_figure_db,
                bandwidth_hz=bandwidth_hz,
                radial_velocity_m_s=v_radial,
            )

            snapshots.append({
                "time_h": t_ns / 3.6e12,
                "gs_name": gs.name,
                "sat_id": best_sat.id,
                **budget,
            })

    return snapshots, gs_names


def _elevation_deg(distance_m: float, altitude_m: float) -> float:
    """
    Approximate elevation angle from GS to satellite.
    Uses law of cosines on the Earth–GS–satellite triangle.
    """
    R = 6_371_000.0
    r_sat = R + altitude_m
    # cos(zenith) = (R² + d² - r_sat²) / (2·R·d)  → elevation = 90 - zenith
    cos_z = (R * R + distance_m * distance_m - r_sat * r_sat) / (2 * R * distance_m)
    cos_z = max(-1.0, min(1.0, cos_z))
    zenith_deg = math.degrees(math.acos(cos_z))
    return max(0.0, 90.0 - zenith_deg)
