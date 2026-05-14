"""
GLONASS constellation model for combined LEO+GLONASS geometry analysis.

GLONASS-M (current operational constellation):
  - 3 orbital planes, RAAN spacing 120 deg
  - 8 satellites per plane, MA spacing 45 deg
  - Inclination: 64.8 deg
  - Altitude: ~19,136 km (quasi-circular)
  - Period: 11h 15.8 min -> 2.131 rev/day
  - Eccentricity: ~0

Reference:
  - GLONASS ICD Edition 5.1, 2008
  - https://www.glonass-iac.ru/
"""

import math
import os
import tempfile

import numpy as np
from sgp4.api import SatrecArray

from aurora.pnt.coverage import (
    _ecef_to_enu,
    _gmst_rad,
    _teme_to_ecef,
    _enu_to_az_el,
    geodetic_to_ecef_km,
    load_satrec_from_tle_file,
)
from aurora.tles.generate_tles_from_scratch import generate_tles_from_scratch_with_sgp

# Operational GLONASS-M constellation parameters
GLONASS_PARAMS = {
    "num_orbits": 3,
    "num_sats_per_orbit": 8,
    "inclination_degree": 64.8,
    "altitude_m": 19_136_000,
    "eccentricity": 1e-7,
    "arg_of_perigee_degree": 0.0,
    # Period = 11h 15.8 min = 40548 s -> 86400/40548 = 2.131 rev/day
    "mean_motion_rev_per_day": 2.131,
    # GLONASS planes use equal RAAN spacing with no inter-plane phase shift
    "phase_diff": False,
}


def generate_glonass_satrec_array(output_dir: str = None) -> tuple[SatrecArray, str]:
    """
    Generate Walker-Delta TLEs matching GLONASS-M and return (SatrecArray, tle_path).

    Args:
        output_dir: Directory to write TLE file. Uses a temp dir if None.

    Returns:
        (SatrecArray, tle_path) — array for SGP4 propagation, path to written TLE file.
    """
    if output_dir is None:
        tle_dir = tempfile.mkdtemp()
    else:
        tle_dir = output_dir
        os.makedirs(tle_dir, exist_ok=True)

    tle_path = os.path.join(tle_dir, "glonass.txt")
    p = GLONASS_PARAMS
    generate_tles_from_scratch_with_sgp(
        tle_path,
        "GLONASS",
        p["num_orbits"],
        p["num_sats_per_orbit"],
        p["phase_diff"],
        p["inclination_degree"],
        p["eccentricity"],
        p["arg_of_perigee_degree"],
        p["mean_motion_rev_per_day"],
    )
    satrecs = load_satrec_from_tle_file(tle_path)
    return SatrecArray(satrecs), tle_path


def compute_glonass_az_el_at_time(
    glonass_array: SatrecArray,
    grid_points: list[tuple],
    jd: float,
    fr: float,
    min_elevation_deg: float = 5.0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Compute GLONASS satellite az/el for all grid points at a single time step.

    Returns list of (az_rad, el_rad) tuples — one per grid point, arrays contain
    only satellites above min_elevation_deg.

    GLONASS uses a lower default mask (5 deg) than LEO (10 deg): MEO geometry
    provides good signal power at shallow elevations.
    """
    e, r_teme, _ = glonass_array.sgp4(
        np.array([jd], dtype=np.float64),
        np.array([fr], dtype=np.float64),
    )
    e = np.asarray(e)[:, 0]
    r_teme = np.asarray(r_teme)[:, 0, :]
    valid = (e == 0)

    gmst = _gmst_rad(jd, fr)
    r_ecef = _teme_to_ecef(r_teme, gmst)

    min_el_rad = math.radians(min_elevation_deg)
    results = []

    for lat, lon, alt_m in grid_points:
        lat_r = math.radians(lat)
        lon_r = math.radians(lon)
        obs_ecef = geodetic_to_ecef_km(lat, lon, alt_m)

        enu = _ecef_to_enu(r_ecef, lat_r, lon_r, obs_ecef)
        az, el = _enu_to_az_el(enu)
        visible = valid & (el >= min_el_rad)
        results.append((az[visible], el[visible]))

    return results
