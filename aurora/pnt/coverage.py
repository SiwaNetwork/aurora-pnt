"""
Fast satellite visibility computation using sgp4 batch API + numpy.

Pipeline per time step:
  1. sgp4 batch propagation → TEME positions (km)
  2. TEME → ECEF via GMST rotation
  3. ECEF → local ENU for each grid point
  4. Elevation / azimuth from ENU
  5. Filter by elevation mask
"""

import math

import numpy as np
from sgp4.api import Satrec, SatrecArray, jday

from aurora.pnt.dop import compute_dop


# ── WGS-72 ellipsoid (matches rest of codebase) ─────────────────────────────
_A_KM = 6378.135
_F = 1.0 / 298.26
_E2 = 2 * _F - _F ** 2


def altitude_to_mean_motion(altitude_m: float) -> float:
    """Compute mean motion (rev/day) from orbital altitude (m)."""
    MU = 398600.4418  # km³/s²
    r_km = _A_KM + altitude_m / 1000.0
    n_rad_s = math.sqrt(MU / r_km ** 3)
    return n_rad_s * 86400.0 / (2.0 * math.pi)


def geodetic_to_ecef_km(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    """Convert geodetic coordinates to ECEF (km). WGS-72."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    N = _A_KM / math.sqrt(1.0 - _E2 * math.sin(lat) ** 2)
    alt_km = alt_m / 1000.0
    x = (N + alt_km) * math.cos(lat) * math.cos(lon)
    y = (N + alt_km) * math.cos(lat) * math.sin(lon)
    z = (N * (1.0 - _E2) + alt_km) * math.sin(lat)
    return np.array([x, y, z])


def _gmst_rad(jd: float, fr: float) -> float:
    """Greenwich Mean Sidereal Time (radians) from Julian date parts."""
    jd_ut1 = jd + fr
    T = (jd_ut1 - 2451545.0) / 36525.0
    gmst_deg = (
        280.46061837
        + 360.98564736629 * (jd_ut1 - 2451545.0)
        + T * T * (0.000387933 - T / 38710000.0)
    )
    return math.radians(gmst_deg % 360.0)


def _teme_to_ecef(r_teme: np.ndarray, gmst: float) -> np.ndarray:
    """Rotate TEME positions (n_sats, 3) to ECEF."""
    cg, sg = math.cos(gmst), math.sin(gmst)
    r_ecef = np.empty_like(r_teme)
    r_ecef[:, 0] = r_teme[:, 0] * cg + r_teme[:, 1] * sg
    r_ecef[:, 1] = -r_teme[:, 0] * sg + r_teme[:, 1] * cg
    r_ecef[:, 2] = r_teme[:, 2]
    return r_ecef


def _ecef_to_enu(sat_ecef: np.ndarray, obs_lat_rad: float,
                 obs_lon_rad: float, obs_ecef: np.ndarray) -> np.ndarray:
    """
    Convert satellite ECEF positions (n_sats, 3) to local ENU vectors
    relative to an observer at obs_ecef (3,).
    """
    delta = sat_ecef - obs_ecef          # (n_sats, 3)
    slat, clat = math.sin(obs_lat_rad), math.cos(obs_lat_rad)
    slon, clon = math.sin(obs_lon_rad), math.cos(obs_lon_rad)

    east  = -slon * delta[:, 0] + clon * delta[:, 1]
    north = -slat * clon * delta[:, 0] - slat * slon * delta[:, 1] + clat * delta[:, 2]
    up    =  clat * clon * delta[:, 0] + clat * slon * delta[:, 1] + slat * delta[:, 2]

    return np.column_stack([east, north, up])


def _enu_to_az_el(enu: np.ndarray):
    """Return (az_rad, el_rad) from ENU vectors (n_sats, 3)."""
    horiz = np.hypot(enu[:, 0], enu[:, 1])
    el = np.arctan2(enu[:, 2], horiz)
    az = np.arctan2(enu[:, 0], enu[:, 1]) % (2.0 * math.pi)
    return az, el


def load_satrec_from_tle_file(tle_filename: str) -> list[Satrec]:
    """
    Parse TLE file (AURORA PNT format) and return list of sgp4 Satrec objects.

    File format:
        <n_orbits> <n_sats_per_orbit>
        <name> <id>
        <TLE line 1>
        <TLE line 2>
        ...
    """
    satrecs = []
    with open(tle_filename) as f:
        f.readline()  # skip header
        while True:
            name_line = f.readline()
            if not name_line:
                break
            line1 = f.readline().strip()
            line2 = f.readline().strip()
            if not line1 or not line2:
                break
            satrecs.append(Satrec.twoline2rv(line1, line2))
    return satrecs


def compute_grid_visibility_at_time(
    sat_array: SatrecArray,
    grid_points: list[tuple],        # [(lat_deg, lon_deg, alt_m), ...]
    jd: float,
    fr: float,
    min_elevation_deg: float = 10.0,
) -> list[dict]:
    """
    Compute visibility and DOP for all grid points at a single time step.

    Args:
        sat_array:          SatrecArray of all satellites
        grid_points:        List of (lat_deg, lon_deg, alt_m) tuples
        jd:                 Julian date (integer part)
        fr:                 Julian date (fractional part)
        min_elevation_deg:  Elevation mask in degrees

    Returns:
        List of dicts (one per grid point):
          {lat, lon, n_sats, pdop, hdop, vdop, gdop, tdop}
          DOP values are None when fewer than 4 satellites are visible.
    """
    # 1. Batch SGP4 propagation → TEME positions
    # SatrecArray.sgp4(jd, fr) where jd/fr have shape (n_times,)
    # returns e:(n_sats,n_times), r:(n_sats,n_times,3).
    # Pass a single time point as shape-(1,) arrays, then squeeze.
    e, r_teme, _ = sat_array.sgp4(
        np.array([jd], dtype=np.float64),
        np.array([fr], dtype=np.float64),
    )
    # Squeeze time dimension → (n_sats,) and (n_sats, 3)
    e = np.asarray(e)[:, 0]           # (n_sats,)
    r_teme = np.asarray(r_teme)[:, 0, :]  # (n_sats, 3) km
    valid = (e == 0)                   # (n_sats,) bool

    # 2. TEME → ECEF
    gmst = _gmst_rad(jd, fr)
    r_ecef = _teme_to_ecef(r_teme, gmst)  # (n_sats, 3) km

    min_el_rad = math.radians(min_elevation_deg)
    results = []

    for lat, lon, alt_m in grid_points:
        lat_r = math.radians(lat)
        lon_r = math.radians(lon)
        obs_ecef = geodetic_to_ecef_km(lat, lon, alt_m)

        # 3. ECEF → ENU
        enu = _ecef_to_enu(r_ecef, lat_r, lon_r, obs_ecef)

        # 4. Az / El
        az, el = _enu_to_az_el(enu)

        # 5. Elevation mask + valid propagation
        visible = valid & (el >= min_el_rad)
        n_vis = int(visible.sum())

        dop = compute_dop(az[visible], el[visible]) if n_vis >= 4 else None

        row = {
            "lat": lat,
            "lon": lon,
            "n_sats": n_vis,
            "pdop": dop["pdop"] if dop else None,
            "hdop": dop["hdop"] if dop else None,
            "vdop": dop["vdop"] if dop else None,
            "gdop": dop["gdop"] if dop else None,
            "tdop": dop["tdop"] if dop else None,
        }
        results.append(row)

    return results
