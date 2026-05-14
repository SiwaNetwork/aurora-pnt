"""
Geographic grids for PNT coverage analysis.

Each phase uses a different grid:
  Phase 1 — Russia territory
  Phase 2 — Eurasia + friendly countries
  Phase 3 — Global

Grid points are (lat_deg, lon_deg, alt_m=0) tuples.
"""

import numpy as np


def make_grid(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    step_deg: float = 5.0,
    alt_m: float = 0.0,
) -> list[tuple[float, float, float]]:
    """Generate a regular lat/lon grid of points."""
    lats = np.arange(lat_min, lat_max + step_deg * 0.5, step_deg)
    lons = np.arange(lon_min, lon_max + step_deg * 0.5, step_deg)
    return [(float(lat), float(lon), alt_m) for lat in lats for lon in lons]


# ── Named grids ──────────────────────────────────────────────────────────────

def russia_grid(step_deg: float = 5.0) -> list[tuple]:
    """Russia territory grid (41°N–82°N, 19°E–170°E)."""
    return make_grid(41, 82, 19, 170, step_deg)


def central_russia_grid(step_deg: float = 2.0) -> list[tuple]:
    """
    Central Russia — European part + Urals (49°N–65°N, 28°E–65°E).
    Covers: Moscow, St. Petersburg, Kazan, Yekaterinburg, Perm, Ufa, Samara.
    Default step 2° for finer resolution on smaller area.
    """
    return make_grid(49, 65, 28, 65, step_deg)


def eurasia_grid(step_deg: float = 5.0) -> list[tuple]:
    """Eurasia + MENA grid (10°N–80°N, -15°E–145°E)."""
    return make_grid(10, 80, -15, 145, step_deg)


def northern_hemisphere_grid(step_deg: float = 5.0) -> list[tuple]:
    """Northern hemisphere grid (0°–90°N, -180°–180°E)."""
    return make_grid(0, 90, -180, 180, step_deg)


def global_grid(step_deg: float = 5.0) -> list[tuple]:
    """Global grid (-90°–90°, -180°–180°)."""
    return make_grid(-90, 90, -180, 180, step_deg)


# Registry for config lookups

GRIDS = {
    "russia": russia_grid,
    "central_russia": central_russia_grid,
    "eurasia": eurasia_grid,
    "northern": northern_hemisphere_grid,
    "global": global_grid,
}


def get_grid(name: str, step_deg: float = 5.0) -> list[tuple]:
    """Return a named grid by string key."""
    if name not in GRIDS:
        raise ValueError(f"Unknown grid '{name}'. Available: {list(GRIDS.keys())}")
    return GRIDS[name](step_deg)


# ── Named ground control stations ────────────────────────────────────────────

# Phase 1 — Russia MCS (Mission Control Stations)
PHASE1_STATIONS = [
    {"name": "Vladimir",       "latitude": 56.1294, "longitude": 40.4067, "elevation_m": 177.0},
    {"name": "Zheleznogorsk",  "latitude": 56.2505, "longitude": 93.5319, "elevation_m": 305.0},
    {"name": "Khabarovsk",     "latitude": 48.4802, "longitude": 135.0719, "elevation_m": 68.0},
]

# Phase 2 — Friendly countries additions
PHASE2_STATIONS = PHASE1_STATIONS + [
    {"name": "Minsk",          "latitude": 53.9045, "longitude": 27.5615, "elevation_m": 220.0},
    {"name": "Almaty",         "latitude": 43.2567, "longitude": 76.9286, "elevation_m": 773.0},
    {"name": "Urumqi",         "latitude": 43.8256, "longitude": 87.6168, "elevation_m": 859.0},
    {"name": "Tehran",         "latitude": 35.6892, "longitude": 51.3890, "elevation_m": 1191.0},
    {"name": "Delhi",          "latitude": 28.6139, "longitude": 77.2090, "elevation_m": 216.0},
    {"name": "Havana",         "latitude": 23.1136, "longitude": -82.3666, "elevation_m": 59.0},
    {"name": "Johannesburg",   "latitude": -26.2041, "longitude": 28.0473, "elevation_m": 1753.0},
]

# Phase 3 — Global coverage additions
PHASE3_STATIONS = PHASE2_STATIONS + [
    {"name": "Buenos_Aires",   "latitude": -34.6037, "longitude": -58.3816, "elevation_m": 25.0},
    {"name": "Nairobi",        "latitude": -1.2921,  "longitude": 36.8219,  "elevation_m": 1661.0},
    {"name": "Jakarta",        "latitude": -6.2088,  "longitude": 106.8456, "elevation_m": 8.0},
    {"name": "Vladivostok",    "latitude": 43.1198,  "longitude": 131.8869, "elevation_m": 28.0},
    {"name": "Murmansk",       "latitude": 68.9585,  "longitude": 33.0827,  "elevation_m": 50.0},
    {"name": "Novosibirsk",    "latitude": 54.9884,  "longitude": 82.9357,  "elevation_m": 162.0},
    {"name": "Yakutsk",        "latitude": 62.0355,  "longitude": 129.6755, "elevation_m": 103.0},
    {"name": "Caracas",        "latitude": 10.4806,  "longitude": -66.9036, "elevation_m": 900.0},
    {"name": "Luanda",         "latitude": -8.8399,  "longitude": 13.2894,  "elevation_m": 6.0},
    {"name": "Yangon",         "latitude": 16.8661,  "longitude": 96.1951,  "elevation_m": 17.0},
]

STATIONS = {
    "phase1": PHASE1_STATIONS,
    "phase2": PHASE2_STATIONS,
    "phase3": PHASE3_STATIONS,
}
