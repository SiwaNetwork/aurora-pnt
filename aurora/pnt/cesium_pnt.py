"""
Animated CesiumJS visualization for LEO PNT constellation.

Generates a self-contained HTML file with:
  - Animated satellite orbits (CZML time-dynamic positions)
  - Ground station markers with labels
  - Interactive time slider (play/pause, speed control)
  - Full 3D globe rotation and zoom
"""

import json
import math
from pathlib import Path

import numpy as np
from astropy.time import Time
from sgp4.api import SatrecArray

from aurora.pnt.coverage import load_satrec_from_tle_file, _gmst_rad

_SEC_PER_DAY = 86400.0

# Satellite color palette (RGBA) — one per orbital plane
_PLANE_COLORS = [
    [80,  200, 255, 255],   # cyan
    [255, 160,  50, 255],   # orange
    [80,  255, 120, 255],   # green
    [255, 100, 220, 255],   # magenta
    [255, 240,  60, 255],   # yellow
    [200, 130, 255, 255],   # violet
]
_TRAIL_ALPHA = 160   # orbit trail opacity


_R_EARTH_M = 6_371_000.0


def _ecef_m(r_teme_km: np.ndarray, jd: float, fr: float) -> list:
    """Convert TEME position (km) → ECEF (m) as [x, y, z]."""
    gmst = _gmst_rad(jd, fr)
    c, s = math.cos(gmst), math.sin(gmst)
    x = (c * r_teme_km[0] + s * r_teme_km[1]) * 1000
    y = (-s * r_teme_km[0] + c * r_teme_km[1]) * 1000
    z = r_teme_km[2] * 1000
    return [round(x, 1), round(y, 1), round(z, 1)]


def _subsatellite_ecef(xyz_ecef_m: list) -> list:
    """Project satellite ECEF position onto Earth surface (nadir point)."""
    x, y, z = xyz_ecef_m
    r = math.sqrt(x*x + y*y + z*z)
    scale = _R_EARTH_M / r
    return [round(x*scale, 1), round(y*scale, 1), round(z*scale, 1)]


def _footprint_radius_m(altitude_m: float, min_elevation_deg: float) -> float:
    """Great-circle radius of the coverage footprint on Earth's surface."""
    eps = math.radians(min_elevation_deg)
    # Nadir half-angle (from satellite) to the coverage edge
    rho = math.asin(_R_EARTH_M * math.cos(eps) / (_R_EARTH_M + altitude_m))
    # Earth central angle (angular radius of footprint)
    lam = math.pi / 2 - eps - rho
    return _R_EARTH_M * lam


def build_czml(
    tle_path: str,
    ground_stations: list,
    label: str,
    duration_h: float = 24.0,
    step_s: float = 60.0,
    speed_multiplier: int = 60,
    altitude_m: float = 1_000_000,
    min_elevation_deg: float = 10.0,
) -> list:
    """
    Build a CZML document for animated satellite visualization.

    Returns list of CZML packets (JSON-serializable).
    """
    satrecs = load_satrec_from_tle_file(tle_path)
    n_sats = len(satrecs)
    sat_array = SatrecArray(satrecs)

    epoch = Time("2000-01-01 00:00:00", scale="tdb")
    jd0 = math.floor(epoch.jd)
    fr0 = epoch.jd - jd0

    times_s = np.arange(0, duration_h * 3600.0, step_s)
    epoch_iso = "2000-01-01T00:00:00Z"
    end_iso   = f"2000-01-0{1 + int(duration_h // 24)}T{int(duration_h % 24):02d}:00:00Z"
    if duration_h == 24.0:
        end_iso = "2000-01-02T00:00:00Z"
    avail = f"{epoch_iso}/{end_iso}"

    # Group satellites by orbital plane (RAAN)
    raas = [s.nodeo for s in satrecs]
    raan_sorted = sorted(set(round(r, 2) for r in raas))

    def plane_color(si):
        rk = round(satrecs[si].nodeo, 2)
        idx = raan_sorted.index(rk) % len(_PLANE_COLORS)
        return _PLANE_COLORS[idx]

    # ── Document packet ───────────────────────────────────────────────────────
    packets = [{
        "id": "document",
        "name": f"LEO PNT — {label}",
        "version": "1.0",
        "clock": {
            "interval": avail,
            "currentTime": epoch_iso,
            "multiplier": speed_multiplier,
            "range": "LOOP_STOP",
            "step": "SYSTEM_CLOCK_MULTIPLIER",
        },
    }]

    # Pre-compute all positions: (n_sats, n_times, 3) in ECEF metres
    print(f"  [cesium] Propagating {n_sats} satellites over {duration_h:.0f}h "
          f"at {step_s:.0f}s steps ({len(times_s)} points)…")

    fp_radius = _footprint_radius_m(altitude_m, min_elevation_deg)
    print(f"  [cesium] Footprint radius: {fp_radius/1000:.0f} km  "
          f"(alt={altitude_m/1000:.0f} km, el≥{min_elevation_deg:.0f}°)")

    pos_cache = {si: [] for si in range(n_sats)}  # si → [t, x, y, z, ...]
    fp_cache  = {si: [] for si in range(n_sats)}  # si → [t, x, y, z, ...] on surface

    for t_s in times_s:
        t_days = t_s / _SEC_PER_DAY
        fr = fr0 + t_days
        jd = jd0 + math.floor(fr)
        fr_r = fr - math.floor(fr)

        e, r_teme, _ = sat_array.sgp4(
            np.array([jd], dtype=np.float64),
            np.array([fr_r], dtype=np.float64),
        )
        e_arr = np.asarray(e)[:, 0]
        r_arr = np.asarray(r_teme)[:, 0, :]  # (n_sats, 3) km

        for si in range(n_sats):
            if e_arr[si] == 0:
                xyz = _ecef_m(r_arr[si], jd, fr_r)
                pos_cache[si].extend([round(t_s, 1)] + xyz)
                fp_cache[si].extend([round(t_s, 1)] + _subsatellite_ecef(xyz))

    # ── Satellite packets ─────────────────────────────────────────────────────
    for si, satrec in enumerate(satrecs):
        col = plane_color(si)
        trail_col = col[:3] + [_TRAIL_ALPHA]
        plane_idx = raan_sorted.index(round(satrec.nodeo, 2))
        raan_deg  = round(math.degrees(satrec.nodeo), 0)
        sat_name  = f"Sat-{si+1:02d}  |  plane {plane_idx+1}  RAAN {raan_deg:.0f}°"

        packets.append({
            "id": f"sat_{si}",
            "name": sat_name,
            "availability": avail,
            "position": {
                "interpolationAlgorithm": "LAGRANGE",
                "interpolationDegree": 5,
                "referenceFrame": "FIXED",
                "epoch": epoch_iso,
                "cartesian": pos_cache[si],
            },
            "point": {
                "color": {"rgba": col},
                "pixelSize": 9,
                "outlineColor": {"rgba": [255, 255, 255, 200]},
                "outlineWidth": 1,
                "heightReference": "NONE",
            },
            "path": {
                "show": [{"interval": avail, "boolean": True}],
                "width": 1.4,
                "material": {"solidColor": {"color": {"rgba": trail_col}}},
                "resolution": 120,
                "leadTime":  [{"number": 1800}],
                "trailTime": [{"number": 1800}],
            },
            "label": {
                "text": sat_name,
                "show": False,
                "font": "11px sans-serif",
                "fillColor": {"rgba": [255, 255, 255, 255]},
                "outlineColor": {"rgba": [0, 0, 0, 200]},
                "outlineWidth": 2,
                "style": "FILL_AND_OUTLINE",
                "pixelOffset": {"cartesian2": [14, 0]},
            },
        })

    # ── Footprint packets ─────────────────────────────────────────────────────
    for si, satrec in enumerate(satrecs):
        col = plane_color(si)
        fill_col    = col[:3] + [35]   # very transparent fill
        outline_col = col[:3] + [200]  # semi-opaque outline

        packets.append({
            "id": f"footprint_{si}",
            "name": f"Coverage — Sat-{si+1:02d}",
            "availability": avail,
            "position": {
                "interpolationAlgorithm": "LAGRANGE",
                "interpolationDegree": 5,
                "referenceFrame": "FIXED",
                "epoch": epoch_iso,
                "cartesian": fp_cache[si],
            },
            "ellipse": {
                "semiMajorAxis": round(fp_radius),
                "semiMinorAxis": round(fp_radius),
                "fill": True,
                "material": {"solidColor": {"color": {"rgba": fill_col}}},
                "outline": True,
                "outlineColor": {"rgba": outline_col},
                "outlineWidth": 1.5,
                "numberOfVerticalLines": 0,
                "heightReference": "CLAMP_TO_GROUND",
            },
        })

    # ── Ground station packets ────────────────────────────────────────────────
    for gs in ground_stations:
        packets.append({
            "id": f"gs_{gs['name']}",
            "name": gs["name"],
            "position": {
                # 5 000 m keeps markers above any terrain; disableDepthTestDistance
                # ensures they always render on top in 3D.  In 2D/Columbus mode only
                # longitude/latitude matter — height is ignored by the flat projection.
                "cartographicDegrees": [gs["longitude"], gs["latitude"], 5000.0]
            },
            "point": {
                "color": {"rgba": [255, 40, 40, 255]},
                "pixelSize": 14,
                "outlineColor": {"rgba": [255, 255, 255, 255]},
                "outlineWidth": 2,
                "disableDepthTestDistance": 1.175494e+38,
            },
            "label": {
                "text": gs["name"],
                "show": True,
                "font": "bold 14px sans-serif",
                "fillColor": {"rgba": [255, 255, 255, 255]},
                "outlineColor": {"rgba": [0, 0, 0, 255]},
                "outlineWidth": 3,
                "style": "FILL_AND_OUTLINE",
                "pixelOffset": {"cartesian2": [0, -20]},
                "verticalOrigin": "BOTTOM",
                "horizontalOrigin": "CENTER",
                "disableDepthTestDistance": 1.175494e+38,
            },
        })

    return packets


def write_cesium_html(
    czml_packets: list,
    output_path: str,
    label: str,
    ion_token: str = "",
) -> str:
    """Embed CZML into a self-contained CesiumJS HTML file."""
    czml_json = json.dumps(czml_packets, separators=(",", ":"))
    speed = czml_packets[0]["clock"]["multiplier"]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>LEO PNT — {label}</title>
  <script src="https://cesium.com/downloads/cesiumjs/releases/1.117/Build/Cesium/Cesium.js"></script>
  <link href="https://cesium.com/downloads/cesiumjs/releases/1.117/Build/Cesium/Widgets/widgets.css" rel="stylesheet">
  <style>
    html, body, #cesiumContainer {{ width:100%; height:100%; margin:0; padding:0; overflow:hidden; }}
    #overlay {{
      position:absolute; top:10px; left:50%; transform:translateX(-50%);
      color:#fff; font-family:sans-serif; font-size:17px; font-weight:bold;
      text-shadow:1px 1px 4px #000; z-index:100; pointer-events:none;
      background:rgba(0,0,0,0.45); padding:6px 18px; border-radius:8px;
    }}
  </style>
</head>
<body>
<div id="cesiumContainer"></div>
<div id="overlay">LEO PNT &mdash; {label} &nbsp;|&nbsp; {speed}&times; speed</div>
<script>
Cesium.Ion.defaultAccessToken = '{ion_token}';

var viewer = new Cesium.Viewer('cesiumContainer', {{
  animation: true,
  timeline: true,
  geocoder: false,
  homeButton: true,
  sceneModePicker: true,
  baseLayerPicker: false,
  navigationHelpButton: true,
  infoBox: true,
  selectionIndicator: true,
  creditContainer: document.createElement('div'),
}});

// Dark space background; real Earth imagery loaded via Ion token above
viewer.scene.backgroundColor = Cesium.Color.BLACK;
viewer.scene.globe.enableLighting = true;

// Load CZML
var czml = {czml_json};
var ds = new Cesium.CzmlDataSource();
ds.load(czml).then(function() {{
  viewer.dataSources.add(ds);
  viewer.camera.setView({{
    destination: Cesium.Cartesian3.fromDegrees(80.0, 55.0, 15000000),
    orientation: {{ heading: 0, pitch: -Cesium.Math.PI_OVER_TWO, roll: 0 }}
  }});
}});

viewer.clock.multiplier = {speed};
viewer.clock.shouldAnimate = true;
</script>
</body>
</html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path


def generate_cesium_visualization(
    tle_path: str,
    ground_stations: list,
    output_dir: str,
    label: str,
    duration_h: float = 24.0,
    step_s: float = 60.0,
    speed_multiplier: int = 60,
    ion_token: str = "",
    altitude_m: float = 1_000_000,
    min_elevation_deg: float = 10.0,
) -> str:
    """Build CZML + write HTML. Returns path to the HTML file."""
    czml = build_czml(
        tle_path, ground_stations, label,
        duration_h, step_s, speed_multiplier,
        altitude_m=altitude_m, min_elevation_deg=min_elevation_deg,
    )
    out_path = str(Path(output_dir) / f"cesium_{label}.html")
    write_cesium_html(czml, out_path, label, ion_token=ion_token)
    print(f"  [cesium] Interactive globe -> {out_path}")
    print(f"  [cesium] Open in browser: file:///{Path(out_path).as_posix()}")
    return out_path
