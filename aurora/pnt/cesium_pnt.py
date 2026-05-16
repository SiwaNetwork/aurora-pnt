"""
Animated CesiumJS visualization for LEO PNT constellation.

Generates a self-contained HTML file with:
  - Animated satellite orbits (CZML time-dynamic positions)
  - Ground station markers with labels
  - Interactive time slider (play/pause, speed control)
  - Full 3D globe rotation and zoom

Offline mode:
  Run `aurora-pnt download-cesium` once to download CesiumJS (~50 MB) into
  assets/cesium/.  After that, all generated HTML files reference the local
  copy and load Natural Earth II imagery from disk — no internet required.
"""

import json
import math
import os
import sys
import urllib.request
import zipfile
import io
from pathlib import Path

import numpy as np
from astropy.time import Time
from sgp4.api import SatrecArray

from aurora.pnt.coverage import load_satrec_from_tle_file, _gmst_rad

_SEC_PER_DAY = 86400.0

# ── Offline CesiumJS support ──────────────────────────────────────────────────
_CESIUM_VERSION = "1.117"

# Local asset directory: <project_root>/assets/cesium/
_CESIUM_LOCAL_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "cesium"
_CESIUM_BUILD     = _CESIUM_LOCAL_DIR / "Build" / "Cesium"
_CESIUM_JS_PATH   = _CESIUM_BUILD / "Cesium.js"

_CDN_BASE = f"https://cesium.com/downloads/cesiumjs/releases/{_CESIUM_VERSION}/Build/Cesium"


def is_cesium_local() -> bool:
    """Return True if CesiumJS has been downloaded to assets/cesium/."""
    return _CESIUM_JS_PATH.exists()


def _rel_or_abs(cesium_build_path: Path, html_path: Path) -> str:
    """
    Return the URL prefix to reference CesiumJS from the given HTML file.

    Uses a relative path when possible (same drive on Windows).
    Falls back to a file:/// absolute URL if drives differ.
    """
    try:
        rel = os.path.relpath(cesium_build_path, html_path.parent)
        return rel.replace("\\", "/")
    except ValueError:
        # Different drives on Windows — use absolute file URL
        return cesium_build_path.as_uri()


def download_cesium(
    target_dir: str | None = None,
    version: str = _CESIUM_VERSION,
) -> bool:
    """
    Download CesiumJS release zip from GitHub and extract to target_dir.

    Default target_dir: <project_root>/assets/cesium/

    The extracted layout is:
      assets/cesium/Build/Cesium/Cesium.js
      assets/cesium/Build/Cesium/Widgets/widgets.css
      assets/cesium/Build/Cesium/Assets/Textures/NaturalEarthII/  ← offline imagery
    """
    target = Path(target_dir) if target_dir else _CESIUM_LOCAL_DIR

    url = (
        f"https://github.com/CesiumGS/cesium/releases/download/"
        f"{version}/Cesium-{version}.zip"
    )

    print(f"  Downloading CesiumJS {version}...")
    print(f"  Source : {url}")
    print(f"  Target : {target}")
    print(f"  Size   : ~50 MB — please wait...")

    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            data = bytearray()
            chunk = 65536
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                data.extend(buf)
                if total:
                    pct = len(data) * 100 // total
                    print(f"\r  Progress: {pct:3d}%  ({len(data)//1024//1024} MB)", end="", flush=True)
        print()
    except Exception as exc:
        print(f"\n  ERROR: download failed — {exc}")
        return False

    print("  Extracting...")
    try:
        with zipfile.ZipFile(io.BytesIO(bytes(data))) as zf:
            zf.extractall(target)
    except Exception as exc:
        print(f"  ERROR: extraction failed — {exc}")
        return False

    js = target / "Build" / "Cesium" / "Cesium.js"
    if js.exists():
        size_mb = js.stat().st_size // (1024 * 1024)
        print(f"  OK: Cesium.js ({size_mb} MB) -> {js}")
        nat_earth = target / "Build" / "Cesium" / "Assets" / "Textures" / "NaturalEarthII"
        if nat_earth.exists():
            print(f"  OK: Natural Earth II imagery available (offline globe ready)")
        return True

    print("  ERROR: Cesium.js not found after extraction")
    return False

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
    """
    Embed CZML into a CesiumJS HTML file.

    If CesiumJS has been downloaded locally (via `aurora-pnt download-cesium`),
    the HTML references local files and uses offline Natural Earth II imagery.
    Otherwise falls back to CDN + Cesium Ion.
    """
    czml_json = json.dumps(czml_packets, separators=(",", ":"))
    speed = czml_packets[0]["clock"]["multiplier"]

    out = Path(output_path).resolve()

    if is_cesium_local():
        prefix = _rel_or_abs(_CESIUM_BUILD, out)
        js_src  = f"{prefix}/Cesium.js"
        css_src = f"{prefix}/Widgets/widgets.css"
        # Natural Earth II ships with Cesium in Assets/Textures/NaturalEarthII
        # Cesium.buildModuleUrl resolves relative to the loaded Cesium.js, so the
        # path is correct as long as the Assets/ folder is next to Cesium.js.
        imagery_js = """
  // Offline: Natural Earth II tiles bundled with CesiumJS (no internet needed)
  Cesium.TileMapServiceImageryProvider.fromUrl(
    Cesium.buildModuleUrl('Assets/Textures/NaturalEarthII')
  ).then(function(provider) {
    viewer.imageryLayers.removeAll();
    viewer.imageryLayers.addImageryProvider(provider);
  });"""
        mode_note = "offline"
        print(f"  [cesium] Using local CesiumJS: {_CESIUM_BUILD}")
    else:
        js_src  = f"{_CDN_BASE}/Cesium.js"
        css_src = f"{_CDN_BASE}/Widgets/widgets.css"
        imagery_js = f"\n  // Online: imagery loaded via Cesium Ion token\n  Cesium.Ion.defaultAccessToken = '{ion_token}';"
        mode_note = "CDN"
        print(f"  [cesium] CesiumJS not found locally — using CDN. Run 'aurora-pnt download-cesium' to enable offline mode.")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>LEO PNT — {label}</title>
  <script src="{js_src}"></script>
  <link href="{css_src}" rel="stylesheet">
  <style>
    html, body, #cesiumContainer {{ width:100%; height:100%; margin:0; padding:0; overflow:hidden; }}
    #overlay {{
      position:absolute; top:10px; left:50%; transform:translateX(-50%);
      color:#fff; font-family:sans-serif; font-size:17px; font-weight:bold;
      text-shadow:1px 1px 4px #000; z-index:100; pointer-events:none;
      background:rgba(0,0,0,0.45); padding:6px 18px; border-radius:8px;
    }}
    #mode-badge {{
      position:absolute; bottom:8px; right:12px;
      color:#aaa; font-family:monospace; font-size:11px;
      background:rgba(0,0,0,0.5); padding:3px 8px; border-radius:4px;
      z-index:100; pointer-events:none;
    }}
  </style>
</head>
<body>
<div id="cesiumContainer"></div>
<div id="overlay">LEO PNT &mdash; {label} &nbsp;|&nbsp; {speed}&times; speed</div>
<div id="mode-badge">CesiumJS {_CESIUM_VERSION} &bull; {mode_note}</div>
<script>
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
  imageryProvider: false,
  creditContainer: document.createElement('div'),
}});

viewer.scene.backgroundColor = Cesium.Color.BLACK;
viewer.scene.globe.enableLighting = true;
{imagery_js}

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

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    return str(out)


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
