"""
Constellation visualization for presentations.

  1. constellation_3d_{label}.png  — Matplotlib 3D globe + orbital rings
  2. ground_tracks_{label}.png     — Cartopy 2D world map, tracks + footprints
"""

import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection
from astropy.time import Time
from sgp4.api import SatrecArray

from aurora.pnt.coverage import load_satrec_from_tle_file, _gmst_rad

_R_EARTH_KM = 6371.0
_SEC_PER_DAY = 86400.0


# ─── helpers ─────────────────────────────────────────────────────────────────

def _jd_fr(jd0: float, fr0: float, t_s: float):
    t_days = t_s / _SEC_PER_DAY
    fr = fr0 + t_days
    jd = jd0 + math.floor(fr)
    return jd, fr - math.floor(fr)


def _teme_to_ecef_vec(r_teme: np.ndarray, jd: float, fr: float) -> np.ndarray:
    """Rotate TEME vector(s) to ECEF. r_teme shape (..., 3)."""
    gmst = _gmst_rad(jd, fr)
    c, s = math.cos(gmst), math.sin(gmst)
    x = c * r_teme[..., 0] + s * r_teme[..., 1]
    y = -s * r_teme[..., 0] + c * r_teme[..., 1]
    z = r_teme[..., 2]
    return np.stack([x, y, z], axis=-1)


def _ecef_to_latlon(xyz: np.ndarray):
    """ECEF (km) → (lat_deg, lon_deg). Works for single vector or (N,3) array."""
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    lat = np.degrees(np.arcsin(np.clip(z / r, -1, 1)))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon


def _batch_positions(sat_array: SatrecArray, n_sats: int,
                     jd0: float, fr0: float, times_s) -> tuple:
    """
    Propagate all satellites at each timestep.

    Returns:
        lats: (n_sats, n_times)  — NaN where SGP4 fails
        lons: (n_sats, n_times)
    """
    n_t = len(times_s)
    lats = np.full((n_sats, n_t), np.nan)
    lons = np.full((n_sats, n_t), np.nan)

    for i, t_s in enumerate(times_s):
        jd, fr = _jd_fr(jd0, fr0, t_s)
        e, r_teme, _ = sat_array.sgp4(
            np.array([jd], dtype=np.float64),
            np.array([fr], dtype=np.float64),
        )
        e = np.asarray(e)[:, 0]
        r_teme = np.asarray(r_teme)[:, 0, :]  # (n_sats, 3)
        valid = e == 0
        if not valid.any():
            continue
        r_ecef = _teme_to_ecef_vec(r_teme, jd, fr)
        lat, lon = _ecef_to_latlon(r_ecef)
        lats[valid, i] = lat[valid]
        lons[valid, i] = lon[valid]

    return lats, lons


def _plane_colors(satrecs):
    """One color per orbital plane (grouped by RAAN)."""
    raas = np.array([s.nodeo for s in satrecs])
    raan_unique = np.unique(np.round(raas, 2))
    n_planes = len(raan_unique)
    palette = plt.cm.tab10(np.linspace(0, min(0.9, (n_planes - 1) * 0.1 + 0.05), n_planes))
    colors = []
    for r in raas:
        idx = int(np.argmin(np.abs(raan_unique - r)))
        colors.append(palette[idx])
    return colors, n_planes


def _footprint_boundary(lat0_deg, lon0_deg, altitude_km, min_el_deg, n=180):
    """Spherical cap boundary for satellite footprint."""
    R = _R_EARTH_KM
    el = math.radians(min_el_deg)
    rho = math.acos(max(-1, min(1, R / (R + altitude_km) * math.cos(el)))) - el

    lat0, lon0 = math.radians(lat0_deg), math.radians(lon0_deg)
    azimuths = np.linspace(0, 2 * math.pi, n)
    lat_b = np.arcsin(np.clip(
        math.sin(lat0) * math.cos(rho) +
        math.cos(lat0) * math.sin(rho) * np.cos(azimuths),
        -1, 1,
    ))
    dlon = np.arctan2(
        np.sin(azimuths) * math.sin(rho) * math.cos(lat0),
        math.cos(rho) - math.sin(lat0) * np.sin(lat_b),
    )
    lon_b = lon0 + dlon
    return np.degrees(lat_b), np.degrees(lon_b)


def _split_antimeridian(lats, lons, thr=180.0):
    """Split lat/lon arrays at antimeridian jumps → list of segment arrays."""
    if len(lats) == 0:
        return [], []
    segs_lat, segs_lon = [[lats[0]]], [[lons[0]]]
    for i in range(1, len(lats)):
        if abs(lons[i] - lons[i - 1]) > thr:
            segs_lat.append([lats[i]])
            segs_lon.append([lons[i]])
        else:
            segs_lat[-1].append(lats[i])
            segs_lon[-1].append(lons[i])
    return [np.array(s) for s in segs_lat], [np.array(s) for s in segs_lon]


def _geodetic_to_xyz(lat_deg, lon_deg, alt_m=0.0):
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    r = _R_EARTH_KM + alt_m / 1000
    return (
        r * math.cos(lat) * math.cos(lon),
        r * math.cos(lat) * math.sin(lon),
        r * math.sin(lat),
    )


# ─── 3D globe ────────────────────────────────────────────────────────────────

def _draw_earth_3d(ax, cam: np.ndarray):
    """
    Earth: ocean sphere + continent outlines (front-hemisphere segments only) + graticule.

    The key trick: continent lines are split into short front-facing segments.
    Each segment's centroid is on the front hemisphere, so matplotlib's painter
    algorithm sorts them in FRONT of the sphere mesh (closer to camera → drawn last).
    """
    import cartopy.io.shapereader as shpreader

    R = _R_EARTH_KM

    # ── Ocean sphere ──────────────────────────────────────────────────────────
    u = np.linspace(0, 2 * math.pi, 80)
    v = np.linspace(0, math.pi, 40)
    xe = R * np.outer(np.cos(u), np.sin(v))
    ye = R * np.outer(np.sin(u), np.sin(v))
    ze = R * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xe, ye, ze, color="#0d3057", alpha=0.90,
                    linewidth=0, antialiased=True, shade=False)

    # ── Graticule (each ring is a closed loop — centroid near globe center so
    #    they may be partially behind sphere; that's fine, just decorative) ───
    th = np.linspace(0, 2 * math.pi, 361)
    for lat_d in (0, 30, 60, -30, -60):
        lat = math.radians(lat_d)
        r_l = R * 1.002 * math.cos(lat)
        z_l = R * 1.002 * math.sin(lat)
        a = 0.50 if lat_d == 0 else 0.22
        lw = 1.1 if lat_d == 0 else 0.5
        ax.plot(r_l * np.cos(th), r_l * np.sin(th),
                np.full(361, z_l), color="#3a7acc", alpha=a, linewidth=lw)
    for lon_d in range(0, 360, 30):
        lon = math.radians(lon_d)
        ph = np.linspace(0, 2 * math.pi, 181)
        r = R * 1.002
        ax.plot(r * np.cos(ph) * math.cos(lon),
                r * np.cos(ph) * math.sin(lon),
                r * np.sin(ph), color="#3a7acc", alpha=0.15, linewidth=0.35)

    # ── Continent outlines — only front-facing short segments ─────────────────
    # Each segment's centroid must have cam·pos > 0 to be closer than the sphere.
    # We emit one ax.plot call per contiguous front-facing run, so every object
    # matplotlib sorts is guaranteed to be in front of the corresponding sphere face.
    try:
        shp = shpreader.natural_earth(resolution="110m", category="physical", name="land")
        R_s = R * 1.018  # 1.8% above surface to beat depth-sort ambiguity

        for record in shpreader.Reader(shp).records():
            geom = record.geometry
            polys = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
            for poly in polys:
                coords = list(poly.exterior.coords)
                if len(coords) < 4:
                    continue
                lons_r = np.radians([c[0] for c in coords])
                lats_r = np.radians([c[1] for c in coords])
                xp = R_s * np.cos(lats_r) * np.cos(lons_r)
                yp = R_s * np.cos(lats_r) * np.sin(lons_r)
                zp = R_s * np.sin(lats_r)
                pts_n = np.stack([xp, yp, zp], axis=1) / R_s  # unit normals
                vis = pts_n @ cam  # positive = facing camera

                seg_x, seg_y, seg_z = [], [], []
                for i in range(len(xp)):
                    d_lon = abs(float(lons_r[i]) - float(lons_r[i - 1])) if i > 0 else 0
                    front = vis[i] > 0.06 and d_lon < math.pi
                    if front:
                        seg_x.append(xp[i]); seg_y.append(yp[i]); seg_z.append(zp[i])
                    else:
                        if len(seg_x) >= 2:
                            ax.plot(seg_x, seg_y, seg_z,
                                    color="#55dd66", alpha=0.95, linewidth=0.85)
                        seg_x, seg_y, seg_z = [], [], []
                if len(seg_x) >= 2:
                    ax.plot(seg_x, seg_y, seg_z,
                            color="#55dd66", alpha=0.95, linewidth=0.85)
    except Exception:
        pass

    # ── Polar axis ────────────────────────────────────────────────────────────
    ax.plot([0, 0], [0, 0], [-R * 1.22, R * 1.22],
            color="#8899bb", alpha=0.50, linewidth=0.7, linestyle="--")
    ax.text(0, 0, R * 1.26, "N", color="#aabbcc", fontsize=9,
            ha="center", va="bottom", fontweight="bold")


def plot_constellation_3d(
    tle_path: str,
    ground_stations: list,
    output_dir: str,
    label: str,
    altitude_m: float = 1_000_000,
    elev: float = None,
    azim: float = None,
) -> str:
    """
    Dark-space 3D globe: continent outlines + orbital rings + footprints + MCS markers.
    View angle is auto-selected to show the orbital plane(s) face-on.
    """
    satrecs = load_satrec_from_tle_file(tle_path)
    n_sats = len(satrecs)
    colors, n_planes = _plane_colors(satrecs)
    alt_km = altitude_m / 1000
    r_sat = _R_EARTH_KM + alt_km

    # ── Auto view angle ────────────────────────────────────────────────────────
    if elev is None:
        elev = 28.0
    if azim is None:
        if ground_stations:
            # Camera aimed slightly west of the mean station longitude so all
            # stations land on the visible front hemisphere (not near the limb)
            mean_lon = float(np.mean([gs["longitude"] for gs in ground_stations]))
            azim = mean_lon - 25.0
        else:
            mean_raan = float(np.mean([s.nodeo for s in satrecs]))
            azim = math.degrees(mean_raan) + 90.0

    # Camera unit vector (viewer direction in ECI frame)
    el_r = math.radians(elev)
    az_r = math.radians(azim)
    cam = np.array([
        math.cos(el_r) * math.cos(az_r),
        math.cos(el_r) * math.sin(az_r),
        math.sin(el_r),
    ])

    fig = plt.figure(figsize=(13, 11), facecolor="#030610")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#030610")

    # ── Earth ─────────────────────────────────────────────────────────────────
    _draw_earth_3d(ax, cam)

    # ── Orbital rings + satellites + footprints ───────────────────────────────
    theta = np.linspace(0, 2 * math.pi, 600)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    plane_done = set()

    for si, satrec in enumerate(satrecs):
        inc  = satrec.inclo   # rad
        raan = satrec.nodeo   # rad
        mo   = satrec.mo      # rad — initial mean anomaly

        ci = math.cos(inc); si_ = math.sin(inc)
        cr = math.cos(raan); sr = math.sin(raan)

        # Full orbital ring in ECI
        xo = r_sat * (cos_t * cr - sin_t * sr * ci)
        yo = r_sat * (cos_t * sr + sin_t * cr * ci)
        zo = r_sat * sin_t * si_

        raan_deg = round(math.degrees(raan), 0)
        new_plane = raan_deg not in plane_done
        plane_done.add(raan_deg)

        ax.plot(xo, yo, zo, color=colors[si], alpha=0.80, linewidth=1.8,
                label=f"Орб. плоскость  RAAN ≈ {raan_deg:.0f}°" if new_plane else None,
                zorder=5)

        # Satellite dot at initial position
        cm0 = math.cos(mo); sm0 = math.sin(mo)
        x0 = r_sat * (cm0 * cr - sm0 * sr * ci)
        y0 = r_sat * (cm0 * sr + sm0 * cr * ci)
        z0 = r_sat * sm0 * si_

        # Glow ring (larger, transparent) + solid dot
        ax.scatter(x0, y0, z0, color=colors[si], s=200, alpha=0.30,
                   depthshade=False, zorder=6)
        ax.scatter(x0, y0, z0, color=colors[si], s=60,
                   edgecolors="white", linewidths=0.8,
                   depthshade=False, zorder=7)

        # Footprint circle on Earth surface (dashed)
        sat_lat = math.degrees(math.asin(max(-1, min(1, z0 / r_sat))))
        sat_lon = math.degrees(math.atan2(y0, x0))
        fp_lats, fp_lons = _footprint_boundary(sat_lat, sat_lon, alt_km, 10.0, n=180)
        fp_lr = np.radians(fp_lats); fp_nr = np.radians(fp_lons)
        Rs = _R_EARTH_KM * 1.007
        xfp = Rs * np.cos(fp_lr) * np.cos(fp_nr)
        yfp = Rs * np.cos(fp_lr) * np.sin(fp_nr)
        zfp = Rs * np.sin(fp_lr)
        ax.plot(xfp, yfp, zfp, color=colors[si], alpha=0.50,
                linewidth=1.0, linestyle="--", zorder=4)

    # ── Ground stations ───────────────────────────────────────────────────────
    # ax.text renders on top of ALL 3D geometry (depth-sorting does not apply
    # to text in matplotlib 3D), so use text for both the star symbol and label.
    STATION_LIFT = 1.20  # 20% above Earth radius — pin tip clearly in space
    for gs in ground_stations:
        xs, ys, zs = _geodetic_to_xyz(
            gs["latitude"], gs["longitude"], gs.get("elevation_m", 0.0)
        )
        xm, ym, zm = xs * STATION_LIFT, ys * STATION_LIFT, zs * STATION_LIFT

        # Pin line: Earth surface → tip
        ax.plot([xs * 1.008, xm], [ys * 1.008, ym], [zs * 1.008, zm],
                color="#ff5555", alpha=0.80, linewidth=1.6)

        # Station marker as unicode ★ text (always renders above 3D surfaces)
        ax.text(xm, ym, zm, "★",
                color="#ff2222", fontsize=20, ha="center", va="center",
                fontweight="bold",
                path_effects=[
                    __import__("matplotlib.patheffects", fromlist=["withStroke"])
                    .withStroke(linewidth=2, foreground="white")
                ])

        # Station name label
        scale_lbl = STATION_LIFT * 1.08
        ax.text(xs * scale_lbl, ys * scale_lbl, zs * scale_lbl + 150,
                gs["name"], color="white", fontsize=9, fontweight="bold",
                ha="center", va="bottom")

    # ── Layout ────────────────────────────────────────────────────────────────
    lim = r_sat * 1.28
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)

    def _plane_word(n):
        if n == 1: return "плоскость"
        if n in (2, 3, 4): return "плоскости"
        return "плоскостей"

    ax.set_title(
        f"LEO PNT  —  {label}\n"
        f"{n_sats} спутников  ·  {alt_km:.0f} км  ·  {n_planes} орбитальных {_plane_word(n_planes)}",
        color="white", fontsize=14, pad=16, fontweight="bold",
    )

    legend_patches = [
        mpatches.Patch(color="#0b2545", label="Земля"),
        mpatches.Patch(color="#55cc66", label="Материки"),
        mpatches.Patch(color="#ff2222", label="Станция управления (MCS)"),
    ]
    seen = {}
    for si, s in enumerate(satrecs):
        rk = round(math.degrees(s.nodeo), 0)
        if rk not in seen:
            seen[rk] = colors[si]
    for rk, col in seen.items():
        legend_patches.append(mpatches.Patch(
            color=col, label=f"Орбитальная плоскость  RAAN ≈ {rk:.0f}°"
        ))

    ax.legend(
        handles=legend_patches, loc="lower left",
        facecolor="#08101e", edgecolor="#334",
        labelcolor="white", fontsize=9, framealpha=0.90,
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out = str(Path(output_dir) / f"constellation_3d_{label}.png")
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="#030610")
    plt.close()
    print(f"  [viz] 3D globe saved → {out}")
    return out


# ─── 2D ground tracks ────────────────────────────────────────────────────────

def plot_ground_tracks(
    tle_path: str,
    ground_stations: list,
    output_dir: str,
    label: str,
    altitude_m: float = 1_000_000,
    min_elevation_deg: float = 10.0,
    duration_h: float = 24.0,
    step_min: float = 2.0,
    show_footprints: bool = True,
) -> str:
    """
    Cartopy 2D map: ground tracks (24h) + coverage footprints at t=0 + MCS.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    satrecs = load_satrec_from_tle_file(tle_path)
    n_sats = len(satrecs)
    colors, n_planes = _plane_colors(satrecs)
    sat_array = SatrecArray(satrecs)

    epoch = Time("2000-01-01 00:00:00", scale="tdb")
    jd0 = math.floor(epoch.jd)
    fr0 = epoch.jd - jd0

    times_s = np.arange(0, duration_h * 3600, step_min * 60)
    lats, lons = _batch_positions(sat_array, n_sats, jd0, fr0, times_s)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(
        figsize=(18, 9),
        subplot_kw={"projection": ccrs.Robinson()},
        facecolor="#07111e",
    )
    ax.set_facecolor("#07111e")
    ax.add_feature(cfeature.OCEAN.with_scale("110m"),   color="#0d2035", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("110m"),    color="#1a2e1a", zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"),
                   edgecolor="#4a8a4a", linewidth=0.45, zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"),
                   edgecolor="#3a6a3a", linewidth=0.25, zorder=2)
    ax.add_feature(cfeature.LAKES.with_scale("110m"),   color="#0d2035", zorder=2)
    ax.gridlines(color="white", alpha=0.12, linewidth=0.3,
                 xlocs=range(-180, 181, 30), ylocs=range(-90, 91, 30))
    ax.set_global()

    pc = ccrs.PlateCarree()

    # ── Ground tracks ─────────────────────────────────────────────────────────
    plane_raas = set()
    for si in range(n_sats):
        valid = ~np.isnan(lats[si])
        if not valid.any():
            continue
        lat_v = lats[si][valid]
        lon_v = lons[si][valid]

        raan_key = round(math.degrees(satrecs[si].nodeo), 0)
        do_label = raan_key not in plane_raas
        plane_raas.add(raan_key)

        segs_lat, segs_lon = _split_antimeridian(lat_v, lon_v)
        for j, (sl, slo) in enumerate(zip(segs_lat, segs_lon)):
            ax.plot(slo, sl, "-", color=colors[si], alpha=0.65, linewidth=0.9,
                    transform=pc,
                    label=f"Orbit plane {raan_key:.0f}° RAAN" if (do_label and j == 0) else None)

        # Satellite position dot at t=0
        ax.plot(lon_v[0], lat_v[0], "o",
                color=colors[si], markersize=5.5, transform=pc, zorder=5)

    # ── Footprints at t=0 ─────────────────────────────────────────────────────
    if show_footprints:
        alt_km = altitude_m / 1000
        for si in range(n_sats):
            if np.isnan(lats[si, 0]) or np.isnan(lons[si, 0]):
                continue
            fp_lat, fp_lon = _footprint_boundary(
                lats[si, 0], lons[si, 0], alt_km, min_elevation_deg
            )
            ax.fill(fp_lon, fp_lat,
                    color=colors[si], alpha=0.10, transform=pc, zorder=3)
            ax.plot(fp_lon, fp_lat, "-",
                    color=colors[si], alpha=0.35, linewidth=0.6,
                    transform=pc, zorder=4)

    # ── Ground stations ───────────────────────────────────────────────────────
    for gs in ground_stations:
        ax.plot(gs["longitude"], gs["latitude"], "*",
                color="#ff3333", markersize=13,
                markeredgecolor="white", markeredgewidth=0.5,
                transform=pc, zorder=8)
        ax.text(
            gs["longitude"] + 2, gs["latitude"] + 1.5, gs["name"],
            color="white", fontsize=8, fontweight="bold",
            transform=pc, zorder=9,
        )

    # ── Style ─────────────────────────────────────────────────────────────────
    ax.set_title(
        f"LEO PNT  —  {label}   |   Ground tracks 24h   |   Footprints at t=0  (el≥{min_elevation_deg}°)",
        color="white", fontsize=13, pad=10,
    )

    handles = [
        mpatches.Patch(color="#ff3333", label="Ground station (MCS)"),
        mpatches.Patch(color="#aaaaaa", label=f"Footprint @ t=0 (el≥{min_elevation_deg}°)"),
    ]
    # Add one patch per orbital plane
    seen_planes = {}
    for si, satrec in enumerate(satrecs):
        rk = round(math.degrees(satrec.nodeo), 0)
        if rk not in seen_planes:
            seen_planes[rk] = colors[si]
    for rk, col in seen_planes.items():
        handles.append(mpatches.Patch(color=col, label=f"Orbital plane RAAN≈{rk:.0f}°"))

    ax.legend(handles=handles, loc="lower left",
              facecolor="#07111e", edgecolor="#334",
              labelcolor="white", fontsize=8)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out = str(Path(output_dir) / f"ground_tracks_{label}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#07111e")
    plt.close()
    print(f"  [viz] Ground track map saved → {out}")
    return out


# ─── convenience: generate both ──────────────────────────────────────────────

def generate_all_visuals(
    tle_path: str,
    ground_stations: list,
    output_dir: str,
    label: str,
    altitude_m: float = 1_000_000,
    min_elevation_deg: float = 10.0,
) -> list[str]:
    """Generate both 3D globe and 2D ground track map."""
    paths = []
    paths.append(plot_constellation_3d(
        tle_path, ground_stations, output_dir, label, altitude_m,
    ))
    paths.append(plot_ground_tracks(
        tle_path, ground_stations, output_dir, label,
        altitude_m, min_elevation_deg,
    ))
    return paths
