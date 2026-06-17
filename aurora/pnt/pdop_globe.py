"""
3D-карта геометрии (PDOP / N_vis) на глобусе — не плоская equirectangular,
а раскраска сферы Земли по фактической геометрии группировки Walker 300/15.

Для сетки (lat, lon) усредняется по нескольким временным снимкам: число
видимых КА (маска 10°) и PDOP = sqrt(tr[(HᵀH)⁻¹]_xyz). Значения наносятся
на поверхность сферы facecolors (две сцены: PDOP и N_vis).
"""

import sys, os, math
from typing import Dict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from aurora.pnt.constellation_anim import (
    _build, _eci, _ecef, R_E, MASK, T_ORB)


def _grid_geometry(nlat=46, nlon=92, n_t=4):
    raan, u0 = _build()
    lats = np.linspace(-89, 89, nlat)
    lons = np.linspace(-180, 180, nlon)
    pdop = np.full((nlat, nlon), np.nan)
    nvis = np.zeros((nlat, nlon))
    times = np.linspace(0, 0.9 * T_ORB, n_t)

    # предрассчёт ECEF спутников по времени
    sats_t = [_ecef(_eci(raan, u0, t), t) for t in times]

    for i, la in enumerate(lats):
        cla = math.cos(math.radians(la)); sla = math.sin(math.radians(la))
        for j, lo in enumerate(lons):
            clo = math.cos(math.radians(lo)); slo = math.sin(math.radians(lo))
            ue = R_E * np.array([cla * clo, cla * slo, sla])
            up = ue / np.linalg.norm(ue)
            # восток/север для матрицы (не нужны — берём ENU через LOS)
            pd_acc, nv_acc, cnt = 0.0, 0, 0
            for sat in sats_t:
                los = sat - ue
                d = np.linalg.norm(los, axis=1)
                losh = los / d[:, None]
                elev = np.arcsin(np.clip(losh @ up, -1, 1))
                vis = elev > MASK
                nv = int(vis.sum()); nv_acc += nv
                if nv >= 4:
                    H = np.column_stack([losh[vis], np.ones(nv)])
                    try:
                        cov = np.linalg.inv(H.T @ H)
                        pd = math.sqrt(max(cov[0, 0] + cov[1, 1] + cov[2, 2], 0))
                        pd_acc += min(pd, 20.0); cnt += 1
                    except np.linalg.LinAlgError:
                        pass
            nvis[i, j] = nv_acc / len(sats_t)
            if cnt:
                pdop[i, j] = pd_acc / cnt
    return lats, lons, pdop, nvis


def _sphere_plot(ax, lats, lons, field, cmap, vmin, vmax, r=1.0):
    LA, LO = np.meshgrid(np.radians(lats), np.radians(lons), indexing="ij")
    x = r * np.cos(LA) * np.cos(LO)
    y = r * np.cos(LA) * np.sin(LO)
    z = r * np.sin(LA)
    norm = plt.Normalize(vmin, vmax)
    fcol = cm.get_cmap(cmap)(norm(np.nan_to_num(field, nan=vmax)))
    ax.plot_surface(x, y, z, rcount=field.shape[0], ccount=field.shape[1],
                    facecolors=fcol, linewidth=0, antialiased=True, shade=False)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.set_axis_off()
    return norm


def run_pdop_globe(output_dir: str, label: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    lats, lons, pdop, nvis = _grid_geometry()

    fig = plt.figure(figsize=(15, 7.5))
    fig.patch.set_facecolor("#05080f")

    ax1 = fig.add_subplot(1, 2, 1, projection="3d"); ax1.set_facecolor("#05080f")
    n1 = _sphere_plot(ax1, lats, lons, pdop, "RdYlGn_r", 1.0, 6.0)
    ax1.view_init(elev=28, azim=-60)
    ax1.set_title("PDOP (автономный, маска 10°)", color="white", fontsize=12)
    m1 = cm.ScalarMappable(norm=n1, cmap="RdYlGn_r")
    cb1 = fig.colorbar(m1, ax=ax1, shrink=0.55, pad=0.02)
    cb1.set_label("PDOP", color="#cfd8e3"); cb1.ax.yaxis.set_tick_params(color="#cfd8e3")
    plt.setp(plt.getp(cb1.ax, "yticklabels"), color="#cfd8e3")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d"); ax2.set_facecolor("#05080f")
    n2 = _sphere_plot(ax2, lats, lons, nvis, "viridis", 4, 22)
    ax2.view_init(elev=28, azim=-60)
    ax2.set_title("Число видимых КА (N_vis)", color="white", fontsize=12)
    m2 = cm.ScalarMappable(norm=n2, cmap="viridis")
    cb2 = fig.colorbar(m2, ax=ax2, shrink=0.55, pad=0.02)
    cb2.set_label("N_vis", color="#cfd8e3"); cb2.ax.yaxis.set_tick_params(color="#cfd8e3")
    plt.setp(plt.getp(cb2.ax, "yticklabels"), color="#cfd8e3")

    fig.suptitle("АВРОРА — геометрия группировки на глобусе (Walker 300/15, i=75°, 1000 км)",
                 color="white", fontsize=13, y=0.97)
    fig.subplots_adjust(left=0.01, right=0.95, top=0.9, bottom=0.03, wspace=0.05)
    path = os.path.join(output_dir, f"pdop_globe_{label}.png")
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)

    vp = pdop[~np.isnan(pdop)]
    return {"image": path,
            "pdop_med": float(np.median(vp)), "pdop_p95": float(np.percentile(vp, 95)),
            "nvis_mean": float(nvis.mean()), "nvis_min": float(nvis.min()),
            "nvis_max": float(nvis.max())}


def print_pdop_globe_summary(label: str, r: Dict) -> None:
    print(f"\n  PDOP globe -- {label}")
    print(f"    PDOP медиана {r['pdop_med']:.2f} / p95 {r['pdop_p95']:.2f}")
    print(f"    N_vis: {r['nvis_min']:.1f}–{r['nvis_max']:.1f} (средн {r['nvis_mean']:.1f})")
    print(f"    Globe: {r['image']}")
