"""
Network metrics simulation runner.

Runs the ISL/GSL topology simulation at each time step and captures snapshots
for downstream metric analysis. Reuses LEOPath building blocks without
modifying the core simulation code.
"""

import math

import networkx as nx
import numpy as np
from astropy import units as astro_units
from astropy.time import Time
from tqdm import tqdm

from aurora.main import (
    calculate_link_params,
    generate_plus_grid_isls,
    setup_ground_stations,
)
from aurora.network_state.helpers import (
    _build_topologies,
    _compute_ground_station_satellites_in_range,
    _compute_isls,
)
from aurora.network_state.gsl_attachment.gsl_attachment_strategies import *  # noqa: F401,F403
from aurora.tles.read_tles import read_tles
from aurora.topology.satellite.satellite import Satellite
from aurora.topology.topology import ConstellationData


_C_M_S = 299_792_458.0  # speed of light, m/s


def run_network_sim(config: dict, tle_path: str) -> list[dict]:
    """
    Run topology simulation over 24 h and return per-step snapshots.

    Each snapshot dict contains:
        time_ns        — simulation time from epoch (int, ns)
        time_h         — simulation time (float, hours)
        gsl_attachment — {gs_name: sat_id | None}  (nearest satellite)
        gsl_distance_m — {gs_name: float}           (GS→sat distance)
        isl_edges      — [(sat_a, sat_b, dist_m), ...]
        n_active_isls  — int
        isl_mean_dist_m — float (mean ISL edge length)
        path_km        — {(gs_a, gs_b): float}      (shortest path length km)
        latency_ms     — {(gs_a, gs_b): float}      (one-way propagation ms)
    """
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

    undirected_isls = generate_plus_grid_isls(
        n_orbits=parsed["n_orbits"],
        n_sats_per_orbit=parsed["n_sats_per_orbit"],
    )

    gs_names = [gs.name for gs in ground_stations]
    gs_ids = [gs.id for gs in ground_stations]

    sim_cfg = config["simulation"]
    end_ns = int(sim_cfg["end_time_hours"] * 3600 * 1e9)
    step_ns = int(sim_cfg["time_step_minutes"] * 60 * 1e9)
    offset_ns = int(sim_cfg.get("offset_ns", 0))

    epoch_t = Time(parsed["epoch"], scale="tdb")
    snapshots = []

    time_steps = list(range(offset_ns, end_ns, step_ns))
    for t_ns in tqdm(time_steps, desc="Network metrics"):
        t_abs = epoch_t + t_ns * astro_units.ns
        time_str = str(t_abs.strftime("%Y/%m/%d %H:%M:%S.%f")[:-3])

        topology, _ = _build_topologies(constellation_data, ground_stations)

        try:
            _compute_isls(topology, undirected_isls, t_abs)
        except ValueError:
            # ISL distance exceeded max — skip step
            continue

        gs_visibility = _compute_ground_station_satellites_in_range(topology, t_abs)

        # Nearest satellite per GS
        gsl_attachment = {}
        gsl_distance = {}
        for gs, vis_list in zip(ground_stations, gs_visibility):
            if vis_list:
                best = min(vis_list, key=lambda x: x[0])
                gsl_attachment[gs.name] = best[1]
                gsl_distance[gs.name] = best[0]
            else:
                gsl_attachment[gs.name] = None
                gsl_distance[gs.name] = None

        # ISL edge snapshot
        isl_edges = []
        sat_node_ids = {s.id for s in satellites}
        for u, v, d in topology.graph.edges(data=True):
            if u in sat_node_ids and v in sat_node_ids:
                isl_edges.append((u, v, d.get("weight", 0.0)))

        isl_dists = [e[2] for e in isl_edges]
        isl_mean = float(np.mean(isl_dists)) if isl_dists else 0.0

        # Shortest path (km) and one-way latency (ms) between every GS pair
        path_km = {}
        latency_ms = {}
        for i, gs_a in enumerate(ground_stations):
            for j, gs_b in enumerate(ground_stations):
                if j <= i:
                    continue
                key = (gs_a.name, gs_b.name)
                try:
                    dist_m = nx.shortest_path_length(
                        topology.graph, source=gs_a.id, target=gs_b.id, weight="weight"
                    )
                    path_km[key] = dist_m / 1000.0
                    latency_ms[key] = dist_m / _C_M_S * 1000.0
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    path_km[key] = None
                    latency_ms[key] = None

        snapshots.append(
            {
                "time_ns": t_ns,
                "time_h": t_ns / 3.6e12,
                "gsl_attachment": gsl_attachment,
                "gsl_distance_m": gsl_distance,
                "isl_edges": isl_edges,
                "n_active_isls": len(isl_edges),
                "isl_mean_dist_m": isl_mean,
                "path_km": path_km,
                "latency_ms": latency_ms,
            }
        )

    return snapshots, gs_names
