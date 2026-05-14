"""
Pass window analysis for ground stations.

For each station: per-timestep satellite count + contiguous visibility intervals.
"""

import csv
from pathlib import Path


def extract_pass_windows(
    station_timeseries: dict,
    threshold: int = 1,
) -> dict:
    """
    Extract contiguous pass windows from per-station (time_s, n_sats) timeseries.

    Args:
        station_timeseries: {station_name: [(t_s, n_sats), ...]}
        threshold: minimum n_sats to count as visible

    Returns:
        {station_name: [{"start_s", "end_s", "start_h", "end_h", "duration_min", "peak_sats"}, ...]}
    """
    return {name: _windows_from_series(ts, threshold) for name, ts in station_timeseries.items()}


def _windows_from_series(timeseries: list, threshold: int) -> list:
    windows = []
    in_window = False
    win_start = win_end = win_peak = 0

    for t_s, n_sats in timeseries:
        if n_sats >= threshold:
            if not in_window:
                in_window = True
                win_start = t_s
                win_peak = n_sats
            else:
                win_peak = max(win_peak, n_sats)
            win_end = t_s
        else:
            if in_window:
                windows.append(_make_window(win_start, win_end, win_peak))
                in_window = False

    if in_window:
        windows.append(_make_window(win_start, win_end, win_peak))

    return windows


def _make_window(start_s: float, end_s: float, peak_sats: int) -> dict:
    step_s = 60.0  # assume 1-min steps; last step counts as full minute
    return {
        "start_s": start_s,
        "end_s": end_s,
        "start_h": round(start_s / 3600, 4),
        "end_h": round(end_s / 3600, 4),
        "duration_min": round((end_s - start_s) / 60 + 1, 1),
        "peak_sats": peak_sats,
    }


def save_pass_windows_csv(windows: dict, output_dir: str, label: str) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = str(Path(output_dir) / f"pass_windows_{label}.csv")
    fields = ["station", "start_h", "end_h", "duration_min", "peak_sats"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for station, wins in windows.items():
            for w in wins:
                writer.writerow({
                    "station": station,
                    "start_h": w["start_h"],
                    "end_h": w["end_h"],
                    "duration_min": w["duration_min"],
                    "peak_sats": w["peak_sats"],
                })
    return path


def save_station_visibility_csv(
    station_timeseries: dict,
    output_dir: str,
    label: str,
) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = str(Path(output_dir) / f"station_visibility_{label}.csv")
    station_names = list(station_timeseries.keys())
    times = [t for t, _ in next(iter(station_timeseries.values()))]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "time_h"] + station_names)
        for i, t_s in enumerate(times):
            row = [t_s, round(t_s / 3600, 4)]
            for name in station_names:
                row.append(station_timeseries[name][i][1])
            writer.writerow(row)
    return path


def print_pass_summary(label: str, windows: dict) -> None:
    sep = "-" * 62
    print(f"\n{sep}")
    print(f"  Pass windows — {label}")
    print(sep)
    for station, wins in windows.items():
        total_min = sum(w["duration_min"] for w in wins)
        print(f"\n  {station}: {len(wins)} window(s),  {total_min:.0f} min/day total")
        if not wins:
            print("    (no passes)")
            continue
        for i, w in enumerate(wins, 1):
            sh = int(w["start_h"])
            sm = int((w["start_h"] - sh) * 60)
            eh = int(w["end_h"])
            em = int((w["end_h"] - eh) * 60)
            print(
                f"    [{i:2d}]  {sh:02d}:{sm:02d} - {eh:02d}:{em:02d}"
                f"   {w['duration_min']:5.1f} min   peak: {w['peak_sats']} sat"
            )
    print(sep)
