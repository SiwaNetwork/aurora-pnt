"""
TESLA MAC / Anti-Spoofing Analysis for АВРОРА.

Models the TESLA (Timed Efficient Stream Loss-tolerant Authentication) protocol
for АВРОРА-T signal authentication. Compatible with Galileo OSNMA architecture.

TESLA key chain: pre-committed hash chain disclosed with delay.
Spoofing protection: receiver can authenticate signals retrospectively.
"""

import math
import os
import csv
import hashlib
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# АВРОРА-T frame parameters (from timing_service.py)
AURORA_T_FRAME_BITS  = 500
AURORA_T_FRAME_S     = 10.0    # seconds per frame
AURORA_T_MAC_BITS    = 128     # TESLA MAC length

# TESLA parameters (modeled on Galileo OSNMA)
TESLA_PARAMS = {
    "key_size_bits":      128,    # AES-128 key
    "mac_size_bits":      128,
    "hash_algo":          "SHA-256",
    "chain_length":       604800, # 1 week of keys (at 1 key/frame)
    "key_disclosure_delay_frames": 6,   # 6 frames = 60 s delay (Galileo: 1-4 subframes)
    "subframe_duration_s": 10.0,
    "auth_delay_s":        60.0,   # time to verify (= disclosure delay)
    "root_key_bits":      256,    # root key from trusted source (NovAtel-style)
}

# Attack scenarios
ATTACKS = {
    "Replay":         "Capture and retransmit authentic signal — blocked by TESLA timestamp",
    "Meaconing":      "Relay real signal with delay — detectable via timing mismatch",
    "Generation":     "Forge signals with wrong PRN — detectable via PRN mismatch",
    "Simulation":     "Full signal simulator — blocked by TESLA MAC (no key disclosure yet)",
    "MITM":           "Man-in-middle signal relay — TESLA hash chain detects alteration",
    "Physical":       "Physical access to satellite — mitigated by root key in MCS only",
}

# Spoofing detection timeline
DETECTION_PHASES = [
    (0,  10,  "Signal acquisition (no authentication yet — vulnerable)"),
    (10, 60,  "Pre-authentication window — cross-check ephemeris/timing"),
    (60, 70,  "First key disclosure received — TESLA MAC verified"),
    (70, 9999,"Authenticated tracking — spoofing attempt detectable"),
]


def tesla_chain_stats(params: Dict) -> Dict:
    """Compute TESLA key chain statistics."""
    n_keys   = params["chain_length"]
    key_bits = params["key_size_bits"]
    mac_bits = params["mac_size_bits"]
    delay_s  = params["key_disclosure_delay_frames"] * params["subframe_duration_s"]

    # Memory to store key chain (compressed: only need current + disclosed keys)
    storage_kb = key_bits / 8 * (params["key_disclosure_delay_frames"] + 1) / 1024

    # Brute force time for AES-128 (reference: 10^18 ops/s cluster)
    bf_ops    = 2 ** key_bits
    bf_years  = bf_ops / (1e18 * 3600 * 24 * 365)

    return {
        "chain_length_keys":    n_keys,
        "chain_lifetime_days":  n_keys * params["subframe_duration_s"] / 86400,
        "auth_delay_s":         delay_s,
        "key_storage_kb":       storage_kb,
        "mac_overhead_bits":    mac_bits,
        "frame_overhead_pct":   100 * mac_bits / AURORA_T_FRAME_BITS,
        "brute_force_years":    bf_years,
        "hash_algo":            params["hash_algo"],
    }


def compute_auth_timeline(
    frame_rate_hz: float = 0.1,   # 1 frame per 10 s
    delay_frames: int = 6,
) -> List[Dict]:
    """Model receiver authentication state over time from cold start."""
    events = []
    t = 0.0
    delay_s = delay_frames / frame_rate_hz

    states = [
        (0,       "UNAUTHENTICATED",  "No signal history — spoof possible"),
        (10,      "TRACKING",         "Signal tracked, ephemeris checked, no MAC yet"),
        (delay_s, "MAC_RECEIVED",     "First TESLA key disclosed — MAC verified"),
        (delay_s + 10, "AUTHENTICATED", "Ongoing authenticated tracking"),
    ]
    for t_event, state, desc in states:
        events.append({"t_s": t_event, "state": state, "description": desc})

    return events


def receiver_vulnerability_window(params: Dict) -> Dict:
    """Time window during which receiver is vulnerable to spoofing."""
    acq_time_s  = 30.0   # typical cold-start acquisition
    auth_delay  = params["auth_delay_s"]
    total_vuln  = acq_time_s + auth_delay
    return {
        "acquisition_s":        acq_time_s,
        "auth_delay_s":         auth_delay,
        "total_vulnerability_s": total_vuln,
        "vulnerability_vs_gps": "GPS C/A: unlimited (no authentication)",
        "vulnerability_vs_osnma": f"Galileo OSNMA: ~{30+60:.0f} s (similar)",
    }


def spoofing_detection_analysis() -> List[Dict]:
    results = []
    for attack, desc in ATTACKS.items():
        blocked = attack not in ("Physical",)
        results.append({
            "attack": attack,
            "description": desc,
            "tesla_blocks": blocked,
            "detection_delay_s": TESLA_PARAMS["auth_delay_s"] if blocked else None,
        })
    return results


def run_tesla_mac_analysis(output_dir: str, label: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    chain   = tesla_chain_stats(TESLA_PARAMS)
    vuln    = receiver_vulnerability_window(TESLA_PARAMS)
    detects = spoofing_detection_analysis()
    timeline = compute_auth_timeline()

    _plot_auth_timeline(timeline, vuln, output_dir, label)
    _plot_attack_summary(detects, output_dir, label)
    _save_tesla_csv(chain, vuln, detects, output_dir, label)

    return {
        "tesla_params": TESLA_PARAMS,
        "chain_stats":  chain,
        "vulnerability": vuln,
        "attack_analysis": detects,
        "timeline":     timeline,
        "frame_s":      AURORA_T_FRAME_S,
        "mac_bits":     AURORA_T_MAC_BITS,
    }


def _plot_auth_timeline(timeline: List[Dict], vuln: Dict,
                        output_dir: str, label: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    colors = {
        "UNAUTHENTICATED": "#e17055",
        "TRACKING":        "#fdcb6e",
        "MAC_RECEIVED":    "#74b9ff",
        "AUTHENTICATED":   "#00b894",
    }
    state_ru = {
        "UNAUTHENTICATED": "НЕ АУТЕНТИФ.",
        "TRACKING":        "СОПРОВОЖДЕНИЕ",
        "MAC_RECEIVED":    "MAC ПОЛУЧЕН",
        "AUTHENTICATED":   "АУТЕНТИФИЦИРОВАН",
    }
    t_max = 120.0
    prev_t = 0.0
    for i, ev in enumerate(timeline):
        t_end = timeline[i+1]["t_s"] if i+1 < len(timeline) else t_max
        t_end = min(t_end, t_max)
        col = colors.get(ev["state"], "#aaa")
        ax.barh(0, t_end - prev_t, left=prev_t, height=0.5,
                color=col, edgecolor="white",
                label=state_ru.get(ev["state"], ev["state"]))
        mid = (prev_t + t_end) / 2
        if t_end - prev_t > 5:
            ax.text(mid, 0, state_ru.get(ev["state"], ev["state"]),
                    ha="center", va="center",
                    fontsize=8, fontweight="bold", color="white")
        prev_t = t_end

    ax.axvline(vuln["acquisition_s"], ls="--", color="#636e72", lw=1)
    ax.axvline(vuln["total_vulnerability_s"], ls="--", color="#6c5ce7", lw=1.5,
               label=f"Аутентификация завершена ({vuln['total_vulnerability_s']:.0f} с)")
    ax.set_xlim(0, t_max)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel("Время от холодного старта (с)")
    ax.set_yticks([])
    ax.set_title(f"АВРОРА-T TESLA MAC — Хронология аутентификации [{label}]")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"tesla_timeline_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_attack_summary(detects: List[Dict], output_dir: str, label: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    attacks = [d["attack"] for d in detects]
    blocked = [1 if d["tesla_blocks"] else 0 for d in detects]
    colors  = ["#00b894" if b else "#e17055" for b in blocked]
    bars    = ax.barh(range(len(attacks)), blocked, color=colors, edgecolor="white")
    ax.set_yticks(range(len(attacks)))
    ax.set_yticklabels(attacks)
    ax.set_xlim(0, 1.4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Блокировано TESLA", "Не блокировано"])
    for i, (b, d) in enumerate(zip(blocked, detects)):
        txt = "БЛОК." if b else "РИСК"
        col = "white" if b else "#e17055"
        ax.text(0.05, i, txt, va="center", fontsize=9, fontweight="bold", color=col)
    ax.set_title(f"АВРОРА-T TESLA MAC — защита от атак [{label}]")
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"tesla_attacks_{label}.png"), dpi=150)
    plt.close(fig)


def _save_tesla_csv(chain, vuln, detects, output_dir, label) -> None:
    path = os.path.join(output_dir, f"tesla_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["parameter", "value"])
        for k, v in chain.items():
            w.writerow([k, v])
        w.writerow(["vulnerability_window_s", vuln["total_vulnerability_s"]])
        w.writerow(["", ""])
        w.writerow(["attack", "tesla_blocks", "detection_delay_s"])
        for d in detects:
            w.writerow([d["attack"], d["tesla_blocks"], d.get("detection_delay_s", "N/A")])


def print_tesla_summary(label: str, result: Dict) -> None:
    sep = "=" * 66

    print(f"\n{sep}")
    print(f"  TESLA MAC Anti-Spoofing -- {label}")
    print(sep)
    c = result["chain_stats"]
    v = result["vulnerability"]

    print(f"  АВРОРА-T Frame: {result['frame_s']:.0f} s, "
          f"MAC: {result['mac_bits']} bits, "
          f"overhead: {c['frame_overhead_pct']:.1f}% of frame")
    print()
    print(f"  TESLA Key Chain:")
    print(f"    Algorithm:         {c['hash_algo']}")
    print(f"    Key size:          {result['tesla_params']['key_size_bits']} bits (AES-128)")
    print(f"    Chain length:      {c['chain_length_keys']:,} keys "
          f"({c['chain_lifetime_days']:.0f} days)")
    print(f"    Key disclosure:    {result['tesla_params']['key_disclosure_delay_frames']} frames "
          f"= {v['auth_delay_s']:.0f} s")
    print(f"    Storage per RX:    {c['key_storage_kb']:.2f} KB")
    print(f"    Brute-force:       {c['brute_force_years']:.1e} years")
    print()
    print(f"  Vulnerability window:")
    print(f"    Acquisition:       {v['acquisition_s']:.0f} s")
    print(f"    Auth delay:        {v['auth_delay_s']:.0f} s")
    print(f"    Total:             {v['total_vulnerability_s']:.0f} s  "
          f"(then fully authenticated)")
    print(f"    GPS C/A:           UNLIMITED (no authentication)")
    print(f"    Galileo OSNMA:     ~90 s (similar architecture)")
    print()
    print(f"  Attack Mitigation:")
    print(f"  {'Attack':<16} {'TESLA blocks':>14} {'Detection delay':>16}")
    print("  " + "-" * 50)
    for d in result["attack_analysis"]:
        blocked = "[BLOCKED]" if d["tesla_blocks"] else "[RISK]"
        delay   = f"{d['detection_delay_s']:.0f} s" if d["tesla_blocks"] else "N/A"
        print(f"  {d['attack']:<16} {blocked:>14} {delay:>16}")
    print(sep)
