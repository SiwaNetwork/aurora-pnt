# AURORA PNT
### Advanced Universal Real-time Orbital Radio-navigation Architecture
**by Shiwa Network**

---

AURORA PNT is a LEO satellite navigation and timing system developed by **Shiwa Network**. It provides precise positioning, navigation, and time synchronization based on a low-Earth-orbit satellite constellation, supporting both autonomous operation (sovereign LPT time scale) and combined mode with GLONASS.

This repository contains the simulation framework used to model, analyze, and visualize all aspects of the AURORA constellation.

---

## Key Characteristics (Phase 4 — Global)

| Parameter | Value |
|---|---|
| Satellites | 300 (15 planes × 20 sat) |
| Altitude | 1000 km |
| Inclination | 75° |
| Type | Walker Delta, phase_diff=true |
| Elevation mask | 10° (LEO) / 5° (GLONASS) |
| PDOP p95 (autonomous) | 5.04 |
| PDOP p95 (LEO+GLONASS) | **1.67** |
| Coverage 4+ satellites | 100% (global, 24 h) |
| Signal advantage vs GPS | **+25 dB (x316 power)** |

---

## Operating Modes

| Mode | Description | CEP (L1+L5) |
|---|---|---|
| A — Autonomous LPT | LEO-only, sovereign time scale, no external dependency | 6.46 m |
| B — Combined LEO+GLONASS | ISB solved as 5th unknown, best accuracy | **1.18 m** |
| C — Combined + SDCM | Mode B + Russian differential corrections | **0.67 m** |

---

## Deployment Roadmap

| Phase | Satellites | Configuration | Coverage | PDOP<6 |
|---|---|---|---|---|
| 0 — Demonstrator | 3 | 1x3, 87 deg | 0% | — |
| 1 — Test | 12 | 3x4, 87 deg | 5% | 2% |
| 2 — Regional | 90 | 9x10, 75 deg | 82% | 62% |
| 3 — Operational (Russia) | **180** | **12x15, 75 deg** | **100%** | **97%** |
| 4 — Global | 300 | 15x20, 75 deg | 100% | 97%+ |

---

## Installation

```bash
pip install aurora-pnt
```

Or from source:

```bash
git clone https://github.com/ShiwaNetwork/aurora-pnt.git
cd aurora-pnt
pip install -e .
```

**Requirements:** Python 3.8+, numpy, matplotlib, sgp4, pyyaml, scipy

---

## Quick Start

```bash
# Run Phase 3 simulation (operational Russia coverage)
aurora-pnt run -c aurora/config/pnt/phase3_operational.yaml -o results/phase3 -l phase3

# Combined LEO+GLONASS (best accuracy)
aurora-pnt combined -c aurora/config/pnt/phase4_global.yaml -o results/combined -l combined

# Autonomous mode (sovereign, no GLONASS)
aurora-pnt combined -c aurora/config/pnt/phase4_global.yaml -o results/autonomous -l autonomous --mode autonomous

# AURORA-T timing service analysis (PTP/NTP grandmaster)
aurora-pnt timing-service -o results/timing -l phase4

# Mixed-clock architecture (OCXO/Rb/Cs per tier)
aurora-pnt clock-arch -o results/clock_arch -l phase4
```

---

## CLI Reference

```bash
aurora-pnt run              # PNT simulation (coverage, PDOP, visibility)
aurora-pnt combined         # Multi-constellation LEO+GLONASS (--mode combined/autonomous/glonass)
aurora-pnt time-scale       # LPT time scale analysis (stability, UERE by mode)
aurora-pnt timing-service   # AURORA-T PTP/NTP grandmaster accuracy
aurora-pnt clock-arch       # Mixed-clock (OCXO/Rb/Cs) ISL chain + holdover analysis
aurora-pnt link-budget      # Link budget (FSPL, Doppler, C/N0)
aurora-pnt network-metrics  # ISL/GSL topology, latency, routing stability
aurora-pnt ranging          # UERE budget, position accuracy
aurora-pnt clock-analysis   # Clock type comparison (TCXO/Rb/Cs/Maser)
aurora-pnt raim             # RAIM integrity (HPL, VPL, availability)
aurora-pnt resilience       # Satellite failure resilience sweep
aurora-pnt cesium           # Interactive CesiumJS globe visualization
aurora-pnt viz              # 3D globe + ground track visualization
aurora-pnt info             # Simulation plan (no execution)
aurora-pnt experiment       # Parametric sweep experiments
```

---

## Project Structure

```
aurora/
├── pnt/                  # Core PNT simulation modules
│   ├── pnt_simulator.py  # Main Walker-Delta simulation engine
│   ├── combined_sim.py   # LEO+GLONASS multi-constellation (ISB)
│   ├── time_scale.py     # LPT time scale, UERE budget by mode
│   ├── timing_service.py # AURORA-T protocol, PTP/NTP accuracy
│   ├── glonass.py        # GLONASS constellation model
│   ├── dop.py            # DOP computation (4- and 5-parameter H matrix)
│   ├── coverage.py       # SGP4 propagation, visibility, footprint
│   ├── raim.py           # RAIM integrity (HPL/VPL)
│   ├── resilience.py     # Satellite failure analysis
│   └── cli.py            # aurora-pnt CLI entry point
├── link_budget/          # FSPL, Doppler, C/N0 per GS-satellite pair
├── network_metrics/      # ISL/GSL topology, Hypatia path stability
├── ranging/              # UERE, position error, clock analysis
└── config/pnt/           # YAML configuration files for all phases
results/                  # Simulation outputs (auto-generated)
```

---

## LPT Time Scale

AURORA operates its own sovereign time scale (LPT — LEO PNT Time), independent of GPS and GLONASS:

```
MCS Master Clock (Cs/H-Maser, Zheleznogorsk)
    |
    +-- ISL chain --> all 300 satellites     sigma_ISL = sqrt(N) x ppb x T_sync
    |
    +-- AURORA-T broadcast (L1/L5, 10 s frame)
         |
         +-- Ground receiver
              +-- 1PPS output          < 5 ns  (with Cs master)
              +-- PTP Grandmaster      IEEE 1588-2019, Class 25
              +-- NTP Stratum-1        RFC 5905, refid="AURA"
```

| Master clock | Timing accuracy | PTP class |
|---|---|---|
| OCXO (1 ppb) | 169.7 ns | Class 33 |
| Rb (0.1 ppb) | 17.0 ns | **Class 25** |
| Cs (0.01 ppb) | **2.2 ns** | **Class 25 — Stratum-0** |

---

## Ground Infrastructure (Phase 4 — 21 stations)

| Region | Stations |
|---|---|
| Russia | Vladimir, Murmansk, Novosibirsk, Zheleznogorsk, Yakutsk, Khabarovsk, Vladivostok |
| CIS | Minsk, Almaty |
| Middle East | Tehran |
| Asia | Urumqi, Delhi, Yangon, Jakarta, Pyongyang, Ulaanbaatar |
| Africa | Luanda, Nairobi, Johannesburg |
| Western Hemisphere | Havana, Buenos Aires |

---

## UERE Budget (L1+L5 dual-frequency)

| Error source | Autonomous | Combined | Combined+SDCM |
|---|---|---|---|
| Clock bias | 3.0 m | 1.5 m | 0.8 m |
| Ephemeris | 0.5 m | 0.1 m | 0.05 m |
| Ionosphere | 0.05 m | 0.05 m | 0.05 m |
| Troposphere | 0.5 m | 0.5 m | 0.3 m |
| Multipath | 0.3 m | 0.3 m | 0.3 m |
| ISB residual | — | 0.5 m | 0.3 m |
| **UERE RSS** | **3.10 m** | **1.69 m** | **0.96 m** |

---

## Signal Advantage vs GPS

| Parameter | GPS (MEO, 20200 km) | AURORA (LEO, 1000 km) | Gain |
|---|---|---|---|
| FSPL at zenith | -166.3 dB | -141.4 dB | **+24.9 dB** |
| FSPL at 10 deg | -177.9 dB | -157.7 dB | **+20.2 dB** |
| Signal power | x1 | **x316 (zenith)** | |
| Jamming resistance | standard | significantly higher | |

The +25 dB advantage enables reception in urban canyons, indoors, and under jamming conditions where GPS fails.

---

## License

Copyright (c) 2026 Shiwa Network. All rights reserved.
See [LICENSE](LICENSE) for details.

---

*AURORA PNT by Shiwa Network — 2026*
