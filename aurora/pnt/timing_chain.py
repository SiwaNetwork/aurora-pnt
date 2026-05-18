"""
Timing Chain Analysis for AURORA PNT.

Models the complete timing chain from master Cs frequency standard
to user 1PPS output, computing Allan deviation, ISL transfer noise,
and end-to-end timing uncertainty at each level.

Architecture:
  Cs anchor → ISL → Rb relay → ISL → OCXO terminal → signal → user receiver

Reference: IEEE Std 1139-2008 (ADEV); Osen, "GNSS Clock Modelling" (2020);
           White, "Precise Timing Using LEO" (2022).
"""

import math
import os
import csv
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Physical constants ────────────────────────────────────────────────────────
C_LIGHT  = 299_792_458.0   # m/s
NS_PER_M = 1e9 / C_LIGHT   # ns / m (≈ 3.336 ns/m)

# ── Clock noise models (ADEV coefficients, 3-term power law) ─────────────────
# ADEV(τ) = √(h_{-2}/τ² + h_{-1}/τ + h_0 + h_1·τ + h_2·τ²)   (simplified)
# Using dominant-term model for each clock type:
#   Cs: white FM (h_0) + flicker FM (h_{-1}): ADEV ~ 5e-12 at τ=1s → 2e-13 at τ=1000s
#   Rb: flicker FM + random walk: ADEV ~ 3e-11 at τ=1s → 1e-12 at 1000s
#   OCXO: white PM + flicker FM: ADEV ~ 1e-10 at τ=1s → 3e-11 at 100s (aging limited)

CLOCKS = {
    "Cs (крейзер)": {
        "label":       "Cs",
        "color":       "#0984e3",
        "adev_1s":     5e-12,    # Allan deviation at 1 s
        "exponent":    -0.5,     # slope in log-log: -0.5 = white FM
        "holdover_ppb_per_day": 0.001,  # ultra-stable, no aging
        "mass_kg":     0.7,
        "power_w":     4.0,
        "desc":        "Cs beam, 1 per plane anchor",
        "n_per_fleet": 15,
    },
    "Rb (ретранслятор)": {
        "label":       "Rb",
        "color":       "#00b894",
        "adev_1s":     3e-11,
        "exponent":    -0.5,
        "holdover_ppb_per_day": 0.05,
        "mass_kg":     0.4,
        "power_w":     3.0,
        "desc":        "Rb vapor cell, 3 per plane relay",
        "n_per_fleet": 45,
    },
    "OCXO (терминал)": {
        "label":       "OCXO",
        "color":       "#e17055",
        "adev_1s":     1e-10,
        "exponent":    -0.5,
        "holdover_ppb_per_day": 0.5,
        "mass_kg":     0.2,
        "power_w":     2.5,
        "desc":        "Oven-controlled XO, 16 per plane terminal",
        "n_per_fleet": 240,
    },
    "TCXO (сравн.)": {
        "label":       "TCXO",
        "color":       "#fdcb6e",
        "adev_1s":     1e-9,
        "exponent":    -0.5,
        "holdover_ppb_per_day": 50.0,
        "mass_kg":     0.05,
        "power_w":     0.1,
        "desc":        "Temperature-compensated XO (user equipment ref.)",
        "n_per_fleet": 0,
    },
}

# ── ISL link parameters ───────────────────────────────────────────────────────
ISL = {
    "range_m":          3_000_000.0,   # typical ISL distance (km)
    "code_noise_m":     0.30,          # code pseudorange noise (1σ, m)
    "phase_noise_m":    0.010,         # carrier phase noise (1σ, m)
    "propagation_err_m": 0.005,        # residual propagation delay uncertainty
    "relativity_err_ns": 0.10,         # Sagnac + relativistic corrections error
}

# ── Timing chain levels ───────────────────────────────────────────────────────
CHAIN = {
    "Cs (master)": {
        "clock":       "Cs (крейзер)",
        "isl_hops":    0,
        "sync_interval_s": 0.0,
        "desc": "Master time scale; defines AURORA Time (ATS)",
    },
    "Rb (relay, 1 hop)": {
        "clock":       "Rb (ретранслятор)",
        "isl_hops":    1,
        "sync_interval_s": 60.0,
        "desc": "Rb relay satellite; synced to Cs via 1 ISL hop",
    },
    "Rb (relay, 3 hops)": {
        "clock":       "Rb (ретранслятор)",
        "isl_hops":    3,
        "sync_interval_s": 60.0,
        "desc": "Rb relay at far end of plane; 3 ISL hops from Cs",
    },
    "OCXO (terminal, 2 hops)": {
        "clock":       "OCXO (терминал)",
        "isl_hops":    2,
        "sync_interval_s": 10.0,
        "desc": "OCXO terminal synced via 2 ISL hops",
    },
    "OCXO (terminal, 6 hops)": {
        "clock":       "OCXO (терминал)",
        "isl_hops":    6,
        "sync_interval_s": 10.0,
        "desc": "OCXO terminal at worst-case 6 ISL hops from Cs",
    },
}

# ── User receiver parameters ──────────────────────────────────────────────────
USER_RECEIVERS = {
    "Геодезия (Survey)": {
        "tcxo_adev_1s":    1e-9,
        "multipath_ns":    0.3,
        "iono_residual_ns": 0.3,  # dual-freq L1+L5
        "tropo_ns":        1.0,
        "sv_clock_ns":     1.0,   # from signal
        "sv_orbit_ns":     1.5,   # from orbit determination
        "desc": "Survey GNSS, dual-freq",
    },
    "Ручной (Handheld)": {
        "tcxo_adev_1s":    5e-9,
        "multipath_ns":    3.0,
        "iono_residual_ns": 5.0,  # single-freq
        "tropo_ns":        2.0,
        "sv_clock_ns":     1.0,
        "sv_orbit_ns":     1.5,
        "desc": "Consumer smartphone, single-freq",
    },
    "Телеком (Timing)": {
        "tcxo_adev_1s":    1e-10,  # external 10 MHz reference
        "multipath_ns":    0.1,
        "iono_residual_ns": 0.1,
        "tropo_ns":        0.3,
        "sv_clock_ns":     1.0,
        "sv_orbit_ns":     1.5,
        "desc": "Telecom timing receiver, external Rb ref.",
    },
}

AURORA_ALT_M = 1_000_000.0   # 1000 km


def adev(clock_key: str, tau_s: float) -> float:
    """Allan deviation for given clock type at averaging interval tau (s)."""
    c = CLOCKS[clock_key]
    return c["adev_1s"] * (tau_s ** c["exponent"])


def isl_transfer_noise_ns(n_hops: int, mode: str = "code") -> float:
    """
    Timing transfer noise (ns, 1σ) accumulated over n ISL hops.

    Each hop adds quadrature noise from: ranging measurement + propagation + Sagnac.
    """
    if n_hops == 0:
        return 0.0
    per_hop_m = {
        "code":  math.sqrt(ISL["code_noise_m"]**2 + ISL["propagation_err_m"]**2),
        "phase": math.sqrt(ISL["phase_noise_m"]**2 + ISL["propagation_err_m"]**2),
    }[mode]
    per_hop_ns = per_hop_m * NS_PER_M
    total_ns   = math.sqrt(n_hops) * per_hop_ns + ISL["relativity_err_ns"]
    return total_ns


def holdover_err_ns(clock_key: str, holdover_s: float) -> float:
    """
    Timing error after holdover_s seconds without sync.

    Uses linear frequency drift model: σ ≈ ADEV(τ) × τ (simplified; actual
    cumulative error = integral of frequency deviation).
    """
    c = CLOCKS[clock_key]
    # ADEV(τ) × τ gives timing error magnitude for white FM / flicker FM
    drift_ns = adev(clock_key, holdover_s) * holdover_s * 1e9
    return drift_ns


def chain_level_budget(level_key: str) -> Dict:
    """
    Compute timing uncertainty budget for one level of the timing chain.

    Returns dict with individual error contributors and total.
    """
    level = CHAIN[level_key]
    clock_key = level["clock"]
    n_hops    = level["isl_hops"]
    sync_s    = level["sync_interval_s"]

    # Clock oscillator error (ADEV over sync interval = free-run noise floor)
    if sync_s > 0:
        osc_err_ns = adev(clock_key, sync_s) * sync_s * 1e9
    else:
        osc_err_ns = adev(clock_key, 1.0) * 1e9   # 1-second noise

    # ISL ranging noise
    isl_ns = isl_transfer_noise_ns(n_hops, mode="code")

    # Relativistic corrections (Sagnac + relativistic frequency shift)
    # AURORA: orbital velocity 7.35 km/s, altitude 1000 km
    # Relativistic freq shift: (gh/c² - v²/2c²) ≈ 5.4e-10 → ±0.2 ns/s if uncompensated
    relativistic_ns = 0.2 if n_hops > 0 else 0.0

    total_ns = math.sqrt(osc_err_ns**2 + isl_ns**2 + relativistic_ns**2)

    return {
        "level":            level_key,
        "clock":            clock_key,
        "n_hops":           n_hops,
        "sync_interval_s":  sync_s,
        "osc_err_ns":       osc_err_ns,
        "isl_ns":           isl_ns,
        "relativistic_ns":  relativistic_ns,
        "total_ns":         total_ns,
    }


def user_timing_budget(level_key: str, user_key: str) -> Dict:
    """
    End-to-end timing budget at user receiver.

    Combines satellite-side errors with propagation and receiver errors.
    """
    sat_budget = chain_level_budget(level_key)
    user  = USER_RECEIVERS[user_key]

    sv_clock_ns  = sat_budget["total_ns"]
    sv_orbit_ns  = user["sv_orbit_ns"]
    prop_iono_ns = user["iono_residual_ns"]
    prop_tropo_ns = user["tropo_ns"]
    multipath_ns  = user["multipath_ns"]

    # User receiver oscillator (OCXO/TCXO noise over 1 PPS integration)
    rx_osc_ns = user["tcxo_adev_1s"] * 1.0 * 1e9   # at τ=1s

    # Total (RSS)
    total_ns = math.sqrt(
        sv_clock_ns**2 + sv_orbit_ns**2 + prop_iono_ns**2
        + prop_tropo_ns**2 + multipath_ns**2 + rx_osc_ns**2
    )

    return {
        "sat_level":       level_key,
        "user_type":       user_key,
        "sv_clock_ns":     sv_clock_ns,
        "sv_orbit_ns":     sv_orbit_ns,
        "iono_ns":         prop_iono_ns,
        "tropo_ns":        prop_tropo_ns,
        "multipath_ns":    multipath_ns,
        "rx_osc_ns":       rx_osc_ns,
        "total_ns":        total_ns,
        "total_m":         total_ns / NS_PER_M,
        "utc_req_ns":      100.0,
        "ptp_class_a_ns":  100.0,
        "viable_100ns":    total_ns <= 100.0,
    }


def run_timing_chain_analysis(
    output_dir: str,
    label: str,
    n_hops_max: int = 12,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    # ── Allan deviation curves ─────────────────────────────────────────────────
    tau_range = np.logspace(-1, 5, 500)   # 0.1 s to 100 000 s

    # ── Chain level budgets ────────────────────────────────────────────────────
    chain_budgets = {k: chain_level_budget(k) for k in CHAIN}

    # ── User timing budgets ────────────────────────────────────────────────────
    user_budgets = {}
    for uk in USER_RECEIVERS:
        user_budgets[uk] = {}
        for lk in CHAIN:
            user_budgets[uk][lk] = user_timing_budget(lk, uk)

    # ── ISL hops sweep ────────────────────────────────────────────────────────
    hops_range    = list(range(0, n_hops_max + 1))
    isl_noise_code  = [isl_transfer_noise_ns(h, "code")  for h in hops_range]
    isl_noise_phase = [isl_transfer_noise_ns(h, "phase") for h in hops_range]

    # ── Holdover curves ───────────────────────────────────────────────────────
    holdover_t = np.logspace(0, 5, 400)   # 1 s to 28 h
    holdover_curves = {
        ck: holdover_err_ns(ck, holdover_t) for ck in CLOCKS
    }

    _plot_adev(tau_range, output_dir, label)
    _plot_isl_transfer(hops_range, isl_noise_code, isl_noise_phase, output_dir, label)
    _plot_holdover(holdover_t, holdover_curves, output_dir, label)
    _plot_chain_budget(chain_budgets, output_dir, label)
    _plot_user_budget(user_budgets, output_dir, label)
    _save_timing_csv(chain_budgets, user_budgets, output_dir, label)

    return {
        "chain_budgets": chain_budgets,
        "user_budgets":  user_budgets,
        "isl_noise_code": isl_noise_code,
        "isl_noise_phase": isl_noise_phase,
        "hops_range":    hops_range,
        "n_hops_max":    n_hops_max,
        "holdover_curves": holdover_curves,
    }


def _adev_tau(clock_key: str, tau_s: float) -> float:
    c = CLOCKS[clock_key]
    return c["adev_1s"] * (tau_s ** c["exponent"])


def _plot_adev(tau_range, output_dir, label):
    fig, ax = plt.subplots(figsize=(11, 6))
    for ck, c in CLOCKS.items():
        adevs = [_adev_tau(ck, t) for t in tau_range]
        ax.loglog(tau_range, adevs, color=c["color"], lw=2, label=f"{c['label']} — {ck}")

    # Reference lines
    ax.axhline(1e-12, ls=":", color="gray", lw=0.8, label="1e-12 (0.001 нс/с)")
    ax.axhline(1e-10, ls=":", color="#ccc", lw=0.8, label="1e-10 (0.1 нс/с)")
    ax.set_xlabel("Интервал усреднения τ (с)")
    ax.set_ylabel("Отклонение Аллана ADEV(τ)")
    ax.set_title(f"AURORA PNT — Отклонение Аллана часов созвездия [{label}]")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"timing_adev_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_isl_transfer(hops_range, isl_code, isl_phase, output_dir, label):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hops_range, isl_code,  "o-", color="#e17055", lw=2,
            label="Кодовые измерения (σ = 0.30 м)")
    ax.plot(hops_range, isl_phase, "s-", color="#0984e3", lw=2,
            label="Фазовые измерения (σ = 0.01 м)")
    ax.axhline(10.0, ls="--", color="#6c5ce7", lw=1.2, label="10 нс (UTC цель)")
    ax.axhline(1.0,  ls=":",  color="#00b894", lw=1.2, label="1 нс (телеком цель)")
    ax.set_xlabel("Количество прыжков ISL")
    ax.set_ylabel("Погрешность синхронизации (нс, 1σ)")
    ax.set_title(f"AURORA — Погрешность передачи времени через ISL [{label}]")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"timing_isl_transfer_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_holdover(holdover_t, holdover_curves, output_dir, label):
    fig, ax = plt.subplots(figsize=(11, 6))
    for ck, c in CLOCKS.items():
        errs = holdover_curves[ck]
        ax.loglog(holdover_t, errs, color=c["color"], lw=2, label=f"{c['label']}")

    # Reference marks
    ax.axvline(60,   ls=":", color="gray", lw=0.8, label="60 с (интервал ISL)")
    ax.axvline(3600, ls=":", color="#ccc", lw=0.8, label="1 ч (пролёт)")
    ax.axhline(10,   ls="--", color="#6c5ce7", lw=1.2, label="10 нс (UTC)")
    ax.axhline(100,  ls="--", color="#e17055", lw=1.2, label="100 нс (ITU-T G.8272)")
    ax.set_xlabel("Время без синхронизации (с)")
    ax.set_ylabel("Накопленная погрешность (нс)")
    ax.set_title(f"AURORA — Деградация хранения времени (holdover) [{label}]")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"timing_holdover_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_chain_budget(chain_budgets, output_dir, label):
    levels = list(chain_budgets.keys())
    osc    = [chain_budgets[l]["osc_err_ns"]  for l in levels]
    isl    = [chain_budgets[l]["isl_ns"]       for l in levels]
    rel    = [chain_budgets[l]["relativistic_ns"] for l in levels]
    total  = [chain_budgets[l]["total_ns"]     for l in levels]

    x = np.arange(len(levels))
    w = 0.25

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - w,   osc,  w, label="Генератор (нс)", color="#74b9ff", edgecolor="white")
    ax.bar(x,       isl,  w, label="ISL (нс)",        color="#00b894", edgecolor="white")
    ax.bar(x + w,   rel,  w, label="Релятивизм (нс)", color="#fdcb6e", edgecolor="white")
    ax.plot(x, total, "D--", color="#e17055", lw=2, ms=7, label="Сумм. погрешность (нс)")

    ax.axhline(10.0, ls=":", color="#6c5ce7", lw=1.2, label="10 нс (UTC)")
    ax.set_xticks(x)
    ax.set_xticklabels([l.split("(")[0].strip() for l in levels], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Погрешность синхронизации (нс, 1σ)")
    ax.set_title(f"AURORA — Бюджет погрешности по уровням цепи синхронизации [{label}]")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"timing_chain_budget_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_user_budget(user_budgets, output_dir, label):
    user_types = list(USER_RECEIVERS.keys())
    levels     = list(CHAIN.keys())

    fig, axes = plt.subplots(1, len(user_types), figsize=(6 * len(user_types), 6), sharey=True)
    if len(user_types) == 1:
        axes = [axes]

    for ax, uk in zip(axes, user_types):
        totals    = [user_budgets[uk][lk]["total_ns"] for lk in levels]
        sv_clocks = [user_budgets[uk][lk]["sv_clock_ns"] for lk in levels]
        x = np.arange(len(levels))

        ax.bar(x, totals, color="#6c5ce7", alpha=0.7, label="Суммарная (нс)")
        ax.bar(x, sv_clocks, color="#0984e3", alpha=0.9, label="Спутн. часы (нс)")
        ax.axhline(100.0, ls="--", color="#e17055", lw=1.2, label="100 нс (UTC)")
        ax.axhline(10.0,  ls=":",  color="#00b894", lw=1.2, label="10 нс (телеком)")
        ax.set_xticks(x)
        ax.set_xticklabels([l.split("(")[0].strip() for l in levels], rotation=30, ha="right", fontsize=7)
        ax.set_title(f"{uk}", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Суммарная погрешность 1PPS (нс, 1σ)")
    fig.suptitle(f"AURORA PNT — Сквозной бюджет погрешности синхронизации [{label}]", fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"timing_user_budget_{label}.png"), dpi=150)
    plt.close(fig)


def _save_timing_csv(chain_budgets, user_budgets, output_dir, label):
    path = os.path.join(output_dir, f"timing_chain_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["level", "clock", "n_hops", "sync_interval_s",
                    "osc_err_ns", "isl_ns", "relativistic_ns", "total_ns"])
        for lk, b in chain_budgets.items():
            w.writerow([lk, b["clock"], b["n_hops"], b["sync_interval_s"],
                        f"{b['osc_err_ns']:.3f}", f"{b['isl_ns']:.3f}",
                        f"{b['relativistic_ns']:.3f}", f"{b['total_ns']:.3f}"])
        w.writerow([])
        w.writerow(["user_type", "sat_level", "total_ns", "viable_100ns"])
        for uk, levels in user_budgets.items():
            for lk, ub in levels.items():
                w.writerow([uk, lk, f"{ub['total_ns']:.2f}", ub["viable_100ns"]])


def print_timing_chain_summary(label: str, result: Dict) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  Timing Chain Analysis -- {label}")
    print(sep)

    print(f"\n  Часы созвездия AURORA:")
    print(f"  {'Тип':<25} {'ADEV(1с)':>12}  {'ADEV(60с)':>12}  {'Хранение 1ч':>12}")
    print(f"  {'':─<65}")
    for ck, c in CLOCKS.items():
        if c["n_per_fleet"] == 0:
            continue
        a1  = adev(ck, 1.0)
        a60 = adev(ck, 60.0)
        ho  = holdover_err_ns(ck, 3600.0)
        print(f"  {ck:<25}  {a1:>10.2e}    {a60:>10.2e}    {ho:>8.1f} нс")

    print(f"\n  Бюджет погрешности по уровням цепи (код ISL):")
    print(f"  {'Уровень':<30} {'Скачки':>6} {'Генер.(нс)':>10} {'ISL(нс)':>10} {'Итого(нс)':>10}")
    print(f"  {'':─<65}")
    for lk, b in result["chain_budgets"].items():
        print(f"  {lk:<30} {b['n_hops']:>6}  {b['osc_err_ns']:>9.2f}  {b['isl_ns']:>9.2f}  {b['total_ns']:>9.2f}")

    print(f"\n  Сквозная погрешность 1PPS у пользователя (нс, 1σ):")
    print(f"  {'Уровень спутника':<30}  ", end="")
    for uk in USER_RECEIVERS:
        print(f"  {uk[:12]:<12}", end="")
    print()
    print(f"  {'':─<70}")
    for lk in CHAIN:
        print(f"  {lk:<30}", end="")
        for uk in USER_RECEIVERS:
            ub = result["user_budgets"][uk][lk]
            ok = "OK" if ub["viable_100ns"] else "--"
            print(f"  {ub['total_ns']:>7.1f} нс [{ok}]", end="")
        print()

    print(sep)
