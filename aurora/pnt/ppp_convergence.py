"""
PPP (Precise Point Positioning) алгоритм и время сходимости АВРОРА.

Реализует упрощённую Kalman-фильтрацию для PPP и моделирует время
сходимости в зависимости от числа наблюдаемых спутников.

Рассчитывает:
  - Время сходимости PPP при разном числе видимых спутников LEO/MEO
  - Kalman-диаграмму ковариации (горизонтальная/вертикальная)
  - Сравнение LEO-PPP (АВРОРА) vs стандартный GPS-PPP
  - Dual-freq vs Single-freq режимы сходимости
  - Зависимость от угла маски приёма (elevation cutoff)

Ссылки:
  Zumberge et al. (1997) — Precise point positioning for the efficient and
    robust analysis of GPS data from large networks.
  Zhang et al. (2020) — LEO constellation-augmented multi-GNSS PPP.
  RTCM SSR (2018) — Standard for State Space Representation messages.
"""

import math, os, csv
from typing import Dict, List, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Параметры орбит ───────────────────────────────────────────────────────────
ALT_LEO_KM  = 1000.0   # АВРОРА
ALT_MEO_KM  = 20200.0  # GPS
R_E_KM      = 6371.0

# ── Шумы измерений (м) ───────────────────────────────────────────────────────
NOISE_CODE_M      = 0.30   # код L1/L5
NOISE_PHASE_M     = 0.003  # фаза
NOISE_IONO_M      = 0.10   # остаточная ионосфера (dual-freq)
NOISE_TROPO_M     = 0.05   # остаточная тропосфера

# ── Начальные неопределённости состояния ─────────────────────────────────────
SIGMA0_POS_M   = 5.0    # начальная неопределённость позиции
SIGMA0_CLK_M   = 1000.0 # начальная неопределённость часов
SIGMA0_ZWD_M   = 0.10   # начальная неопределённость ZWD
SIGMA0_AMB_CYC = 100.0  # начальная неопределённость несущей

# ── Шум процесса ─────────────────────────────────────────────────────────────
Q_POS_M_S   = 0.001   # случайное блуждание позиции (м/с^0.5, статичный приёмник)
Q_CLK_M_S   = 10.0    # нестабильность часов
Q_ZWD_M_S   = 0.0001  # медленное изменение ZWD

# ── Угловая скорость спутников ────────────────────────────────────────────────
def orbital_period_s(alt_km: float) -> float:
    mu = 3.986004418e14
    r = (R_E_KM + alt_km) * 1e3
    return 2 * math.pi * math.sqrt(r**3 / mu)


def angular_rate_deg_s(alt_km: float) -> float:
    T = orbital_period_s(alt_km)
    return 360.0 / T  # градусов/сек видимости


def visibility_duration_s(alt_km: float, elev_cutoff_deg: float = 5.0) -> float:
    """Время прохождения одного спутника через видимое окно (над горизонтом)."""
    r_sat = (R_E_KM + alt_km) * 1e3
    r_e   = R_E_KM * 1e3
    half_angle = math.acos(r_e / r_sat * math.cos(math.radians(elev_cutoff_deg))) \
                 - math.radians(elev_cutoff_deg)
    fraction = 2 * half_angle / (2 * math.pi)
    return orbital_period_s(alt_km) * fraction


def n_visible(alt_km: float, n_total: int, elev_cutoff_deg: float = 5.0) -> float:
    """Среднее число одновременно видимых спутников."""
    vis_dur = visibility_duration_s(alt_km, elev_cutoff_deg)
    T = orbital_period_s(alt_km)
    return n_total * (vis_dur / T)


# ── Упрощённый Kalman PPP ─────────────────────────────────────────────────────
def simulate_ppp_convergence(n_sats: int, dt_s: float = 30.0,
                             dual_freq: bool = True,
                             max_epochs: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Симулирует сходимость PPP через последовательные Kalman-шаги.
    Возвращает (times_min, sigma_h_m, sigma_v_m).

    Состояние: [x, y, z, clk, zwD] + n_sats несущих
    """
    n_state = 4 + n_sats  # позиция (3) + часы + ZWD + несущие

    # Начальная ковариация
    P = np.zeros(n_state)
    P[0:3] = SIGMA0_POS_M ** 2
    P[3]   = SIGMA0_CLK_M ** 2
    if n_state > 4:
        P[4]   = SIGMA0_ZWD_M ** 2
    for i in range(5, n_state):
        P[i] = SIGMA0_AMB_CYC ** 2

    # Измерительный шум
    meas_noise = NOISE_CODE_M ** 2
    if dual_freq:
        meas_noise = (NOISE_CODE_M ** 2 + NOISE_IONO_M ** 2)
    meas_noise += NOISE_PHASE_M ** 2 + NOISE_TROPO_M ** 2

    # Шум процесса
    Q = np.zeros(n_state)
    Q[0:3] = (Q_POS_M_S * dt_s) ** 2
    Q[3]   = (Q_CLK_M_S * dt_s) ** 2
    if n_state > 4:
        Q[4] = (Q_ZWD_M_S * dt_s) ** 2

    sigma_h_list = []
    sigma_v_list = []
    times_list   = []

    for epoch in range(max_epochs):
        t_min = epoch * dt_s / 60.0

        # Шаг прогноза: P_{k+1} = P_k + Q
        P += Q

        # Шаг обновления: каждый спутник даёт 2 наблюдения (код + фаза)
        for sat in range(n_sats):
            # Псевдодальность
            H_pos = 1.0 / math.sqrt(3)   # усреднённый геометрический вес
            H_code = H_pos ** 2 * (P[0] + P[1] + P[2]) + P[3]
            K_code = H_code / (H_code + meas_noise)
            P[0:3] = P[0:3] * (1 - K_code * H_pos ** 2)
            P[3]   = P[3]   * (1 - K_code)

            # Несущая (с разрешением неоднозначности)
            if n_state > 5 + sat:
                H_phase = H_pos ** 2 * (P[0] + P[1] + P[2]) + P[3] + P[5 + sat]
                K_phase = H_phase / (H_phase + NOISE_PHASE_M ** 2)
                P[0:3] = P[0:3] * (1 - K_phase * H_pos ** 2)
                P[5 + sat] = P[5 + sat] * (1 - K_phase)

        sigma_h = math.sqrt((P[0] + P[1]) / 2.0)   # горизонт. (RMS)
        sigma_v = math.sqrt(P[2])                    # вертикаль.

        sigma_h_list.append(sigma_h)
        sigma_v_list.append(sigma_v)
        times_list.append(t_min)

    return (np.array(times_list),
            np.array(sigma_h_list),
            np.array(sigma_v_list))


def convergence_time_min(sigma_h: np.ndarray, times: np.ndarray,
                         threshold_m: float = 0.10) -> float:
    """Время сходимости: первый эпоха, когда sigma_h < threshold."""
    idx = np.where(sigma_h < threshold_m)[0]
    if len(idx) == 0:
        return times[-1]
    return times[idx[0]]


def run_ppp_convergence_analysis(output_dir: str, label: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    dt_s = 30.0  # 30-сек эпохи

    # Сценарии числа видимых спутников
    n_sats_leo_typical = int(n_visible(ALT_LEO_KM, 300))   # АВРОРА полное
    n_sats_meo_typical = int(n_visible(ALT_MEO_KM, 24))    # GPS
    n_sats_leo_ioc = int(n_visible(ALT_LEO_KM, 60))        # IOC фаза (60 спутников)

    scenarios = {
        f"АВРОРА FOC ({n_sats_leo_typical} сп., dual)":
            simulate_ppp_convergence(min(n_sats_leo_typical, 20), dt_s, dual_freq=True),
        f"АВРОРА IOC ({n_sats_leo_ioc} сп., dual)":
            simulate_ppp_convergence(min(n_sats_leo_ioc, 8), dt_s, dual_freq=True),
        f"GPS only ({n_sats_meo_typical} сп., dual)":
            simulate_ppp_convergence(min(n_sats_meo_typical, 8), dt_s, dual_freq=True),
        f"АВРОРА FOC (single-freq)":
            simulate_ppp_convergence(min(n_sats_leo_typical, 20), dt_s, dual_freq=False),
    }
    colors = ["#00b894", "#fdcb6e", "#e17055", "#74b9ff"]

    conv_times = {name: convergence_time_min(sc[1], sc[0]) for name, sc in scenarios.items()}

    _plot_convergence_comparison(scenarios, colors, output_dir, label)
    _plot_convergence_bar(conv_times, output_dir, label)
    _plot_kalman_covariance(scenarios, colors, output_dir, label)
    _plot_leo_vs_meo_geometry(output_dir, label)
    _save_csv(scenarios, conv_times, output_dir, label)

    return {
        "scenarios":    {k: {"conv_min": v} for k, v in conv_times.items()},
        "n_vis_leo":    n_sats_leo_typical,
        "n_vis_meo":    n_sats_meo_typical,
        "vis_dur_leo_s": visibility_duration_s(ALT_LEO_KM),
        "vis_dur_meo_s": visibility_duration_s(ALT_MEO_KM),
    }


def _plot_convergence_comparison(scenarios, colors, output_dir, label):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for (name, (times, sh, sv)), color in zip(scenarios.items(), colors):
        ax1.plot(times, sh * 100, color=color, lw=2, label=name)
        ax2.plot(times, sv * 100, color=color, lw=2, label=name)

    for ax, title, th in [(ax1, "Горизонтальная точность", 10.0),
                           (ax2, "Вертикальная точность", 15.0)]:
        ax.axhline(th, ls="--", color="#2d3436", lw=1.2, label=f"{th} см порог")
        ax.set_xlabel("Время (мин)")
        ax.set_ylabel("Среднеквадр. ошибка (см)")
        ax.set_title(f"АВРОРА — PPP {title} [{label}]")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, min(ax.get_ylim()[1], 300))

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"ppp_convergence_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_convergence_bar(conv_times, output_dir, label):
    fig, ax = plt.subplots(figsize=(9, 5))
    names  = list(conv_times.keys())
    times  = list(conv_times.values())
    colors = ["#00b894", "#fdcb6e", "#e17055", "#74b9ff"]
    bars   = ax.barh(names, times, color=colors[:len(names)], edgecolor="white", height=0.5)
    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f"{t:.1f} мин", va="center", fontsize=10, fontweight="bold")
    ax.axvline(5.0,  ls="--", color="#00b894", lw=1.5, label="5 мин (цель LEO)")
    ax.axvline(20.0, ls=":",  color="#e17055", lw=1.5, label="20 мин (типовой GPS)")
    ax.set_xlabel("Время сходимости PPP (мин, σ_H < 10 см)")
    ax.set_title(f"АВРОРА — Время сходимости PPP [{label}]")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"ppp_conv_bar_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_kalman_covariance(scenarios, colors, output_dir, label):
    """Нормированная диагональ ковариационной матрицы."""
    fig, ax = plt.subplots(figsize=(10, 6))
    first_name = list(scenarios.keys())[0]
    times, sh, sv = scenarios[first_name]
    pos_var  = sh ** 2
    vert_var = sv ** 2
    ax.semilogy(times, pos_var,  color="#0984e3", lw=2, label="Горизонт. дисперсия (м²)")
    ax.semilogy(times, vert_var, color="#6c5ce7", lw=2, label="Вертикал. дисперсия (м²)")
    ax.axhline(0.01, ls="--", color="#00b894", lw=1.2, label="0.01 м² (σ=10 см)")
    ax.set_xlabel("Время (мин)")
    ax.set_ylabel("Дисперсия позиции (м²)")
    ax.set_title(f"АВРОРА — Kalman PPP: эволюция ковариации [{label}]  {first_name}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"ppp_kalman_cov_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_leo_vs_meo_geometry(output_dir, label):
    """Число видимых спутников и время прохода для разных орбит."""
    alts = np.linspace(500, 25000, 200)
    n_leo_30 = [n_visible(a, 300) for a in alts]
    n_meo_24 = [n_visible(a, 24)  for a in alts]
    vis_dur  = [visibility_duration_s(a) / 60 for a in alts]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(alts / 1000, n_leo_30, color="#00b894", lw=2, label="300 спутников (АВРОРА)")
    ax1.plot(alts / 1000, n_meo_24, color="#e17055", lw=2, label="24 спутника (GPS)")
    ax1.axvline(ALT_LEO_KM / 1000, ls="--", color="#0984e3", lw=1.2, label="АВРОРА 1000 км")
    ax1.axvline(ALT_MEO_KM / 1000, ls="--", color="#fdcb6e", lw=1.2, label="GPS 20200 км")
    ax1.set_xlabel("Высота орбиты (тыс. км)")
    ax1.set_ylabel("Ср. число видимых спутников")
    ax1.set_title(f"Геометрия созвездий [{label}]")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(alts / 1000, vis_dur, color="#6c5ce7", lw=2)
    ax2.axvline(ALT_LEO_KM / 1000, ls="--", color="#0984e3", lw=1.2, label="АВРОРА 1000 км")
    ax2.axvline(ALT_MEO_KM / 1000, ls="--", color="#fdcb6e", lw=1.2, label="GPS 20200 км")
    ax2.set_xlabel("Высота орбиты (тыс. км)")
    ax2.set_ylabel("Время видимости одного спутника (мин)")
    ax2.set_title(f"Время прохода спутника через зенит [{label}]")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"ppp_leo_vs_meo_{label}.png"), dpi=150)
    plt.close(fig)


def _save_csv(scenarios, conv_times, output_dir, label):
    path = os.path.join(output_dir, f"ppp_convergence_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "conv_time_min", "vis_dur_leo_min", "vis_dur_meo_min"])
        vis_leo = visibility_duration_s(ALT_LEO_KM) / 60
        vis_meo = visibility_duration_s(ALT_MEO_KM) / 60
        for name, t in conv_times.items():
            w.writerow([name, f"{t:.2f}", f"{vis_leo:.1f}", f"{vis_meo:.1f}"])


def print_ppp_summary(label: str, result: Dict) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  PPP Convergence Analysis -- {label}")
    print(sep)
    print(f"  Видимые спутники LEO 1000 км (300 сп.): {result['n_vis_leo']:.0f}")
    print(f"  Видимые спутники MEO 20200 км (24 сп.): {result['n_vis_meo']:.0f}")
    print(f"  Время прохода LEO: {result['vis_dur_leo_s']/60:.1f} мин")
    print(f"  Время прохода MEO: {result['vis_dur_meo_s']/60:.1f} мин")
    print()
    for name, sc in result["scenarios"].items():
        print(f"  {name:<45} {sc['conv_min']:>6.1f} мин")
    print(sep)
