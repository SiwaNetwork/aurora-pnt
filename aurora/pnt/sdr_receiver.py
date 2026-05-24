"""
Программно-определяемый приёмник (SDR) AURORA PNT — захват и слежение.

Реализует полный конвейер обработки сигнала AURORA L1 BOC(1,1):
1) Генерация принимаемого сигнала на промежуточной частоте f_IF = 4 МГц
   при частоте дискретизации 16,368 МГц с заданными C/N₀ и Доплером.
2) Параллельный 2D-захват (Doppler × code-phase) через FFT
   (Sutton et al. 1997 — параллельный поиск через корреляцию по FFT).
3) Замыкание петель: PLL Costas (BW = 1,4 Гц) + DLL EML (BW = 0,5 Гц,
   разнос 0,5 чипа).
4) Метрики времени до первого фикса (TTFF) vs C/N₀.

Ссылки:
  Kaplan E. & Hegarty C. (2017) — Understanding GPS/GNSS: Principles and
    Applications, 3rd ed. Artech House (глл. 5-8).
  Borre K. et al. (2007) — A Software-Defined GPS and Galileo Receiver:
    Single-Frequency Approach. Birkhäuser.
  Sutton E. (1997) — Parallel code-phase search acquisition.
  ESA GNSS Software Receivers (2015) — Acquisition & Tracking algorithms.
"""

import os
import csv
import math
from typing import Dict, Tuple, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Сигнальные параметры AURORA L1 ─────────────────────────────────────────
F_IF       = 4.0e6           # промежуточная частота, Гц
F_SAMPLE   = 16.368e6        # частота дискретизации, Гц (16 × чип. скор.)
F_CHIP_L1  = 1.023e6         # чиповая скорость L1 C/A / BOC(1,1) PRN
N_CHIPS_CA = 1023            # длина дальномерного кода 1 мс (C/A)
T_CODE     = 1.0e-3          # период кода, с

# Бюджет линии AURORA на L1 — из §10
CN0_BUDGET_DBHZ = 52.6       # C/N₀ в зените, дБ-Гц

# Слежение
PLL_BW_HZ = 1.4
DLL_BW_HZ = 0.5
EML_SPACING_CHIPS = 0.5

PALETTE = {
    "orange": "#e17055",
    "yellow": "#fdcb6e",
    "blue":   "#0984e3",
    "green":  "#00b894",
    "purple": "#6c5ce7",
    "lightb": "#74b9ff",
}


# ── Простой C/A-генератор для демонстрации (стандарт GPS L1 C/A) ───────────

def _ca_code(prn: int = 1) -> np.ndarray:
    """GPS L1 C/A код длины 1023 chip, биполярный {-1, +1}."""
    g2_phase = {1: (2, 6), 2: (3, 7), 3: (4, 8), 4: (5, 9), 5: (1, 9)}
    t1, t2 = g2_phase.get(prn, (2, 6))

    def lfsr10(taps):
        reg = np.ones(10, dtype=np.int8)
        out = np.empty(1023, dtype=np.int8)
        for i in range(1023):
            out[i] = reg[9]
            fb = 0
            for t in taps:
                fb ^= reg[t - 1]
            reg = np.concatenate([[fb], reg[:-1]])
        return out

    g1 = lfsr10((3, 10))
    g2 = lfsr10((2, 3, 6, 8, 9, 10))
    g2d = np.bitwise_xor(np.roll(g2, -(t1 - 1)), np.roll(g2, -(t2 - 1)))
    code = np.bitwise_xor(g1, g2d)
    return np.where(code == 1, -1, 1).astype(np.float32)


def _upsample_code(code: np.ndarray, fs: float, fchip: float,
                   n_samples: int, code_phase_chips: float = 0.0) -> np.ndarray:
    """Передискретизация чипового кода до n_samples с произвольной фазой."""
    n_code = len(code)
    t = np.arange(n_samples) / fs
    chip_idx = (t * fchip + code_phase_chips) % n_code
    idx = chip_idx.astype(np.int64)
    return code[idx]


def _boc11_subcarrier(fs: float, n_samples: int) -> np.ndarray:
    """Поднесущая BOC(1,1) — квадратная волна частоты 1,023 МГц."""
    t = np.arange(n_samples) / fs
    return np.sign(np.sin(2 * np.pi * F_CHIP_L1 * t)).astype(np.float32)


# ── Генерация принимаемого сигнала ─────────────────────────────────────────

def _generate_received(duration_s: float, cn0_dbhz: float,
                       doppler_hz: float, code_phase_chips: float,
                       prn: int = 1, fs: float = F_SAMPLE,
                       use_boc: bool = True,
                       rng_seed: int = 12345) -> np.ndarray:
    """Возвращает сигнал I-канала (реальный) на ПЧ с шумом."""
    rng = np.random.default_rng(rng_seed)
    n = int(duration_s * fs)
    t = np.arange(n) / fs

    code = _ca_code(prn)
    s_code = _upsample_code(code, fs, F_CHIP_L1, n, code_phase_chips)

    # Несущая ПЧ + Доплер
    carr = np.cos(2 * np.pi * (F_IF + doppler_hz) * t).astype(np.float32)

    sig = s_code * carr
    if use_boc:
        sig *= _boc11_subcarrier(fs, n)

    # Уровень шума под заданное C/N₀
    # Мощность сигнала = 0,5 (косинус на едине), C = 0,5
    # N₀ = C / (10^(C/N₀/10)) → дисперсия шума = N₀ · BW = N₀ · fs/2
    cn0_lin = 10 ** (cn0_dbhz / 10)
    n0 = 0.5 / cn0_lin
    sigma = math.sqrt(n0 * fs / 2)
    noise = rng.standard_normal(n).astype(np.float32) * sigma
    return (sig + noise).astype(np.float32)


# ── Захват: параллельный 2D-поиск через FFT ────────────────────────────────

def _acquisition(rx: np.ndarray, prn: int = 1, fs: float = F_SAMPLE,
                 doppler_range_hz: float = 50_000.0,
                 doppler_step_hz: float = 500.0) -> Dict:
    """2D-захват: код-фаза × Доплер. Возвращает корр. матрицу и пик."""
    n_samples_per_ms = int(fs * T_CODE)
    rx_1ms = rx[:n_samples_per_ms].astype(np.float64)

    code = _ca_code(prn)
    code_up = _upsample_code(code, fs, F_CHIP_L1, n_samples_per_ms, 0.0)
    sub = _boc11_subcarrier(fs, n_samples_per_ms)
    local_code = (code_up * sub).astype(np.float64)
    F_code = np.fft.fft(local_code).conj()

    dopplers = np.arange(-doppler_range_hz, doppler_range_hz + 1, doppler_step_hz)
    t = np.arange(n_samples_per_ms) / fs

    corr_matrix = np.zeros((len(dopplers), n_samples_per_ms), dtype=np.float32)
    for i, fd in enumerate(dopplers):
        carr_i = np.cos(2 * np.pi * (F_IF + fd) * t)
        carr_q = np.sin(2 * np.pi * (F_IF + fd) * t)
        ri = rx_1ms * carr_i
        rq = rx_1ms * carr_q
        Ri = np.fft.fft(ri)
        Rq = np.fft.fft(rq)
        ci = np.fft.ifft(Ri * F_code).real
        cq = np.fft.ifft(Rq * F_code).real
        corr_matrix[i] = (ci ** 2 + cq ** 2).astype(np.float32)

    idx = np.unravel_index(np.argmax(corr_matrix), corr_matrix.shape)
    peak = float(corr_matrix[idx])
    # peak-to-side ratio (исключим окрестность пика 4 чипа)
    chip_samples = int(fs / F_CHIP_L1)
    mask = np.ones_like(corr_matrix, dtype=bool)
    di, dj = idx
    for ii in range(max(0, di - 1), min(corr_matrix.shape[0], di + 2)):
        j0 = max(0, dj - 2 * chip_samples)
        j1 = min(corr_matrix.shape[1], dj + 2 * chip_samples)
        mask[ii, j0:j1] = False
    side = float(np.max(corr_matrix[mask]))
    psr = peak / side if side > 0 else float("inf")

    return {
        "corr_matrix": corr_matrix,
        "dopplers_hz": dopplers,
        "code_phase_samples": np.arange(n_samples_per_ms),
        "peak_doppler_hz": float(dopplers[idx[0]]),
        "peak_code_phase_samples": int(idx[1]),
        "peak_code_phase_chips": float(idx[1] / fs * F_CHIP_L1),
        "peak": peak,
        "side": side,
        "psr": psr,
        "detected": psr > 2.5,        # типовой порог
    }


# ── Слежение: PLL Costas + DLL EML ─────────────────────────────────────────

def _tracking(rx: np.ndarray, acq: Dict,
              duration_track_s: float = 0.05,
              fs: float = F_SAMPLE,
              prn: int = 1) -> Dict:
    """Замыкание петель PLL/DLL. Возвращает логи измерений."""
    n_samples_per_ms = int(fs * T_CODE)
    n_total = int(duration_track_s * fs)
    rx = rx[:n_total].astype(np.float64)

    code = _ca_code(prn)
    code_up = _upsample_code(code, fs, F_CHIP_L1, n_samples_per_ms, 0.0)
    sub_full = _boc11_subcarrier(fs, n_total)

    # ── Параметры петель (BL=BW; коэффициенты второго порядка) ─────────────
    # Дискретный фильтр PI: K1 = 8·ζ·ω₀·T/(4+...) — упрощ. форма из Kaplan
    # ω₀ = BL / 0.53 для ζ=0.707, BL — шумовая полоса
    def pi_coeffs(bn, zeta=0.707, dt=T_CODE):
        wn = bn / 0.53
        k1 = 2 * zeta * wn * dt
        k2 = wn ** 2 * dt ** 2
        return k1, k2

    k1_pll, k2_pll = pi_coeffs(PLL_BW_HZ)
    k1_dll, k2_dll = pi_coeffs(DLL_BW_HZ)

    # Инициализация из захвата
    f_carr = F_IF + acq["peak_doppler_hz"]
    phi    = 0.0
    code_phase = acq["peak_code_phase_chips"]
    code_freq  = F_CHIP_L1

    n_ms = int(duration_track_s / T_CODE)
    pll_err_log   = np.zeros(n_ms)
    dll_err_log   = np.zeros(n_ms)
    doppler_log   = np.zeros(n_ms)
    code_phase_log = np.zeros(n_ms)
    ip_log        = np.zeros(n_ms)
    qp_log        = np.zeros(n_ms)

    pll_int = 0.0
    dll_int = 0.0

    for k in range(n_ms):
        i0 = k * n_samples_per_ms
        i1 = i0 + n_samples_per_ms
        seg = rx[i0:i1]
        sub_seg = sub_full[i0:i1]

        # Локальные несущие
        t_seg = (np.arange(n_samples_per_ms) + i0) / fs
        carr_i = np.cos(2 * np.pi * f_carr * t_seg + phi)
        carr_q = np.sin(2 * np.pi * f_carr * t_seg + phi)

        # Локальные коды E/P/L (с поднесущей BOC)
        cp_E = _upsample_code(code, fs, code_freq, n_samples_per_ms,
                              code_phase - EML_SPACING_CHIPS / 2)
        cp_P = _upsample_code(code, fs, code_freq, n_samples_per_ms,
                              code_phase)
        cp_L = _upsample_code(code, fs, code_freq, n_samples_per_ms,
                              code_phase + EML_SPACING_CHIPS / 2)

        # I/Q корреляторы (с BOC sub-carrier)
        seg_d_i = seg * carr_i * sub_seg
        seg_d_q = seg * carr_q * sub_seg
        IE = float(np.sum(seg_d_i * cp_E))
        QE = float(np.sum(seg_d_q * cp_E))
        IP = float(np.sum(seg_d_i * cp_P))
        QP = float(np.sum(seg_d_q * cp_P))
        IL = float(np.sum(seg_d_i * cp_L))
        QL = float(np.sum(seg_d_q * cp_L))

        # PLL Costas: atan2(QP/IP)/2π — фазовая ошибка в циклах
        pll_err = math.atan2(QP, IP) / (2 * math.pi) if IP != 0 else 0.0

        # DLL EML: нормированный
        E = math.sqrt(IE * IE + QE * QE)
        L = math.sqrt(IL * IL + QL * QL)
        dll_err = (E - L) / (E + L + 1e-12) * 0.5  # ошибка в чипах

        # PI-фильтры (рекуррентные)
        pll_int += k2_pll * pll_err
        f_carr  += k1_pll * pll_err + pll_int

        dll_int += k2_dll * dll_err
        code_freq_new = F_CHIP_L1 + k1_dll * dll_err + dll_int
        # Обновим фазу кода (учитывая, что за 1 мс отсчётов было n_per_ms)
        code_phase = (code_phase + (code_freq_new - F_CHIP_L1) * T_CODE * N_CHIPS_CA / N_CHIPS_CA) % N_CHIPS_CA
        # NB: чиповая частота меняется слабо, удобнее обновить фазу так:
        code_phase = (code_phase - dll_err) % N_CHIPS_CA
        code_freq = code_freq_new

        # Журналы
        pll_err_log[k]    = pll_err
        dll_err_log[k]    = dll_err
        doppler_log[k]    = f_carr - F_IF
        code_phase_log[k] = code_phase
        ip_log[k]         = IP
        qp_log[k]         = QP

        # Обновление фазы несущей: phi += 2π · f_carr · T_code   (но
        # учтено в t_seg для следующей итерации через i0)

    return {
        "pll_err": pll_err_log,
        "dll_err": dll_err_log,
        "doppler": doppler_log,
        "code_phase": code_phase_log,
        "ip": ip_log,
        "qp": qp_log,
        "t_ms": np.arange(n_ms),
    }


# ── TTFF в зависимости от C/N₀ ─────────────────────────────────────────────

def _ttff_vs_cn0(cn0_dbhz_list: List[float], n_runs: int = 5,
                 rng_seed_base: int = 1000) -> Dict:
    """Оценка медианы и std TTFF для cold start vs C/N₀.

    Модель: TTFF ≈ T_search + T_pull-in.
    T_search для холодного: ~ N_doppler_bins × T_dwell.
    Доплер до ±50 кГц, шаг 500 Гц → 200 ячеек × 1 мс корр × n_inc
    (некогерентное накопление), всего ~ 12-30 с для слабых сигналов.
    """
    results = {"cn0": [], "ttff_median": [], "ttff_std": [], "success_pct": []}

    for cn0 in cn0_dbhz_list:
        ttff_runs = []
        success = 0
        # Полуэмпирическая модель TTFF на основе количества некогерентных
        # накоплений, необходимых для надёжного захвата по C/N₀
        # (Kaplan p. 235, формула для P_d ≥ 0.9)
        # n_inc ~ ceil(20^(40/cn0)) — экспоненциальный рост к слабым сигналам
        for r in range(n_runs):
            rng = np.random.default_rng(rng_seed_base + int(cn0 * 100) + r)
            cn0_lin = 10 ** (cn0 / 10)
            # Требуемое произведение T_int · C/N₀ ≥ 18 дБ-Гц·с для P_d ≥ 0,9,
            # P_fa ≤ 1e-3 (Kaplan & Hegarty Ch. 8, табл. 8.3).
            # T_int_total = 10^(18/10) / cn0_lin
            req_int_s = (10 ** (18 / 10)) / cn0_lin          # сек/ячейку
            req_int_s = max(req_int_s, 1.0e-3)
            # Холодный поиск: ±50 кГц / 500 Гц = 200 Доплер-ячеек × параллельная
            # FFT по код. фазе (1023 × 2). Альманаха нет, эфемерид нет.
            n_dopp_bins = int(100_000 / 500)
            # Время сбора эфемерид с навигационного сообщения L1C: 18 с
            # (TOW + clock + 1-ой страницы). Это нижний предел для cold start.
            t_ephem = 18.0
            t_search = n_dopp_bins * req_int_s
            # Pull-in PLL после захвата
            t_pullin = 2.0 + rng.exponential(1.0)
            t_total  = t_search + t_pullin + t_ephem
            # Вероятность детектирования: монотонно растёт от 0 на 25 дБ-Гц
            p_detect = 1.0 / (1.0 + math.exp(-(cn0 - 30.0) / 2.0))
            if rng.random() < p_detect:
                success += 1
                ttff_runs.append(min(t_total, 300.0))
            else:
                ttff_runs.append(300.0)                      # пропуск
        results["cn0"].append(cn0)
        results["ttff_median"].append(float(np.median(ttff_runs)))
        results["ttff_std"].append(float(np.std(ttff_runs)))
        results["success_pct"].append(100.0 * success / n_runs)

    return results


# ── Главный запуск ─────────────────────────────────────────────────────────

def run_sdr_receiver_analysis(output_dir: str, label: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    # 1) Сигнал 10 мс (захват + слежение)
    duration_s   = 0.010                                    # 10 мс
    doppler_true = 25_000.0
    cp_true      = 412.0                                    # чипы
    prn          = 1

    rx = _generate_received(duration_s, CN0_BUDGET_DBHZ, doppler_true,
                            cp_true, prn=prn, rng_seed=2026)

    # 2) Захват
    acq = _acquisition(rx, prn=prn, doppler_step_hz=500.0)

    # 3) Слежение (50 мс синтетического сигнала)
    rx_track = _generate_received(0.050, CN0_BUDGET_DBHZ, doppler_true,
                                  cp_true, prn=prn, rng_seed=2027)
    trk = _tracking(rx_track, acq, duration_track_s=0.050)

    # 4) TTFF vs C/N₀
    cn0_list = np.arange(30.0, 56.0, 2.5).tolist()           # 11 точек
    ttff = _ttff_vs_cn0(cn0_list, n_runs=5)

    # 5) Метрики
    # медиана TTFF при 52,6 дБ-Гц (интерполяция)
    arr_cn0  = np.array(ttff["cn0"])
    arr_med  = np.array(ttff["ttff_median"])
    ttff_at_budget = float(np.interp(CN0_BUDGET_DBHZ, arr_cn0, arr_med))

    results = {
        "cn0_budget_dbhz":   CN0_BUDGET_DBHZ,
        "doppler_true_hz":   doppler_true,
        "doppler_acq_hz":    acq["peak_doppler_hz"],
        "doppler_acq_err_hz": acq["peak_doppler_hz"] - doppler_true,
        "code_phase_true_chips": cp_true,
        "code_phase_acq_chips":  acq["peak_code_phase_chips"],
        "psr": acq["psr"],
        "detected": acq["detected"],
        "pll_err_std":  float(np.std(trk["pll_err"][10:])),
        "dll_err_std":  float(np.std(trk["dll_err"][10:])),
        "doppler_track_final_hz": float(trk["doppler"][-1]),
        "ttff_vs_cn0":   ttff,
        "ttff_at_budget_s": ttff_at_budget,
    }

    _plot_acquisition_3d(acq, output_dir, label)
    _plot_tracking(trk, output_dir, label)
    _plot_ttff_vs_cn0(ttff, results, output_dir, label)
    _plot_eye_diagram(trk, output_dir, label)
    _save_csv(ttff, results, output_dir, label)

    return results


# ── Визуализация ────────────────────────────────────────────────────────────

def _plot_acquisition_3d(acq: Dict, output_dir: str, label: str) -> None:
    cm = acq["corr_matrix"]
    dopp = acq["dopplers_hz"]
    # ограничим по код. фазе ±200 семплов вокруг пика для читаемости
    peak_j = acq["peak_code_phase_samples"]
    j0 = max(0, peak_j - 300)
    j1 = min(cm.shape[1], peak_j + 300)
    cm_show = cm[:, j0:j1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # heatmap
    im = ax1.imshow(cm_show, aspect="auto",
                    extent=[j0 / F_SAMPLE * F_CHIP_L1,
                            j1 / F_SAMPLE * F_CHIP_L1,
                            dopp[-1] / 1000, dopp[0] / 1000],
                    cmap="viridis")
    ax1.set_xlabel("Фаза кода (чипы)")
    ax1.set_ylabel("Доплер (кГц)")
    ax1.set_title("Корреляц. функция захвата (2D)")
    plt.colorbar(im, ax=ax1, label="|R|²")
    ax1.plot(acq["peak_code_phase_chips"], acq["peak_doppler_hz"] / 1000,
             "r*", ms=15, markeredgecolor="white", label=f"Пик: {acq['psr']:.1f}σ")
    ax1.legend(fontsize=9)

    # срез по код. фазе для пик. Доплера
    peak_i = np.argmin(np.abs(dopp - acq["peak_doppler_hz"]))
    ax2.plot(cm[peak_i], color=PALETTE["green"], lw=1.2)
    ax2.axvline(peak_j, color=PALETTE["orange"], ls="--", lw=1.2,
                label=f"Пик при коде {acq['peak_code_phase_chips']:.1f} ч.")
    ax2.set_xlabel("Образец фазы кода")
    ax2.set_ylabel("|R|²")
    ax2.set_title(f"Срез при Доплере = {acq['peak_doppler_hz']/1000:.2f} кГц")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.suptitle(
        f"Захват AURORA L1 — C/N₀ = {CN0_BUDGET_DBHZ:.1f} дБ-Гц, PSR = {acq['psr']:.2f} [{label}]"
    )
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"sdr_acquisition_3d_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_tracking(trk: Dict, output_dir: str, label: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    ax = axes[0, 0]
    ax.plot(trk["t_ms"], trk["pll_err"], color=PALETTE["blue"], lw=1.2)
    ax.set_xlabel("Время (мс)")
    ax.set_ylabel("Ошибка PLL (цикл)")
    ax.set_title("Дискриминатор PLL (Costas)")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(trk["t_ms"], trk["dll_err"], color=PALETTE["green"], lw=1.2)
    ax.set_xlabel("Время (мс)")
    ax.set_ylabel("Ошибка DLL (чип)")
    ax.set_title("Дискриминатор DLL (EML)")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(trk["t_ms"], trk["doppler"] / 1000, color=PALETTE["orange"], lw=1.4)
    ax.axhline(25.0, color=PALETTE["purple"], ls="--", lw=1.0,
               label="Истинный Доплер = 25 кГц")
    ax.set_xlabel("Время (мс)")
    ax.set_ylabel("Измер. Доплер (кГц)")
    ax.set_title("Измерения Доплера в PLL")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(trk["t_ms"], trk["code_phase"], color=PALETTE["purple"], lw=1.2)
    ax.set_xlabel("Время (мс)")
    ax.set_ylabel("Фаза кода (чип)")
    ax.set_title("Измерения фазы кода (DLL)")
    ax.grid(alpha=0.3)

    fig.suptitle(f"Слежение AURORA L1 — PLL/DLL петли [{label}]")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"sdr_tracking_loops_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_ttff_vs_cn0(ttff: Dict, res: Dict,
                      output_dir: str, label: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    cn0 = ttff["cn0"]
    med = ttff["ttff_median"]
    std = ttff["ttff_std"]
    ax.errorbar(cn0, med, yerr=std, fmt="-o", color=PALETTE["green"],
                lw=2, capsize=4, label="Cold start медиана ± σ")
    ax.axvline(CN0_BUDGET_DBHZ, ls="--", color=PALETTE["blue"], lw=1.3,
               label=f"Бюджет AURORA: {CN0_BUDGET_DBHZ:.1f} дБ-Гц")
    ax.axhline(res["ttff_at_budget_s"], ls=":", color=PALETTE["orange"],
               lw=1.3, label=f"TTFF при бюджете: {res['ttff_at_budget_s']:.1f} с")
    ax.set_xlabel("C/N₀ (дБ-Гц)")
    ax.set_ylabel("TTFF (с)")
    ax.set_yscale("log")
    ax.set_title(f"Время до первого фикса vs C/N₀ — Cold Start [{label}]")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"sdr_ttff_vs_cn0_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_eye_diagram(trk: Dict, output_dir: str, label: str) -> None:
    """Глазковая диаграмма на выходе I-коррелятора (бит на символ)."""
    # фрагменты по 2 мс из IP (стабильные после settling)
    ip = trk["ip"][10:]
    if len(ip) < 4:
        # дополним до минимума, чтобы plot не падал
        ip = np.concatenate([ip, np.zeros(4)])
    n_per_eye = 2
    n_eyes = len(ip) // n_per_eye
    fig, ax = plt.subplots(figsize=(10, 5))
    for k in range(n_eyes):
        seg = ip[k * n_per_eye:(k + 1) * n_per_eye]
        ax.plot(np.arange(n_per_eye), seg, color=PALETTE["blue"],
                alpha=0.35, lw=1.0)
    ax.axhline(0, color="#2d3436", lw=0.6)
    ax.set_xlabel("Время (мс, окно 2 мс)")
    ax.set_ylabel("Значение I-коррелятора (a.u.)")
    ax.set_title(f"Глазковая диаграмма выхода I-коррелятора [{label}]")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"sdr_eye_diagram_{label}.png"), dpi=150)
    plt.close(fig)


def _save_csv(ttff: Dict, res: Dict, output_dir: str, label: str) -> None:
    path = os.path.join(output_dir, f"sdr_receiver_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mode", "cn0_dbhz", "ttff_median_s", "ttff_std_s", "success_pct"])
        for c, m, s, p in zip(ttff["cn0"], ttff["ttff_median"],
                              ttff["ttff_std"], ttff["success_pct"]):
            mode = "cold"
            w.writerow([mode, f"{c:.1f}", f"{m:.2f}", f"{s:.2f}", f"{p:.0f}"])
        w.writerow([])
        w.writerow(["summary", "cn0_budget_dbhz", f"{res['cn0_budget_dbhz']:.1f}",
                    "ttff_at_budget_s", f"{res['ttff_at_budget_s']:.2f}"])
        w.writerow(["acq_psr", f"{res['psr']:.2f}",
                    "doppler_err_hz", f"{res['doppler_acq_err_hz']:.1f}",
                    "tracking_dll_std_chip", f"{res['dll_err_std']:.4f}"])


# ── Итоговый отчёт ─────────────────────────────────────────────────────────

def print_sdr_receiver_summary(label: str, results: Dict) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  SDR Receiver Analysis -- {label}")
    print(sep)
    print(f"  Захват AURORA L1 (C/N0 = {results['cn0_budget_dbhz']:.1f} дБ-Гц):")
    print(f"    Истинный Доплер  : {results['doppler_true_hz']/1000:.2f} кГц")
    print(f"    Захвач. Доплер   : {results['doppler_acq_hz']/1000:.2f} кГц "
          f"(ошибка {results['doppler_acq_err_hz']:+.0f} Гц)")
    print(f"    Истинн. фаза кода: {results['code_phase_true_chips']:.1f} ч.")
    print(f"    Захвач. фаза кода: {results['code_phase_acq_chips']:.1f} ч.")
    print(f"    PSR (peak-to-side): {results['psr']:.2f}  "
          f"({'OK' if results['detected'] else 'НЕ ЗАХВАЧЕН'})")
    print(f"  Слежение (50 мс):")
    print(f"    PLL ошибка sigma : {results['pll_err_std']:.4f} цикл")
    print(f"    DLL ошибка sigma : {results['dll_err_std']:.4f} чип")
    print(f"  TTFF:")
    print(f"    При C/N0 = {results['cn0_budget_dbhz']:.1f} дБ-Гц: "
          f"{results['ttff_at_budget_s']:.1f} с")
    arr_cn0 = results['ttff_vs_cn0']['cn0']
    arr_med = results['ttff_vs_cn0']['ttff_median']
    for c, m in zip(arr_cn0, arr_med):
        print(f"    C/N0 {c:.1f} дБ-Гц -> TTFF {m:.1f} с")
    print(sep)
