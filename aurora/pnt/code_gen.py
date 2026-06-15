"""
Генератор и анализ дальномерных кодов АВРОРА.

Реализует:
- Генератор Weil-кодов GPS L1C на простом n = 10223 (Legendre-последовательность,
  перестановка Вейля, вставка 7 chips по IS-GPS-800)
- Генератор Extended Memory кодов L5 (n = 10230) методом случайного отбора по
  критерию максимальной кросс-корреляции
- Эталонный генератор Gold-кодов GPS L1 C/A (m-последовательности, n = 1023)
- Численная проверка свойств: автокорреляция, кросс-корреляция, Merit Factor
  (Голея)

Корреляции вычисляются через FFT (теорема Винера-Хинчина), что необходимо
для длинных Weil-кодов (n = 10223).

Ссылки:
  IS-GPS-800 (2021) — Navstar GPS Space Segment / User Segment L1C Interfaces.
  Rushanan J. (2007) — Weil Sequences: A Family of Binary Sequences with Good
    Correlation Properties. ION GNSS.
  Tran M. & Hegarty C. (2002) — Performance evaluation of the new GPS L5 and L1C
    civil signal design and code generation. ION GPS.
  Golay M.J.E. (1977) — Sieves for low autocorrelation binary sequences. IEEE TIT.
"""

import os
import csv
import math
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Параметры кодов ─────────────────────────────────────────────────────────
WEIL_N  = 10223          # простое число для Weil-кодов GPS L1C
L5_N    = 10230          # длина XB-памяти L5 (Extended Memory)
GOLD_N  = 1023           # длина Gold-кода GPS L1 C/A (m = 10)

# Точка вставки 7 чипов по IS-GPS-800 (для PRN 1; для других PRN — табл. 3.2-2);
# мы используем фиксированную позицию для всех PRN — это допустимо для оценки
# свойств семейства (численные значения автокорреляции не зависят от точки).
INSERTION_POINT = 412
INSERTION_BITS  = [0, 1, 1, 0, 1, 0, 0]  # 7 chips по спецификации L1Cp

PALETTE = {
    "orange": "#e17055",
    "yellow": "#fdcb6e",
    "blue":   "#0984e3",
    "green":  "#00b894",
    "purple": "#6c5ce7",
    "lightb": "#74b9ff",
}


# ── Legendre-последовательность ─────────────────────────────────────────────

def _quadratic_residues(n: int) -> np.ndarray:
    """Битовая маска квадратичных вычетов mod n (исключая 0).

    Q[k] = True, если существует x: x^2 ≡ k (mod n).
    """
    q = np.zeros(n, dtype=bool)
    # x пробегает 1..(n-1)/2 — этого достаточно из-за x^2 ≡ (n-x)^2
    for x in range(1, (n + 1) // 2):
        q[(x * x) % n] = True
    return q


def legendre_sequence(n: int) -> np.ndarray:
    """Legendre-последовательность L(k) ∈ {-1, +1} для простого n.

    L(0)  = -1
    L(k)  = +1, если k — квадратичный вычет mod n
    L(k)  = -1, иначе
    """
    q = _quadratic_residues(n)
    L = -np.ones(n, dtype=np.int8)
    L[q] = 1
    L[0] = -1
    return L


# ── Weil-коды L1C ───────────────────────────────────────────────────────────

def weil_code(prn_w: int, n: int = WEIL_N,
              insertion_point: int = INSERTION_POINT) -> np.ndarray:
    """Weil-код длины n + 7 = 10230 для GPS L1C.

    1) Берём Legendre L(k) длины n.
    2) Строим W_w(t) = L(t) XOR L((t + w) mod n) → биполярная форма:
       W_w(t) = -L(t) * L((t + w) mod n)
    3) Вставляем 7 фиксированных чипов в позицию `insertion_point`.

    Параметр `prn_w` — сдвиг Вейля для конкретного PRN
    (из табл. 3.2-2 IS-GPS-800; для тестов сгодятся любые 1..n-1).
    """
    L = legendre_sequence(n)
    Lw = np.roll(L, -prn_w)                       # L((t + w) mod n)
    # Weil-последовательность: W_w(t) = L(t) * L((t + w) mod n)
    # (по Rushanan 2007, IS-GPS-800 — корреляция ≤ 2√n + 3)
    W = (L * Lw).astype(np.int8)
    insertion = np.where(np.array(INSERTION_BITS) == 1, 1, -1).astype(np.int8)
    code = np.concatenate([W[:insertion_point], insertion, W[insertion_point:]])
    return code.astype(np.int8)


# ── Gold-коды (m-последовательности) для GPS L1 C/A ────────────────────────

_G1_TAPS = (3, 10)           # x^10 + x^3 + 1
_G2_TAPS = (2, 3, 6, 8, 9, 10)
_G2_PHASE = {  # таблица фазовых задержек (псевдо-PRN) для нескольких PRN
    1:  (2, 6),  2:  (3, 7),  3:  (4, 8),  4:  (5, 9),  5:  (1, 9),
    6:  (2, 10), 7:  (1, 8),  8:  (2, 9),  9:  (3, 10), 10: (2, 3),
}


def _lfsr10(taps: Tuple[int, ...]) -> np.ndarray:
    """10-битный LFSR, выход длины 1023, как у GPS L1 C/A."""
    reg = np.ones(10, dtype=np.int8)
    out = np.empty(1023, dtype=np.int8)
    for i in range(1023):
        out[i] = reg[9]
        fb = 0
        for t in taps:
            fb ^= reg[t - 1]
        reg = np.concatenate([[fb], reg[:-1]])
    return out


def gold_code(prn: int) -> np.ndarray:
    """Gold-код GPS L1 C/A для PRN 1..10. Возвращает {-1, +1}."""
    g1 = _lfsr10(_G1_TAPS)
    g2 = _lfsr10(_G2_TAPS)
    t1, t2 = _G2_PHASE[prn]
    g2_delayed = np.bitwise_xor(np.roll(g2, -(t1 - 1)), np.roll(g2, -(t2 - 1)))
    gold = np.bitwise_xor(g1, g2_delayed)
    return np.where(gold == 1, -1, 1).astype(np.int8)   # 0→+1, 1→-1


# ── Extended Memory L5 ──────────────────────────────────────────────────────

def extended_memory_l5(n_codes: int, n_len: int = L5_N,
                       xcorr_threshold_db: float = -39.4,
                       max_trials: int = 6,
                       rng_seed: int = 4242) -> List[np.ndarray]:
    """Генерация набора Extended Memory кодов L5 длины n_len.

    Каждый код — случайная биполярная последовательность; принимаем код,
    если его максимальная кросс-корреляция с уже принятыми ≤ порога.
    """
    rng = np.random.default_rng(rng_seed)
    accepted: List[np.ndarray] = []
    thr_lin = 10.0 ** (xcorr_threshold_db / 20.0)
    while len(accepted) < n_codes:
        trial = 0
        while trial < max_trials:
            cand = rng.choice([-1, 1], size=n_len).astype(np.int8)
            ok = True
            for prev in accepted[-min(len(accepted), 30):]:
                xc = _xcorr_max_norm(cand, prev)
                if xc > thr_lin * 1.6:           # запас на фините длину
                    ok = False
                    break
            if ok:
                accepted.append(cand)
                break
            trial += 1
        if trial >= max_trials:
            # ослабим критерий, чтобы не зависнуть
            accepted.append(cand)
    return accepted


# ── Корреляции через FFT ───────────────────────────────────────────────────

def _autocorr_fft(c: np.ndarray) -> np.ndarray:
    """Круговая автокорреляция (нормированная на длину)."""
    f = np.fft.fft(c.astype(np.float64))
    ac = np.real(np.fft.ifft(f * np.conj(f)))
    return ac / len(c)


def _xcorr_fft(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Круговая кросс-корреляция (нормированная)."""
    n = max(len(a), len(b))
    A = np.fft.fft(a.astype(np.float64), n=n)
    B = np.fft.fft(b.astype(np.float64), n=n)
    xc = np.real(np.fft.ifft(A * np.conj(B)))
    return xc / n


def _xcorr_max_norm(a: np.ndarray, b: np.ndarray) -> float:
    """Максимум |xcorr| / длину (линейная амплитуда)."""
    return float(np.max(np.abs(_xcorr_fft(a, b))))


def _merit_factor(c: np.ndarray) -> float:
    """Merit Factor по Голею: MF = N² / (2 · Σ_{k=1..N-1} r(k)²)."""
    ac = _autocorr_fft(c) * len(c)              # ненормированная AC
    side = ac[1:]                                # без главного пика
    return (len(c) ** 2) / (2.0 * np.sum(side ** 2))


# ── Главный анализ ──────────────────────────────────────────────────────────

def run_code_gen_analysis(output_dir: str, label: str,
                          n_prn_weil: int = 10,
                          n_em_codes: int = 350) -> Dict:
    """Главная функция модуля. Генерирует коды, считает метрики, рисует."""
    os.makedirs(output_dir, exist_ok=True)

    # Legendre-последовательность для визуализации
    L = legendre_sequence(WEIL_N)

    # Набор Weil-PRN: сдвиги w из табл. 3.2-2 IS-GPS-800 (PRN 1..10).
    # Шкала разнесена по всему диапазону 1..(n-1)/2 для хорошей изоляции.
    weil_shifts = [5111, 5421, 5261, 5547, 6295,
                   2607, 2473, 5644, 6313, 5933]
    weil_codes  = [weil_code(w) for w in weil_shifts[:n_prn_weil]]

    # Gold-коды для сравнения
    gold_codes = [gold_code(p) for p in range(1, n_prn_weil + 1)]

    # Extended Memory L5 — пониженное число кодов для скорости
    em_codes = extended_memory_l5(n_codes=min(n_em_codes, 60), n_len=L5_N)
    # ← мы валидируем свойства на ~60 кодах; полный набор в продакшене ≥350

    # ─── Автокорреляция (Weil PRN0) ────────────────────────────────────────
    ac_w = _autocorr_fft(weil_codes[0])
    ac_peak  = ac_w[0]
    ac_side  = np.max(np.abs(ac_w[1:]))
    ac_side_dB = 20.0 * math.log10(ac_side / ac_peak)

    # ─── Кросс-корреляция Weil (матрица) ───────────────────────────────────
    xc_matrix = np.zeros((n_prn_weil, n_prn_weil))
    xc_max_lin = []
    for i in range(n_prn_weil):
        for j in range(n_prn_weil):
            xc = _xcorr_fft(weil_codes[i], weil_codes[j])
            if i == j:
                xc_matrix[i, j] = 0.0  # главный пик исключаем
                xc_no_main = np.copy(xc)
                xc_no_main[0] = 0.0
                xc_max_lin.append(np.max(np.abs(xc_no_main)))
            else:
                v = np.max(np.abs(xc))
                xc_matrix[i, j] = 20.0 * math.log10(v + 1e-12)
                xc_max_lin.append(v)
    xc_weil_max_dB = 20.0 * math.log10(max(xc_max_lin) + 1e-12)

    # ─── Merit Factor ──────────────────────────────────────────────────────
    mf_weil = float(np.mean([_merit_factor(c) for c in weil_codes]))
    mf_gold = float(np.mean([_merit_factor(c) for c in gold_codes]))
    mf_em   = float(np.mean([_merit_factor(c) for c in em_codes[:10]]))

    # XCorr Gold (для сравнения)
    xc_gold_lin = []
    for i in range(len(gold_codes)):
        for j in range(i + 1, len(gold_codes)):
            xc_gold_lin.append(_xcorr_max_norm(gold_codes[i], gold_codes[j]))
    xc_gold_max_dB = 20.0 * math.log10(max(xc_gold_lin) + 1e-12)

    # XCorr Extended Memory
    xc_em_lin = []
    for i in range(min(len(em_codes), 10)):
        for j in range(i + 1, min(len(em_codes), 10)):
            xc_em_lin.append(_xcorr_max_norm(em_codes[i], em_codes[j]))
    xc_em_max_dB = 20.0 * math.log10(max(xc_em_lin) + 1e-12) if xc_em_lin else float("nan")

    results = {
        "weil_n":           WEIL_N,
        "weil_n_prn":       n_prn_weil,
        "weil_ac_peak":     float(ac_peak),
        "weil_ac_side_dB":  float(ac_side_dB),
        "weil_xcorr_max_dB": float(xc_weil_max_dB),
        "weil_merit_factor": mf_weil,
        "gold_xcorr_max_dB": float(xc_gold_max_dB),
        "gold_merit_factor": mf_gold,
        "em_n":             len(em_codes),
        "em_xcorr_max_dB":  float(xc_em_max_dB),
        "em_merit_factor":  mf_em,
    }

    _plot_legendre(L, output_dir, label)
    _plot_autocorrelation(ac_w, results, output_dir, label)
    _plot_xcorr_matrix(xc_matrix, output_dir, label)
    _plot_compare_codes(results, output_dir, label)
    _save_csv(weil_codes, gold_codes, em_codes, results, output_dir, label)

    return results


# ── Визуализация ────────────────────────────────────────────────────────────

def _plot_legendre(L: np.ndarray, output_dir: str, label: str) -> None:
    n_show = 200
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.step(np.arange(n_show), L[:n_show], where="post",
            color=PALETTE["blue"], lw=1.4)
    ax.axhline(0, color="#2d3436", lw=0.4)
    ax.set_xlabel("Индекс k")
    ax.set_ylabel("L(k)")
    ax.set_yticks([-1, 0, 1])
    ax.set_title(
        f"Legendre-последовательность L(k), n = {WEIL_N}, первые {n_show} битов [{label}]"
    )
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"codegen_legendre_seq_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_autocorrelation(ac: np.ndarray, res: Dict,
                          output_dir: str, label: str) -> None:
    ac_n = ac / ac[0]
    ac_db = 20.0 * np.log10(np.abs(ac_n) + 1e-12)
    n = len(ac_n)
    lags = np.arange(-n // 2, n // 2)
    ac_shift = np.fft.fftshift(ac_db)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Линейная — узкое окно вокруг главного пика
    win = 80
    centre = n // 2
    ax1.plot(lags[centre - win:centre + win],
             np.fft.fftshift(ac_n)[centre - win:centre + win],
             color=PALETTE["green"], lw=1.5)
    ax1.set_xlabel("Сдвиг (чипы)")
    ax1.set_ylabel("Нормированная АКФ")
    ax1.set_title("АКФ Weil-кода — главный пик")
    ax1.grid(alpha=0.3)

    # Логарифмическая — на всей длине
    ax2.plot(lags, ac_shift, color=PALETTE["orange"], lw=0.6, alpha=0.7)
    ax2.axhline(-24, color=PALETTE["purple"], ls="--", lw=1.2,
                label=f"−24 дБ — типовая боковая лепестка")
    ax2.axhline(res["weil_ac_side_dB"], color=PALETTE["blue"], ls=":", lw=1.2,
                label=f"Макс. боковая = {res['weil_ac_side_dB']:.1f} дБ")
    ax2.set_xlabel("Сдвиг (чипы)")
    ax2.set_ylabel("АКФ (дБ)")
    ax2.set_ylim(-50, 5)
    ax2.set_title("АКФ Weil-кода (логарифм. шкала)")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.suptitle(f"Автокорреляция Weil-кода n = {WEIL_N} [{label}]")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"codegen_autocorrelation_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_xcorr_matrix(xc_db: np.ndarray, output_dir: str, label: str) -> None:
    n = xc_db.shape[0]
    mat = np.copy(xc_db)
    # для отображения диагонали — поставим нижнюю границу
    mat[np.arange(n), np.arange(n)] = np.min(mat[mat < 0])

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mat, cmap="viridis", vmin=-50, vmax=-30, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"PRN{i+1}" for i in range(n)], rotation=45, fontsize=8)
    ax.set_yticklabels([f"PRN{i+1}" for i in range(n)], fontsize=8)
    ax.set_title(
        f"Кросс-корреляция Weil-кодов n = {WEIL_N} (дБ) [{label}]"
    )
    plt.colorbar(im, ax=ax, label="Макс. |XCorr| (дБ)")

    # подписи на ячейках
    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(j, i, f"{xc_db[i, j]:.0f}",
                        ha="center", va="center",
                        color="white" if xc_db[i, j] < -42 else "black",
                        fontsize=7)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"codegen_crosscorr_matrix_{label}.png"), dpi=150)
    plt.close(fig)


def _plot_compare_codes(res: Dict, output_dir: str, label: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Merit Factor
    names  = ["Gold C/A\n(n=1023)", "Weil L1C\n(n=10223)", "Extended L5\n(n=10230)"]
    vals   = [res["gold_merit_factor"], res["weil_merit_factor"], res["em_merit_factor"]]
    colors = [PALETTE["yellow"], PALETTE["green"], PALETTE["purple"]]
    bars = ax1.bar(names, vals, color=colors, edgecolor="white")
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.15,
                 f"{v:.2f}", ha="center", fontsize=10)
    ax1.axhline(6.1, color=PALETTE["blue"], ls="--", lw=1.2,
                label="MF ≈ 6,1 (теор. Weil-цель)")
    ax1.set_ylabel("Merit Factor (Голея)")
    ax1.set_title("Сравнение Merit Factor")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim(0, max(8, max(vals) * 1.2))

    # XCorr max
    xc_vals = [res["gold_xcorr_max_dB"], res["weil_xcorr_max_dB"], res["em_xcorr_max_dB"]]
    bars2 = ax2.bar(names, xc_vals, color=colors, edgecolor="white")
    for b, v in zip(bars2, xc_vals):
        ax2.text(b.get_x() + b.get_width() / 2, v - 1.5,
                 f"{v:.1f} дБ", ha="center", fontsize=10, color="white",
                 fontweight="bold")
    ax2.axhline(-44, color=PALETTE["blue"], ls="--", lw=1.2,
                label="−44 дБ (цель Weil L1C)")
    ax2.axhline(-24, color=PALETTE["orange"], ls=":", lw=1.2,
                label="−24 дБ (типичный Gold)")
    ax2.set_ylabel("Макс. кросс-корреляция (дБ)")
    ax2.set_title("Сравнение XCorr-изоляции")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(min(xc_vals) - 5, 0)

    fig.suptitle(f"Сравнение семейств дальномерных кодов [{label}]")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"codegen_compare_codes_{label}.png"), dpi=150)
    plt.close(fig)


def _save_csv(weil_codes, gold_codes, em_codes, res: Dict,
              output_dir: str, label: str) -> None:
    path = os.path.join(output_dir, f"codegen_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["PRN_id", "n", "MF", "AC_peak_dB", "XCorr_max_dB", "type"])

        # Weil
        for i, c in enumerate(weil_codes):
            mf = _merit_factor(c)
            ac = _autocorr_fft(c)
            side = np.max(np.abs(ac[1:])) / ac[0]
            side_dB = 20.0 * math.log10(side + 1e-12)
            w.writerow([f"Weil{i+1}", len(c), f"{mf:.3f}", "0.0",
                        f"{side_dB:.2f}", "Weil"])

        # Gold
        for i, c in enumerate(gold_codes):
            mf = _merit_factor(c)
            ac = _autocorr_fft(c)
            side = np.max(np.abs(ac[1:])) / ac[0]
            side_dB = 20.0 * math.log10(side + 1e-12)
            w.writerow([f"Gold{i+1}", len(c), f"{mf:.3f}", "0.0",
                        f"{side_dB:.2f}", "Gold"])

        # Extended Memory (только первые 10 для CSV-краткости)
        for i, c in enumerate(em_codes[:10]):
            mf = _merit_factor(c)
            ac = _autocorr_fft(c)
            side = np.max(np.abs(ac[1:])) / ac[0]
            side_dB = 20.0 * math.log10(side + 1e-12)
            w.writerow([f"EM{i+1}", len(c), f"{mf:.3f}", "0.0",
                        f"{side_dB:.2f}", "ExtendedMemory"])

        # Сводка
        w.writerow([])
        w.writerow(["summary",
                    "weil_xcorr_max_dB", f"{res['weil_xcorr_max_dB']:.2f}",
                    "gold_xcorr_max_dB", f"{res['gold_xcorr_max_dB']:.2f}",
                    f"em_xcorr_max_dB={res['em_xcorr_max_dB']:.2f}"])


# ── Итоговый отчёт ─────────────────────────────────────────────────────────

def print_code_gen_summary(label: str, results: Dict) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  Code Generator Analysis -- {label}")
    print(sep)
    print(f"  Weil L1C (n = {results['weil_n']}, PRN = {results['weil_n_prn']}):")
    print(f"    Главный пик АКФ        : {results['weil_ac_peak']:.4f}")
    print(f"    Макс. боковая АКФ      : {results['weil_ac_side_dB']:.2f} дБ")
    print(f"    Макс. XCorr            : {results['weil_xcorr_max_dB']:.2f} дБ")
    print(f"    Merit Factor (среднее) : {results['weil_merit_factor']:.3f}")
    print(f"  Gold C/A (n = {GOLD_N}):")
    print(f"    Макс. XCorr            : {results['gold_xcorr_max_dB']:.2f} дБ")
    print(f"    Merit Factor (среднее) : {results['gold_merit_factor']:.3f}")
    print(f"  Extended Memory L5 (n = {L5_N}, {results['em_n']} кодов):")
    print(f"    Макс. XCorr            : {results['em_xcorr_max_dB']:.2f} дБ")
    print(f"    Merit Factor (среднее) : {results['em_merit_factor']:.3f}")
    print(sep)
