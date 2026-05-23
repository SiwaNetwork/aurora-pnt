"""
Реестр рисков AURORA PNT и количественная оценка.

Анализирует:
- 25 идентифицированных рисков по 6 категориям (технические,
  программные, внешние, космическая среда, кибер, эксплуатация)
- Оценку P × S (вероятность × последствия) и уровень риска
  (Low / Medium / High / Critical)
- Матрицу рисков 5×5, топ-10 по приоритету, распределение по категориям
- Burn-down: ожидаемое снижение суммарного риска после мер (-50% за 18 мес.)

Ссылки:
  ECSS-M-ST-80C (2008) — Space project management — Risk management. ECSS.
  ISO 31000:2018 — Risk management. Guidelines. ISO/TC 262.
  NASA/SP-2011-3422 — Risk Management Handbook. NASA HQ.
"""

import sys, os, csv
from typing import Dict, List, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Палитра проекта ──────────────────────────────────────────────────────────
PALETTE = ["#e17055", "#fdcb6e", "#0984e3", "#00b894",
           "#6c5ce7", "#74b9ff", "#dfe6e9", "#2d3436"]

# ── Категории рисков и цвета ─────────────────────────────────────────────────
CATEGORIES = {
    "Технический":      "#e17055",
    "Программный":      "#fdcb6e",
    "Внешний":          "#0984e3",
    "Косм. среда":      "#6c5ce7",
    "Кибер":            "#00b894",
    "Эксплуатация":     "#74b9ff",
}

# ── Реестр рисков AURORA PNT (25 шт.) ────────────────────────────────────────
# поля: id, категория, описание, P, S, владелец, мера митигации
RISKS: List[Dict] = [
    # ─── Технические ────────────────────────────────────────────────────────
    {"id": "T-01", "cat": "Технический",
     "desc": "Отказ Cs-часов на КА", "P": 2, "S": 3,
     "owner": "Бортовой компл.",
     "action": "Резерв 3×Cs + 2×Rb + OCXO, фолбэк до Rb/OCXO"},
    {"id": "T-02", "cat": "Технический",
     "desc": "Радиационное повреждение БЦВМ", "P": 3, "S": 4,
     "owner": "ЭКБ / БЦВМ",
     "action": "Rad-hard ЭКБ, TMR, ECC память, перепроектирование Phase 1"},
    {"id": "T-03", "cat": "Технический",
     "desc": "Многократный отказ ISL канала", "P": 2, "S": 3,
     "owner": "ISL/связь",
     "action": "4 канала ISL на КА, mesh-резервирование через 8 соседей"},
    {"id": "T-04", "cat": "Технический",
     "desc": "Превышение бюджета массы при ОКР", "P": 3, "S": 3,
     "owner": "Системотехника",
     "action": "Резерв 15% массы T1, утилизация запаса топлива"},
    {"id": "T-05", "cat": "Технический",
     "desc": "Несоответствие SISA требованию <0,5 м", "P": 2, "S": 4,
     "owner": "POD/часы",
     "action": "Параллельный POD на двух движках, ISL ranging как backup"},
    {"id": "T-06", "cat": "Технический",
     "desc": "Деградация солн. батарей быстрее планируемой",
     "P": 2, "S": 2, "owner": "СЭП",
     "action": "Запас 25% по мощности EOL, оптимизация цикла заряда"},

    # ─── Программные / организационные ──────────────────────────────────────
    {"id": "P-01", "cat": "Программный",
     "desc": "Срыв сроков ОКР", "P": 3, "S": 4,
     "owner": "Руководство проекта",
     "action": "Резерв 6 мес. на ОКР, ежеквартальный аудит CDR"},
    {"id": "P-02", "cat": "Программный",
     "desc": "Удорожание свыше 1,5× LCC", "P": 3, "S": 4,
     "owner": "Финансовый блок",
     "action": "Resp-стратегия закупок, индексация контрактов, EV-учёт"},
    {"id": "P-03", "cat": "Программный",
     "desc": "Уход ключевого персонала", "P": 3, "S": 3,
     "owner": "HR",
     "action": "Программа удержания, дублирование компетенций, opcionon"},
    {"id": "P-04", "cat": "Программный",
     "desc": "Конфликт со «Сферой» Роскосмоса", "P": 2, "S": 3,
     "owner": "Руководство",
     "action": "Соглашение о разделении частот L1/L5 vs L2/L6, координация"},

    # ─── Внешние / регуляторные ─────────────────────────────────────────────
    {"id": "E-01", "cat": "Внешний",
     "desc": "Отказ МСЭ в выделении частот", "P": 2, "S": 5,
     "owner": "Регуляторика",
     "action": "Coordination filing 7 лет до старта, резервная полоса L-band"},
    {"id": "E-02", "cat": "Внешний",
     "desc": "Задержка ГКРЧ", "P": 3, "S": 3,
     "owner": "Регуляторика",
     "action": "Параллельная подача заявок РЭС, лоббирование ФАС"},
    {"id": "E-03", "cat": "Внешний",
     "desc": "Запрет на использование иностранных РН (санкции)",
     "P": 4, "S": 4, "owner": "Запусковый сегмент",
     "action": "Опора на «Союз-2», «Ангара-1.2/А5», запас 30% по слотам"},
    {"id": "E-04", "cat": "Внешний",
     "desc": "Срыв поставок Cs-стандартов РИРВ", "P": 2, "S": 4,
     "owner": "Часовой блок",
     "action": "Контракт с двумя поставщиками Cs, стратегический запас 60 шт."},
    {"id": "E-05", "cat": "Внешний",
     "desc": "Эмбарго на rad-hard ЭКБ", "P": 4, "S": 4,
     "owner": "ЭКБ",
     "action": "Импортозамещение «Микрон»/«Ангстрем», rad-tolerant COTS+TMR"},
    {"id": "E-06", "cat": "Внешний",
     "desc": "Геополитический запрет наземных станций за рубежом",
     "P": 3, "S": 3, "owner": "Наземный сегмент",
     "action": "RU-only сеть из 21 МКС + ISL-only POD как fallback"},

    # ─── Космическая среда ──────────────────────────────────────────────────
    {"id": "S-01", "cat": "Косм. среда",
     "desc": "Столкновение с космическим мусором", "P": 2, "S": 5,
     "owner": "Полёт. динамика",
     "action": "СПД-25/55, авто-манёвр Pc>1e-4, страх. 25% запас Δv"},
    {"id": "S-02", "cat": "Косм. среда",
     "desc": "Внеплановая солн. вспышка (SEU storm)", "P": 3, "S": 3,
     "owner": "БЦВМ/АРМ",
     "action": "Safe-mode, TMR, watchdog, восстан. за <1 ч"},
    {"id": "S-03", "cat": "Косм. среда",
     "desc": "Сход сторонней КА в зоне созвездия", "P": 2, "S": 4,
     "owner": "Полёт. динамика",
     "action": "Мониторинг 18 SPCS, COLA <72 ч, манёвр Δv≤5 см/с"},

    # ─── Кибер / безопасность ───────────────────────────────────────────────
    {"id": "C-01", "cat": "Кибер",
     "desc": "Компрометация TESLA-ключей", "P": 2, "S": 4,
     "owner": "ИБ",
     "action": "Ротация ключей <30 с, HSM на MCS, разделение полномочий"},
    {"id": "C-02", "cat": "Кибер",
     "desc": "Спуфинг сигнала AURORA", "P": 3, "S": 3,
     "owner": "ИБ / приёмники",
     "action": "Аутентиф. навигационного сообщения OSNMA + TESLA, RF-watermark"},
    {"id": "C-03", "cat": "Кибер",
     "desc": "DoS на MCS центр", "P": 3, "S": 3,
     "owner": "ИБ / MCS",
     "action": "Гео-резерв MCS (Москва + Красноярск), DDoS-защита"},

    # ─── Эксплуатационные ───────────────────────────────────────────────────
    {"id": "O-01", "cat": "Эксплуатация",
     "desc": "Отказ MCS без резерва", "P": 1, "S": 5,
     "owner": "Наземный сегмент",
     "action": "Резервный MCS hot-standby, RPO=0, RTO<5 мин"},
    {"id": "O-02", "cat": "Эксплуатация",
     "desc": "Несинхронизация LPT с UTC >1 мкс", "P": 2, "S": 3,
     "owner": "Часовой блок / РИРВ",
     "action": "Дублирование UTC(SU) канала, 3-уровн. сравнение TWSTFT"},
    {"id": "O-03", "cat": "Эксплуатация",
     "desc": "Отказ сети RSN > 5 станций", "P": 2, "S": 3,
     "owner": "Наземный сегмент",
     "action": "Резерв 5 ст. горячего pool, ISL-only POD как backup PPP"},
]


# ── Уровень риска по оценке P×S ──────────────────────────────────────────────
def risk_level(score: int) -> str:
    if score > 20: return "Critical"
    if score >= 13: return "High"
    if score >= 6:  return "Medium"
    return "Low"


def level_color(level: str) -> str:
    return {"Low":      "#00b894",
            "Medium":   "#fdcb6e",
            "High":     "#e17055",
            "Critical": "#6c5ce7"}[level]


def run_risks_analysis(output_dir: str, label: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    enriched = []
    for r in RISKS:
        score = r["P"] * r["S"]
        item = {**r, "score": score, "level": risk_level(score)}
        enriched.append(item)

    # сортировка по убыванию score (приоритет)
    enriched.sort(key=lambda x: x["score"], reverse=True)

    by_cat: Dict[str, List[Dict]] = {c: [] for c in CATEGORIES}
    for it in enriched:
        by_cat[it["cat"]].append(it)

    by_level = {"Low": 0, "Medium": 0, "High": 0, "Critical": 0}
    for it in enriched:
        by_level[it["level"]] += 1

    total_score = sum(it["score"] for it in enriched)

    results = {
        "risks":          enriched,
        "by_category":    by_cat,
        "by_level":       by_level,
        "total_score":    total_score,
        "n_total":        len(enriched),
        "top10":          enriched[:10],
    }

    _plot_matrix(enriched, output_dir, label)
    _plot_top10(enriched[:10], output_dir, label)
    _plot_categories(by_cat, by_level, output_dir, label)
    _plot_burndown(total_score, output_dir, label)
    _save_csv(enriched, output_dir, label)
    return results


# ── График 1: матрица рисков 5×5 ─────────────────────────────────────────────
def _plot_matrix(risks: List[Dict], output_dir: str, label: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))

    # фоновая сетка цветов по уровням
    grid = np.zeros((5, 5))
    for p in range(1, 6):
        for s in range(1, 6):
            grid[p - 1, s - 1] = p * s

    # дискретный colormap по уровням
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(["#00b894", "#fdcb6e", "#e17055", "#6c5ce7"])
    bnds = [0, 6, 13, 21, 26]
    norm = BoundaryNorm(bnds, cmap.N)
    ax.imshow(grid, cmap=cmap, norm=norm, origin="lower",
              extent=(0.5, 5.5, 0.5, 5.5), alpha=0.45, aspect="auto")

    # точки рисков (с jitter, чтобы id не накладывались)
    rng = np.random.default_rng(7)
    pos = {}
    for r in risks:
        key = (r["P"], r["S"])
        pos.setdefault(key, 0)
        n = pos[key]
        pos[key] += 1
        dx = (n % 3 - 1) * 0.18 + rng.uniform(-0.04, 0.04)
        dy = (n // 3 - 1) * 0.18 + rng.uniform(-0.04, 0.04)
        ax.scatter(r["S"] + dx, r["P"] + dy,
                   s=160, color=CATEGORIES[r["cat"]],
                   edgecolor="white", lw=1.2, zorder=3)
        ax.text(r["S"] + dx, r["P"] + dy, r["id"],
                ha="center", va="center", fontsize=7.2,
                color="#2d3436", fontweight="bold", zorder=4)

    ax.set_xticks(range(1, 6))
    ax.set_yticks(range(1, 6))
    ax.set_xticklabels(["1\nпренебр.", "2\nмалое", "3\nсредн.",
                        "4\nкрупн.", "5\nкатастр."])
    ax.set_yticklabels(["1\nоч. низк.", "2\nнизк.", "3\nсредн.",
                        "4\nвысок.", "5\nоч. высок."])
    ax.set_xlabel("Последствия (S)")
    ax.set_ylabel("Вероятность (P)")
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.5, 5.5)
    ax.set_title(f"Матрица рисков AURORA PNT 5×5  [{label}]\n"
                 f"Всего рисков: {len(risks)},  сумма P×S = "
                 f"{sum(r['score'] for r in risks)}")
    ax.grid(alpha=0.25, color="white")

    # легенда категорий
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, label=n) for n, c in CATEGORIES.items()]
    handles += [
        Patch(facecolor="#00b894", alpha=0.45, label="Low (<6)"),
        Patch(facecolor="#fdcb6e", alpha=0.45, label="Medium (6-12)"),
        Patch(facecolor="#e17055", alpha=0.45, label="High (13-20)"),
        Patch(facecolor="#6c5ce7", alpha=0.45, label="Critical (>20)"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="upper left",
              bbox_to_anchor=(1.02, 1.0))

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"risks_matrix_{label}.png"), dpi=150)
    plt.close(fig)


# ── График 2: топ-10 рисков ──────────────────────────────────────────────────
def _plot_top10(top10: List[Dict], output_dir: str, label: str) -> None:
    labels = [f"{r['id']} — {r['desc'][:42]}" for r in top10]
    scores = [r["score"] for r in top10]
    colors = [CATEGORIES[r["cat"]] for r in top10]

    fig, ax = plt.subplots(figsize=(13, 7))
    bars = ax.barh(labels, scores, color=colors,
                   edgecolor="white", height=0.65)
    for bar, r in zip(bars, top10):
        ax.text(bar.get_width() + 0.2,
                bar.get_y() + bar.get_height() / 2,
                f"P×S = {r['score']}  ({r['level']})",
                va="center", fontsize=9, color="#2d3436")

    ax.axvline(6,  ls=":",  color="#fdcb6e", lw=1.2, label="порог Medium")
    ax.axvline(13, ls="--", color="#e17055", lw=1.4, label="порог High")
    ax.axvline(21, ls="-.", color="#6c5ce7", lw=1.4, label="порог Critical")

    ax.set_xlabel("Оценка приоритета P × S")
    ax.set_title(f"Топ-10 рисков AURORA PNT по приоритету  [{label}]")
    ax.invert_yaxis()
    ax.set_xlim(0, max(scores) * 1.35)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"risks_top10_{label}.png"), dpi=150)
    plt.close(fig)


# ── График 3: категории — pie + bar ──────────────────────────────────────────
def _plot_categories(by_cat: Dict[str, List[Dict]],
                     by_level: Dict[str, int],
                     output_dir: str, label: str) -> None:
    cats   = list(by_cat.keys())
    counts = [len(by_cat[c]) for c in cats]
    sums   = [sum(r["score"] for r in by_cat[c]) for c in cats]
    colors = [CATEGORIES[c] for c in cats]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # pie: количество рисков по категории
    wedges, texts, ats = ax1.pie(
        counts, labels=cats, colors=colors,
        autopct=lambda p: f"{p:.0f}%\n({int(round(p * sum(counts) / 100))})",
        startangle=90, pctdistance=0.72,
        wedgeprops=dict(edgecolor="white", linewidth=1.5))
    for t in texts: t.set_fontsize(9)
    for at in ats: at.set_fontsize(8)
    ax1.set_title(f"Распределение рисков по категориям\n(всего {sum(counts)})")

    # bar: сумма P×S по категориям
    bars = ax2.bar(cats, sums, color=colors, edgecolor="white", lw=1.5)
    for bar, v, n in zip(bars, sums, counts):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.5,
                 f"Σ={v}\nn={n}",
                 ha="center", fontsize=9, color="#2d3436")
    ax2.set_ylabel("Суммарный P × S по категории")
    ax2.set_title(f"Суммарный приоритет по категориям  [{label}]")
    ax2.set_xticks(range(len(cats)))
    ax2.set_xticklabels(cats, rotation=20, ha="right", fontsize=9)
    ax2.set_ylim(0, max(sums) * 1.20)
    ax2.grid(axis="y", alpha=0.3)

    # подпись по уровням
    lvl_str = "  |  ".join(f"{lv}: {by_level[lv]}"
                            for lv in ["Critical", "High", "Medium", "Low"])
    fig.suptitle(f"AURORA PNT — Реестр рисков  [{label}]\n{lvl_str}",
                 fontsize=11, y=1.02)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"risks_categories_{label}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── График 4: burn-down (снижение риска во времени) ──────────────────────────
def _plot_burndown(total_score: int, output_dir: str, label: str) -> None:
    months = np.arange(0, 37)
    # экспоненциальное снижение: -50% за 18 мес. ⇒ τ = 18/ln(2)
    tau = 18.0 / np.log(2)
    planned = total_score * np.exp(-months / tau)
    # «оптимистичный» сценарий: -70% за 18 мес.
    tau_opt = 18.0 / np.log(1 / 0.30)
    optim = total_score * np.exp(-months / tau_opt)
    # «консервативный»: -30% за 18 мес.
    tau_con = 18.0 / np.log(1 / 0.70)
    conserv = total_score * np.exp(-months / tau_con)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(months, planned,  color="#0984e3", lw=2.5,
            label="План:  −50% за 18 мес. (целевой)")
    ax.plot(months, optim,    color="#00b894", lw=2.0, ls="--",
            label="Оптим.: −70% за 18 мес.")
    ax.plot(months, conserv,  color="#e17055", lw=2.0, ls=":",
            label="Консерв.: −30% за 18 мес.")
    ax.fill_between(months, optim, conserv, color="#74b9ff", alpha=0.15,
                    label="Коридор неопределённости")

    ax.axvline(18, ls="--", color="#2d3436", lw=1.0)
    ax.text(18.4, total_score * 0.93, "18 мес.\n(контроль КП-1)",
            fontsize=9, color="#2d3436")
    ax.axhline(total_score * 0.5, ls=":", color="#6c5ce7", lw=1.0)
    ax.text(0.6, total_score * 0.51, f"50% = {total_score * 0.5:.0f}",
            fontsize=9, color="#6c5ce7")

    ax.set_xlabel("Месяц от начала программы митигации")
    ax.set_ylabel("Суммарный остаточный риск (Σ P × S)")
    ax.set_title(f"Burn-down остаточного риска AURORA PNT  [{label}]\n"
                 f"Стартовое значение Σ P×S = {total_score}")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 36)
    ax.set_ylim(0, total_score * 1.05)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"risks_burndown_{label}.png"), dpi=150)
    plt.close(fig)


# ── CSV ──────────────────────────────────────────────────────────────────────
def _save_csv(risks: List[Dict], output_dir: str, label: str) -> None:
    path = os.path.join(output_dir, f"risks_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["id", "категория", "описание",
                    "P", "S", "PxS", "уровень", "владелец", "мера"])
        for r in risks:
            w.writerow([r["id"], r["cat"], r["desc"],
                        r["P"], r["S"], r["score"], r["level"],
                        r["owner"], r["action"]])


# ── Текстовый отчёт ──────────────────────────────────────────────────────────
def print_risks_summary(label: str, results: Dict) -> None:
    sep = "=" * 78
    print(f"\n{sep}")
    print(f"  AURORA PNT -- Risks Register  --  {label}")
    print(sep)
    print(f"  Всего рисков:      {results['n_total']}")
    print(f"  Суммарный Σ P×S:   {results['total_score']}")
    print(f"  Распределение по уровням:")
    for lv in ["Critical", "High", "Medium", "Low"]:
        print(f"    {lv:<10}{results['by_level'][lv]:>3}")
    print(f"\n  Распределение по категориям:")
    for cat, items in results["by_category"].items():
        s = sum(it["score"] for it in items)
        print(f"    {cat:<14}  n = {len(items):>2},  Σ P×S = {s:>3}")
    print(f"\n  Топ-10 по приоритету:")
    print(f"  {'#':<3}{'id':<6}{'кат.':<14}{'P':>2}  {'S':>2}  {'P×S':>4} "
          f"{'уровень':<10}описание")
    print(f"  {'─' * 74}")
    for i, r in enumerate(results["top10"], 1):
        desc = r["desc"][:34]
        print(f"  {i:<3}{r['id']:<6}{r['cat']:<14}"
              f"{r['P']:>2}  {r['S']:>2}  {r['score']:>4} "
              f"{r['level']:<10}{desc}")
    print(sep)
