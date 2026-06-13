"""
Линтер связности документов AURORA PNT.

Проверяет:
  - ВСЕ документы (ТП, роадмап, ОКР) — на эмодзи/галочки (ГОСТ-оформление);
  - технический проект — дополнительно на старые/завышенные числа,
    устаревшую часовую схему и штампы версий документа.

ОКР-документы (SAD/ICD/ТЗ) версионируются легитимно, поэтому проверка версий
к ним НЕ применяется.

Запуск:  python check_tp.py          (CLI, exit 1 при проблемах)
         python check_tp.py --hook   (неблокирующий Stop-хук)
"""

import re
import sys

TP = "AURORA_PNT_Technical_Project.md"
# Формальные ГОСТ-документы (ОКР). Роадмап — рабочий планировочный документ,
# в его статус-таблицах допускаются галочки, поэтому он сюда не входит.
DOCS = [
    "docs/SAD_AURORA.md",
    "docs/ICD_SIS_AURORA-001.md",
    "docs/ICD_ISL_AURORA-002.md",
    "docs/ICD_TTC_AURORA-003.md",
    "docs/TZ_AURORA.md",
]

# Эмодзи/галочки — запрещены во всех документах
EMOJI = [
    (r"[✅❌⚠✓✗\U0001F534\U0001F7E2\U0001F7E0]", "эмодзи/галочка"),
]

# Только для ТП: устаревшая часовая схема, старые числа, штампы версий
TP_ONLY = [
    ("1× Cs, 2× Rb", "старая часовая схема (Cs+2Rb на КА)"),
    ("Cs-мастер", "старый Cs-мастер"),
    ("Cs-стандарт", "старый Cs-стандарт (должен быть CSAC/H-мазер)"),
    ("Symmetricom", "старый эталон Symmetricom (должен быть Ч1-1008)"),
    ("Quantum-18 (ВРЕМЯ-Ч)", "неверная атрибуция Quantum (это ШИВА НЕТВОРК)"),
    ("0,114", "старый URE 0,114 (должно 0,45)"),
    ("0,279", "старый UERE 0,279 (должно 0,70)"),
    ("(42 сп", "старый N_vis 42 (должно 14)"),
    ("N_vis 42", "старый N_vis 42 (должно 14)"),
    ("Ф1 (18 сп", "старый план фаз 18 (должно 12)"),
    ("Ф2 (60 сп", "старый план фаз 60 (должно 90)"),
    ("Версия 1.0", "штамп версии документа"),
    ("**Версия:**", "штамп версии документа"),
]

ALLOWED_CONTEXT = [
    "цезий на борту не",   # «...цезий на борту не применяется»
    "версия 5.1",          # ГЛОНАСС ИКД (версия источника)
    "| Версия |",          # столбец «Версия» в матрице стандартов
]


def scan(path, patterns):
    problems = []
    try:
        lines = open(path, encoding="utf-8").read().splitlines()
    except FileNotFoundError:
        return problems
    for i, line in enumerate(lines, 1):
        if any(a in line for a in ALLOWED_CONTEXT):
            continue
        for pat, desc in patterns:
            if pat.startswith("[") or "\\u" in pat:
                if re.search(pat, line):
                    problems.append((path, i, desc, line.strip()[:60]))
            elif pat in line:
                problems.append((path, i, desc, line.strip()[:60]))
    return problems


def main():
    hook_mode = "--hook" in sys.argv  # неблокирующий режим для Stop-хука
    try:  # консоль может быть cp1251 — не падаем на печати юникода
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    problems = scan(TP, EMOJI + TP_ONLY)
    for d in DOCS:
        problems += scan(d, EMOJI)

    if hook_mode:
        if problems:
            import json
            head = "; ".join(f"{p.split('/')[-1]}:{ln} {desc}"
                             for p, ln, desc, _ in problems[:6])
            msg = f"check_tp: {len(problems)} проблем(ы) связности — {head}"
            print(json.dumps({"systemMessage": msg}, ensure_ascii=False))
        return 0

    if problems:
        print(f"check_tp: НАЙДЕНО {len(problems)} проблем(ы) связности:")
        for p, ln, desc, snip in problems:
            print(f"  {p}:{ln}: {desc}  |  {snip}")
        return 1
    print("check_tp: связность ОК — все документы чисты (эмодзи/старые числа/версии).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
