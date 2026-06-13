"""
Линтер связности технического проекта AURORA PNT.

Проверяет AURORA_PNT_Technical_Project.md на:
  - эмодзи/галочки (запрещены ГОСТ-оформлением);
  - старые/завышенные числа и устаревшую часовую схему (должны быть исправлены);
  - штампы версий документа.

Запуск:  python check_tp.py
Код возврата: 0 — чисто; 1 — найдены проблемы.
"""

import re
import sys

MD = "AURORA_PNT_Technical_Project.md"

# (паттерн, пояснение). Паттерны — обычный поиск подстроки, если не regex=True.
BANNED = [
    # эмодзи / галочки
    (r"[✅❌⚠✓✗\U0001F534\U0001F7E2\U0001F7E0]", "эмодзи/галочка"),
    # устаревшая часовая схема
    ("1× Cs, 2× Rb", "старая часовая схема (Cs+2Rb на КА)"),
    ("Cs-мастер", "старый Cs-мастер"),
    ("Cs-стандарт", "старый Cs-стандарт (должен быть CSAC/H-мазер)"),
    ("Symmetricom", "старый эталон Symmetricom (должен быть Ч1-1008)"),
    ("Quantum-18 (ВРЕМЯ-Ч)", "неверная атрибуция Quantum (это ШИВА НЕТВОРК)"),
    # старые завышенные точностные числа
    ("0,114", "старый URE 0,114 (должно 0,45)"),
    ("0,279", "старый UERE 0,279 (должно 0,70)"),
    ("(42 сп", "старый N_vis 42 (должно 14)"),
    ("N_vis 42", "старый N_vis 42 (должно 14)"),
    # старый план фаз
    ("Ф1 (18 сп", "старый план фаз 18 (должно 12)"),
    ("Ф2 (60 сп", "старый план фаз 60 (должно 90)"),
    # штампы версий
    ("Версия 1.0", "штамп версии документа"),
    ("**Версия:**", "штамп версии документа"),
]

# Разрешённые исключения (легитимные совпадения подстрок)
ALLOWED_CONTEXT = [
    "цезий на борту не",          # «...цезий на борту не применяется»
    "версия 5.1",                 # ГЛОНАСС ИКД, версия источника
    "| Версия |",                 # столбец «Версия» в матрице стандартов
]


def main():
    hook_mode = "--hook" in sys.argv  # неблокирующий режим для Stop-хука
    try:
        text = open(MD, encoding="utf-8").read()
    except FileNotFoundError:
        if not hook_mode:
            print(f"check_tp: файл {MD} не найден", file=sys.stderr)
        return 0 if hook_mode else 1

    problems = []
    for i, line in enumerate(text.splitlines(), 1):
        if any(a in line for a in ALLOWED_CONTEXT):
            continue
        for pat, desc in BANNED:
            if pat.startswith("[") or "\\u" in pat:
                if re.search(pat, line):
                    problems.append((i, desc, line.strip()[:70]))
            elif pat in line:
                problems.append((i, desc, line.strip()[:70]))

    if hook_mode:
        # systemMessage только при проблемах; всегда exit 0 (не блокирует Stop)
        if problems:
            import json
            head = "; ".join(f"стр.{ln} {desc}" for ln, desc, _ in problems[:6])
            msg = f"check_tp: в ТП {len(problems)} проблем(ы) связности — {head}"
            print(json.dumps({"systemMessage": msg}, ensure_ascii=False))
        return 0

    if problems:
        print(f"check_tp: НАЙДЕНО {len(problems)} проблем(ы) связности:")
        for ln, desc, snip in problems:
            print(f"  стр. {ln}: {desc}  |  {snip}")
        return 1
    print("check_tp: связность ОК — эмодзи/старых чисел/версий не найдено.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
