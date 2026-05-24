# Contributing to AURORA PNT

Спасибо за интерес к проекту AURORA PNT! Данный документ описывает процесс
сотрудничества по разработке системы.

---

## Структура проекта

```
LEOPath/
├── aurora/pnt/              # Основные модули моделирования (60+ файлов)
│   ├── *.py                 # Каждый модуль: run_X_analysis() + print_X_summary()
│   └── cli.py               # Точка входа CLI: aurora-pnt <команда>
├── tests/                   # Pytest-тесты (149+ тестов)
│   ├── test_*.py            # Юнит и smoke-тесты модулей
│   └── conftest.py          # Общие фикстуры
├── docs/                    # ОКР-документация
│   ├── SAD_AURORA.md        # Software Architecture Document
│   ├── ICD_SIS_*.md         # Signal-in-Space ICD
│   ├── ICD_ISL_*.md         # Inter-Satellite Link ICD
│   ├── ICD_TTC_*.md         # TT&C ICD
│   └── TZ_AURORA.md         # Техническое задание (ГОСТ 19.201-78)
├── results/                 # Результаты симуляций (PNG, CSV, HTML)
├── .github/workflows/       # CI/CD pipelines
├── AURORA_PNT_Technical_Project.md  # Главный технический проект (64 раздела)
├── AURORA_PNT_GOST.docx     # DOCX-версия (ГОСТ 2.105/7.32)
└── AURORA_PNT_GOST.pdf      # PDF-версия для печати
```

---

## Шаблон нового модуля

Все модули в `aurora/pnt/` следуют единому шаблону. Пример — `aurora/pnt/rtk_ppp.py`.

```python
"""
<Русский докстринг: что считает модуль + ссылки на литературу>
"""
import sys, math, os, csv
from typing import Dict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# UTF-8 stdout/stderr на Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Константы / параметры ──
PARAMETERS = {...}

# ── Основная функция ──
def run_X_analysis(output_dir: str, label: str) -> Dict:
    """<Что делает>"""
    os.makedirs(output_dir, exist_ok=True)
    results = {...}
    _plot_1(...); _plot_2(...); _plot_3(...)
    _save_csv(results, output_dir, label)
    return results

# ── Графики ──
def _plot_X(..., output_dir, label):
    fig, ax = plt.subplots(figsize=(11, 6))
    ...
    ax.set_title(f"<Русское название> [{label}]")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"<prefix>_{label}.png"), dpi=150)
    plt.close(fig)

# ── Сводка ──
def print_X_summary(label: str, results: Dict) -> None:
    sep = "=" * 70
    print(f"\n{sep}\n  <Title> -- {label}\n{sep}")
    ...
```

**Правила:**
- Все подписи matplotlib на **русском языке** (acronyms PDOP/UERE/CEP — латиницей)
- Палитра по умолчанию:
  `"#e17055","#fdcb6e","#0984e3","#00b894","#6c5ce7","#74b9ff"`
- numpy/matplotlib only — без scipy, если возможно
- PNG: `<имя>_{label}.png`, CSV: `<имя>_{label}.csv`
- Файлы должны открываться/закрываться чисто (`plt.close(fig)` обязательно)

---

## Регистрация в CLI

Каждый новый модуль регистрируется в `aurora/pnt/cli.py`:

```python
def cmd_<new_module>(args):
    """<Описание>"""
    from aurora.pnt.<new_module> import run_X_analysis, print_X_summary
    label = args.label or "phase5"
    output_dir = args.output or "results/<dir>"
    print(f"\n  Running ... analysis: {label}")
    r = run_X_analysis(output_dir, label)
    print_X_summary(label, r)
```

И parser в секции `for _name, _help, _odir in [...]`.

---

## Запуск тестов

```bash
# Все тесты
pytest tests/

# Только быстрые smoke-тесты
pytest tests/ -m smoke

# Без медленных тестов
pytest tests/ -m "not slow"

# Конкретный модуль
pytest tests/test_constellation_physics.py -v

# С покрытием
pytest tests/ --cov=aurora --cov-report=html
```

---

## Стиль кода

- **Python 3.10+**
- **Русский язык** в комментариях, докстрингах, plot labels
- **Английский язык** допустим в идентификаторах кода, ключах словарей
- **Type hints** — желательно для публичных функций (`run_X_analysis` точно)
- **Линт:** `ruff check aurora/pnt/ --select=E9,F63,F7,F82` (только критичные ошибки)

---

## Процесс изменений

1. **Создайте feature-ветку:** `git checkout -b feature/<краткое-описание>`
2. **Сделайте изменения** с тестами
3. **Запустите локально:** `pytest tests/`
4. **Регенерируйте DOCX** (если меняли техпроект): `python convert_gost.py`
5. **Закоммитьте** с осмысленным сообщением на русском
6. **Push + PR** в `main`

### Соглашения по commit messages

```
<тип>: <краткое описание (≤72 символа)>

<развёрнутое описание (если нужно)>

Co-Authored-By: ...
```

Типы:
- `feat:` — новая функциональность
- `fix:` — исправление ошибки
- `docs:` — обновление документации
- `test:` — добавление/изменение тестов
- `refactor:` — рефакторинг без изменения поведения
- `chore:` — рутинные задачи (CI, deps, форматирование)

---

## Документирование изменений в технический проект

При изменении любого расчётного модуля, который ссылается из техпроекта:
1. Обновите соответствующий раздел в `AURORA_PNT_Technical_Project.md`
2. Регенерируйте DOCX: `python convert_gost.py`
3. (Опционально) Регенерируйте PDF: `python -c "from docx2pdf import convert; convert('AURORA_PNT_GOST.docx', 'AURORA_PNT_GOST.pdf')"`
4. Обновите CHANGELOG.md

---

## CI/CD

При каждом push в ветку `feature/**`, `develop`, `main` (а также PR) запускается:

- **`ci-pnt.yml`** (лёгкий):
  - pytest на Python 3.10, 3.11, 3.12
  - ruff lint на критичные ошибки
  - Сборка DOCX как артефакт

- **`ci.yml`** (полный):
  - Полный coverage (codecov)
  - Docker-сборка
  - Релизы

---

## Безопасность

- **Не коммитьте** ключи, пароли, токены (`*.env`, `*.key`, `*.pem`)
- **Не загружайте** PII или внутренние документы
- При обнаружении уязвимости — сообщите приватно

---

## Лицензия

Copyright (c) 2026 Shiwa Network. См. [LICENSE](LICENSE).

---

## Контакты

- Главный конструктор: см. CODEOWNERS
- Issues: https://github.com/SiwaNetwork/aurora-pnt/issues
- Pull Requests: https://github.com/SiwaNetwork/aurora-pnt/pulls
