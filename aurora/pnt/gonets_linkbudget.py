"""
Бюджет линии пользователя для hosted-варианта на ГОНЕЦ-М1 (1500 км) в сравнении
с базовой орбитой АВРОРЫ (1000 км). Переиспользует методику и константы модуля
link_budget.py (та же EIRP, шумовая модель, термошум псевдодальности), меняя лишь
высоту орбиты. Часть исследования АВРОРА-ГОНЕЦ-001 (НЕ часть основного ТП).

Запуск:  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python aurora/pnt/gonets_linkbudget.py
"""

import sys, os, importlib.util

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Загружаем link_budget.py напрямую (минуя пакет aurora.pnt с зависимостью sgp4)
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("lb", os.path.join(_here, "link_budget.py"))
lb = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(lb)

ALT_AURORA = 1_000_000.0
ALT_GONETS = 1_500_000.0
ELEVS = [10, 30, 90]
TERMS = ["Survey", "Handheld"]
SIGS  = ["L1", "L5"]


def compare():
    print("Бюджет линии: АВРОРА 1000 км vs ГОНЕЦ-М1 1500 км "
          "(та же методика link_budget.py)\n")
    print(f"{'Сигнал/Терминал':18s} {'Место':>6s} "
          f"{'C/N0 1000км':>11s} {'C/N0 1500км':>11s} {'Δ, дБ':>7s}")
    deltas = []
    for sig in SIGS:
        for term in TERMS:
            for el in ELEVS:
                a = lb.compute_link_budget(sig, term, el, ALT_AURORA)["cn0_db_hz"]
                g = lb.compute_link_budget(sig, term, el, ALT_GONETS)["cn0_db_hz"]
                d = g - a
                deltas.append(d)
                tag = f"{sig}/{term}"
                print(f"{tag:18s} {el:5d}° {a:11.1f} {g:11.1f} {d:7.2f}")
    print(f"\nСредняя потеря C/N0 при переходе 1000→1500 км: "
          f"{sum(deltas)/len(deltas):.2f} дБ "
          f"(зенит ≈ 3,5 дБ; ниже по углу места — больше из-за роста дальности).")

    # Анти-джам запас (из §10/§65 ТП): на 1000 км Сервис А +11 дБ, Сервис Б +23 дБ
    # над MEO-ГНСС. На 1500 км вычитаем потерю на трассе (зенит).
    z = lb.compute_link_budget("L1", "Survey", 90, ALT_GONETS)["cn0_db_hz"] \
        - lb.compute_link_budget("L1", "Survey", 90, ALT_AURORA)["cn0_db_hz"]
    print(f"\nАнти-джам запас над MEO-ГНСС (зенит, скорректирован на {z:+.1f} дБ):")
    print(f"  Сервис А: +11 дБ → {11+z:+.1f} дБ;  Сервис Б: +23 дБ → {23+z:+.1f} дБ")
    return deltas


def main():
    out = os.path.abspath(os.path.join(_here, "..", "..", "results", "gonets"))
    compare()
    # Рисунки бюджета линии для 1500 км (та же функция, что у АВРОРЫ)
    res = lb.run_link_budget_analysis(out, "gonets1500", altitude_m=ALT_GONETS)
    print(f"\nРисунки бюджета линии (1500 км): {out}\\link_budget_*_gonets1500.png")
    # Контрольная сводка
    lb.print_link_budget_summary("ГОНЕЦ-М1 hosted (1500 км)", res)


if __name__ == "__main__":
    main()
