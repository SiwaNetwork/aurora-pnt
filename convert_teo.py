"""
Сборка ТЭО (АВРОРА-ТЭО-001) в ГОСТ-DOCX + PDF из docs/TEO_AURORA.md.

Переиспользует пайплайн convert_gost.py (вёрстка ГОСТ 2.105, парсер MD→DOCX,
экспорт PDF через Word COM) со своим титульным листом ТЭО. Пути рисунков в
исходнике записаны относительно docs/ («../results/…»); парсер convert_gost
резолвит их от корня проекта, поэтому перед сборкой «../results/» → «results/».

Запуск:  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python convert_teo.py [--no-pdf]
"""

import os, sys
import convert_gost as cg

BASE_DIR = cg.BASE_DIR
MD_IN    = os.path.join(BASE_DIR, "docs", "TEO_AURORA.md")
DOCX_OUT = os.path.join(BASE_DIR, "docs", "TEO_AURORA_GOST.docx")
PDF_OUT  = os.path.join(BASE_DIR, "docs", "TEO_AURORA_GOST.pdf")
TMP_MD   = os.path.join(BASE_DIR, ".teo_build_tmp.md")


def add_teo_title(doc):
    cg._cp(doc, "РОССИЙСКАЯ ФЕДЕРАЦИЯ", bold=True, before=18)
    cg._cp(doc, "ООО «ШИВА НЕТВОРК» (ShiwaNetwork)", before=4)
    cg._cp(doc, "Основание: приказ ООО «ШИВА НЕТВОРК» № 11/2026", size=cg.SZ_SM, before=4)
    cg._cp(doc, "")
    cg._cp(doc, "НАУЧНО-ИССЛЕДОВАТЕЛЬСКАЯ РАБОТА «СИЯНИЕ»", size=cg.Pt(15), bold=True, before=22)
    cg._cp(doc, "")
    cg._cp(doc, "АВРОРА", size=cg.Pt(22), bold=True, before=8)
    cg._cp(doc, "Низкоорбитальная система навигации, позиционирования", size=cg.Pt(14), bold=True)
    cg._cp(doc, "и синхронизации", size=cg.Pt(14), bold=True)
    cg._cp(doc, "")
    cg._cp(doc, "ТЕХНИКО-ЭКОНОМИЧЕСКОЕ ОБОСНОВАНИЕ", size=cg.Pt(18), bold=True, before=14)
    cg._cp(doc, "Документ: АВРОРА-ТЭО-001 (предварительная редакция)", bold=True, before=8)
    cg._cp(doc, "")
    cg._cp(doc, "Составлен в соответствии с ГОСТ Р 15.201-2000, ГОСТ 2.105-2019;", size=cg.SZ_SM, italic=True, before=30)
    cg._cp(doc, "методическими рекомендациями по оценке эффективности", size=cg.SZ_SM, italic=True)
    cg._cp(doc, "инвестиционных проектов (РФ)", size=cg.SZ_SM, italic=True)
    cg._cp(doc, "")
    cg._cp(doc, "Москва — 2026", bold=True, before=12)
    doc.add_page_break()


def main():
    print("AURORA TEO — ГОСТ Р 15.201-2000 / 2.105-2019")
    # Переписать пути рисунков ../results/ → results/ (парсер резолвит от корня).
    md = open(MD_IN, encoding="utf-8").read()
    md = md.replace("](../results/", "](results/")
    with open(TMP_MD, "w", encoding="utf-8") as f:
        f.write(md)
    try:
        doc = cg.setup_doc()
        cnt = cg.Counters()
        cg.add_page_num(doc.sections[0])
        print("  Титульный лист ТЭО...")
        add_teo_title(doc)
        print("  Оглавление...")
        cg.add_toc(doc)
        print("  Парсинг текста ТЭО...")
        cg.parse(TMP_MD, doc, cnt)
        print(f"  Статистика: рисунков={cnt.fig}, таблиц={cnt.tbl}, формул={cnt.eq}")
        doc.save(DOCX_OUT)
        print(f"  Готово! {os.path.getsize(DOCX_OUT)//1024} КБ: {DOCX_OUT}")
    finally:
        if os.path.exists(TMP_MD):
            os.remove(TMP_MD)

    if "--no-pdf" not in sys.argv:
        print("  Экспорт PDF…")
        cg.export_pdf(DOCX_OUT, PDF_OUT)


if __name__ == "__main__":
    main()
