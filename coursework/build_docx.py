"""Build the coursework from the HSE title template and measured JSON results."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION_START
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=ROOT / "coursework/output")
OUT = parser.parse_args().output
OUT.mkdir(exist_ok=True, parents=True)
DATA = ROOT / "benchmarks/results/full"
S = json.loads((DATA / "summary.json").read_text())
META = json.loads((DATA / "meta.json").read_text())
TEMPLATE = ROOT / "coursework/title_template.docx"
TARGETS = {
    "exp_minus_one": "exp(x)−1",
    "log": "ln(x)",
    "exp_e_minus_x": "exp(e−x)",
    "e_minus_exp": "e−exp(x)",
}
TITLE = "Устойчивость обучения символьных EML моделей при инициализации и итеративном уточнении"
FINAL = OUT / "Курсовая_Белов_EML_TRM.docx"
shutil.copy2(TEMPLATE, FINAL)
doc = Document(FINAL)
for name in ("Title", "Heading 1", "Heading 2", "Caption", "TOC Heading"):
    if name not in doc.styles:
        doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
# Retain the reference title structure while filling designated slots.
replace = {
    8: "Белов Сергей Алексеевич",
    9: "Студент 2 курса группы 25ТИДИ",
    12: TITLE,
    13: "Курсовая работа",
    14: "",
    15: "по направлению подготовки 09.03.04 Программная инженерия",
    16: "образовательная программа",
    17: "«Технологии искусственного и дополненного интеллекта»",
}
for i, text in replace.items():
    p = doc.paragraphs[i]
    p.text = text
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in p.runs:
        r.font.name = "Times New Roman"
        r.font.size = Pt(14)
        r.font.color.rgb = RGBColor(0, 0, 0)
        r.bold = i == 12
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.line_spacing = 1.0
    if i == 12:
        p.style = "Title"
# Remove obsolete empty graduation wording paragraph and redundant gaps.
for i in (14, 11):
    p = doc.paragraphs[i]
    if not p.text:
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1
# The source table is used only for supervisor alignment; optional roles are omitted.
t = doc.tables[0]
for floating in list(t._tbl.tblPr.findall(qn("w:tblpPr"))):
    t._tbl.tblPr.remove(floating)
t.cell(0, 0).text = ""
t.cell(
    0, 1
).text = "Руководитель\n________________________\nучёная степень и должность\n________________________\nИ.О. Фамилия"
for row in t.rows:
    row.height = None
    for c in row.cells:
        for p in c.paragraphs:
            p.paragraph_format.first_line_indent = Cm(0)
            p.paragraph_format.line_spacing = 1
            p.paragraph_format.space_after = Pt(0)
            for r in p.runs:
                r.font.name = "Times New Roman"
                r.font.size = Pt(14)
p = doc.add_paragraph("Нижний Новгород, 2026")
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(60)
for r in p.runs:
    r.font.name = "Times New Roman"
    r.font.size = Pt(14)
# New body section has the geometry explicitly prescribed in the 2026 guidelines.
s = doc.add_section(WD_SECTION_START.NEW_PAGE)
s.page_width = Cm(21)
s.page_height = Cm(29.7)
s.left_margin = Cm(3)
s.right_margin = Cm(1.5)
s.top_margin = Cm(2)
s.bottom_margin = Cm(2)
s.footer_distance = Cm(1)
s.footer.is_linked_to_previous = False
p = s.footer.paragraphs[0]
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
fld = OxmlElement("w:fldSimple")
fld.set(qn("w:instr"), "PAGE")
p._p.append(fld)
num = OxmlElement("w:pgNumType")
num.set(qn("w:start"), "2")
s._sectPr.append(num)
for style in ["Normal", "Title", "Heading 1", "Heading 2", "Caption"]:
    st = doc.styles[style]
    st.font.name = "Times New Roman"
    st.font.color.rgb = RGBColor(0, 0, 0)
    st.font.size = Pt(14)
    st.paragraph_format.first_line_indent = Cm(0)
    st.paragraph_format.space_before = Pt(0)
    st.paragraph_format.space_after = Pt(0)
normal = doc.styles["Normal"]
normal.paragraph_format.first_line_indent = Cm(1.25)
normal.paragraph_format.line_spacing = 1.5
normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
for level, size in [("Heading 1", 16), ("Heading 2", 14)]:
    st = doc.styles[level]
    st.font.size = Pt(size)
    st.font.bold = True
    st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    st.paragraph_format.keep_with_next = True
    st.paragraph_format.space_after = Pt(14)
    st.paragraph_format.space_before = Pt(0 if level == "Heading 1" else 20)
    st.paragraph_format.line_spacing = 1.0
    st.paragraph_format.page_break_before = level == "Heading 1"
    outline = OxmlElement("w:outlineLvl")
    outline.set(qn("w:val"), "0" if level == "Heading 1" else "1")
    st.element.get_or_add_pPr().append(outline)
# Title styles must not pick up body indentation.
for p in doc.paragraphs[:-1]:
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.line_spacing = 1
# TOC, updated in native Word after authoring.
p = doc.add_paragraph("Оглавление", "Heading 1")
p.style = "TOC Heading"
for run in p.runs:
    run.font.name = "Times New Roman"
    run.font.size = Pt(16)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0, 0, 0)
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.page_break_before = False
p = doc.add_paragraph()
p.paragraph_format.first_line_indent = Cm(0)
f = OxmlElement("w:fldSimple")
f.set(qn("w:instr"), 'TOC \\o "1-2" \\h \\z \\u')
p._p.append(f)
settings = doc.settings.element
u = OxmlElement("w:updateFields")
u.set(qn("w:val"), "true")
settings.append(u)


def para(text):
    p = doc.add_paragraph(text)
    p.paragraph_format.widow_control = True
    return p


def table(caption, headers, rows, widths=None):
    p = para(caption)
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.keep_with_next = True
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    t = doc.add_table(rows=1, cols=len(headers))
    t.autofit = False
    if widths:
        for c, w in zip(t.columns, widths, strict=False):
            c.width = Cm(w)
    for cell, txt in zip(t.rows[0].cells, headers, strict=False):
        cell.text = txt
    repeat = OxmlElement("w:tblHeader")
    t.rows[0]._tr.get_or_add_trPr().append(repeat)
    for row in rows:
        for c, txt in zip(t.add_row().cells, row, strict=False):
            c.text = str(txt)
    for i, row in enumerate(t.rows):
        pr = row._tr.get_or_add_trPr()
        pr.append(OxmlElement("w:cantSplit"))
        for c in row.cells:
            tcpr = c._tc.get_or_add_tcPr()
            borders = OxmlElement("w:tcBorders")
            for edge in ["top", "left", "bottom", "right"]:
                el = OxmlElement("w:" + edge)
                el.set(qn("w:val"), "single")
                el.set(qn("w:sz"), "4")
                el.set(qn("w:color"), "BFBFBF")
                borders.append(el)
            tcpr.append(borders)
            if i == 0:
                sh = OxmlElement("w:shd")
                sh.set(qn("w:fill"), "E7E7E7")
                tcpr.append(sh)
            for p in c.paragraphs:
                p.paragraph_format.first_line_indent = Cm(0)
                p.paragraph_format.line_spacing = 1
                p.paragraph_format.space_before = Pt(4)
                p.paragraph_format.space_after = Pt(4)
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for r in p.runs:
                    r.font.size = Pt(11)
                    r.bold = i == 0
    para("")


def figure(name, caption):
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Cm(0)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(DATA / name), width=Cm(16))
    p = para(caption)
    p.paragraph_format.first_line_indent = Cm(0)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.line_spacing = 1
    for r in p.runs:
        r.font.size = Pt(12)


def cells(methods):
    return [
        r
        for name in TARGETS
        for m, b in methods
        for r in S["summary"]
        if (r["target"], r["method"], r["beta"]) == (name, m, b)
    ]


def render_token(token):
    if token == "targets_table":
        table(
            "Таблица 1. Целевые функции и эталонные деревья",
            ["Функция", "Представление через E", "Глубина"],
            [
                ["exp(x)−1", "E(x,E(1,1))", 2],
                ["ln(x)", "E(1,E(E(1,x),1))", 3],
                ["exp(e−x)", "E(E(1,E(x,1)),1)", 3],
                ["e−exp(x)", "E(1,E(E(x,1),1))", 3],
            ],
            [3.5, 10, 2.5],
        )
    elif token == "run_audit":
        a = S["audit"]
        elapsed = (META["finished"] - META["started"]) / 60
        para(
            f"Выполнено {a['episodes']} эпизодов и {a['rows']} отдельных проходов обучения. Каждый проход содержит 1000 эпох. Число досрочных остановок из-за нечисловой потери составило {a['early_stops']}. Аудит фактических эпох подтвердил равенство бюджета: по 3000 эпох для IR и RF и по 1000 для диагностического старта."
        )
        para(
            f"Вычисления выполнены в среде Python {META['python']}, PyTorch {META['torch']}, NumPy {META['numpy']} под Windows на процессоре с 8 физическими ядрами и 16 логическими потоками. Использовались четыре процесса по одному потоку PyTorch. Длительность серии по часам — {elapsed:.1f} мин; сумма зарегистрированных длительностей обучения — {a['total_training_seconds'] / 60:.1f} мин. Различие связано с параллельным выполнением."
        )
        para(
            "Пилотные 20 запусков не включены в эти числа и не используются при оценке основных гипотез. Воспроизводимый протокол и исходные данные размещены в репозитории [7]."
        )
    elif token == "methods_table":
        rows = []
        for r in cells([("IR", 0), ("IR", 0.99), ("RF", 0), ("RF", 0.99)]):
            rows.append(
                [
                    TARGETS[r["target"]],
                    r["method"] + (" + EMA" if r["beta"] else ""),
                    f"{r['k']}/{r['n']}",
                    f"{r['low']:.2f}–{r['high']:.2f}",
                    r["structural"],
                    f"{r['seconds']:.2f}",
                ]
            )
        para(
            "В таблице 2 приведены успехи выбранного кандидата, интервалы Уилсона, число структурных совпадений и медианное время эпизода. Время измерено при параллельном запуске и не является чистой оценкой накладных расходов EMA."
        )
        table(
            "Таблица 2. Поиск формулы при бюджете 3000 эпох",
            ["Функция", "Метод", "Успех", "95% ДИ", "Структ.", "Время, с"],
            rows,
            [3.2, 2.5, 1.8, 3, 2, 2.5],
        )
    elif token == "methods_figure":
        figure("methods.png", "Рисунок 1. Доли успеха методов и 95-процентные интервалы Уилсона")
    elif token == "basin_table":
        rows = []
        for r in cells([("cold", 0), ("k0", 0), ("k1", 0), ("k3", 0)]):
            rows.append(
                [
                    TARGETS[r["target"]],
                    r["method"],
                    f"{r['k']}/{r['n']}",
                    "не применяется" if r["method"] == "k0" else f"{r['low']:.2f}–{r['high']:.2f}",
                ]
            )
        para(
            "Таблица 3 содержит результаты одного прохода длительностью 1000 эпох. Обозначение cold соответствует случайному старту, k0, k1 и k3 — числу повреждённых активных входов эталона. Для k0 статистический интервал не интерпретируется, поскольку старт детерминирован."
        )
        table(
            "Таблица 3. Чувствительность к начальному состоянию",
            ["Функция", "Старт", "Успех", "95% ДИ"],
            rows,
            [4.5, 3, 3, 4.5],
        )
    elif token == "basin_figure":
        figure(
            "basin.png",
            "Рисунок 2. Восстановление при повреждении входов; пунктиром показан холодный старт",
        )
    elif token == "comparisons_table":
        rows = []
        for c in S["comparisons"]:
            if c["comparison"] == "basin":
                continue
            rows.append(
                [
                    TARGETS[c["target"]],
                    c["comparison"],
                    f"{c['delta'] * 100:+.1f}",
                    "нет расхождений"
                    if c["low"] is None
                    else f"{c['low'] * 100:+.1f}; {c['high'] * 100:+.1f}",
                ]
            )
        para(
            "В таблице 4 разности приведены в процентных пунктах. EMA означает IR с EMA минус IR без EMA; RF — RF минус IR без EMA; RF_EMA — RF с EMA минус IR с EMA. Интервалы получены парным бутстрепом, не скорректированы на множественные сравнения и интерпретируются как разведочные."
        )
        table(
            "Таблица 4. Парные разности долей успеха",
            ["Функция", "Сравнение", "Разность, п.п.", "95% интервал, п.п."],
            rows,
            [4.5, 3, 3, 5],
        )
    elif token == "ablation_table":
        ablation = json.loads((ROOT / "benchmarks/results/ablation/summary.json").read_text())
        rows = [
            [r["method"], f"{r['k']}/{r['n']}", f"{r['low']:.2f}–{r['high']:.2f}", "3000"]
            for r in ablation["summary"]
        ]
        table(
            "Таблица 5. Контроль на новых seed для ln(x)",
            ["Метод", "Успех", "95% ДИ", "Эпох на эпизод"],
            rows,
            [4, 4, 4, 4],
        )
    elif token == "baseline_table":
        rows = [
            [TARGETS[r["target"]], f"{r['interpolation']:.3e}", f"{r['extrapolation']:.3e}"]
            for r in S["baseline"]
        ]
        table(
            "Таблица 6. Максимальные ошибки полинома степени 5",
            ["Функция", "Интерполяция", "Экстраполяция"],
            rows,
            [6, 5, 5],
        )
    elif token in [
        "methods_discussion",
        "basin_discussion",
        "comparisons_discussion",
        "ablation_discussion",
        "conclusion",
    ]:
        for p in json.loads((ROOT / "coursework/discussion.json").read_text(encoding="utf-8-sig"))[
            token
        ]:
            para(p)
    elif token == "references":
        refs = [
            "Odrzywołek A. All elementary functions from a single binary operator. 2026. arXiv:2603.21852v1. URL: https://arxiv.org/abs/2603.21852v1 (дата обращения: 05.11.2026).",
            "Jolicoeur-Martineau A. Less is More: Recursive Reasoning with Tiny Networks. 2025. arXiv:2510.04871v1. URL: https://arxiv.org/abs/2510.04871v1 (дата обращения: 05.11.2026).",
            "Cranmer M., Sanchez-Gonzalez A., Battaglia P. et al. Discovering Symbolic Models from Deep Learning with Inductive Biases // Advances in Neural Information Processing Systems. 2020. Vol. 33. URL: https://arxiv.org/abs/2006.11287 (дата обращения: 05.11.2026).",
            "Kingma D. P., Ba J. Adam: A Method for Stochastic Optimization // International Conference on Learning Representations. 2015. URL: https://arxiv.org/abs/1412.6980 (дата обращения: 05.11.2026).",
            "Wilson E. B. Probable Inference, the Law of Succession, and Statistical Inference // Journal of the American Statistical Association. 1927. Vol. 22, № 158. P. 209–212. URL: https://www.jstor.org/stable/2276774 (дата обращения: 05.11.2026).",
            "Efron B. Bootstrap Methods: Another Look at the Jackknife // The Annals of Statistics. 1979. Vol. 7, № 1. P. 1–26. DOI: 10.1214/aos/1176344552.",
            "EML. Исходный код и материалы экспериментов. URL: https://github.com/clinch0/EML/tree/research/ema-coursework (дата обращения: 05.11.2026).",
        ]
        for i, r in enumerate(refs, 1):
            para(f"{i}. {r}")
    elif token == "software":
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        para(
            "Студент: Белов Сергей Алексеевич, 2 курс, группа 25ТИДИ. Направление 09.03.04 «Программная инженерия», образовательная программа «Технологии искусственного и дополненного интеллекта»."
        )
        para(
            "Репозиторий: https://github.com/clinch0/EML. Ветка: research/ema-coursework. Код эксперимента: "
            + META["commit"]
            + ". Версия материалов: "
            + commit
            + "."
        )
        para(
            "Среда: Python 3.12, PyTorch CPU, NumPy, SymPy, Matplotlib, pytest и Ruff. Команды выполняются из корня репозитория. Для нового воспроизведения указан отдельный каталог, чтобы не использовать сохранённые эпизоды."
        )
        for command in [
            'python -m pip install -e ".[dev]"',
            "python -m pytest -q",
            "python -m ruff check .",
            "python benchmarks/coursework_experiments.py --runs 30 --jobs 4 --out benchmarks/results/reproduction",
            "python benchmarks/coursework_summary.py",
        ]:
            p = para(command)
            p.paragraph_format.first_line_indent = Cm(0)
            p.paragraph_format.line_spacing = 1
            for r in p.runs:
                r.font.size = Pt(11)
        para(
            "Сценарий сводки по умолчанию читает benchmarks/results/full. Для анализа повторного прогона необходимо передать его каталог параметром --out. Сохранены исходные CSV, сводные таблицы, графики и метаданные."
        )
        para(
            "Проверочный список исследовательского протокола: математическая постановка, эталонные деревья, warm start, EMA, последовательное уточнение, независимые перезапуски, равный бюджет, независимая проверка, статистическая сводка, воспроизводимый код — выполнены 10 из 10 пунктов (100%). Это оценка данного протокола; соответствие официальному индивидуальному заданию проверяется после его согласования с руководителем."
        )
        para(
            "Руководитель: ____________________. Подпись: ____________________. Дата: ____________________."
        )


text = (ROOT / "coursework/text.md").read_text(encoding="utf-8-sig")
text = text.replace(
    "95-процентный интервал Уилсона.", "95-процентный интервал Уилсона [5]."
).replace("Бутстреп повторно выбирает", "Бутстреп [6] повторно выбирает")
for block in text.split("\n\n"):
    block = block.strip()
    if not block:
        continue
    if block.startswith("## "):
        doc.add_paragraph(block[3:], "Heading 2")
    elif block.startswith("# "):
        doc.add_paragraph(block[2:], "Heading 1")
    elif block.startswith("[formula] "):
        p = para(block[10:])
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(8)
    elif block.startswith("[") and block.endswith("]"):
        render_token(block[1:-1])
    else:
        para(block)
doc.core_properties.title = TITLE
doc.core_properties.author = "Белов Сергей Алексеевич"
doc.core_properties.subject = "Курсовая работа, 25ТИДИ, EML и итеративное уточнение"
doc.save(FINAL)
print(FINAL)
