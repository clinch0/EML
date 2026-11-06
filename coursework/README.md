# Воспроизведение курсовой

Исследование: устойчивость обучения EML-моделей, EMA и последовательное уточнение.
Протокол зафиксирован в docs/PROTOCOL_FULL.md до основных измерений.

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
python benchmarks/coursework_experiments.py --runs 30 --jobs 4 --out benchmarks/results/reproduction
python benchmarks/coursework_summary.py --out benchmarks/results/reproduction
```

Для повторения с другими параметрами используйте новый каталог: отдельные CSV используются для возобновления эпизодов. При сравнении результатов не смешивайте разные конфигурации.

Файлы публикации: benchmarks/results/full/runs.csv, summary.json, summary.csv, comparisons.csv, polynomial.csv, meta.json, methods.png, basin.png.

Исходный текст — coursework/text.md. Интерпретация измерений — coursework/discussion.json. Сборка документа требует python-docx и исходного титульного шаблона coursework/title_template.docx:

```powershell
python -m pip install python-docx
python coursework/build_docx.py --output coursework/output
```

В Word обновить оглавление целиком. Готовый документ проверяется визуально после экспорта в PDF. ФИО и степень руководителя не выдумываются: если не указаны, поля остаются для заполнения.

Вклад автоматизации раскрыт в специальном разделе. До защиты студент должен самостоятельно воспроизвести запуск и уметь объяснить каждый этап. История Git отражает фактическую работу, даты не изменяются.

Дополнительный контроль:

```powershell
python benchmarks/coursework_experiments.py --ablation --runs 30 --jobs 4 --out benchmarks/results/ablation
python benchmarks/ablation_summary.py
```

`ablation_summary.py` читает фиксированный каталог `benchmarks/results/ablation`. Чтобы полностью повторить серию, используйте чистую копию репозитория без сохранённых поэпизодных файлов или новый каталог с последующей настройкой пути сводки. Объединённые `runs.csv` содержат все исходные наблюдения; отдельные файлы эпизодов исключены из Git как дубликаты.
