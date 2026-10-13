# eml-sr

Symbolic regression on EML trees. / Символьная регрессия на EML-деревьях.

`eml(x, y) = exp(x) - ln(y)`, grammar `S -> 1 | eml(S, S)` (plus the variable `x` as a leaf).
From the constant `1` and the operator `eml` elementary functions can be expressed
(Odrzywolek, arXiv:2603.21852, 2026), e.g. `e = eml(1, 1)`, `exp(x) = eml(x, 1)`,
`ln(x) = eml(1, eml(eml(1, x), 1))`.

---

## English

### What it does
Given samples `(x, y)`, eml-sr trains a differentiable "master tree" of EML nodes, snaps it to an
exact discrete EML tree, converts the tree to a readable sympy formula (when possible) and draws
the tree and the fit. A brute-force enumerator and a library of known functions are used as a
baseline and for recognising subtrees.

### Installation
```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build of torch
pip install -e ".[dev]"
```
Requires Python >= 3.10, numpy, torch (CPU), sympy, matplotlib; pytest and ruff for development.

### Example
```python
import numpy as np
from eml_sr import EMLRegressor

x = np.linspace(0.5, 3.0, 40)
model = EMLRegressor(depth=3, n_restarts=50, seed=0).fit(x, np.log(x))
print(model.formula_, model.success_)  # log(x) True
print(model.tree_)  # eml(1, eml(eml(1, x), 1))
```
Command line (CSV with columns `x,y`, header optional):
```bash
eml-sr fit data.csv --depth 3 --restarts 50 --out result/
```
It prints the formula and writes `result/tree.png`, `result/fit.png` and `result/result.json`
(exit code 0 if the snapped tree reproduces the data within `tol`, 1 otherwise, 2 on input errors).

Example scripts: `python examples/recover_exp.py`, `python examples/recover_ln.py`
(figures go to `examples/output/`).

### Tests, lint, benchmark
```bash
pytest
ruff check . && ruff format --check .
python benchmarks/run_benchmark.py --quick        # 5 runs per (target, depth)
python benchmarks/run_benchmark.py --runs 50      # fuller measurement
python scripts/completion.py                      # completion of docs/REQUIREMENTS.md
```

### Limitations
- One input variable only; `fit` raises `NotImplementedError` for `X` with several columns.
- Supported domain: `x > 0`. Computations are complex128 with the principal branch of `log`.
- `safe_exp` clamps the real part of its argument to `[-50, 50]`; `safe_log` replaces `|z| < 1e-12`
  by `1e-12`. Formulas that need `±inf` (e.g. `ln 0 = -inf` as an intermediate value) are **not
  supported**.
- A master tree of depth `d` can only represent trees of depth `<= d`. In particular `ln(x)`
  (depth 3) cannot be found with `depth=2`; use `depth=3`.
- Training is stochastic: a single run often fails, so multistart (`n_restarts`) is essential.
  Success rates measured by the quick benchmark are below.
- The library recognises only the listed known functions; with 8 leaves several of them (`x + 1`,
  `2*x`, `x^2`, `-x`, `1/x`, `sqrt(x)`, `-1`, `2`) are reported as "not found". If no verified
  readable form is obtained, the raw expression is returned with `readable=False`.

### Quick benchmark (measured)
`python benchmarks/run_benchmark.py --quick`: 5 independent single runs (no multistart) per cell,
2000 epochs, 40 points on `x in [0.5, 2]`, 95 % Wilson intervals; Python 3.12.3, torch 2.14.1+cpu,
4 vCPU. Raw data: `benchmarks/results/quick.csv`. With 5 runs the intervals are wide.

| target | depth | representable | successes | rate | 95 % CI | s/run |
| --- | --- | --- | --- | --- | --- | --- |
| exp(x) | 2 | yes | 4/5 | 0.80 | [0.38, 0.96] | 1.46 |
| exp(x) | 3 | yes | 4/5 | 0.80 | [0.38, 0.96] | 1.89 |
| ln(x) | 2 | no | 0/5 | 0.00 | [0.00, 0.43] | 1.37 |
| ln(x) | 3 | yes | 1/5 | 0.20 | [0.04, 0.62] | 1.87 |
| exp(exp(x)) | 2 | yes | 5/5 | 1.00 | [0.57, 1.00] | 1.34 |
| exp(exp(x)) | 3 | yes | 2/5 | 0.40 | [0.12, 0.77] | 1.89 |
| e - ln(x) | 2 | yes | 4/5 | 0.80 | [0.38, 0.96] | 1.34 |
| e - ln(x) | 3 | yes | 4/5 | 0.80 | [0.38, 0.96] | 1.85 |
| exp(x) - 1 | 2 | yes | 3/5 | 0.60 | [0.23, 0.88] | 1.33 |
| exp(x) - 1 | 3 | yes | 0/5 | 0.00 | [0.00, 0.43] | 1.86 |
| exp(x) - ln(x) | 2 | yes | 4/5 | 0.80 | [0.38, 0.96] | 1.34 |
| exp(x) - ln(x) | 3 | yes | 3/5 | 0.60 | [0.23, 0.88] | 1.86 |

See `docs/ARCHITECTURE.md`, `docs/DECISIONS.md`, `docs/REQUIREMENTS.md`, `docs/SOFTWARE_INFO.md`.

---

## Русский

### Что делает пакет
По выборке `(x, y)` eml-sr обучает дифференцируемое «мастер-дерево» из EML-узлов, округляет его
до точного дискретного EML-дерева, переводит дерево в читаемую формулу sympy (если это удаётся) и
рисует дерево и график аппроксимации. Полный перебор деревьев и библиотека известных функций
служат базовой линией и используются для распознавания поддеревьев.

### Установка
```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-сборка torch
pip install -e ".[dev]"
```
Нужны Python >= 3.10, numpy, torch (CPU), sympy, matplotlib; для разработки pytest и ruff.

### Пример
```python
import numpy as np
from eml_sr import EMLRegressor

x = np.linspace(0.5, 3.0, 40)
model = EMLRegressor(depth=3, n_restarts=50, seed=0).fit(x, np.log(x))
print(model.formula_, model.success_)  # log(x) True
print(model.tree_)  # eml(1, eml(eml(1, x), 1))
```
Консольная утилита (CSV со столбцами `x,y`, заголовок необязателен):
```bash
eml-sr fit data.csv --depth 3 --restarts 50 --out result/
```
Печатает формулу и сохраняет `result/tree.png`, `result/fit.png`, `result/result.json`
(код возврата 0, если округлённое дерево воспроизводит данные с точностью `tol`, иначе 1;
2 при ошибке входных данных).

Примеры: `python examples/recover_exp.py`, `python examples/recover_ln.py`
(рисунки сохраняются в `examples/output/`).

### Тесты, линтер, бенчмарк
```bash
pytest
ruff check . && ruff format --check .
python benchmarks/run_benchmark.py --quick        # 5 запусков на (функция, глубина)
python benchmarks/run_benchmark.py --runs 50      # более полный замер
python scripts/completion.py                      # процент выполнения docs/REQUIREMENTS.md
```

### Ограничения
- Только одна входная переменная; для `X` с несколькими столбцами `fit` выбрасывает
  `NotImplementedError`.
- Область определения: `x > 0`. Вычисления в complex128, логарифм по главной ветви.
- `safe_exp` ограничивает вещественную часть аргумента отрезком `[-50, 50]`; `safe_log` заменяет
  `|z| < 1e-12` на `1e-12`. Формулы, которым нужны `±inf` (например, `ln 0 = -inf` как
  промежуточное значение), **не поддерживаются**.
- Мастер-дерево глубины `d` представляет только деревья глубины `<= d`. В частности, `ln(x)`
  (глубина 3) не может быть найден при `depth=2`; нужна `depth=3`.
- Обучение стохастично: отдельный запуск часто неудачен, поэтому мультистарт (`n_restarts`)
  обязателен. Измеренные доли успеха приведены в таблице быстрого бенчмарка выше (раздел
  English, «Quick benchmark»), исходные данные в `benchmarks/results/quick.csv`.
- Библиотека распознаёт только перечисленные известные функции; при 8 листьях часть из них
  (`x + 1`, `2*x`, `x^2`, `-x`, `1/x`, `sqrt(x)`, `-1`, `2`) честно помечается как «не найдено».
  Если проверенную читаемую форму получить не удалось, возвращается сырое выражение и
  `readable=False`.

Подробности: `docs/ARCHITECTURE.md`, `docs/DECISIONS.md`, `docs/REQUIREMENTS.md`,
`docs/SOFTWARE_INFO.md`.
