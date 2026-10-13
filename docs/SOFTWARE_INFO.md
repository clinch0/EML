# Software information

| Field | Value |
| --- | --- |
| Name | eml-sr |
| Version | 0.1.0 |
| Repository URL | https://github.com/clinch0/EML |
| Branch | `cursor/eml-sr-implementation-520e` |
| Tag | `v0.1.0` (release commit; resolve with `git rev-parse v0.1.0`) |
| Last code commit before the release-docs commit | `58984c1` (Stage 5b: benchmark, examples) |
| Tagged release commit | `1bb421c` (`v0.1.0`) |
| License | MIT (declared in `pyproject.toml`) |

The release commit itself cannot contain its own hash; use `git rev-parse v0.1.0` or
`git log -1 --format=%H v0.1.0` to obtain it.

## Requirements
- Python >= 3.10 (developed and tested with 3.12.3)
- numpy >= 1.24 (tested 2.5.4), torch >= 2.0 CPU (tested 2.14.1+cpu), sympy >= 1.12 (tested 1.14.0),
  matplotlib >= 3.7 (tested 3.11.2)
- Development: pytest >= 7.4 (tested 9.1.1), ruff >= 0.5 (tested 0.17.0)
- No GPU, graphviz or network access needed at run time.

## Build / install
```bash
git clone <clone URL of sergejbelov289/eml> eml-sr && cd eml-sr
git checkout v0.1.0
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```
On Debian/Ubuntu `python3-venv` (e.g. `python3.12-venv`) may be needed for `python -m venv`.

## Run
```bash
eml-sr fit data.csv --depth 3 --restarts 50 --out result/
python examples/recover_exp.py
python examples/recover_ln.py
python benchmarks/run_benchmark.py --quick
```

## Test and lint
```bash
pytest
ruff check .
ruff format --check .
python scripts/completion.py
```
