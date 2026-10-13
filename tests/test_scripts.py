import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load(rel):
    spec = importlib.util.spec_from_file_location(Path(rel).stem, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_wilson_interval_known_values():
    bench = _load("benchmarks/run_benchmark.py")
    low, high = bench.wilson_interval(0, 5)
    assert low == 0.0 and high == pytest.approx(0.4345, abs=1e-4)
    low, high = bench.wilson_interval(5, 5)
    assert low == pytest.approx(0.5655, abs=1e-4) and high == 1.0
    low, high = bench.wilson_interval(10, 20)
    assert low < 0.5 < high and low + high == pytest.approx(1.0)
    with pytest.raises(ValueError):
        bench.wilson_interval(0, 0)


def test_benchmark_targets_cover_spec():
    bench = _load("benchmarks/run_benchmark.py")
    names = {t.name for t in bench.TARGETS}
    assert {"exp(x)", "ln(x)", "exp(exp(x))"} <= names
    assert 5 <= len(names) <= 6
    x = np.linspace(0.5, 2.0, 9)
    for t in bench.TARGETS:
        assert np.allclose(t.known_tree.evaluate(x), t.fn(x))


def test_completion_counts(tmp_path):
    comp = _load("scripts/completion.py")
    text = "- [x] R1 a\n- [ ] R2 b\n- [X] R3 c\nnot an item [x]\n"
    assert comp.count_items(text) == (2, 3, ["R2"])
    p = tmp_path / "req.md"
    p.write_text(text, encoding="utf-8")
    assert comp.main([str(p)]) == 0


def test_requirements_file_parses():
    comp = _load("scripts/completion.py")
    done, total, _ = comp.count_items((ROOT / "docs" / "REQUIREMENTS.md").read_text())
    assert total >= 40 and 0 <= done <= total
