import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from coursework_experiments import targets, task
from coursework_summary import paired, wilson


def test_reference_functions():
    x = np.linspace(0.5, 2.5, 100)
    for fn, tree, depth in targets().values():
        assert tree.depth() == depth
        np.testing.assert_allclose(tree.evaluate(x, safe=False), fn(x), atol=1e-9, rtol=0)


def test_episode_budgets_and_common_first_round(tmp_path):
    records = []
    for method in ("IR", "RF"):
        path = task(("log", method, 0.0, 19, 2, str(tmp_path)))
        with open(path, encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 3
        assert sum(int(r["epochs"]) for r in rows) == 6
        records.append(rows[0])
    for key in ("tree", "loss_before_snap", "max_error", "seed"):
        assert records[0][key] == records[1][key]


def test_intervals_and_pairing():
    lo, hi = wilson(0, 30)
    assert abs(lo) < 1e-12
    assert 0.11 < hi < 0.12
    assert paired([0, 1], [0, 1])["low"] is None
    assert paired([1] * 30, [0] * 30)["delta"] == 1


def test_ablation_budget(tmp_path):
    for method in ("RF0", "LONG"):
        path = task(("log", method, 0.0, 19, 2, str(tmp_path)))
        with open(path, encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        assert sum(int(r["epochs"]) for r in rows) == 6
