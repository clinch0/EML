import json
import subprocess
import sys

import numpy as np
import pytest
import sympy as sp

from eml_sr.api import EMLRegressor
from eml_sr.cli import load_csv, main
from eml_sr.search.library import X_SYMBOL

XS = np.linspace(0.5, 2.5, 25)


@pytest.fixture(scope="module")
def fitted_exp():
    return EMLRegressor(depth=2, n_restarts=10, seed=0).fit(XS, np.exp(XS))


def test_fit_exp(fitted_exp):
    m = fitted_exp
    assert m.success_
    assert m.readable_
    assert m.formula_ == "exp(x)"
    assert m.formula_expr_ == sp.exp(X_SYMBOL)
    assert m.tree_.to_str() == "eml(x, 1)"
    assert m.max_error_ < 1e-8
    assert len(m.runs_) >= 1 and m.runs_[-1].success


def test_predict_and_score(fitted_exp):
    pred = fitted_exp.predict(XS)
    assert pred.dtype == np.float64
    assert np.allclose(pred, np.exp(XS))
    assert fitted_exp.score(XS, np.exp(XS)) == pytest.approx(1.0)
    assert fitted_exp.score(XS, np.exp(XS) + 1.0) < 1.0


def test_column_vector_input():
    m = EMLRegressor(depth=2, n_restarts=10, seed=0).fit(XS.reshape(-1, 1), np.exp(XS))
    assert m.success_
    assert m.predict(XS.reshape(-1, 1)).shape == (len(XS),)


def test_multivariate_not_implemented():
    with pytest.raises(NotImplementedError, match="one input variable"):
        EMLRegressor(depth=2).fit(np.ones((5, 2)), np.ones(5))


def test_shape_mismatch_and_unfitted():
    with pytest.raises(ValueError):
        EMLRegressor(depth=2).fit(XS, XS[:-1])
    with pytest.raises(RuntimeError):
        EMLRegressor().predict(XS)


def test_default_hyperparameters():
    m = EMLRegressor()
    assert (m.depth, m.n_restarts, m.seed) == (3, 50, 0)


def _write_csv(path, x, y, header=True):
    rows = (["x,y"] if header else []) + [f"{a:.17g},{b:.17g}" for a, b in zip(x, y, strict=True)]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_load_csv(tmp_path):
    p = tmp_path / "d.csv"
    _write_csv(p, XS, np.exp(XS), header=False)
    x, y = load_csv(p)
    assert np.allclose(x, XS) and np.allclose(y, np.exp(XS))
    p3 = tmp_path / "d3.csv"
    p3.write_text("a,b,y\n1,2,3\n", encoding="utf-8")
    with pytest.raises(NotImplementedError):
        load_csv(p3)


def test_cli_fit(tmp_path, capsys):
    data = tmp_path / "data.csv"
    _write_csv(data, XS, np.exp(XS))
    out = tmp_path / "result"
    code = main(["fit", str(data), "--depth", "2", "--restarts", "10", "--out", str(out)])
    assert code == 0
    printed = capsys.readouterr().out
    assert "formula: exp(x)" in printed
    assert (out / "tree.png").stat().st_size > 1000
    assert (out / "fit.png").stat().st_size > 1000
    summary = json.loads((out / "result.json").read_text())
    assert summary["success"] and summary["formula"] == "exp(x)"


def test_cli_multivariate_error(tmp_path, capsys):
    data = tmp_path / "data.csv"
    data.write_text("x1,x2,y\n1,2,3\n2,3,4\n", encoding="utf-8")
    assert main(["fit", str(data), "--out", str(tmp_path / "r")]) == 2
    assert "only one input column" in capsys.readouterr().err


def test_cli_module_entry(tmp_path):
    data = tmp_path / "data.csv"
    _write_csv(data, XS, np.exp(XS))
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "eml_sr.cli",
            "fit",
            str(data),
            "--depth",
            "2",
            "--restarts",
            "10",
            "--out",
            str(tmp_path / "r"),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    assert "formula: exp(x)" in proc.stdout
