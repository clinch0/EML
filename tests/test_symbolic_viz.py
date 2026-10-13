import numpy as np
import pytest
import sympy as sp

from eml_sr.core.tree import ONE, Eml, X
from eml_sr.search.library import X_SYMBOL, build_library
from eml_sr.symbolic.convert import convert, tree_to_sympy
from eml_sr.viz.plot import plot_fit, plot_tree

LN_TREE = Eml(ONE, Eml(Eml(ONE, X), ONE))
EXP_TREE = Eml(X, ONE)


@pytest.fixture(scope="module")
def library():
    return build_library()


def test_raw_translation():
    assert tree_to_sympy(EXP_TREE) == sp.exp(X_SYMBOL)
    assert tree_to_sympy(Eml(ONE, X)) == sp.E - sp.log(X_SYMBOL)


def test_exp_tree_readable(library):
    res = convert(EXP_TREE, library)
    assert res.readable
    assert res.expr == sp.exp(X_SYMBOL)
    assert res.text == "exp(x)"


def test_ln_tree_readable(library):
    res = convert(LN_TREE, library)
    assert res.readable
    assert res.expr == sp.log(X_SYMBOL)
    assert "ln(x)" in res.recognized
    assert res.text == "log(x)"


def test_identity_tree_readable(library):
    tree = Eml(ONE, Eml(Eml(ONE, Eml(X, ONE)), ONE))  # ln(exp(x))
    assert convert(tree, library).expr == X_SYMBOL


def test_x_minus_one_from_library(library):
    tree = library.tree_for("x - 1")
    assert tree is not None
    res = convert(tree, library)
    assert res.readable and sp.simplify(res.expr - (X_SYMBOL - 1)) == 0


def test_composite_expression_is_exact(library):
    tree = Eml(Eml(X, X), Eml(ONE, X))
    res = convert(tree, library)
    fn = sp.lambdify(X_SYMBOL, res.expr, "numpy")
    pts = np.linspace(0.5, 2.0, 7)
    assert np.allclose(fn(pts.astype(complex)), tree.evaluate(pts))


def test_unverifiable_tree_returns_raw_not_readable(library):
    tree = Eml(Eml(Eml(X, ONE), ONE), ONE)  # exp(exp(exp(x))) overflows at x = 4.4
    res = convert(tree, library)
    assert not res.readable
    assert res.expr == res.raw == tree_to_sympy(tree)


def test_custom_verification_points(library):
    tree = Eml(Eml(Eml(X, ONE), ONE), ONE)
    res = convert(tree, library, points=np.linspace(0.1, 1.0, 5))
    assert res.readable
    assert res.expr == sp.exp(sp.exp(sp.exp(X_SYMBOL)))


def test_unreadable_when_too_complex(library):
    tree = Eml(Eml(Eml(X, X), Eml(ONE, X)), Eml(X, Eml(X, Eml(ONE, X))))
    res = convert(tree, library, max_ops=1)
    assert not res.readable
    assert res.expr == res.raw


def test_plot_tree_saves_png(tmp_path):
    out = tmp_path / "tree.png"
    fig = plot_tree(LN_TREE, path=out)
    assert out.exists() and out.stat().st_size > 1000
    assert len(fig.axes) == 1


def test_plot_fit_saves_png(tmp_path):
    x = np.linspace(0.5, 3.0, 20)
    out = tmp_path / "fit.png"
    plot_fit(x, np.log(x), LN_TREE, label="log(x)", path=out)
    assert out.exists() and out.stat().st_size > 1000
