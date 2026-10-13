import numpy as np
import pytest

from eml_sr.core.tree import ONE, Eml, Leaf, X, from_rpn, to_rpn

XS = np.linspace(0.1, 10.0, 57)
LN_TREE = Eml(ONE, Eml(Eml(ONE, X), ONE))


def test_eml_1_1_is_e():
    assert np.allclose(Eml(ONE, ONE).evaluate(XS), np.e, rtol=1e-14, atol=0)


def test_eml_x_1_is_exp():
    assert np.allclose(Eml(X, ONE).evaluate(XS), np.exp(XS), rtol=1e-14, atol=0)


def test_ln_identity():
    assert np.allclose(LN_TREE.evaluate(XS), np.log(XS), rtol=1e-12, atol=1e-14)
    assert np.allclose(LN_TREE.evaluate(XS, safe=False), np.log(XS), rtol=1e-12, atol=1e-14)


def test_evaluate_complex128():
    assert LN_TREE.evaluate(XS).dtype == np.complex128


def test_rpn_ln_length_7():
    assert LN_TREE.to_rpn() == ["1", "1", "x", "E", "1", "E", "E"]
    assert len(to_rpn(LN_TREE)) == 7


@pytest.mark.parametrize(
    "tree",
    [ONE, X, Eml(X, ONE), LN_TREE, Eml(Eml(X, X), Eml(ONE, Eml(X, ONE)))],
)
def test_rpn_roundtrip(tree):
    assert from_rpn(to_rpn(tree)) == tree
    assert from_rpn(" ".join(tree.to_rpn())) == tree


def test_size_depth_leaves_str():
    assert LN_TREE.size() == 7
    assert LN_TREE.n_leaves() == 4
    assert LN_TREE.depth() == 3
    assert X.depth() == 0
    assert Eml(X, ONE).depth() == 1
    assert LN_TREE.to_str() == "eml(1, eml(eml(1, x), 1))"
    assert str(Eml(X, ONE)) == "eml(x, 1)"


@pytest.mark.parametrize("bad", ["", "E", "1 E", "1 x", "1 y E"])
def test_from_rpn_invalid(bad):
    with pytest.raises(ValueError):
        from_rpn(bad)


def test_leaf_kind_validation():
    with pytest.raises(ValueError):
        Leaf("two")  # type: ignore[arg-type]


def test_trees_hashable():
    assert len({Eml(X, ONE), Eml(X, ONE), LN_TREE}) == 2
