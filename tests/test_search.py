import numpy as np
import pytest

from eml_sr.core.tree import ONE, Eml, X
from eml_sr.search.enumerate import (
    DEFAULT_MAX_LEAVES,
    enumerate_trees,
    fingerprint_key,
    fingerprint_values,
)
from eml_sr.search.library import KNOWN_FUNCTIONS, Library, build_library


@pytest.fixture(scope="module")
def enum8():
    return enumerate_trees()


def test_default_max_leaves_is_8(enum8):
    assert DEFAULT_MAX_LEAVES == 8
    assert enum8.max_leaves == 8
    assert max(enum8.n_leaves) == 8


def test_level_one_and_ordering(enum8):
    assert enum8.by_leaves(1) == [ONE, X]
    assert enum8.n_leaves == sorted(enum8.n_leaves)


def test_finds_e_at_two_leaves(enum8):
    tree = enum8.find_function(lambda z: np.full_like(z, np.e))
    assert tree == Eml(ONE, ONE)
    assert tree.n_leaves() == 2


def test_finds_exp_at_two_leaves(enum8):
    tree = enum8.find_function(np.exp)
    assert tree == Eml(X, ONE)
    assert tree.n_leaves() == 2


def test_finds_ln_at_four_leaves(enum8):
    tree = enum8.find_function(np.log)
    assert tree is not None
    assert tree.n_leaves() == 4
    assert tree == Eml(ONE, Eml(Eml(ONE, X), ONE))


def test_ln_not_found_with_three_leaves():
    assert enumerate_trees(3).find_function(np.log) is None


def test_fingerprints_unique_and_finite(enum8):
    assert np.all(np.isfinite(enum8.values))
    keys = {fingerprint_key(v) for v in enum8.values}
    assert len(keys) == len(enum8.trees)


def test_shortest_tree_kept(enum8):
    # exp(x) = eml(x, 1) = eml(x, ln(e)); only the shortest one may be stored.
    ln_e = Eml(ONE, Eml(Eml(ONE, Eml(ONE, ONE)), ONE))
    longer = Eml(X, ln_e)
    assert np.allclose(fingerprint_values(longer), np.exp(enum8.points))
    assert longer not in set(enum8.trees)


def test_fingerprint_key_rejects_non_finite():
    assert fingerprint_key(np.array([1.0, np.inf])) is None
    assert fingerprint_key(np.array([1.0, np.nan])) is None


def test_invalid_max_leaves():
    with pytest.raises(ValueError):
        enumerate_trees(0)


@pytest.fixture(scope="module")
def library(enum8):
    return Library(enum8)


def test_library_known_entries(library):
    assert library.tree_for("x") == X
    assert library.tree_for("exp(x)") == Eml(X, ONE)
    assert library.tree_for("ln(x)") == Eml(ONE, Eml(Eml(ONE, X), ONE))
    assert library.tree_for("e") == Eml(ONE, ONE)
    for entry in library.entries.values():
        if entry.found:
            assert np.allclose(fingerprint_values(entry.tree), entry.function.values())


def test_library_honest_not_found():
    lib = build_library(max_leaves=3)
    assert lib.tree_for("ln(x)") is None
    assert not lib.entries["ln(x)"].found
    assert lib.tree_for("exp(x)") == Eml(X, ONE)


def test_library_contains_required_names():
    names = {f.name for f in KNOWN_FUNCTIONS}
    assert {"x", "exp(x)", "ln(x)", "x - 1", "x + 1", "2*x", "x^2"} <= names


def test_library_match_tree(library):
    assert library.match_tree(Eml(ONE, Eml(Eml(ONE, X), ONE))).name == "ln(x)"
    assert library.match_tree(Eml(X, ONE)).name == "exp(x)"
    assert library.match_tree(Eml(Eml(X, X), Eml(X, X))) is None
