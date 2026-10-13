# Requirements checklist

`[x]` is set only for requirements that are implemented **and** verified (by a test, a lint run
or an actual command run). `scripts/completion.py` computes the completion percentage from the
lines of this file that match `- [x] Rn` / `- [ ] Rn`.

## Infrastructure
- [x] R1 `pyproject.toml`, `src/eml_sr/` layout, Python >= 3.10, deps numpy/torch/sympy/matplotlib, dev deps pytest/ruff.
- [x] R2 `ruff check` reports no issues.
- [x] R3 `pytest` passes.
- [x] R4 Git commit after each stage, tag `v0.1.0`.
- [x] R5 Type hints, English docstrings, fixed seeds, no global mutable state.

## Safe operations (core/safe_ops.py)
- [x] R6 `safe_exp`: real part of the exponent clamped to `[-L, L]`, `L = 50` by default, configurable.
- [x] R7 `safe_log`: `|z| < eps` replaced by `eps`, configurable, behaviour documented.
- [x] R8 Torch versions clamp real/imaginary parts separately, no in-place ops, autograd works, NaN is not silently replaced by 0.
- [x] R9 Formulas requiring `±inf` (e.g. `ln 0`) are documented as unsupported.
- [x] R10 Test: `safe_exp` / `safe_log` give no NaN on very large and very small inputs.

## Tree (core/tree.py)
- [x] R11 `Node`: `Leaf(one | x)` and `Eml(left, right)`; `evaluate`, `size`, `depth`, `to_rpn`, `from_rpn`, `to_str`; complex128, principal log branch.
- [x] R12 Test: identities `eml(1,1)=e`, `eml(x,1)=exp(x)`, `ln(x)=eml(1,eml(eml(1,x),1))` for `x > 0`.
- [x] R13 Test: `from_rpn(to_rpn(t)) == t`; RPN of `ln(x)` has length 7.

## Search (search/)
- [x] R14 Bottom-up enumeration by number of leaves with fingerprint de-duplication (points 0.5, 1.3, 2.1, 3.7, 4.4, rounding 1e-9), nan/inf trees dropped, shortest tree kept, max leaves 8 by default (configurable).
- [x] R15 Test: enumeration finds `e` at n=2, `exp(x)` at n=2, `ln(x)` at n=4.
- [x] R16 Library of known functions (name -> function), fingerprint matching against enumeration, honest "not found" beyond N leaves.

## Master tree (model/master_tree.py)
- [x] R17 Depth-d full binary tree of `2^d - 1` eml nodes; bottom inputs `a*1 + b*x`, other inputs `a*1 + b*x + c*f`; weights `softmax(logits / tau)`; complex128.
- [x] R18 Test: number of logits equals `5*2^d - 6` (14 for d=2, 34 for d=3).
- [x] R19 `forward(x)` and `to_discrete()` (argmax -> one-hot -> Node tree).

## Training (model/trainer.py)
- [x] R20 Loss `mean |pred - y|^2`, Adam with configurable lr, init noise std `sigma`.
- [x] R21 Phases: soft training, hardening (tau annealing tau0 -> tau_min and/or entropy penalty), snapping + verification; success = max error of snapped tree < tol (1e-8).
- [x] R22 Multistart: seeds seed, seed+1, ...; stop at first success, otherwise best by error; `RunResult(seed, success, loss_before_snap, loss_after_snap, epochs, seconds)`.
- [x] R23 Test: depth-2 training with multistart succeeds on `exp(x)` (fixed seed).
- [ ] R24 Test: depth-2 training with multistart succeeds on `ln(x)` (fixed seed). **Not achievable:** a depth-2 master tree only represents trees of depth <= 2, while `ln x` needs depth 3; `tests/test_model.py::test_depth2_cannot_represent_ln` checks all 144 snapped depth-2 configurations. Replaced by R25 (depth 3).
- [x] R25 Test: training with multistart succeeds on `ln(x)` at the smallest master depth that can represent it.

## Symbolic and visualisation
- [x] R26 Tree -> sympy (`eml(a,b) = exp(a) - log(b)`), sympy simplification.
- [x] R27 Subtree recognition via library fingerprints; raw expression + `readable=False` when no readable form is obtained.
- [x] R28 Test: exp-tree and ln-tree are converted to a readable form.
- [x] R29 Matplotlib tree drawing (eml nodes, leaves 1 and x), no graphviz dependency.
- [x] R30 Plot "data vs formula".

## API and CLI
- [x] R31 `EMLRegressor(depth=3, n_restarts=50, epochs, tau0, tau_min, sigma, seed=0)`; `fit` accepts X of shape (n,) or (n,1), raises `NotImplementedError` for several variables.
- [x] R32 `predict`, `score` (R^2), attributes `formula_`, `tree_`, `success_`, `runs_`.
- [x] R33 CLI `eml-sr fit data.csv --depth 3 --restarts 50 --out result/` prints the formula and saves PNGs.
- [x] R34 Tests for API and CLI on a small example.

## Benchmark, examples, docs
- [x] R35 `benchmarks/run_benchmark.py`: exp(x), ln(x), exp(exp(x)) + 2-3 others, depths 2 and 3, N runs each, Wilson CI, CSV output, `--quick`, no pre-written results.
- [x] R36 `examples/`: recovery of exp(x) and ln(x) saving figures.
- [x] R37 README in Russian and English: install, example, limitations, tests, benchmark.
- [x] R38 `docs/SOFTWARE_INFO.md`: repository URL field, branch/tag/commit hash, requirements, build/run/test commands.
- [x] R39 `docs/ARCHITECTURE.md`: modules and data flow.
- [x] R40 `docs/DECISIONS.md` with the decisions taken.
- [x] R41 `scripts/completion.py` computes the completion percentage from this file.
