# Decisions and assumptions

Decisions taken while implementing the specification, in the order they were made.

## Stage 0 — scaffold
- **D1. Build backend.** setuptools with a `src/` layout; the CLI is exposed as the
  `eml-sr` console script (`eml_sr.cli:main`).
- **D2. Ruff rules.** `E, F, W, I, UP, B, D, ANN, NPY` with the Google docstring convention.
  Tests are exempt from docstring/annotation rules (`D`, `ANN`) to keep them short.
- **D3. Git.** The repository already existed, so no `git init` was run; work is committed per
  stage on branch `cursor/eml-sr-implementation-520e`.
- **D4. Environment.** A local `.venv` with the CPU wheel of torch
  (`--index-url https://download.pytorch.org/whl/cpu`) is used for development.

## Stage 1 — core
- **D5. One function for both backends.** `safe_exp` / `safe_log` dispatch on the argument type:
  torch tensors use the torch path (autograd-friendly), everything else goes through NumPy.
  Both paths return complex128 and have identical semantics.
- **D6. `safe_exp` clamps only the real part** of the argument to `[-L, L]` (`L = 50`). The
  imaginary part does not affect the magnitude, so it is not clamped. In torch the real and
  imaginary parts are handled separately (`torch.clamp` does not accept complex tensors) and
  recombined with `torch.complex`, without in-place operations.
- **D7. `safe_log` default `eps = 1e-12`.** Entries with `|z| < eps` are replaced by the real
  number `eps`. Smaller values (e.g. `1e-30`) produce gradients of order `1/eps` that destabilise
  Adam; `1e-12` worked in the prototype. Both thresholds are function arguments.
- **D8. Tree representation.** Frozen dataclasses `Leaf(kind)` and `Eml(left, right)` so trees are
  immutable, hashable and comparable with `==`. RPN tokens: `1`, `x`, `E`.
- **D9. `size()` = total number of nodes = RPN length** (7 for `ln x`); `n_leaves()` counts leaves
  (used by enumeration); `depth()` is 0 for a leaf (so `exp(x) = eml(x, 1)` has depth 1 and
  `ln x` has depth 3).
- **D10. `Node.evaluate(x, safe=True)`.** Uses the guarded operations by default; `safe=False`
  uses plain `numpy.exp/log` so that enumeration can detect and drop trees producing `nan/inf`.

## Stage 2 — search
- **D11. Enumeration fingerprints use plain `exp`/`log`** (not the guarded versions), so overflow
  (`inf`) and `ln 0 = -inf` are visible and such trees are dropped, as the spec requires. With the
  guarded operations a clamped tree would get a fake finite fingerprint.
- **D12. Vectorised enumeration.** For each split `(i, n - i)` all pairs are evaluated at once
  with NumPy broadcasting from the cached fingerprints of the children; only unique trees are
  materialised. Ties are broken deterministically by split index, then row-major order.
- **D13. Fingerprint key** = real and imaginary parts rounded to 9 decimals (`+ 0.0` to merge
  `-0.0` with `0.0`). Library/target **matching** uses `numpy.isclose` with
  `rtol = atol = 1e-9` instead of exact keys, so a value near a rounding boundary still matches.
- **D14. Library contents.** `x, e, 0, -1, 2, exp(x), ln(x), x - 1, x + 1, 2*x, x^2, -x, 1/x,
  sqrt(x), exp(x) - 1, exp(exp(x)), e - ln(x), exp(x) - ln(x)`. With the default 8 leaves the
  following are **not found** (reported as `tree is None`): `-1, 2, x + 1, 2*x, x^2, -x, 1/x,
  sqrt(x)`. They are kept in the list so the "not found" path is exercised honestly; they need
  more leaves (and some known constructions rely on `ln 0 = -inf`, which is unsupported).

## Stage 3 — model
- **D15. Level-wise vectorised master tree.** `levels[k]` holds the logits of level `k` with shape
  `(2^k, 2, c)` (`c = 2` on the bottom level, `3` elsewhere); node `j` of level `k` feeds from
  children `2j` (left input) and `2j+1` (right input). One softmax per input. Logit count
  `5*2^d - 6` is checked for `d = 1..5`.
- **D16. `forward(x, tau=None)`** uses the attribute `tau` set by the trainer, so `forward(x)` has
  the signature from the spec. `to_discrete()` returns a `Node`; `forward_discrete()` evaluates
  the same one-hot snapping in torch.
- **D17. Private RNG per run.** Each run creates `torch.Generator().manual_seed(seed)`; no global
  `torch.manual_seed` is called (no global state).
- **D18. Default hyper-parameters** (chosen from a small prototype sweep, not tuned per target):
  `epochs = 2000` (half soft, half hardening), `lr = 0.05`, `tau0 = 1`, `tau_min = 0.01`,
  `sigma = 1`, `entropy_weight = 0` (the penalty is implemented but did not help in the prototype
  on `ln x` at depth 3: 1/20 successes with and without it), `tol = 1e-8`.
- **D19. Success / "best" criterion.** Success = `max |tree(x) - y| < tol` for the snapped tree,
  evaluated with the guarded operations. When no run succeeds, the run with the smallest max error
  (ties: smallest snapped MSE) is returned.
- **D20. Non-finite loss** stops the run early; the epochs actually done are recorded and the
  current logits are snapped. Nothing is replaced by zeros.
- **D21. `ln x` at master depth 2 is impossible by construction.** A depth-2 master tree has 3 eml
  nodes and its snapped trees have depth <= 2, whereas the shortest `ln x` tree
  `eml(1, eml(eml(1, x), 1))` has depth 3. An exhaustive test over all `3^2 * 2^4 = 144` snapped
  depth-2 configurations confirms that none equals `ln x`. The spec's test "depth-2 training on
  `ln x` succeeds" therefore cannot be written honestly; it is replaced by a depth-3 test and the
  requirement R24 is left unchecked.

## Stage 4 — symbolic and viz
- **D22. Top-down recognition.** The largest subtree matching a library fingerprint wins; its
  children are not inspected further. Leaves are already readable.
- **D23. Verification instead of trust.** After `sympy.simplify` the expression is evaluated with
  `lambdify` and compared with the tree (`rtol = 1e-8`, `atol = 1e-10`) on the verification points
  (default: fingerprint points; the API passes the training `x`). If the comparison fails or the
  tree is not finite there, the raw translation is returned with `readable=False`. This guards
  against branch-cut rewrites by sympy and against wrong recognitions.
- **D24. Readability criterion.** `readable = verified and (whole tree recognised or
  count_ops(simplified) <= 12)`. The threshold `max_ops = 12` is a parameter. The returned
  expression is always an exact rewrite of the tree, never a fitted guess.
- **D25. `x` is a positive real sympy symbol**, matching the supported domain `x > 0`, which lets
  sympy apply `log(exp(x)) = x` and `exp(a - log(x)) = exp(a)/x` legitimately.
- **D26. Plots use `matplotlib.figure.Figure` directly** (no pyplot), so there is no global backend
  state and no GUI requirement; graphviz is not used.

## Stage 5 — API, CLI, benchmark, docs
- **D27. Extra estimator attributes.** Besides `formula_`, `tree_`, `success_`, `runs_` the
  estimator exposes `formula_expr_` (sympy), `readable_` and `max_error_`, because they are
  produced anyway and are needed by the CLI summary. Extra constructor arguments: `lr`,
  `entropy_weight`, `tol`, `library_leaves`.
- **D28. `predict` returns the real part** of the complex tree value (targets are real).
  `score` is the usual R^2; for constant `y` it returns 1.0 for a perfect fit and 0.0 otherwise.
- **D29. Library is not cached globally.** `EMLRegressor.fit` builds it once per call (~0.2 s for
  8 leaves) to avoid module-level mutable state.
- **D30. CSV format.** Two numeric columns `x, y`; a non-numeric first row is treated as a header;
  more columns -> `NotImplementedError` message and exit code 2. Exit code 1 means "fitted but not
  exact" (outputs are still written).
- **D31. Benchmark measures single runs.** Each of the `N` runs per (target, depth) is one
  `train_once` call (seed `seed + i`, no multistart), so the reported rate is the per-run success
  probability; multistart success is `1 - (1 - p)^k`. `--quick` uses `N = 5`, default `N = 30`.
  Data: 40 points on `[0.5, 2]` (keeps `exp(exp(x))` far from the clamp `L = 50`). Additional
  targets: `e - ln(x)`, `exp(x) - 1`, `exp(x) - ln(x)`. The column `representable` comes from the
  depth of each target's known tree (`ln(x)` is not representable at depth 2, see D21).
- **D32. Benchmark results are committed** (`benchmarks/results/quick.csv`) and quoted in the
  README exactly as measured; example figures (`examples/output/`) are generated, not committed.
- **D33. `SOFTWARE_INFO.md` commit hash.** A commit cannot contain its own hash, so the document
  records the last code commit before the release-docs commit and explains how to resolve the tag.
