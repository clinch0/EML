# Architecture

## Modules

| Module | Responsibility |
| --- | --- |
| `core/safe_ops.py` | `safe_exp` (real part of the argument clamped to `[-L, L]`, `L = 50`) and `safe_log` (`|z| < eps` -> `eps`, principal branch). One function per op for NumPy and torch (autograd-safe, no in-place ops). |
| `core/tree.py` | Immutable trees `Leaf("one" \| "x")`, `Eml(left, right)`: `evaluate` (complex128, guarded or plain ops), `size`, `n_leaves`, `depth`, `to_rpn`, `from_rpn`, `to_str`. |
| `search/enumerate.py` | Bottom-up enumeration by number of leaves; fingerprints at `x = 0.5, 1.3, 2.1, 3.7, 4.4` rounded to `1e-9`; trees with `nan/inf` dropped; the shortest tree per fingerprint kept. |
| `search/library.py` | Known functions (name, NumPy function, sympy expression) matched to their shortest enumerated tree; lookup of a tree/values by fingerprint; honest "not found". |
| `model/master_tree.py` | `MasterTree(depth)`: `2^d - 1` soft eml nodes, inputs `softmax(logits / tau)` over `{1, x}` (bottom) or `{1, x, child}`; `5*2^d - 6` logits; `forward`, `forward_discrete`, `to_discrete`. |
| `model/trainer.py` | `TrainConfig`, `train_once` (soft phase -> tau annealing / entropy penalty -> snapping -> check `max error < tol`), `fit_multistart` (seeds `seed, seed+1, ...`, stop at first success), `RunResult`. |
| `symbolic/convert.py` | Tree -> sympy (`exp(a) - log(b)`), library recognition of subtrees, `sympy.simplify`, numeric verification, `readable` flag. |
| `viz/plot.py` | `plot_tree` (matplotlib, no graphviz) and `plot_fit` (data vs formula). |
| `api.py` | `EMLRegressor` (`fit`, `predict`, `score`, `formula_`, `tree_`, `success_`, `runs_`, ...). |
| `cli.py` | `eml-sr fit data.csv --depth 3 --restarts 50 --out result/`. |

## Data flow

```
(x, y) ──► EMLRegressor.fit
             │
             ├─► trainer.fit_multistart ──► train_once × k
             │        MasterTree.forward (safe_exp/safe_log, softmax(logits/tau))
             │        Adam on mean|pred - y|^2, tau: tau0 -> tau_min
             │        MasterTree.to_discrete() ──► Node tree ──► max error < tol ?
             │
             ├─► best RunResult.tree  ──► tree_
             │
             └─► symbolic.convert(tree, Library)
                      tree_to_sympy ─► recognise subtrees (search.library fingerprints,
                      built from search.enumerate) ─► simplify ─► verify on x ─► formula_, readable_

viz.plot_tree(tree_), viz.plot_fit(x, y, tree_)  ◄── cli.py / examples
```

`benchmarks/run_benchmark.py` calls `train_once` directly (single runs, no multistart) to measure
per-run success rates.
