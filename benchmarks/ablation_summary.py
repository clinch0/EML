"""Summarize the fresh-seed ablation and audit actual budgets."""

import csv
import json
from collections import defaultdict
from pathlib import Path

from coursework_summary import paired, wilson

root = Path("benchmarks/results/ablation")
rows = list(csv.DictReader((root / "runs.csv").open()))
episodes = defaultdict(list)
for row in rows:
    episodes[(row["method"], int(row["episode_seed"]))].append(row)
methods = defaultdict(list)
for (method, seed), runs in episodes.items():
    assert sum(int(r["epochs"]) for r in runs) == 3000
    best = min(runs, key=lambda r: float(r["max_error"]))
    methods[method].append((seed, best["verified_success"] == "True"))
summary = []
for method, items in methods.items():
    items.sort()
    k = sum(x[1] for x in items)
    lo, hi = wilson(k, len(items))
    summary.append(dict(method=method, k=k, n=len(items), low=lo, high=hi))
comparisons = []
for a, b in [("RF", "IR"), ("RF", "RF0"), ("RF", "LONG"), ("RF0", "LONG")]:
    comparisons.append(
        dict(a=a, b=b, **paired([x[1] for x in methods[a]], [x[1] for x in methods[b]]))
    )
result = dict(
    summary=summary,
    comparisons=comparisons,
    rows=len(rows),
    episodes=len(episodes),
    total_epochs=sum(int(r["epochs"]) for r in rows),
)
(root / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
