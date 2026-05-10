"""Compare snapshotted eval runs and print aggregate / per-category /
per-question diffs. Each snapshot represents a different pipeline state
(see ../REPRODUCE.md):

  results-baseline.json     — pre-fix, pre-retrofit
  results-bold.json         — gold-list fixes + lead-with-term retrofit (bold)
  results-pre-hybrid.json   — bold stripped (no-bold canonical for 2-strategy era)
  results-hybrid.json       — + hybrid retrieval (BM25 + dense, weighted score fusion)
  results-rerank.json       — + cross-encoder rerank (canonical, == results.json)

Strategies in each run vary: early runs only have preamble + chunk; later
runs add hybrid and hybrid_rerank. The script handles missing strategies
gracefully — only metrics that exist in all loaded runs are diffed.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent

SNAPSHOTS = [
    ("baseline",    "results-baseline.json"),
    ("bold",        "results-bold.json"),
    ("no-bold",     "results-pre-hybrid.json"),
    ("hybrid",      "results-hybrid.json"),
    ("rerank",      "results-rerank.json"),
]

ALL_STRATEGIES = ("preamble", "chunk", "hybrid", "hybrid_rerank")


def load_all() -> dict[str, dict]:
    out = {}
    for name, fn in SNAPSHOTS:
        p = HERE / fn
        if p.exists():
            out[name] = json.loads(p.read_text())
    return out


def has_strategy(run: dict, strat: str) -> bool:
    return strat in run["aggregate"]


def main() -> int:
    runs = load_all()
    if not runs:
        raise SystemExit("no result snapshots found")
    snap_names = list(runs.keys())

    # === AGGREGATE ===
    print(f"=== AGGREGATE — {' / '.join(snap_names)} ===\n")
    print(f"{'strategy':<14} {'metric':<14}", end="")
    for n in snap_names:
        print(f" {n:>12}", end="")
    print()

    metrics = ("recall@1", "recall@3", "recall@5", "mrr", "avg_top5_tokens")
    for strat in ALL_STRATEGIES:
        any_has = any(has_strategy(runs[n], strat) for n in snap_names)
        if not any_has:
            continue
        for k in metrics:
            print(f"{strat:<14} {k:<14}", end="")
            for n in snap_names:
                if has_strategy(runs[n], strat):
                    v = runs[n]["aggregate"][strat][k]
                    fmt = "{:>12,.3f}" if k != "avg_top5_tokens" else "{:>12,.0f}"
                    print(f" {fmt.format(v)}", end="")
                else:
                    print(f" {'—':>12}", end="")
            print()
        print()

    # === BY CATEGORY (recall@5) ===
    print(f"=== BY CATEGORY (recall@5) ===\n")
    cats = ("fact", "cross-domain", "staleness", "synthesis")
    for cat in cats:
        print(f"  {cat}")
        for strat in ALL_STRATEGIES:
            any_has = any(
                has_strategy(runs[n], strat) and
                cat in runs[n]["aggregate"].get("by_category", {})
                for n in snap_names
            )
            if not any_has:
                continue
            print(f"    {strat:<14}", end="")
            for n in snap_names:
                ag = runs[n]["aggregate"]
                if strat in ag and cat in ag.get("by_category", {}):
                    v = ag["by_category"][cat][strat]["recall@5"]
                    print(f"  {n}: {v:.3f}", end="")
                else:
                    print(f"  {n}: —    ", end="")
            print()
        print()

    # === HEADLINE: canonical aggregate ===
    canonical_name = snap_names[-1]
    canonical = runs[canonical_name]["aggregate"]
    print(f"=== CANONICAL ({canonical_name}) — full aggregate ===\n")
    print(f"{'strategy':<14} {'r@1':>6} {'r@3':>6} {'r@5':>6} {'MRR':>6} {'top5_tok':>10}")
    for strat in ALL_STRATEGIES:
        if strat in canonical:
            a = canonical[strat]
            print(f"{strat:<14} {a['recall@1']:>6.3f} {a['recall@3']:>6.3f} "
                  f"{a['recall@5']:>6.3f} {a['mrr']:>6.3f} {a['avg_top5_tokens']:>10,.0f}")
    naive = canonical.get("naive_per_query_tokens")
    if naive:
        print(f"{'naive':<14} {'1.000':>6} {'1.000':>6} {'1.000':>6} {'1.000':>6} "
              f"{naive:>10,}  (full-corpus, per query)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
