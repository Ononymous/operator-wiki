"""
Retrieval-strategy eval harness for the operator-wiki vault.

Runs four retrieval strategies on the gold QA set, plus a naive token-cost
baseline:

  1. naive          — read every indexable page in full (token-cost baseline)
  2. preamble       — dense vector retrieval over (title + preamble) only
  3. chunk          — dense vector retrieval over ~1500-char body chunks
  4. hybrid         — preamble dense + body BM25, weighted score fusion
  5. hybrid_rerank  — strategy 4 candidates rescored by a cross-encoder

Strategies 2 and 3 share the same model and ChromaDB pipeline; they differ
only in what gets embedded. Strategy 4 adds BM25 (proper-noun robustness).
Strategy 5 adds cross-encoder rerank (synthesis-query lift, staleness regression).

For each question, computes recall@1/3/5 and MRR against gold pages, plus the
input-token cost of consuming the top-5 retrieved documents. The naive baseline
contributes a single per-query corpus-read cost for the amortization comparison.

Output: results.json next to this file, with per-question detail and aggregate
metrics.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import chromadb
import ollama
import tiktoken

HERE = Path(__file__).parent
SEARCH_DIR = HERE.parent / "vault-vector-search"
sys.path.insert(0, str(SEARCH_DIR))
from config import (  # noqa: E402
    CHROMA_DIR as PREAMBLE_CHROMA,
    COLLECTION_NAME as PREAMBLE_COLLECTION,
    EMBED_MODEL,
    VAULT_ROOT,
)
from index import collect_pages  # noqa: E402
from hybrid import hybrid_search  # noqa: E402
from rerank import hybrid_rerank_search  # noqa: E402

CHUNK_CHROMA = HERE / "chroma_chunks"
CHUNK_COLLECTION = "wiki_chunks"
GOLD_PATH = HERE / "gold-qa.jsonl"
RESULTS_PATH = HERE / "results.json"

TOP_K = 10           # depth retrieved for recall scoring
CHUNK_OVERSAMPLE = 30  # chunks fetched before dedup-to-pages
TOKEN_ENC = "cl100k_base"


def embed(text: str) -> list[float]:
    return ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]


def retrieve_preamble(query: str, top_k: int = TOP_K) -> list[dict]:
    coll = chromadb.PersistentClient(path=str(PREAMBLE_CHROMA)).get_collection(
        PREAMBLE_COLLECTION
    )
    res = coll.query(
        query_embeddings=[embed(query)],
        n_results=top_k,
        include=["metadatas", "documents", "distances"],
    )
    out = []
    for m, doc, dist in zip(res["metadatas"][0], res["documents"][0], res["distances"][0]):
        out.append({"rel_path": m["rel_path"], "score": 1.0 - float(dist), "doc": doc})
    return out


def retrieve_chunk(query: str, top_k: int = TOP_K) -> list[dict]:
    coll = chromadb.PersistentClient(path=str(CHUNK_CHROMA)).get_collection(
        CHUNK_COLLECTION
    )
    res = coll.query(
        query_embeddings=[embed(query)],
        n_results=CHUNK_OVERSAMPLE,
        include=["metadatas", "documents", "distances"],
    )
    seen: dict[str, dict] = {}
    for m, doc, dist in zip(res["metadatas"][0], res["documents"][0], res["distances"][0]):
        rp = m["rel_path"]
        score = 1.0 - float(dist)
        # Take the highest-scoring chunk per page
        if rp not in seen or seen[rp]["score"] < score:
            seen[rp] = {"rel_path": rp, "score": score, "doc": doc}
    ranked = sorted(seen.values(), key=lambda x: -x["score"])[:top_k]
    return ranked


def retrieve_hybrid(query: str, top_k: int = TOP_K) -> list[dict]:
    hits = hybrid_search(query, top_k=top_k)
    return [{"rel_path": h.rel_path, "score": h.score, "doc": h.excerpt}
            for h in hits]


def retrieve_hybrid_rerank(query: str, top_k: int = TOP_K) -> list[dict]:
    hits = hybrid_rerank_search(query, top_k=top_k)
    return [{"rel_path": h.rel_path, "score": h.rerank_score, "doc": h.excerpt}
            for h in hits]


_PREAMBLE_DOC_CACHE: dict[str, str] | None = None


def _preamble_doc(rel_path: str) -> str:
    """Look up the preamble-collection document for a given rel_path.
    This is what each strategy is taken to "surface" for downstream LLM triage,
    making token cost directly comparable across retrieval strategies."""
    global _PREAMBLE_DOC_CACHE
    if _PREAMBLE_DOC_CACHE is None:
        coll = chromadb.PersistentClient(path=str(PREAMBLE_CHROMA)).get_collection(
            PREAMBLE_COLLECTION
        )
        got = coll.get(include=["documents"])
        _PREAMBLE_DOC_CACHE = dict(zip(got["ids"], got["documents"]))
    return _PREAMBLE_DOC_CACHE.get(rel_path, "")


def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(retrieved[:k]) & set(gold)) / len(set(gold))


def mrr(retrieved: list[str], gold: list[str]) -> float:
    gold_set = set(gold)
    for i, p in enumerate(retrieved, 1):
        if p in gold_set:
            return 1.0 / i
    return 0.0


def naive_per_query_tokens(enc) -> int:
    total = 0
    for p in collect_pages():
        full = (VAULT_ROOT / p["rel_path"]).read_text(encoding="utf-8")
        total += len(enc.encode(full))
    return total


def main() -> int:
    enc = tiktoken.get_encoding(TOKEN_ENC)

    qa = []
    with open(GOLD_PATH) as f:
        for line in f:
            qa.append(json.loads(line))
    print(f"[eval] {len(qa)} questions")

    print("[eval] computing naive per-query token cost (full-corpus read)…")
    t0 = time.perf_counter()
    naive_cost = naive_per_query_tokens(enc)
    print(f"[eval] naive per-query: {naive_cost:,} tokens "
          f"({time.perf_counter() - t0:.1f}s)")

    results: dict = {"top_k": TOP_K, "questions": [], "aggregate": {}}

    for q in qa:
        per_q = {
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "gold": q["gold"],
        }
        for name, fn in (
            ("preamble", retrieve_preamble),
            ("chunk", retrieve_chunk),
            ("hybrid", retrieve_hybrid),
            ("hybrid_rerank", retrieve_hybrid_rerank),
        ):
            hits = fn(q["question"])
            paths = [h["rel_path"] for h in hits]
            # Uniform top-5 cost: tokens of the preamble of each surfaced
            # page. Same metric across strategies — "what the consumer reads
            # for triage at top-5."
            top5_preamble_tokens = sum(
                len(enc.encode(_preamble_doc(p))) for p in paths[:5]
            )
            # Strategy-native cost: tokens of whatever document the strategy
            # itself returned (preamble for preamble-strategy, chunk for
            # chunk-strategy, mixed for hybrid). Kept for reference.
            top5_native_tokens = sum(len(enc.encode(h["doc"])) for h in hits[:5])
            per_q[name] = {
                "top10": paths,
                "scores": [round(h["score"], 4) for h in hits],
                "recall@1": recall_at_k(paths, q["gold"], 1),
                "recall@3": recall_at_k(paths, q["gold"], 3),
                "recall@5": recall_at_k(paths, q["gold"], 5),
                "mrr": mrr(paths, q["gold"]),
                "top5_tokens": top5_preamble_tokens,
                "top5_native_tokens": top5_native_tokens,
            }
        results["questions"].append(per_q)
        # Per-question summary line for the eval log
        p = per_q["preamble"]; c = per_q["chunk"]
        h = per_q["hybrid"]; r = per_q["hybrid_rerank"]
        print(f"  {q['id']} [{q['category']:11}] "
              f"P r@5={p['recall@5']:.2f}  "
              f"C r@5={c['recall@5']:.2f}  "
              f"H r@5={h['recall@5']:.2f} mrr={h['mrr']:.2f}  "
              f"R r@5={r['recall@5']:.2f} mrr={r['mrr']:.2f}")

    # Aggregate
    for strat in ("preamble", "chunk", "hybrid", "hybrid_rerank"):
        n = len(results["questions"])
        agg = {}
        for k in (1, 3, 5):
            agg[f"recall@{k}"] = round(
                sum(q[strat][f"recall@{k}"] for q in results["questions"]) / n, 4
            )
        agg["mrr"] = round(
            sum(q[strat]["mrr"] for q in results["questions"]) / n, 4
        )
        agg["avg_top5_tokens"] = round(
            sum(q[strat]["top5_tokens"] for q in results["questions"]) / n, 1
        )
        results["aggregate"][strat] = agg

    # By-category breakdown — useful for the report
    by_cat: dict[str, dict[str, list]] = {}
    for q in results["questions"]:
        by_cat.setdefault(q["category"], {
            "preamble": [], "chunk": [], "hybrid": [], "hybrid_rerank": [],
        })
        for strat in ("preamble", "chunk", "hybrid", "hybrid_rerank"):
            by_cat[q["category"]][strat].append(q[strat])
    cat_agg = {}
    for cat, strats in by_cat.items():
        cat_agg[cat] = {}
        for strat, lst in strats.items():
            cat_agg[cat][strat] = {
                "n": len(lst),
                "recall@1": round(sum(x["recall@1"] for x in lst) / len(lst), 3),
                "recall@5": round(sum(x["recall@5"] for x in lst) / len(lst), 3),
                "mrr": round(sum(x["mrr"] for x in lst) / len(lst), 3),
            }
    results["aggregate"]["by_category"] = cat_agg
    results["aggregate"]["naive_per_query_tokens"] = naive_cost

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n[eval] wrote {RESULTS_PATH}")

    # Summary table
    print("\n=== AGGREGATE ===")
    print(f"{'strategy':<14} {'r@1':>6} {'r@3':>6} {'r@5':>6} {'MRR':>6} {'top5_tok':>10}")
    for strat in ("preamble", "chunk", "hybrid", "hybrid_rerank"):
        a = results["aggregate"][strat]
        print(f"{strat:<14} {a['recall@1']:>6.3f} {a['recall@3']:>6.3f} "
              f"{a['recall@5']:>6.3f} {a['mrr']:>6.3f} {a['avg_top5_tokens']:>10,.0f}")
    print(f"{'naive':<14} {'1.000':>6} {'1.000':>6} {'1.000':>6} {'1.000':>6} "
          f"{naive_cost:>10,}  (full-corpus, per query)")

    print("\n=== BY CATEGORY (recall@5) ===")
    for cat, strats in cat_agg.items():
        p = strats["preamble"]; c = strats["chunk"]
        h = strats["hybrid"]; r = strats["hybrid_rerank"]
        print(f"  {cat:<13} (n={p['n']:2})  P {p['recall@5']:.3f}   "
              f"C {c['recall@5']:.3f}   H {h['recall@5']:.3f}   R {r['recall@5']:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
