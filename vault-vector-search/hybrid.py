"""
Hybrid retrieval: dense preamble + body BM25, fused via Reciprocal Rank Fusion.

The two retrievers index different surfaces:
  - dense (ChromaDB):  title + "## For future Claude" preamble per page
  - BM25:              title + full body per page

This is the architectural payoff for keeping preambles short and curated:
the dense side captures semantic intent, while BM25 over bodies catches
proper-noun queries (people, model names, paper names) where dense
embeddings of niche terms degenerate to "what is X" generic matches.

`hybrid_search(query, top_k)` returns ranked unique pages.

CLI:
    python hybrid.py "what is OpenPiton" --top-k 5
"""
from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import chromadb
import numpy as np
import ollama

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_MODEL,
    HYBRID_FAN,
    W_BM25,
    W_DENSE,
)
from index_bm25 import BM25_FILE, tokenize_query

DEFAULT_FAN = HYBRID_FAN
DEFAULT_W_DENSE = W_DENSE
DEFAULT_W_BM25 = W_BM25


@dataclass
class HybridHit:
    rel_path: str
    title: str
    score: float          # weighted normalized combined score
    dense_rank: int | None
    bm25_rank: int | None
    excerpt: str


_BM25_CACHE: dict | None = None


def _load_bm25() -> dict:
    global _BM25_CACHE
    if _BM25_CACHE is None:
        with open(BM25_FILE, "rb") as f:
            _BM25_CACHE = pickle.load(f)
    return _BM25_CACHE


def _embed_query(query: str) -> list[float]:
    return ollama.embeddings(model=EMBED_MODEL, prompt=query)["embedding"]


def _dense_top(query: str, fan: int) -> list[dict]:
    coll = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(
        COLLECTION_NAME
    )
    res = coll.query(
        query_embeddings=[_embed_query(query)],
        n_results=fan,
        include=["metadatas", "documents", "distances"],
    )
    return [
        {
            "rel_path": m["rel_path"],
            "title": m["title"],
            "score": 1.0 - float(dist),
            "doc": doc,
        }
        for m, doc, dist in zip(
            res["metadatas"][0], res["documents"][0], res["distances"][0]
        )
    ]


def _bm25_top(query: str, fan: int) -> list[dict]:
    d = _load_bm25()
    bm25 = d["bm25"]
    metas = d["metas"]
    scores = bm25.get_scores(tokenize_query(query))
    idx = np.argsort(scores)[::-1][:fan]
    return [
        {
            "rel_path": metas[i]["rel_path"],
            "title": metas[i]["title"],
            "score": float(scores[i]),
            "doc": "",
        }
        for i in idx
    ]


def _minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo = min(scores.values())
    hi = max(scores.values())
    span = hi - lo
    if span <= 0:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / span for k, v in scores.items()}


def hybrid_search(
    query: str,
    top_k: int = 10,
    fan: int = DEFAULT_FAN,
    w_dense: float = DEFAULT_W_DENSE,
    w_bm25: float = DEFAULT_W_BM25,
) -> list[HybridHit]:
    """Hybrid search via weighted convex combination of min-max-normalized scores.

    Min-max normalization within each retriever's top-fan keeps the strongest
    signal in either retriever near 1.0, which preserves the high-confidence
    BM25-only top-1 cases (e.g., niche proper nouns the embedder doesn't know)
    that RRF with k=60 would otherwise dominate via dual-retriever co-occurrence.
    """
    dense_hits = _dense_top(query, fan)
    bm25_hits = _bm25_top(query, fan)

    dense_raw = {h["rel_path"]: h["score"] for h in dense_hits}
    bm25_raw = {h["rel_path"]: h["score"] for h in bm25_hits}

    dense_norm = _minmax(dense_raw)
    bm25_norm = _minmax(bm25_raw)

    dense_rank = {h["rel_path"]: r for r, h in enumerate(dense_hits, 1)}
    bm25_rank = {h["rel_path"]: r for r, h in enumerate(bm25_hits, 1)}

    title: dict[str, str] = {}
    excerpt: dict[str, str] = {}
    for h in dense_hits:
        title.setdefault(h["rel_path"], h["title"])
        excerpt.setdefault(h["rel_path"], h["doc"])
    for h in bm25_hits:
        title.setdefault(h["rel_path"], h["title"])
        excerpt.setdefault(h["rel_path"], "")

    paths = set(dense_norm) | set(bm25_norm)
    combined = {
        p: w_dense * dense_norm.get(p, 0.0) + w_bm25 * bm25_norm.get(p, 0.0)
        for p in paths
    }
    fused = sorted(combined.items(), key=lambda kv: -kv[1])[:top_k]
    return [
        HybridHit(
            rel_path=rp,
            title=title[rp],
            score=score,
            dense_rank=dense_rank.get(rp),
            bm25_rank=bm25_rank.get(rp),
            excerpt=excerpt[rp][:500].rstrip(),
        )
        for rp, score in fused
    ]


def format_hits(query: str, hits: list[HybridHit]) -> str:
    lines = [f"Hybrid top {len(hits)} for: {query!r}", ""]
    for i, h in enumerate(hits, 1):
        d = f"d#{h.dense_rank}" if h.dense_rank else "d#-"
        b = f"b#{h.bm25_rank}" if h.bm25_rank else "b#-"
        lines.append(f"{i}. [[{h.rel_path}]] — {h.title}")
        lines.append(f"   score={h.score:.4f}  ({d}, {b})")
        if h.excerpt:
            lines.append(f"   {h.excerpt[:200]}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--fan", type=int, default=DEFAULT_FAN)
    args = parser.parse_args()
    print(format_hits(args.query, hybrid_search(args.query, args.top_k, args.fan)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
