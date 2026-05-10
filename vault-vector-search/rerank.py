"""
Cross-encoder rerank on top of hybrid retrieval.

Independent of `hybrid.py` and `index.py` — the only entry point used here is
`hybrid.hybrid_search`. This is a standalone trial: deletable as a unit if
the rerank step doesn't pay off.

Pipeline:
    query
      ├──→ hybrid_search(query, top_k=30)        (BM25 + dense, weighted RRF-equivalent)
      ├──→ fetch each candidate's preamble doc   (from the dense ChromaDB collection)
      ├──→ cross_encoder.predict(query, doc)     (one forward pass per candidate)
      └──→ sort by rerank score → take top-K

Defaults to `cross-encoder/ms-marco-MiniLM-L-6-v2` — small (~80 MB), fast
(~50 ms/query for 30 candidates on M2), the canonical English reranker. Swap
via env var `RERANKER_MODEL`. The research recommendation `BAAI/bge-reranker-v2-m3`
is heavier (~568 MB) but stronger; either works.

CLI:
    python rerank.py "what is OpenPiton" --top-k 5
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import chromadb

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    RERANK_FAN,
    RERANKER_MODEL,
)
from hybrid import hybrid_search

DEFAULT_FAN = RERANK_FAN
DEFAULT_TOP_K = 10


@dataclass
class RerankHit:
    rel_path: str
    title: str
    rerank_score: float
    hybrid_rank: int
    hybrid_score: float
    excerpt: str


# Lazy state: we don't import sentence_transformers at module load time so
# that callers running with RERANK_FAN=0 never pay the ~1 GB torch + transformers
# import cost. The CrossEncoder is loaded on first non-zero-fan call only.
_CE_CACHE = None
_PREAMBLE_DOC_CACHE: dict[str, str] | None = None


def _load_ce():
    """Lazily import sentence-transformers and load the cross-encoder.
    Only ever called when RERANK_FAN > 0."""
    global _CE_CACHE
    if _CE_CACHE is None:
        from sentence_transformers import CrossEncoder  # heavy import — deferred
        _CE_CACHE = CrossEncoder(RERANKER_MODEL, max_length=512)
    return _CE_CACHE


def _preamble_doc(rel_path: str) -> str:
    """Look up the preamble-collection document for a rel_path. The
    cross-encoder needs page text to score; we use the curated preamble as
    the document representation (uniform across BM25-only and dense hits)."""
    global _PREAMBLE_DOC_CACHE
    if _PREAMBLE_DOC_CACHE is None:
        coll = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(
            COLLECTION_NAME
        )
        got = coll.get(include=["documents"])
        _PREAMBLE_DOC_CACHE = dict(zip(got["ids"], got["documents"]))
    return _PREAMBLE_DOC_CACHE.get(rel_path, "")


def hybrid_rerank_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    fan: int = DEFAULT_FAN,
) -> list[RerankHit]:
    # Disable-rerank short-circuit. When fan <= 0, return hybrid results in
    # RerankHit shape — no cross-encoder load, no torch import, no model cost.
    # Lets RERANK_FAN=0 in config.py serve as a one-knob disable for the entire
    # rerank stage and its ~1 GB dependency.
    if fan <= 0:
        hits = hybrid_search(query, top_k=top_k)
        return [
            RerankHit(
                rel_path=h.rel_path,
                title=h.title,
                rerank_score=h.score,        # hybrid score in lieu of rerank score
                hybrid_rank=i + 1,
                hybrid_score=h.score,
                excerpt=h.excerpt,
            )
            for i, h in enumerate(hits)
        ]

    candidates = hybrid_search(query, top_k=fan)
    if not candidates:
        return []

    docs = [_preamble_doc(c.rel_path) for c in candidates]
    pairs = [(query, d if d else c.title) for c, d in zip(candidates, docs)]

    ce = _load_ce()
    scores = ce.predict(pairs, show_progress_bar=False)

    rescored = [
        RerankHit(
            rel_path=c.rel_path,
            title=c.title,
            rerank_score=float(scores[i]),
            hybrid_rank=i + 1,
            hybrid_score=c.score,
            excerpt=docs[i][:500].rstrip() if docs[i] else "",
        )
        for i, c in enumerate(candidates)
    ]
    rescored.sort(key=lambda h: -h.rerank_score)
    return rescored[:top_k]


def format_hits(query: str, hits: list[RerankHit]) -> str:
    lines = [f"Hybrid+rerank top {len(hits)} for: {query!r}", ""]
    for i, h in enumerate(hits, 1):
        lines.append(f"{i}. [[{h.rel_path}]] — {h.title}")
        lines.append(
            f"   ce={h.rerank_score:+.3f}  (hybrid#{h.hybrid_rank}, "
            f"hybrid_score={h.hybrid_score:.3f})"
        )
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
    print(format_hits(args.query, hybrid_rerank_search(args.query, args.top_k, args.fan)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
