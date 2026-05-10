"""
Vector search over the indexed vault. Importable function plus a CLI
entrypoint for smoke testing without going through MCP.

Usage:
    python search.py "what is preamble triage"
    python search.py "compile-once wiki" --top-k 10
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import chromadb
import ollama

from config import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL


@dataclass
class Hit:
    rel_path: str
    title: str
    score: float
    has_preamble: bool
    excerpt: str


def _collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(COLLECTION_NAME)


def _embed(query: str) -> list[float]:
    return ollama.embeddings(model=EMBED_MODEL, prompt=query)["embedding"]


def search_wiki(query: str, top_k: int = 5) -> list[Hit]:
    coll = _collection()
    res = coll.query(
        query_embeddings=[_embed(query)],
        n_results=top_k,
        include=["metadatas", "documents", "distances"],
    )
    metas = res["metadatas"][0]
    docs = res["documents"][0]
    dists = res["distances"][0]

    hits = []
    for m, d, dist in zip(metas, docs, dists):
        excerpt = d.split("\n", 1)[1].strip() if "\n" in d else d.strip()
        hits.append(Hit(
            rel_path=m["rel_path"],
            title=m["title"],
            score=1.0 - float(dist),
            has_preamble=bool(m["has_preamble"]),
            excerpt=excerpt[:500].rstrip(),
        ))
    return hits


def format_hits(query: str, hits: list[Hit]) -> str:
    lines = [f"Top {len(hits)} matches for: {query!r}", ""]
    for i, h in enumerate(hits, 1):
        lines.append(f"{i}. [[{h.rel_path}]] — {h.title}")
        lines.append(f"   score={h.score:.3f}  preamble={'yes' if h.has_preamble else 'no'}")
        lines.append(f"   {h.excerpt}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Natural-language query")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    hits = search_wiki(args.query, top_k=args.top_k)
    print(format_hits(args.query, hits))
    return 0


if __name__ == "__main__":
    sys.exit(main())
