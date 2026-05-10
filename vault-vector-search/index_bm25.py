"""
Build a BM25 index over wiki page bodies for the hybrid retrieval mode.

Each indexable page contributes one BM25 record over (title + body). Persisted
as pickle to `bm25_index/bm25.pkl` next to this file. Re-runs are full
rebuilds (BM25Okapi doesn't support incremental updates and the corpus is
small enough that full rebuild is sub-second).

Usage:
    python index_bm25.py
"""
from __future__ import annotations

import pickle
import re
import sys
from pathlib import Path

from rank_bm25 import BM25Okapi

from config import TITLE_BOOST, VAULT_ROOT
from index import collect_pages, split_frontmatter

BM25_DIR = Path(__file__).parent / "bm25_index"
BM25_FILE = BM25_DIR / "bm25.pkl"

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Minimal English-question-stopword list. Applied at query time only — keeping
# stopwords in the corpus preserves IDF statistics. Removing them from the
# query stops noisy "what/is/the" tokens from inflating scores of unrelated
# pages on multi-word questions.
_QUERY_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "to", "for", "in", "on", "with",
    "as", "by", "at", "from", "is", "are", "was", "were", "be", "been",
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "do", "does", "did", "this", "that", "these", "those", "it", "its",
    "i", "you", "he", "she", "we", "they", "them",
    "differ", "different", "between",
})


def tokenize(text: str) -> list[str]:
    """Lowercase + alphanumeric runs. Robust enough for BM25; no stemming —
    we want proper-noun matches like "OpenPiton" → "openpiton" preserved."""
    return _TOKEN_RE.findall(text.lower())


def tokenize_query(text: str) -> list[str]:
    """Query-time tokenization: drop English stopwords so the discriminating
    tokens (proper nouns, named techniques) carry the BM25 score."""
    return [t for t in tokenize(text) if t not in _QUERY_STOPWORDS]


def main() -> int:
    BM25_DIR.mkdir(parents=True, exist_ok=True)
    pages = collect_pages()
    print(f"[bm25] indexing {len(pages)} pages")

    # Title-field boost (TITLE_BOOST in config.py): repeat the title that many
    # times before the body so its tokens get term-frequency credit
    # proportional to a separate title field. Without this, short entity pages
    # lose to longer pages that wikilink the entity from their bodies — own-page
    # mention TF is dominated by reference TF.
    corpus_tokens: list[list[str]] = []
    metas: list[dict] = []
    for p in pages:
        full = (VAULT_ROOT / p["rel_path"]).read_text(encoding="utf-8")
        _, body = split_frontmatter(full)
        title_repeat = ("\n".join([p["title"]] * TITLE_BOOST))
        text = f"{title_repeat}\n\n{body}"
        corpus_tokens.append(tokenize(text))
        metas.append({"rel_path": p["rel_path"], "title": p["title"]})

    bm25 = BM25Okapi(corpus_tokens)

    with open(BM25_FILE, "wb") as f:
        pickle.dump({"bm25": bm25, "metas": metas}, f)
    print(f"[bm25] wrote {BM25_FILE} "
          f"(avg doc len: {sum(len(t) for t in corpus_tokens) / len(corpus_tokens):.0f} tokens)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
