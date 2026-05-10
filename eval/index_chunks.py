"""
Build the chunk-RAG ChromaDB collection used by strategy 3 of the eval.

For each indexable page, the body (frontmatter stripped, preamble retained) is
split into ~1500-char chunks at paragraph boundaries. Each chunk is embedded
with the same model as the preamble collection so the head-to-head only
varies the document granularity.

The collection lives in `eval/chroma_chunks/` to stay separate from the
production preamble collection in `vault-vector-search/chroma_db/`.

Usage:
    python index_chunks.py              # incremental
    python index_chunks.py --rebuild    # drop the collection first
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import chromadb
import ollama

# Reuse the page walker + frontmatter parser from the preamble indexer
HERE = Path(__file__).parent
SEARCH_DIR = HERE.parent / "vault-vector-search"
sys.path.insert(0, str(SEARCH_DIR))
from index import collect_pages, split_frontmatter  # noqa: E402
from config import CHUNK_SIZE, EMBED_MODEL  # noqa: E402

CHROMA_DIR = HERE / "chroma_chunks"
COLLECTION_NAME = "wiki_chunks"


def chunk_text(text: str, size: int = CHUNK_SIZE) -> list[str]:
    """Split body into ~size-char chunks at paragraph (\\n\\n) boundaries.
    Falls back to a hard char split when a paragraph alone exceeds size."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, buf = [], ""
    for para in paragraphs:
        if len(para) >= size:
            if buf:
                chunks.append(buf.strip())
                buf = ""
            for i in range(0, len(para), size):
                chunks.append(para[i:i + size].strip())
            continue
        if len(buf) + len(para) + 2 > size and buf:
            chunks.append(buf.strip())
            buf = para
        else:
            buf = f"{buf}\n\n{para}".strip() if buf else para
    if buf:
        chunks.append(buf.strip())
    return [c for c in chunks if c]


def embed(text: str) -> list[float]:
    return ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if args.rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    coll = client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    pages = collect_pages()
    print(f"[chunks] eligible pages: {len(pages)}")

    # Re-derive (rel_path, body) for each page; collect_pages gives us the
    # combined embed_doc, but we need the raw body for chunking.
    from config import VAULT_ROOT
    docs_to_index = []
    for p in pages:
        full = (VAULT_ROOT / p["rel_path"]).read_text(encoding="utf-8")
        _, body = split_frontmatter(full)
        # Embed the title + body together so the title appears in every chunk
        # query (parity with the preamble collection's "# {title}\n\n..." form)
        for i, ch in enumerate(chunk_text(body)):
            doc = f"# {p['title']}\n\n{ch}"
            docs_to_index.append({
                "id": f"{p['rel_path']}::chunk{i}",
                "doc": doc,
                "rel_path": p["rel_path"],
                "title": p["title"],
                "chunk_idx": i,
                "hash": hashlib.sha256(doc.encode("utf-8")).hexdigest(),
            })

    got = coll.get(include=["metadatas"])
    existing = dict(zip(got["ids"], got["metadatas"]))

    fresh = [d for d in docs_to_index
             if existing.get(d["id"], {}).get("hash") != d["hash"]]

    print(f"[chunks] total chunks: {len(docs_to_index)}  "
          f"(avg {len(docs_to_index)/max(len(pages),1):.1f}/page)")
    print(f"[chunks] embedding {len(fresh)} chunks "
          f"({len(docs_to_index) - len(fresh)} unchanged)")

    if not fresh:
        return 0

    ids, embeds, docs, metas = [], [], [], []
    for i, d in enumerate(fresh, 1):
        embeds.append(embed(d["doc"]))
        ids.append(d["id"])
        docs.append(d["doc"])
        metas.append({
            "rel_path": d["rel_path"],
            "title": d["title"],
            "chunk_idx": d["chunk_idx"],
            "hash": d["hash"],
        })
        if i % 50 == 0:
            print(f"[chunks]   {i}/{len(fresh)}")

    coll.upsert(ids=ids, embeddings=embeds, documents=docs, metadatas=metas)
    print(f"[chunks] wrote {len(fresh)} chunk embeddings → {CHROMA_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
