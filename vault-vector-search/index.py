"""
Walk the vault, extract title + "## For future Claude" preamble per eligible
page, embed via Ollama, and persist to a ChromaDB collection.

Re-runs are idempotent: pages whose embedded text hasn't changed since the
last run are skipped.

Usage:
    python index.py             # incremental
    python index.py --rebuild   # drop the collection first
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import chromadb
import ollama

from config import (
    ALLOW_PREFIXES,
    ALLOW_TOP_LEVEL,
    CHROMA_DIR,
    COLLECTION_NAME,
    DENY_DIR_NAMES,
    DENY_FILES,
    EMBED_MODEL,
    VAULT_ROOT,
)

_PREAMBLE_RE = re.compile(r"^##\s+For future Claude\s*$", re.MULTILINE)
_HEADING_RE = re.compile(r"^##\s", re.MULTILINE)
_FRONTMATTER_TITLE_RE = re.compile(r"^title:\s*(.+)$", re.MULTILINE)
_H1_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)


def is_eligible(path: Path) -> bool:
    if any(part in DENY_DIR_NAMES for part in path.parts):
        return False
    rel = path.relative_to(VAULT_ROOT).as_posix()
    if rel in DENY_FILES:
        return False
    if rel in ALLOW_TOP_LEVEL:
        return True
    return any(rel.startswith(p) for p in ALLOW_PREFIXES)


def split_frontmatter(text: str) -> tuple[str, str]:
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            return parts[1], parts[2]
    return "", text


def extract_title(frontmatter: str, body: str, fallback: str) -> str:
    m = _FRONTMATTER_TITLE_RE.search(frontmatter)
    if m:
        return m.group(1).strip().strip('"\'')
    m = _H1_RE.search(body)
    if m:
        return m.group(1).strip()
    return fallback


def extract_preamble(body: str) -> str | None:
    m = _PREAMBLE_RE.search(body)
    if not m:
        return None
    after = body[m.end():]
    nxt = _HEADING_RE.search(after)
    section = after[:nxt.start()] if nxt else after
    return section.strip() or None


def build_doc(title: str, preamble: str | None, body: str) -> str:
    if preamble:
        return f"# {title}\n\n{preamble}"
    snippet = body.strip()[:600].strip()
    return f"# {title}\n\n{snippet}"


def hash_doc(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def collect_pages() -> list[dict]:
    pages = []
    for md in VAULT_ROOT.rglob("*.md"):
        if not is_eligible(md):
            continue
        try:
            text = md.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        frontmatter, body = split_frontmatter(text)
        title = extract_title(frontmatter, body, fallback=md.stem)
        preamble = extract_preamble(body)
        doc = build_doc(title, preamble, body)
        pages.append({
            "rel_path": md.relative_to(VAULT_ROOT).as_posix(),
            "title": title,
            "has_preamble": preamble is not None,
            "doc": doc,
            "hash": hash_doc(doc),
        })
    return pages


def embed(text: str) -> list[float]:
    resp = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return resp["embedding"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true",
                        help="Drop the collection before indexing.")
    args = parser.parse_args()

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if args.rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"[index] dropped collection {COLLECTION_NAME}")
        except Exception:
            pass

    coll = client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    got = coll.get(include=["metadatas"])
    existing = dict(zip(got["ids"], got["metadatas"]))

    pages = collect_pages()
    print(f"[index] eligible pages: {len(pages)}")

    fresh, with_preamble = [], 0
    for p in pages:
        if p["has_preamble"]:
            with_preamble += 1
        prev = existing.get(p["rel_path"])
        if prev and prev.get("hash") == p["hash"]:
            continue
        fresh.append(p)

    print(f"[index] with preamble: {with_preamble}/{len(pages)} "
          f"({100*with_preamble/max(len(pages),1):.0f}%)")
    print(f"[index] embedding {len(fresh)} pages "
          f"({len(pages) - len(fresh)} unchanged)")

    if not fresh:
        return 0

    ids, embeds, docs, metas = [], [], [], []
    for i, p in enumerate(fresh, 1):
        embeds.append(embed(p["doc"]))
        ids.append(p["rel_path"])
        docs.append(p["doc"])
        metas.append({
            "rel_path": p["rel_path"],
            "title": p["title"],
            "has_preamble": p["has_preamble"],
            "hash": p["hash"],
        })
        if i % 25 == 0:
            print(f"[index]   {i}/{len(fresh)}")

    coll.upsert(ids=ids, embeddings=embeds, documents=docs, metadatas=metas)
    print(f"[index] wrote {len(fresh)} embeddings → {CHROMA_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
