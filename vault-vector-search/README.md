# vault-vector-search

Local hybrid-retrieval MCP server over an Obsidian vault, used as the **fallback retrieval layer** for the operator-grade memory architecture.

Stack: [Ollama](https://ollama.com/) + `nomic-embed-text` (768-dim, runs on Apple Metal or CUDA) + [ChromaDB](https://www.trychroma.com/) (cosine, local persistent) + [`rank-bm25`](https://pypi.org/project/rank-bm25/) (BM25 over titles + bodies) + [`sentence-transformers`](https://www.sbert.net/) cross-encoder rerank + [MCP](https://modelcontextprotocol.io/) (stdio server).

The retrieval pipeline:

```
query
 ├──→ dense (preamble) ──┐
 │                       ├──→ weighted score fusion (top-30) ──→ cross-encoder rerank ──→ top-K
 └──→ BM25 (title+body) ─┘     hybrid.py                          rerank.py
```

Each layer is its own file and can be used standalone. The deployed system (`server.py`) currently exposes `search_wiki_tool` over the dense-only path; switching to hybrid+rerank in the MCP requires changing one import.

## What it indexes

For each eligible markdown page, the indexer extracts:

- The page **title** (frontmatter `title:` → first H1 → filename)
- The body of the `## For future Claude` preamble, if present
- Otherwise, the first ~600 chars of the page body as a fallback

The (title + preamble) string is embedded with `nomic-embed-text` and stored in ChromaDB with cosine distance. Re-runs are idempotent — pages whose embedded text hasn't changed are skipped.

The indexing scope is an **allow-list** in `config.py` (`03-research/wiki/` plus three top-level docs) so eval results are reproducible by anyone running the public repo.

## Setup

```bash
# 1. Ollama + embedding model
brew install ollama
brew services start ollama          # or: ollama serve
ollama pull nomic-embed-text

# 2. Python deps (Python 3.12)
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Use

```bash
# Build the index (incremental; --rebuild drops + reindexes)
.venv/bin/python index.py

# CLI smoke test
.venv/bin/python search.py "compile-once wiki vs RAG"

# Run as an MCP server (stdio)
.venv/bin/python server.py
```

To register with Claude Code:

```bash
claude mcp add vault-vector-search -- /abs/path/to/.venv/bin/python /abs/path/to/server.py
```

## Config

| Env var | Default | Meaning |
|---|---|---|
| `VAULT_ROOT` | repo root (one level up from this directory) | Vault root |
| `VAULT_EMBED_MODEL` | `nomic-embed-text` | Ollama model name |

## Files

- `config.py` — paths, model, indexing scope
- `index.py` — dense preamble indexer (Ollama → ChromaDB cosine)
- `index_bm25.py` — BM25 indexer over (title × 5 boost) + body; pickled to `bm25_index/bm25.pkl`
- `search.py` — dense-only search function + CLI entrypoint
- `hybrid.py` — weighted score fusion of dense preamble + BM25 body
- `rerank.py` — cross-encoder rerank on top of hybrid (default: `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- `server.py` — MCP wrapper (`search_wiki` tool)
- `test_server.py` — E2E stdio smoke test

## Why this exists

The **vector fallback** layer of the operator-grade memory architecture. The agent's primary retrieval path is `INDEX.md` → wiki indexes → 2–3 candidate pages; this MCP server kicks in when that index-first path doesn't surface a good candidate. The eval in `../eval/` compares hybrid retrieval against dense-only and naive full-read on a hand-written gold QA set.
