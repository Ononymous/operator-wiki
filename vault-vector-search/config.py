"""
Central tuning surface for the operator-wiki retrieval stack.

Every knob the system exposes lives here. Override defaults via:
  - direct edit of this file (recommended for permanent tuning)
  - environment variables for the few settings that have them

Knobs are grouped by stage. Each module (`index.py`, `index_bm25.py`,
`hybrid.py`, `rerank.py`, `eval/index_chunks.py`) imports from this file —
do not introduce duplicate constants in those modules.
"""
import os
from pathlib import Path

# ============================================================
# 1. Vault location
# ============================================================
# Defaults to the repo root (one level up from this file). Override with the
# VAULT_ROOT env var to point at any Obsidian vault.
VAULT_ROOT = Path(os.environ.get(
    "VAULT_ROOT",
    str(Path(__file__).resolve().parent.parent),
))


# ============================================================
# 2. Embedding model (dense retrieval — strategies 2, 3, 4, 5)
# ============================================================
# Ollama-hosted embedding model. nomic-embed-text is 768-dim, 8192 ctx, runs
# on Apple Metal or CUDA. Override via VAULT_EMBED_MODEL env var.
EMBED_MODEL = os.environ.get("VAULT_EMBED_MODEL", "nomic-embed-text")


# ============================================================
# 3. Indexing scope (which pages get indexed)
# ============================================================
# Allow-list: only paths matching one of these prefixes (or top-level files
# in ALLOW_TOP_LEVEL) are indexed.
ALLOW_PREFIXES = (
    "03-research/wiki/",
)

# Top-level files to include alongside ALLOW_PREFIXES. Empty by default —
# routing/policy docs are reached via the index router, not vector search.
ALLOW_TOP_LEVEL = ()

# Directory names anywhere in the path that disqualify a file (build outputs,
# trash, dot-dirs, virtual envs).
DENY_DIR_NAMES = {".obsidian", ".git", ".trash", "node_modules", ".venv", "chroma_db"}

# Routing/meta pages within ALLOW_PREFIXES that should not be retrieval
# candidates: their bodies aggregate vocabulary from many topics and dominate
# retrieval via lexical match rather than semantic relevance.
DENY_FILES = (
    "03-research/wiki/index.md",
    "03-research/wiki/index-llms.md",
    "03-research/wiki/index-kb.md",
    "03-research/wiki/index-systems.md",
    "03-research/wiki/log.md",
)


# ============================================================
# 4. ChromaDB (dense indexes — preamble + chunks)
# ============================================================
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "wiki_preambles"


# ============================================================
# 5. Chunking (eval baseline — strategy 3 only)
# ============================================================
# Chars per chunk for the chunk-RAG eval baseline.
CHUNK_SIZE = 1500


# ============================================================
# 6. BM25 (strategy 4 — hybrid retrieval)
# ============================================================
# Title-field boost. The page title is repeated TITLE_BOOST times before the
# body at index time. Without this (TITLE_BOOST=1), short entity pages lose
# to longer pages that wikilink the entity from their bodies — own-page
# mention TF gets dominated by reference TF. 5 is a good default; raise if
# proper-noun queries still under-rank, lower if titles over-dominate.
TITLE_BOOST = 5


# ============================================================
# 7. Hybrid score fusion (strategy 4)
# ============================================================
# Per-retriever fan-out: how many candidates each of {dense, BM25} contributes
# to the fusion stage before scoring. Higher = more recall (more candidates
# considered) at marginal cost.
HYBRID_FAN = 30

# Weights for the convex combination of min-max-normalized dense and BM25
# scores. Defaults are equal weight; raise W_BM25 (e.g., 0.6) if your queries
# are extremely proper-noun-heavy or short, raise W_DENSE if queries are
# longer and more semantic.
W_DENSE = 0.5
W_BM25 = 0.5


# ============================================================
# 8. Cross-encoder rerank (strategy 5, optional)
# ============================================================
# How many hybrid candidates the cross-encoder rescores. The CE pass is the
# dominant query-time cost (~5–10 ms per pair on CPU MiniLM-L-6); 30 is a
# reasonable balance. Setting to 0 disables rerank entirely (the
# `hybrid_rerank` strategy in run.py will degenerate to hybrid).
RERANK_FAN = 30

# Cross-encoder reranker model. Defaults to `ms-marco-MiniLM-L-6-v2` (~22M
# params, ~80 MB). Heavier alternatives:
#   - "BAAI/bge-reranker-base"     (~278 MB, stronger)
#   - "BAAI/bge-reranker-v2-m3"    (~568 MB, multilingual, strongest)
# Override via RERANKER_MODEL env var.
RERANKER_MODEL = os.environ.get(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
