# operator-wiki

A starter template for an **operator-grade Obsidian vault**: a persistent memory architecture for a long-running LLM coding agent (Claude Code or any MCP-compatible runtime) that operates across multiple workstreams — coursework, research, project work, life logistics — from a single vault.

This is a **structural template**, not a content corpus. Fork it, populate the empty folders with your own notes, and the architecture's rules + retrieval stack do the rest.

## What's an "operator-grade" vault?

The architecture's design point is a single agent reading and writing across the full breadth of one person's working life — not just one corpus or one workflow. Optimization target is *operator continuity* (the agent re-acquires context fast across sessions and across domains), not just retrieval accuracy on a single corpus. See [`CLAUDE.md`](./CLAUDE.md) for the full operating spec.

The contribution is a set of **portable rules** encoded in `CLAUDE.md` plus a hybrid retrieval stack (`vault-vector-search/`). Both work on any markdown corpus.

---

## Setup

### 0. System prerequisites

| Requirement | Why | macOS install |
|---|---|---|
| Python **3.12** | retrieval stack + eval | `brew install python@3.12` |
| [Homebrew](https://brew.sh/) | install Ollama and Python | follow the install script on brew.sh |
| [Claude Code](https://claude.com/claude-code) | the agent runtime that reads `CLAUDE.md` | `brew install --cask claude-code` (or use any MCP-compatible runtime) |
| [Ollama](https://ollama.com/) | hosts the local embedding model | `brew install ollama` |

Linux is supported (tested on Apple Silicon macOS; CUDA works in place of Metal for the embedding model). The cross-encoder reranker (optional strategy 5) is the only piece that requires `torch`; everything else runs on stdlib + small libraries.

### 1. Clone the template into your vault location

```bash
git clone <fork-url> my-vault
cd my-vault
```

The cloned directory becomes your Obsidian vault root.

### 2. Pull the embedding model

```bash
brew services start ollama          # background daemon (or `ollama serve` in a separate terminal)
ollama pull nomic-embed-text         # ~280 MB, 768-dim, runs on Apple Metal or CUDA
```

### 3. Install Python dependencies

```bash
cd vault-vector-search
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` covers everything for retrieval strategies 1–5:

| Package | For | Size |
|---|---|---|
| `chromadb` | dense vector store | ~50 MB |
| `mcp` | MCP server protocol | ~5 MB |
| `ollama` | client for the local embedding daemon | ~1 MB |
| `numpy` | numerics | ~30 MB |
| `rank-bm25` | BM25 (strategy 4) | ~50 KB pure Python |
| `sentence-transformers` | cross-encoder rerank (strategy 5) — pulls `torch` + `transformers`, ~1 GB total | heaviest dep |

If you don't want strategy 5, comment out `sentence-transformers` in `requirements.txt` before installing — you'll save ~1 GB and can still run strategies 1–4.

### 4. (Optional) Install Claude Code skills

The vault uses Claude Code skills to bridge tooling. The 6 **custom slash commands** specific to this architecture (`/capture`, `/now`, `/query`, `/resume`, `/save`, `/synth`) are already shipped at [`.claude/commands/`](./.claude/commands/) — they work out of the box when you open the vault root in Claude Code.

Recommended **third-party skills** (install separately if you want their integrations):

| Skill | What it does | Where to get it |
|---|---|---|
| `obsidian-cli` | wikilink-aware search/read/backlinks operations on the vault — preferred over `Bash(grep)` | install via `claude skill install` (see `find-skills`) or [kepano/obsidian-skills](https://github.com/kepano/obsidian-skills) |
| `obsidian-markdown` | author Obsidian Flavored Markdown (callouts, embeds, properties) | same skill collection as `obsidian-cli` |
| `defuddle` | extract clean markdown from web pages — used by `/ingest` for URLs | same skill collection or [kepano/defuddle](https://github.com/kepano/defuddle) |
| `second-brain-ingest` | ingest source documents into wiki pages (rewrite-not-append) | [obsidian-second-brain skill suite](https://github.com/eugeniughelbur/obsidian-second-brain) — `second-brain-*` skills there |
| `second-brain-query` | answer questions against the wiki | same suite |
| `second-brain-lint` | health-check the wiki (broken links, orphans, contradictions) | same suite |

You can use `find-skills` (a generic skill-discovery helper) to explore what's available. None of these are strictly required — the vault works without them, just less ergonomic.

### 5. Edit the placeholders

```bash
$EDITOR CLAUDE.md           # fill in `## Identity` and `## Right now`
$EDITOR CRITICAL_FACTS.md   # fill in `## Identity` and `## Active commitments`
```

These two files are the always-loaded operator state. Keep `CRITICAL_FACTS.md` under ~120 tokens.

### 6. Build the indexes

```bash
cd vault-vector-search
.venv/bin/python index.py             # dense preamble index (sub-second per page on Metal)
.venv/bin/python index_bm25.py        # BM25 index over (title × 5 boost) + body
```

You can skip these until you have content — empty vault = empty index.

### 7. Open in Obsidian and start writing

```bash
open -a Obsidian .   # macOS — opens the vault root in Obsidian
```

Drop quick notes into `00-inbox/` (use `/capture` for fast capture). Run `/sort-inbox` periodically to route them. Use `templates/` as the format reference for new wiki pages.

---

## Configuration

All tunable knobs live in [`vault-vector-search/config.py`](./vault-vector-search/config.py). Edit that file to change behavior; modules don't have duplicate constants.

### Knobs at a glance

| Knob | Default | Section | Effect |
|---|---|---|---|
| `VAULT_ROOT` | repo root | 1. Vault | Vault root path. Override with `VAULT_ROOT` env var. |
| `EMBED_MODEL` | `nomic-embed-text` | 2. Embedding | Ollama model used for dense retrieval. Override via `VAULT_EMBED_MODEL`. |
| `ALLOW_PREFIXES` | `("03-research/wiki/",)` | 3. Indexing scope | Path prefixes that get indexed. Add domains here. |
| `ALLOW_TOP_LEVEL` | `()` | 3. Indexing scope | Top-level files to include alongside prefixes. |
| `DENY_DIR_NAMES` | `{".obsidian", ".git", ...}` | 3. Indexing scope | Directory names to skip anywhere in the path. |
| `DENY_FILES` | the 5 routing pages | 3. Indexing scope | Specific files within scope that must not be retrieval candidates. |
| `CHUNK_SIZE` | `1500` | 5. Chunking | Chars per chunk for the chunk-RAG eval baseline (strategy 3). |
| **`TITLE_BOOST`** | `5` | 6. BM25 | How many times the page title is repeated before the body at index time. Critical for proper-noun queries — short pages otherwise lose to longer reference pages. |
| **`HYBRID_FAN`** | `30` | 7. Fusion | Per-retriever candidates contributed to score fusion. Higher = more recall at marginal cost. |
| **`W_DENSE`** | `0.5` | 7. Fusion | Weight on dense (normalized) score. Raise for longer / more semantic queries. |
| **`W_BM25`** | `0.5` | 7. Fusion | Weight on BM25 (normalized) score. Raise for proper-noun-heavy / shorter queries. |
| **`RERANK_FAN`** | `30` | 8. Rerank | How many hybrid candidates the cross-encoder rescores. Set to `0` to disable rerank entirely. |
| **`RERANKER_MODEL`** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 8. Rerank | Cross-encoder model. Override via `RERANKER_MODEL` env var. Heavier alternatives: `BAAI/bge-reranker-base` (~278 MB), `BAAI/bge-reranker-v2-m3` (~568 MB). |

### Common tuning recipes

**Disable rerank entirely** (skip the 1 GB `sentence-transformers` dep):
```python
# config.py
RERANK_FAN = 0
```
…and comment out `sentence-transformers` in `requirements.txt`.

**Bias toward proper-noun queries** (your queries are mostly people / paper / model names):
```python
W_DENSE = 0.4
W_BM25 = 0.6
```

**Bias toward semantic queries** (your queries are full-sentence questions about concepts):
```python
W_DENSE = 0.7
W_BM25 = 0.3
```

**Increase recall at higher cost** (you want top-5 to almost certainly contain the right page):
```python
HYBRID_FAN = 60
RERANK_FAN = 60
```

**Add a new wiki domain** (e.g., a `concepts-bio/` split for biology):
```python
ALLOW_PREFIXES = (
    "03-research/wiki/",
    # everything under wiki is already covered; no change needed unless you
    # want to index pages outside that prefix
)
```

---

## Layout

```
my-vault/
├── CLAUDE.md            # operating rules — the actual contribution
├── INDEX.md             # routing layer (always-loaded)
├── CRITICAL_FACTS.md    # always-loaded ~120-token operator state
├── 00-inbox/            # fast capture, sorted later
├── 01-self/             # personal/private notes
├── 02-academics/        # course material
├── 03-research/wiki/    # the wiki: concepts, entities, sources, syntheses
├── 04-projects/         # one folder per code project (each with its own CLAUDE.md)
├── 05-life/             # life/plans/ideas
├── 06-tasks/now.md      # active task list
├── 07-reference/        # external-tool reference docs
├── logs/                # session history
├── templates/           # 4 example wiki pages illustrating the format
├── .claude/commands/    # 6 custom slash commands (capture, now, query, resume, save, synth)
├── vault-vector-search/ # local hybrid-retrieval MCP server + tunable config
├── eval/                # evaluation harness (see eval/README.md)
└── tools/               # vault lint + preamble audit
```

## The retrieval stack

The MCP server in `vault-vector-search/` is a **fallback** retrieval layer — your primary path is the agent reading `INDEX.md`, then the wiki indexes, then 2–3 candidate pages' preambles. Vector search kicks in when that index-first path doesn't surface a good candidate.

Four retrieval modes are available (all share the same model + ChromaDB):

| mode | implementation | when to use |
|---|---|---|
| dense (preamble) | `index.py` + `search.py` | default; sub-millisecond, ~50KB pure-Python deps |
| dense (chunk) | `eval/index_chunks.py` | comparison baseline only |
| **hybrid** | `index_bm25.py` + `hybrid.py` | **recommended** — adds BM25 to catch proper-noun queries dense embeddings miss; +24pp recall@5 over chunk-RAG at the same cost |
| hybrid + cross-encoder rerank | `rerank.py` | optional opt-in for synthesis-heavy queries; +1.7pp r@5 at ~200× higher per-query latency and a ~1GB extra dependency |

Detailed reproduction in [`REPRODUCE.md`](./REPRODUCE.md).

## Rules to read before writing your first note

The most load-bearing rules in `CLAUDE.md`:

1. **Every content page gets a `## For future Claude` preamble** at creation. 2–3 sentences. Sentence 1 leads with the page's primary searchable term so vector retrieval finds it.
2. **Routing pages don't get preambles** — `index*.md`, `log.md`, `INDEX.md`, `CLAUDE.md`, `CRITICAL_FACTS.md`. They're navigation, not retrieval candidates; preambles on them dominate vector search via lexical match.
3. **Rewrite-not-append on ingest** — `/ingest` should update 5–15 existing pages, not create siblings.
4. **Use `[[wikilinks]]`** for every entity / concept / project mention; create stubs for broken targets.
5. **Confidence markers** in frontmatter; inline date stamps on benchmark claims.

The 4 example pages in `templates/` illustrate each note type.

## Acknowledgments

Borrows patterns from prior LLM+Obsidian systems including `obsidian-second-brain`, `obsidian-llm-wiki-local`, `tldw_server`, `OB1`, `llm-wiki-agent`, `eclaire`, and `obsidian-ava`.
