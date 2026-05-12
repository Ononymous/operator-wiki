# Reproducing the eval

How to run the four-strategy retrieval evaluation on your own populated vault. All scripts honor `VAULT_ROOT` (defaults to the repo root); none require it to be set if you run from inside the cloned vault.

## 0. Prerequisites

- macOS or Linux (tested on macOS 14, Apple Silicon)
- Python **3.12** (3.13 may work)
- [Homebrew](https://brew.sh/) (or another package manager that can install Ollama)

The eval scope is `03-research/wiki/` plus the three top-level routing docs (`INDEX.md`, `CRITICAL_FACTS.md`, `CLAUDE.md`). Routing pages within the wiki (`index*.md`, `log.md`) are excluded from the vector index.

## 1. Install Ollama and pull the embedding model

```bash
brew install ollama
brew services start ollama          # or: ollama serve  (foreground)
ollama pull nomic-embed-text         # 137M params, ~280 MB, runs on Apple Metal
```

`nomic-embed-text` is a 768-dim embedding model with an 8192-token context window. Throughput on M2 Pro Metal: ~50–150 short-text docs/sec.

## 2. Set up the Python environment

```bash
cd vault-vector-search
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install tiktoken                # eval-only — input-token accounting
```

`requirements.txt` covers all four retrieval strategies. The `sentence-transformers` line pulls in `torch` and `transformers` (~1 GB total) — only required if you run strategy 5 (cross-encoder rerank). To skip strategy 5, comment out that line and the rerank import in `eval/run.py`.

## 3. Build the dense preamble index (strategy 2)

The system's primary retrieval index. One vector per content page: `# Title` + `## For future Claude` block. Routing pages are excluded.

```bash
cd vault-vector-search
.venv/bin/python index.py --rebuild
```

Sub-second per page on Apple Metal. Persisted under `vault-vector-search/chroma_db/`.

Smoke test:

```bash
.venv/bin/python search.py "<a query about your wiki content>"
```

## 4. Build the chunk-RAG vector index (strategy 3, eval baseline only)

One vector per ~1500-char body chunk. Persisted separately to keep the deployed system clean.

```bash
cd ../eval
../vault-vector-search/.venv/bin/python index_chunks.py --rebuild
```

## 5. Build the BM25 index (strategy 4 — hybrid retrieval)

Pure Python; one record per page over (title × 5 boost + body). Sub-second.

```bash
cd ../vault-vector-search
.venv/bin/python index_bm25.py
```

Persisted as `vault-vector-search/bm25_index/bm25.pkl`. Strategy 5 uses the same BM25 index — no separate indexing step.

## 6. Build your gold QA set

The `eval/gold-qa.jsonl` file is corpus-specific — you write 20–30 questions about *your* wiki content. See [`eval/README.md`](./eval/README.md) for the format and the recommended category mix. `eval/gold-qa.example.jsonl` shows the schema with placeholder questions.

## 7. Run the eval

```bash
cd eval
../vault-vector-search/.venv/bin/python run.py
```

Output:

- Per-question line, one per question across all four retrieval strategies (P/C/H/R)
- Aggregate table — recall@1/3/5, MRR, top-5 token cost
- By-category breakdown (fact / cross-domain / staleness / synthesis)
- `results.json` with full per-question detail

## 8. (Optional) Resource benchmark

```bash
cd eval
../vault-vector-search/.venv/bin/python benchmark_resources.py
```

Measures index build time, on-disk size, per-query latency for each strategy at N ∈ {20, 50, 100, full} subsamples of your corpus. Output: `benchmark.json` plus a printed table. Use this to characterize how your operating point shifts as the wiki grows.

## 9. (Optional) Lint & audit

```bash
cd ../tools
python lint_vault.py             # broken wikilinks, orphans, missing-from-index, missing preambles
python audit_preambles.py        # flag preambles that don't lead with the page's primary searchable term
```

Both run on stdlib Python; no venv required.

## Reproducibility notes

- **Embeddings are deterministic** modulo Ollama version + model checksum. Re-running `index.py` on the same inputs produces byte-identical vectors and byte-identical retrieval rankings.
- **ChromaDB cosine space** is set explicitly via `metadata={"hnsw:space": "cosine"}` on collection creation. Both collections (preamble + chunks) use the same space so scores are directly comparable.
- **BM25** uses `rank-bm25`'s `BM25Okapi` with default parameters (k1=1.5, b=0.75). Title-field boost is `TITLE_BOOST = 5` (the title is repeated 5× before the body at index time). Query-time stopword filtering uses a small English question-stopword set (see `vault-vector-search/index_bm25.py`).
- **Hybrid score fusion** uses min-max-normalized weighted sum with `w_dense = w_bm25 = 0.5` and `fan = 30` candidates per retriever (see `vault-vector-search/hybrid.py`). RRF with k=60 was tested and rejected — at that constant, dual-retriever modest-rank hits dominate single-retriever top-1 hits, which fails for proper-noun queries where BM25 alone has the right answer.
- **Cross-encoder reranker** is `cross-encoder/ms-marco-MiniLM-L-6-v2` with `max_length = 512`. Override via `RERANKER_MODEL` env var; `BAAI/bge-reranker-base` and `BAAI/bge-reranker-v2-m3` are drop-in replacements (heavier, possibly stronger).
- **Token counts** use OpenAI's `cl100k_base` tokenizer via `tiktoken`. This is a stand-in for the production LLM's tokenizer; absolute numbers are approximate but order-of-magnitude differences between strategies are robust to the choice.
- **Random seeding is irrelevant** — none of the retrieval strategies sample. Re-runs are fully deterministic.
- **GPU vs CPU.** Reference numbers below were produced on Apple M2 Pro (Metal for embeddings, CPU for BM25 and the cross-encoder MiniLM). Running on CUDA produces identical rankings; only wall-clock changes.

## Key findings from a reference run

(Numbers from a 153-page personal wiki. Your numbers will vary depending on corpus size, vocabulary, and query distribution.)

0. **Naive read-all is the upper-bound baseline.** Feeding every page in the vault into the model's context per query trivially reaches r@5 = 1.0 on any gold set — but the cost is ~172 k input tokens per query on the reference 153-page wiki, roughly 180× hybrid's per-query token count. This is the bar the other strategies are trying to approximate at a fraction of the cost.
1. **Lead-with-search-term in preamble content gives a free precision win** — sentence 1 of the preamble must start with the page's primary searchable term, not bury it mid-sentence.
2. **Hybrid retrieval is non-negotiable for proper-noun-heavy personal corpora.** Dense alone hits a ~0.66 r@5 ceiling at small N (per BEIR small-N studies, LIMIT benchmark, Toloka enterprise eval). Adding BM25 with min-max-normalized score fusion lifts r@5 to ~0.82 with a +58 pp jump on the fact category alone.
3. **Cross-encoder rerank is not a uniform improvement.** Strong gains on synthesis-style queries (+23 pp r@5) and on r@1/MRR; real regression on staleness queries because off-the-shelf rerankers are trained on relevance benchmarks, not recency. Production routing fix: skip rerank for staleness-tagged queries.
4. **Hybrid is the recommended canonical operating point.** It captures ~98% of rerank's r@5 at sub-millisecond per-query latency (vs. ~320 ms for rerank on the reference run) and a ~50 KB pure-Python dependency vs ~1 GB.
5. **Rerank's per-query cost is constant in N, not linear.** An earlier framing called the marginal recall-per-millisecond of hybrid→rerank "~10,000× worse" than preamble→hybrid; that conflated absolute query latency with amortized per-document cost. The proper model is `T_rerank(N) = T_hybrid(N) + RERANK_FAN × T_ce`, where the second term is independent of N (a fixed number of cross-encoder forward passes per query). Per-query latency therefore plateaus rather than scaling, and per-document amortized cost decays as `1/N`. For interactive use at the scales considered here, the absolute ~320 ms per-query hit is what matters — hence rerank shipping as opt-in. For batch use over a large corpus, the per-document cost effectively vanishes.
