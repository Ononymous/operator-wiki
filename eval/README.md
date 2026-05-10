# eval

Reproducible evaluation harness for the four retrieval strategies (preamble, chunk, hybrid, hybrid+rerank) plus the naive token-cost baseline. The harness is corpus-agnostic — only the gold QA set and result snapshots differ from corpus to corpus.

## To run on your own vault

1. Fill in `gold-qa.jsonl` with 20–30 questions about *your* wiki content (see [`gold-qa.example.jsonl`](./gold-qa.example.jsonl) for format).
2. Build the indexes:

   ```bash
   cd ../vault-vector-search
   .venv/bin/python index.py             # dense preamble
   .venv/bin/python index_bm25.py        # BM25
   ```

3. Build the chunk-RAG eval baseline (only used for the chunk strategy comparison):

   ```bash
   cd ../eval
   ../vault-vector-search/.venv/bin/python index_chunks.py --rebuild
   ```

4. Run the harness:

   ```bash
   ../vault-vector-search/.venv/bin/python run.py
   ```

   Output: per-question retrieval traces, aggregate metrics, by-category breakdown, `results.json`.

5. Optional — measure how each strategy scales as your library grows:

   ```bash
   ../vault-vector-search/.venv/bin/python benchmark_resources.py
   ```

## What `gold-qa.jsonl` should contain

One JSON object per line:

```json
{"id": "Q01", "category": "fact", "question": "...", "gold": ["path/to/page.md", "..."]}
```

- **`category`**: one of `fact`, `cross-domain`, `staleness`, `synthesis` (used for the by-category breakdown)
- **`gold`**: the **full** set of pages that legitimately answer the question. Recall@k counts a hit if **any** gold page appears in the top-k retrieved set.
- Aim for a mix: ~33% fact lookups, ~33% cross-domain (1–3 page answer), ~16% staleness ("most recent X"), ~16% synthesis (big-picture)

A reference 30-question gold set averages 2.4 gold pages per question.

## Files

| File | Purpose |
|---|---|
| `gold-qa.example.jsonl` | 5-question worked example showing format |
| `gold-qa.jsonl` | TODO — your own gold QA |
| `run.py` | The harness — runs all 4 strategies, writes `results.json` |
| `index_chunks.py` | Build the chunk-RAG ChromaDB collection (strategy 3 only) |
| `compare_results.py` | Diff multiple `results-*.json` snapshots side-by-side |
| `benchmark_resources.py` | Measure index build time, size, per-query latency at varying N |
| `chroma_chunks/` | Strategy-3 vector collection (gitignored) |

## Reference numbers

(Produced on a 153-page personal wiki — your numbers will differ depending on corpus size, vocabulary, and query distribution.)

| strategy | recall@5 | MRR | top-5 tokens |
|---:|---:|---:|---:|
| preamble | 0.559 | 0.628 | 960 |
| chunk | 0.660 | 0.612 | 1,059 |
| hybrid | 0.821 | 0.825 | 954 |
| hybrid + rerank | 0.838 | 0.864 | 966 |
| naive (full-read) | 1.000 | 1.000 | 165,736 |

The marginal recall-per-millisecond of upgrading **preamble → hybrid** is ~10,000× better than **hybrid → rerank**. Most deployments should land on hybrid as canonical.
