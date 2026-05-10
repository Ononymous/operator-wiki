"""
Resource benchmark — how does each strategy scale as the library size N grows?

Measures, for N ∈ {20, 50, 100, 153}, the index build time, on-disk size, and
per-query latency for each of:

  - preamble dense       (the deployed primary; nomic-embed-text + ChromaDB)
  - chunk dense          (eval-only baseline)
  - BM25                 (rank-bm25 over title × 5 + body)
  - hybrid               (preamble dense + BM25 + score fusion)
  - hybrid + rerank      (cross-encoder ms-marco-MiniLM-L-6-v2 over hybrid top-30)

The argument from this benchmark is the resource side of the report's
hybrid-vs-rerank pareto: rerank delivers ~2 pp r@5 over hybrid at >5×
per-query latency and a ~1 GB extra dep. The benchmark quantifies that.

Output: `eval/benchmark.json` + printed table.

Usage:
    python benchmark_resources.py
    python benchmark_resources.py --sizes 20,50,100,153
    python benchmark_resources.py --queries 10        # subsample question set
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

import chromadb
import numpy as np
import ollama
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

HERE = Path(__file__).parent
SEARCH_DIR = HERE.parent / "vault-vector-search"
sys.path.insert(0, str(SEARCH_DIR))

from config import EMBED_MODEL, VAULT_ROOT  # noqa: E402
from index import collect_pages, split_frontmatter  # noqa: E402
from index_bm25 import tokenize, tokenize_query  # noqa: E402
from index_chunks import chunk_text  # noqa: E402

GOLD_PATH = HERE / "gold-qa.jsonl"
RESULTS_PATH = HERE / "benchmark.json"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_SIZES = [20, 50, 100, 153]
QUERY_FAN = 30  # how many candidates each retriever surfaces before fusion
RERANK_FAN = 30  # how many candidates the cross-encoder rescores


def du_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def fmt_size(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / (1024*1024):.2f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def embed(text: str) -> list[float]:
    return ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]


def median_ms(seq: list[float]) -> float:
    return round(statistics.median(seq) * 1000, 2)


def p95_ms(seq: list[float]) -> float:
    s = sorted(seq)
    return round(s[max(0, int(len(s) * 0.95) - 1)] * 1000, 2)


def benchmark_at_n(
    n: int,
    queries: list[str],
    all_pages: list[dict],
    cross_encoder: CrossEncoder,
    bench_root: Path,
) -> dict:
    pages = all_pages[:n]
    out: dict = {"n": n}

    # ----- BUILD: preamble dense -----
    pre_dir = bench_root / f"pre_{n}"
    if pre_dir.exists():
        shutil.rmtree(pre_dir)
    pre_dir.mkdir(parents=True)
    t0 = time.perf_counter()
    pre_client = chromadb.PersistentClient(path=str(pre_dir))
    pre_coll = pre_client.get_or_create_collection(
        "pre", metadata={"hnsw:space": "cosine"}
    )
    pre_embeds = [embed(p["doc"]) for p in pages]
    pre_coll.upsert(
        ids=[p["rel_path"] for p in pages],
        embeddings=pre_embeds,
        documents=[p["doc"] for p in pages],
        metadatas=[
            {"rel_path": p["rel_path"], "title": p["title"]} for p in pages
        ],
    )
    pre_build = time.perf_counter() - t0
    pre_size = du_size(pre_dir)

    # ----- BUILD: chunk dense -----
    chunk_dir = bench_root / f"chunk_{n}"
    if chunk_dir.exists():
        shutil.rmtree(chunk_dir)
    chunk_dir.mkdir(parents=True)
    t0 = time.perf_counter()
    chunk_client = chromadb.PersistentClient(path=str(chunk_dir))
    chunk_coll = chunk_client.get_or_create_collection(
        "chunks", metadata={"hnsw:space": "cosine"}
    )
    chunk_ids: list[str] = []
    chunk_embeds_l: list[list[float]] = []
    chunk_docs: list[str] = []
    chunk_metas: list[dict] = []
    for p in pages:
        full = (VAULT_ROOT / p["rel_path"]).read_text(encoding="utf-8")
        _, body = split_frontmatter(full)
        for i, ch in enumerate(chunk_text(body)):
            doc = f"# {p['title']}\n\n{ch}"
            chunk_ids.append(f"{p['rel_path']}::chunk{i}")
            chunk_embeds_l.append(embed(doc))
            chunk_docs.append(doc)
            chunk_metas.append({
                "rel_path": p["rel_path"], "title": p["title"], "chunk_idx": i,
            })
    chunk_coll.upsert(
        ids=chunk_ids, embeddings=chunk_embeds_l,
        documents=chunk_docs, metadatas=chunk_metas,
    )
    chunk_build = time.perf_counter() - t0
    chunk_size = du_size(chunk_dir)

    # ----- BUILD: BM25 -----
    t0 = time.perf_counter()
    corpus_tokens: list[list[str]] = []
    bm_metas: list[dict] = []
    for p in pages:
        full = (VAULT_ROOT / p["rel_path"]).read_text(encoding="utf-8")
        _, body = split_frontmatter(full)
        title_repeat = "\n".join([p["title"]] * 5)
        text = f"{title_repeat}\n\n{body}"
        corpus_tokens.append(tokenize(text))
        bm_metas.append({"rel_path": p["rel_path"], "title": p["title"]})
    bm25 = BM25Okapi(corpus_tokens)
    bm25_pkl = bench_root / f"bm25_{n}.pkl"
    with open(bm25_pkl, "wb") as f:
        pickle.dump({"bm25": bm25, "metas": bm_metas}, f)
    bm25_build = time.perf_counter() - t0
    bm25_size = du_size(bm25_pkl)

    # Cache query embeddings; we want to attribute embedding cost separately
    # from retrieval cost.
    qe_times = []
    q_embeds: list[list[float]] = []
    for q in queries:
        t0 = time.perf_counter()
        v = embed(q)
        qe_times.append(time.perf_counter() - t0)
        q_embeds.append(v)

    # Preamble doc map for rerank
    got = pre_coll.get(include=["documents"])
    pre_doc_map = dict(zip(got["ids"], got["documents"]))

    # ----- QUERY: preamble dense -----
    pre_query_times = []
    for qv in q_embeds:
        t0 = time.perf_counter()
        pre_coll.query(
            query_embeddings=[qv],
            n_results=10,
            include=["metadatas", "documents", "distances"],
        )
        pre_query_times.append(time.perf_counter() - t0)

    # ----- QUERY: chunk dense -----
    chunk_query_times = []
    for qv in q_embeds:
        t0 = time.perf_counter()
        chunk_coll.query(
            query_embeddings=[qv],
            n_results=QUERY_FAN,
            include=["metadatas", "documents", "distances"],
        )
        chunk_query_times.append(time.perf_counter() - t0)

    # ----- QUERY: BM25 only (excluding embed cost — BM25 doesn't need it) -----
    bm25_query_times = []
    for q in queries:
        t0 = time.perf_counter()
        scores = bm25.get_scores(tokenize_query(q))
        np.argsort(scores)[::-1][:QUERY_FAN]
        bm25_query_times.append(time.perf_counter() - t0)

    # ----- QUERY: hybrid (preamble dense + BM25 + minmax fuse) -----
    hybrid_query_times = []
    for q, qv in zip(queries, q_embeds):
        t0 = time.perf_counter()
        # Dense fan-out
        pre_res = pre_coll.query(
            query_embeddings=[qv], n_results=QUERY_FAN,
            include=["metadatas", "documents", "distances"],
        )
        # BM25 fan-out
        bm_scores = bm25.get_scores(tokenize_query(q))
        bm_idx = np.argsort(bm_scores)[::-1][:QUERY_FAN]
        # Minmax + weighted sum (mirroring hybrid.py)
        dense_raw = {
            pre_res["metadatas"][0][i]["rel_path"]: 1.0 - float(pre_res["distances"][0][i])
            for i in range(len(pre_res["metadatas"][0]))
        }
        bm_raw = {bm_metas[i]["rel_path"]: float(bm_scores[i]) for i in bm_idx}
        # cheap normalization
        def norm(d):
            if not d: return d
            lo, hi = min(d.values()), max(d.values())
            sp = hi - lo or 1.0
            return {k: (v - lo) / sp for k, v in d.items()}
        dn, bn = norm(dense_raw), norm(bm_raw)
        all_paths = set(dn) | set(bn)
        comb = {p: 0.5 * dn.get(p, 0) + 0.5 * bn.get(p, 0) for p in all_paths}
        sorted(comb.items(), key=lambda kv: -kv[1])[:10]
        hybrid_query_times.append(time.perf_counter() - t0)

    # ----- QUERY: hybrid + cross-encoder rerank -----
    rerank_query_times = []
    for q, qv in zip(queries, q_embeds):
        t0 = time.perf_counter()
        # Same hybrid retrieval
        pre_res = pre_coll.query(
            query_embeddings=[qv], n_results=QUERY_FAN,
            include=["metadatas", "documents", "distances"],
        )
        bm_scores = bm25.get_scores(tokenize_query(q))
        bm_idx = np.argsort(bm_scores)[::-1][:QUERY_FAN]
        dense_raw = {
            pre_res["metadatas"][0][i]["rel_path"]: 1.0 - float(pre_res["distances"][0][i])
            for i in range(len(pre_res["metadatas"][0]))
        }
        bm_raw = {bm_metas[i]["rel_path"]: float(bm_scores[i]) for i in bm_idx}
        def norm(d):
            if not d: return d
            lo, hi = min(d.values()), max(d.values())
            sp = hi - lo or 1.0
            return {k: (v - lo) / sp for k, v in d.items()}
        dn, bn = norm(dense_raw), norm(bm_raw)
        all_paths = set(dn) | set(bn)
        comb = {p: 0.5 * dn.get(p, 0) + 0.5 * bn.get(p, 0) for p in all_paths}
        candidates = [p for p, _ in sorted(comb.items(), key=lambda kv: -kv[1])[:RERANK_FAN]]
        # CE rescoring
        pairs = [(q, pre_doc_map.get(c, "")) for c in candidates]
        cross_encoder.predict(pairs, show_progress_bar=False)
        rerank_query_times.append(time.perf_counter() - t0)

    out["preamble"] = {
        "build_time_s": round(pre_build, 3),
        "build_size_bytes": pre_size,
        "build_size_human": fmt_size(pre_size),
        "query_time_ms_median": median_ms(pre_query_times),
        "query_time_ms_p95": p95_ms(pre_query_times),
    }
    out["chunk"] = {
        "build_time_s": round(chunk_build, 3),
        "build_size_bytes": chunk_size,
        "build_size_human": fmt_size(chunk_size),
        "n_chunks": len(chunk_ids),
        "query_time_ms_median": median_ms(chunk_query_times),
        "query_time_ms_p95": p95_ms(chunk_query_times),
    }
    out["bm25"] = {
        "build_time_s": round(bm25_build, 3),
        "build_size_bytes": bm25_size,
        "build_size_human": fmt_size(bm25_size),
        "query_time_ms_median": median_ms(bm25_query_times),
        "query_time_ms_p95": p95_ms(bm25_query_times),
    }
    out["hybrid"] = {
        "query_time_ms_median": median_ms(hybrid_query_times),
        "query_time_ms_p95": p95_ms(hybrid_query_times),
    }
    out["hybrid_rerank"] = {
        "query_time_ms_median": median_ms(rerank_query_times),
        "query_time_ms_p95": p95_ms(rerank_query_times),
    }
    out["query_embed_ms_median"] = median_ms(qe_times)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default=",".join(str(s) for s in DEFAULT_SIZES))
    parser.add_argument("--queries", type=int, default=30,
                        help="number of gold questions to use (default: all)")
    args = parser.parse_args()

    sizes = sorted({int(s) for s in args.sizes.split(",") if s.strip()})

    pages = collect_pages()
    print(f"[bench] all eligible pages: {len(pages)}")
    sizes = [s for s in sizes if s <= len(pages)]
    print(f"[bench] N values to test: {sizes}")

    qa = []
    with open(GOLD_PATH) as f:
        for line in f:
            qa.append(json.loads(line))
    queries = [q["question"] for q in qa][: args.queries]
    print(f"[bench] using {len(queries)} queries")

    print("[bench] loading cross-encoder…")
    ce = CrossEncoder(RERANKER_MODEL, max_length=512)

    all_results = {"sizes": sizes, "n_queries": len(queries),
                   "embedding_model": EMBED_MODEL, "reranker_model": RERANKER_MODEL,
                   "runs": []}

    with tempfile.TemporaryDirectory(prefix="bench_") as tmp:
        bench_root = Path(tmp)
        for n in sizes:
            print(f"\n[bench] === N = {n} ===")
            t0 = time.perf_counter()
            r = benchmark_at_n(n, queries, pages, ce, bench_root)
            print(f"[bench] N={n} done in {time.perf_counter() - t0:.1f}s")
            all_results["runs"].append(r)

    RESULTS_PATH.write_text(json.dumps(all_results, indent=2))
    print(f"\n[bench] wrote {RESULTS_PATH}")

    # Pretty table
    print("\n=== BUILD TIME (s) ===")
    print(f"{'N':>6}  {'preamble':>10} {'chunk':>10} {'BM25':>10}")
    for r in all_results["runs"]:
        print(f"{r['n']:>6}  "
              f"{r['preamble']['build_time_s']:>10.2f} "
              f"{r['chunk']['build_time_s']:>10.2f} "
              f"{r['bm25']['build_time_s']:>10.3f}")

    print("\n=== INDEX SIZE ===")
    print(f"{'N':>6}  {'preamble':>12} {'chunk':>12} {'BM25':>12}")
    for r in all_results["runs"]:
        print(f"{r['n']:>6}  "
              f"{r['preamble']['build_size_human']:>12} "
              f"{r['chunk']['build_size_human']:>12} "
              f"{r['bm25']['build_size_human']:>12}")

    print("\n=== PER-QUERY MEDIAN (ms) — excludes shared query-embedding cost ===")
    print(f"{'N':>6}  {'embed':>8} {'pre':>8} {'chunk':>8} {'BM25':>8} {'hybrid':>8} {'rerank':>10}")
    for r in all_results["runs"]:
        print(f"{r['n']:>6}  "
              f"{r['query_embed_ms_median']:>8.1f} "
              f"{r['preamble']['query_time_ms_median']:>8.2f} "
              f"{r['chunk']['query_time_ms_median']:>8.2f} "
              f"{r['bm25']['query_time_ms_median']:>8.2f} "
              f"{r['hybrid']['query_time_ms_median']:>8.2f} "
              f"{r['hybrid_rerank']['query_time_ms_median']:>10.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
