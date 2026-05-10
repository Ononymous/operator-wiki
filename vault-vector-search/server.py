"""
MCP server exposing vector search over the indexed vault.

Tool:
    search_wiki(query: str, top_k: int = 5) -> str

Returns a formatted ranked list with rel_path, title, cosine score, preamble
flag, and a short excerpt for triage.

Run as an MCP server (stdio):
    python server.py

Register with Claude Code:
    claude mcp add vault-vector-search -- /abs/path/to/.venv/bin/python /abs/path/to/server.py
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from search import format_hits, search_wiki

mcp = FastMCP("vault-vector-search")


@mcp.tool()
def search_wiki_tool(query: str, top_k: int = 5) -> str:
    """Vector-search the Obsidian wiki for pages relevant to a query.

    Returns a ranked list of candidates with title, relative path, cosine
    score, preamble flag, and a short excerpt for triage. Use as a fallback
    when index/preamble navigation does not surface a candidate, or as a
    sanity check against the primary retrieval path.

    Args:
        query: Natural-language query.
        top_k: Number of candidates to return (default 5, max 20).
    """
    top_k = max(1, min(int(top_k), 20))
    hits = search_wiki(query, top_k=top_k)
    return format_hits(query, hits)


if __name__ == "__main__":
    mcp.run()
