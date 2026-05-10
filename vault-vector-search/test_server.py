"""End-to-end smoke test of the MCP server via stdio: spawn server.py,
list tools, call search_wiki with one query, verify a result is returned."""
import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> int:
    here = Path(__file__).parent
    params = StdioServerParameters(
        command=str(here / ".venv" / "bin" / "python"),
        args=[str(here / "server.py")],
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            tool_names = [t.name for t in tools.tools]
            print(f"[test] tools: {tool_names}")
            assert "search_wiki_tool" in tool_names, tool_names

            res = await session.call_tool(
                "search_wiki_tool",
                {"query": "compile-once wiki vs RAG", "top_k": 3},
            )
            text = res.content[0].text
            print("[test] tool output (first 600 chars):")
            print(text[:600])
            assert "Top 3 matches" in text
            assert "compile-once" in text.lower() or "wiki-pattern" in text.lower()

    print("[test] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
