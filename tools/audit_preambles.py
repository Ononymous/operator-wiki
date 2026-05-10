"""
Audit `## For future Claude` preambles for the lead-with-search-term rule
(CLAUDE.md AI-First Note Rule 1).

Heuristic: the preamble's first sentence should start with the page's primary
searchable term. We approximate by checking whether any significant token from
the H1 title appears in the first 80 chars of the preamble. The check is noisy
— some pages legitimately lead with a synonym (e.g. ai-first-note-format leads
with "preamble triage"), and the audit will flag those as false positives. The
output is a candidate list, not a fix list — humans/agents must judge each.

Usage:
    python audit_preambles.py                  # plain report
    python audit_preambles.py --json paths.json # write paths-only JSON list
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

VAULT = Path(os.environ.get(
    "VAULT_ROOT",
    str(Path(__file__).resolve().parent.parent),
))
WIKI = VAULT / "03-research" / "wiki"

PREAMBLE_RE = re.compile(r"## For future Claude\s*\n(.+?)(?=\n##|\n---)", re.DOTALL)
H1_RE = re.compile(r"^#\s+(.+?)$", re.MULTILINE)
STOPWORDS = {
    "a", "an", "the", "of", "and", "to", "for", "in", "on", "with",
    "as", "by", "at", "from", "vs", "its", "is", "are",
}
ROUTING_FILES = {
    "index.md", "index-llms.md", "index-kb.md", "index-systems.md", "log.md",
}


def title_terms(h1: str) -> list[str]:
    """Extract significant terms from an H1, dropping subtitles after — or :,
    parenthetical aliases, and stopwords."""
    h1 = re.sub(r"\([^)]*\)", "", h1)
    h1 = re.sub(r"[—:].*$", "", h1)
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9'-]*", h1)
    return [t for t in tokens if t.lower() not in STOPWORDS and len(t) > 1]


def preamble_lead(text: str, n: int = 80) -> str | None:
    m = PREAMBLE_RE.search(text)
    if not m:
        return None
    return m.group(1).strip()[:n]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", help="write violations as JSON list to this path")
    args = parser.parse_args()

    violations, ok = [], []
    for md in WIKI.rglob("*.md"):
        if md.name in ROUTING_FILES:
            continue
        text = md.read_text(encoding="utf-8")
        h1m = H1_RE.search(text)
        if not h1m:
            continue
        h1 = h1m.group(1).strip()
        lead = preamble_lead(text)
        if lead is None:
            continue
        rel = md.relative_to(VAULT).as_posix()
        terms = title_terms(h1)
        # Acronyms in parens: e.g. "(GRPO)" → "GRPO"
        terms += re.findall(r"\(([A-Z][A-Z0-9-]+)\)", h1)
        leads = any(t.lower() in lead.lower() for t in terms)
        if leads:
            ok.append(rel)
        else:
            violations.append({"rel_path": rel, "h1": h1, "preamble_lead": lead})

    print(f"OK       : {len(ok)}")
    print(f"VIOLATES : {len(violations)}")
    print()
    for v in violations:
        print(f"  {v['rel_path']}")
        print(f"    H1:  {v['h1'][:60]}")
        print(f"    Pre: {v['preamble_lead']}")

    if args.json:
        Path(args.json).write_text(json.dumps([v["rel_path"] for v in violations], indent=2))
        print(f"\nWrote {len(violations)} paths → {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
