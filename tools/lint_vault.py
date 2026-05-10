"""
Vault lint: broken wikilinks, orphan pages, slug collisions, missing-index
entries, and content pages without `## For future Claude` preambles.

Operates on the split-domain wiki layout used by this template
(`03-research/wiki/{concepts,entities,sources,synthesis}-{kb,llms,systems}/`).
Cross-references files in `01-self/`, `02-academics/`, `04-projects/`,
`05-life/`, `07-reference/` for wikilink resolution.

Usage:
    python lint_vault.py
    VAULT_ROOT=/path/to/vault python lint_vault.py
"""
from __future__ import annotations

import os
import re
import sys
from collections import defaultdict
from pathlib import Path

VAULT = Path(os.environ.get(
    "VAULT_ROOT",
    str(Path(__file__).resolve().parent.parent),
))
WIKI = VAULT / "03-research" / "wiki"
ROOT_DOCS = ("INDEX.md", "CRITICAL_FACTS.md", "CLAUDE.md")
EXTRA_DIRS = ("01-self", "02-academics", "04-projects", "05-life",
              "07-reference", "Default")
INDEX_FILES = ("index.md", "index-llms.md", "index-kb.md", "index-systems.md")

LINK_RE = re.compile(r"\[\[([^\]\|#]+?)(?:#[^\]\|]+)?(?:\|[^\]]+)?\]\]")
H1_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
PREAMBLE_RE = re.compile(r"^##\s+For future Claude\s*$", re.MULTILINE)

# Targets that show up inside template/example fragments — skip them
TEMPLATE_PLACEHOLDERS = {
    "a", "b", "c", "page-a", "page-b", "page-c", "page-name", "linked",
    "wikilinks", "sources/x", "concepts/x", "schedule",
}


def slugify(s: str) -> str:
    s = s.strip().lower()
    s = s.rsplit("/", 1)[-1]
    s = re.sub(r"\.md$", "", s)
    return s


def slug_alts(s: str) -> set[str]:
    base = slugify(s)
    return {base, base.replace(" ", "-"), base.replace("-", " ")}


def strip_code_blocks(text: str) -> str:
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`[^`]*`", "", text)
    return text


def main() -> int:
    page_index: dict[str, str] = {}
    duplicates: dict[str, list[str]] = defaultdict(list)

    for md in WIKI.rglob("*.md"):
        slug = slugify(md.stem)
        rel = md.relative_to(VAULT).as_posix()
        if slug in page_index:
            duplicates[slug].append(rel)
            duplicates[slug].append(page_index[slug])
        else:
            page_index[slug] = rel

    for top in ROOT_DOCS:
        p = VAULT / top
        if p.exists():
            page_index[slugify(p.stem)] = p.relative_to(VAULT).as_posix()

    for extra in EXTRA_DIRS:
        d = VAULT / extra
        if not d.exists():
            continue
        for md in d.rglob("*.md"):
            slug = slugify(md.stem)
            page_index.setdefault(slug, md.relative_to(VAULT).as_posix())

    inbound: dict[str, set[str]] = defaultdict(set)
    broken: list[tuple[str, str]] = []
    pages_scanned: list[str] = []

    for md in WIKI.rglob("*.md"):
        rel = md.relative_to(VAULT).as_posix()
        try:
            text = md.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        pages_scanned.append(rel)
        scan_text = strip_code_blocks(text)
        for m in LINK_RE.finditer(scan_text):
            target = m.group(1).strip().rstrip("\\")
            if not target or target in TEMPLATE_PLACEHOLDERS:
                continue
            found = None
            for alt in slug_alts(target):
                if alt in page_index:
                    found = alt
                    break
            if found:
                inbound[found].add(rel)
            else:
                broken.append((rel, target))

    for top in ROOT_DOCS:
        p = VAULT / top
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        scan_text = strip_code_blocks(text)
        for m in LINK_RE.finditer(scan_text):
            target = m.group(1).strip().rstrip("\\")
            if not target or target in TEMPLATE_PLACEHOLDERS:
                continue
            for alt in slug_alts(target):
                if alt in page_index:
                    inbound[alt].add(p.name)
                    break

    orphans: list[str] = []
    for md in WIKI.rglob("*.md"):
        if md.name in INDEX_FILES or md.name == "log.md":
            continue
        slug = slugify(md.stem)
        if not inbound.get(slug):
            orphans.append(md.relative_to(VAULT).as_posix())

    indexed: set[str] = set()
    for ifn in INDEX_FILES:
        ip = WIKI / ifn
        if not ip.exists():
            continue
        text = ip.read_text(encoding="utf-8")
        for m in LINK_RE.finditer(text):
            target = m.group(1).strip()
            for alt in slug_alts(target):
                if alt in page_index:
                    indexed.add(alt)
                    break
    missing_from_index: list[str] = []
    for md in WIKI.rglob("*.md"):
        if md.name in INDEX_FILES or md.name == "log.md":
            continue
        if slugify(md.stem) not in indexed:
            missing_from_index.append(md.relative_to(VAULT).as_posix())

    no_preamble: list[str] = []
    for md in WIKI.rglob("*.md"):
        if md.name in INDEX_FILES or md.name == "log.md":
            continue
        text = md.read_text(encoding="utf-8")
        if not PREAMBLE_RE.search(text):
            no_preamble.append(md.relative_to(VAULT).as_posix())

    print(f"Vault scanned: {len(pages_scanned)} wiki pages "
          f"+ {sum(1 for r in ROOT_DOCS if (VAULT / r).exists())} root docs")
    print(f"Page index size: {len(page_index)}")
    print()

    by_target: dict[str, list[str]] = defaultdict(list)
    for src, tgt in broken:
        by_target[tgt].append(src)

    print(f"=== BROKEN WIKILINKS  ({len(by_target)} unique targets, {len(broken)} total refs) ===")
    for tgt, srcs in sorted(by_target.items(), key=lambda kv: -len(kv[1])):
        print(f"  [[{tgt}]]  ({len(srcs)} refs)")
        for s in sorted(set(srcs))[:5]:
            print(f"      from {s}")
        if len(set(srcs)) > 5:
            print(f"      ... +{len(set(srcs)) - 5} more")
    print()

    print(f"=== DUPLICATE FILENAME SLUGS  ({len(duplicates)}) ===")
    for slug, paths in duplicates.items():
        print(f"  {slug}: {sorted(set(paths))}")
    print()

    print(f"=== ORPHAN PAGES (no inbound wikilinks)  ({len(orphans)}) ===")
    for o in sorted(orphans):
        print(f"  {o}")
    print()

    print(f"=== PAGES MISSING FROM INDEX FILES  ({len(missing_from_index)}) ===")
    for m in sorted(missing_from_index):
        print(f"  {m}")
    print()

    print(f"=== CONTENT PAGES MISSING `## For future Claude`  ({len(no_preamble)}) ===")
    for n in sorted(no_preamble):
        print(f"  {n}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
