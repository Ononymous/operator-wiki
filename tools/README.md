# tools

Cross-cutting scripts that audit the wiki (not part of the deployed system).

| Script | Purpose |
|---|---|
| `lint_vault.py` | Broken wikilinks, orphan pages, slug collisions, pages missing from index files, content pages missing `## For future Claude` preambles. |
| `audit_preambles.py` | Flags preambles that don't lead with the page's primary searchable term — the AI-First Note Rule 1 check. Heuristic; can over-flag synonyms. |

## Usage

Both scripts honor `VAULT_ROOT`:

```bash
# from repo/tools
python lint_vault.py
python audit_preambles.py
python audit_preambles.py --json /tmp/violations.json   # paths-only JSON
```

Both run on Python 3.12+ with no external dependencies (stdlib only).
