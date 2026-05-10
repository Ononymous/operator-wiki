# INDEX

The always-loaded routing layer. Keep this short — it's read on most queries.

## Active

- [[CRITICAL_FACTS]] — minimum-viable operator state (always-loaded)
- [[06-tasks/now|now]] — active tasks
- [[03-research/wiki/index|wiki/index]] — research wiki entry point

## How to extend

When you add a new wiki domain split (e.g., a new `concepts-{domain}/` group), add a one-line link here. Do not list every individual page — the wiki indexes (`wiki/index.md`, `wiki/index-{domain}.md`) handle that.

## Conventions

- This file is a **routing page** — it carries no `## For future Claude` preamble and is excluded from the vector index.
- Keep entries one-line. Long descriptions belong on the linked page itself.
