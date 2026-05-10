# 03-research/wiki

The research wiki — the load-bearing knowledge structure. Every page here follows the AI-First Note Rules in `../../CLAUDE.md`.

## Subfolders

| Folder | Contents |
|---|---|
| `concepts-{kb,llms,systems}/` | Concept pages — ideas, paradigms, design principles. Domain split by `kb` (knowledge-base systems), `llms` (language models / NLP / ML), `systems` (computer systems & architecture). |
| `entities/` | People, organizations, products. Cross-domain (people may publish across `llms` and `systems`). |
| `sources-{kb,llms,systems}/` | One-page summaries of source material — papers, blog posts, GitHub repos. Domain split mirrors `concepts-`. |
| `synthesis/` | Synthesis pages connecting 3+ existing pages. Built by `/synth`. |

## Routing pages (no preamble, excluded from vector index)

- `index.md` — top-level wiki entry point
- `index-{kb,llms,systems}.md` — per-domain indexes
- `log.md` — append-only event log (synth runs, ingests, lint passes)

## When to add a new domain split

See the **Index split** section of `../../CLAUDE.md` — roughly, when a single domain crosses ~40 pages or the wiki crosses ~150 total pages.
