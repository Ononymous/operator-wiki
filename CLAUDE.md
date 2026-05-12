# operator-wiki — Claude Code instructions

This file is the operating spec for an operator-grade Obsidian vault: rules an LLM agent (Claude Code or any MCP-compatible runtime) follows when reading, writing, and querying the vault. **Fork this repo, populate the empty folders with your own content, and edit the `Identity` and `Right now` sections below.**

## Identity (TODO: fill in)

<!-- Two or three lines about who you are and what you work on. The model uses this for tone and to avoid asking redundant background questions every session. Example:

> Software engineer at $COMPANY. Working on [topic]. Located in [city]. Collaborates with Claude Code daily.

-->

## Right now (TODO: update weekly)

<!-- A short rolling list of active deadlines and priorities. The most useful version is 3–6 lines. Example:

> - Active project: [name]
> - Immediate deadline: [thing] by [date]
> - Top task: check 06-tasks/now.md

-->

## Vault map

| Need | Location |
|---|---|
| Fast capture | `00-inbox/` |
| Personal / private notes | `01-self/` |
| Course or coursework material | `02-academics/[course]/` |
| Research papers / concepts (the wiki) | `03-research/wiki/` → `INDEX.md` first |
| Code projects | `04-projects/[name]/CLAUDE.md` (one-per-project) |
| Life / plans / ideas | `05-life/` |
| Active task list | `06-tasks/now.md` |
| External-tool reference docs | `07-reference/` |
| Session history | `logs/` |

## Tool priority — use in this order

1. `obsidian-cli` skill (or any MCP wrapper for Obsidian) — `search query="..."` understands tags, frontmatter, wikilinks
2. `obsidian-cli` `read file="note-name"` — wikilink-style resolution when path is unknown
3. `obsidian-cli` `backlinks file="..."` — traverse the knowledge graph; use before proposing synthesis
4. `Read` — direct file read when exact path is already known
5. `Bash(grep/find)` — bulk raw pattern matching only when the obsidian-cli skill is insufficient
6. Never use `Bash(cat)` — use `Read` instead
7. Never use `Bash(ls)` to discover vault structure — use search

## Query rules — READ THIS BEFORE EVERY QUERY

1. Read `INDEX.md` first — never scan folders blindly
2. Check `03-research/wiki/synthesis/` before drilling into individual pages
3. Drill into 2–3 relevant pages max, then answer
4. Never auto-read private folders (`01-self/`, journal-style content) — wait for explicit request
5. For code projects: check the project's own knowledge artifacts (e.g., a code-graph if one exists) BEFORE reading `src/`
6. Routing pages (`index*.md`, `log.md`, `INDEX.md`, `CLAUDE.md`, `CRITICAL_FACTS.md`) carry no `## For future Claude` preamble — they are navigation, not retrieval candidates. If you encounter a *content* page that lacks a preamble, add one and flag it.

## Auto-synthesis pipeline

After completing any query or ingest, silently check all three:
- Did I read ≥4 wiki pages to answer a single question?
- Did an ingest touch ≥3 existing concept pages with no synthesis page connecting them?
- Did I notice a cross-domain connection I've explained before in a prior context?

If any trigger fires AND no synthesis page already covers it, append this line at the very end of the response (compact, one line):

```
→ Synth: [topic in 5 words] · [[page-a]] + [[page-b]] + [[page-c]] · say "y" to build
```

On any affirmative reply (y / yes / go / build it): immediately run `/synth` for that topic. Propose at most one synthesis per response.

## Note format

- Filenames: kebab-case (`chain-of-thought.md`, not `Chain of Thought.md`)
- Always use `[[wikilinks]]` not `[markdown](links)` for internal links
- Frontmatter on every new note:
  ```yaml
  ---
  title:
  tags: []
  created: YYYY-MM-DD
  type: [note|concept|entity|source|log|task]
  confidence: stated|high|medium|speculation   # omit for non-research notes
  ---
  ```
- Minimum 2 `[[wikilinks]]` per permanent note

## AI-First Note Rules

Every content page follows these at creation (routing pages are exempt — see Query rule 6):

1. **"For future Claude" preamble** — `## For future Claude` block right after the H1. 2–3 sentences, ~50–100 tokens.
   - **Sentence 1 leads with the page's load-bearing search term** (e.g., "Defines the **preamble triage** retrieval pattern…") so vector retrieval surfaces this page rather than peripheral matches that share incidental vocabulary.
   - **Sentence 2** is a key detail: origin (authors+year for papers), the central claim, or the canonical example.
   - **Sentence 3** starts with "Consult when…" — name 2–3 specific scenarios.
   - Routing pages never get one — adding a preamble to an index/log/policy doc puts it in the search collection, where it dominates by vocabulary breadth.
2. **Self-contained** — no reliance on backlinks for context; every page must make sense pulled cold
3. **Confidence markers** — `confidence: stated|high|medium|speculation` in frontmatter; inline `(as of YYYY-MM, source)` on benchmark claims
4. **Mandatory wikilinks** — every entity / concept / project mention gets `[[linked]]`; create stubs for missing targets
5. **Rewrite-not-append** — on ingest, update existing pages rather than creating new ones; one ingest touches 5–15 pages
6. **Bi-temporal source prompt** — when reading or editing an entity page representing a paper, model, or research artifact (`type: entity` and tagged `paper|model|llm|architecture` or similar) that lacks `timeline:` frontmatter, prompt the user for the arXiv URL or paper source before the next ingest of that entity. Goal: populate `timeline:` (event_date, learned_date, source, confidence) so historical queries work. Skip if the user says "skip" or the source is unavailable. Do not block the current task on this — ask once, log the response, move on.

## Index split — when to add a new domain

The default wiki layout splits by domain at `03-research/wiki/concepts-{kb,llms,systems}/`, `entities/`, `sources-{kb,llms,systems}/`, `synthesis/`. Add a new `concepts-{domain}/` + `sources-{domain}/` split when:
- A new domain accumulates ≥40 pages and starts producing irrelevant top-N hits in cross-domain queries, OR
- The wiki crosses ~150 total pages and a single flat `index.md` becomes its own bottleneck

Re-run `tools/audit_preambles.py` and `tools/lint_vault.py` after restructuring to catch broken wikilinks and missing preambles.

## Whole-vault vector-search fallback — when to stop walking the folder tree

The folder-tree-plus-domain-split policy above stays the default. It breaks down for queries that legitimately span the entire vault — for example, "which of the methods I have read about could plausibly help with X?" or "do any of my course notes contradict this finding?". Once the vault crosses ~1000 total pages, walking even a well-pruned folder tree to answer those queries becomes prohibitively expensive: the index router itself becomes the bottleneck.

At that scale, route cross-cutting queries through whole-vault hybrid retrieval over the same Chroma + BM25 indexes that the per-branch retrievers already use. This is a different default mode on the same `search_wiki` endpoint, not a separate retrieval system — no code change required, just a mode flip.

A useful side-effect of the fallback: the folder tree never mixes domains, but the embedding space has no such partition. A query mentioning "neurons" can pull both a neuroscience source and a deep-learning page, because the representations share enough structure to land near each other. Two knobs decide which way that goes:
- **Retrieval threshold** — raise it to suppress weak cross-domain hits, lower it to invite them.
- **Indexed scope** (`ALLOW_PREFIXES`, `DENY_FILES` in `vault-vector-search/config.py`) — narrow to a single domain for strict in-domain search, widen across related domains for deliberate cross-domain mixing.

Both knobs are tunable per query, so a single session can shift between "find everything in this project on X" and "surface anything from anywhere on X" without re-shaping the underlying vault.

## Model selection (suggestion)

- Ingest / summarize / index updates → smaller / faster model (Claude Haiku or equivalent)
- Cross-note synthesis / query answers → primary model (Claude Sonnet or equivalent)

## Session commands

These are the slash commands that operationalize the vault. Implement as Claude Code skills (`.claude/skills/`) or invoke as plain prompts.

### `/resume`

1. Read the 2 most recent files in `logs/`
2. Read `06-tasks/now.md`
3. State: what's active, what's pending, what changed

### `/save [topic]`

1. Create `logs/YYYY-MM-DD-[topic].md` with: what happened, decisions made, pending / next steps, notes modified
2. Update `06-tasks/now.md` if tasks changed
3. Update `INDEX.md` if new wiki pages were created

### `/ingest [file or url]`

Process into `03-research/wiki/` — create or update concept pages, entity pages, source summary.

**Rewrite-not-append:** prefer updating existing concept / entity pages with new information over creating new pages. One ingest should touch 5–15 existing pages. Only create a new page when the topic is genuinely absent from the wiki. Update `03-research/wiki/index.md` and root `INDEX.md`.

### `/synth [topic]`

Build a synthesis page that connects 3+ existing wiki pages on a topic. Live under `03-research/wiki/synthesis/`. Surface new connections that aren't in any single source page; do not just summarize.

### `/sort-inbox`

Process everything in `00-inbox/` — route each note to the right folder. Ask before moving anything ambiguous.

### `/now`

Read `06-tasks/now.md` only. No other files. State the active task list.

### `/capture [text]`

Drop a quick note into `00-inbox/` with timestamped filename. To be sorted later by `/sort-inbox`.

### `/query [question]`

Answer a question against the wiki, following the **Query rules** above. Read `INDEX.md` and `03-research/wiki/synthesis/` first.
