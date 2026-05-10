# 04-projects

One subfolder per code project. Each project gets its own `CLAUDE.md` with project-specific operating rules — they override or extend the root `CLAUDE.md` for work inside that subfolder.

Suggested layout per project:

```
04-projects/[name]/
├── CLAUDE.md      # project-specific instructions
├── PLAN.md        # planning / design notes
├── src/           # actual code (or a clone of the repo)
└── README.md      # public-facing intro
```

Code projects should also point the agent at their own knowledge artifacts (e.g., a code-graph) before reading source. See **Query rule 5** in the root `CLAUDE.md`.
