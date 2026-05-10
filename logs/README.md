# logs

Session history. One file per significant working session, named `YYYY-MM-DD-[topic].md`. Created by `/save` (see `CLAUDE.md`).

Each log captures: what happened, decisions made, pending / next steps, notes modified. `/resume` reads the 2 most recent files here to reconstitute operator state at the start of a new session.

This folder is not in the vector index — it's accessed by date order via `/resume`, not by similarity.
