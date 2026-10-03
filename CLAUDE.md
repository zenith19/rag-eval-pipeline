# rag-eval-pipeline

Grounded question answering over the ~48 NLP papers behind a master's thesis on Bengali
coreference resolution. The point of the project is not that RAG works — it is that retrieval
quality is **measured**, with a from-scratch harness (recall@k, hit-rate@k, MRR).

Plan: four releases — R1 correct+ship, R2 measure+improve, R3 operate, R4 harden.
Decisions live in `docs/decisions/`.

## Environment

- **Use `.venv`.** `myenv/` is a stale, half-built environment left over from a directory move;
  its scripts point at a path that no longer exists and it is missing most dependencies.
- Qdrant runs in Docker: `docker compose up -d` (REST on :6333).
- Generation uses Amazon Bedrock (`eu-central-1`) and needs AWS credentials.

## Invariants — breaking these corrupts data silently

1. **One retrieval path.** Every consumer — eval harness, API, agent, MCP tools, label helper —
   goes through `DenseRetriever.retrieve_chunks()`. `retrieve()` is derived from it so offline
   scores and production results cannot diverge. Never add a second query site.
   Guarded by `tests/test_retrieval_parity.py`. See `docs/decisions/001`.
2. **Never rename or delete files in `data/`.** `make_chunk_id()` hashes `"<filename>:<index>"`,
   so a rename silently invalidates every gold label pointing into that file. *Adding* files is
   safe. Six of the original nine filenames contain literal newlines — leave them alone.
3. **Changing `CHUNK_SIZE` / `CHUNK_OVERLAP` re-labels the eval set.** Chunk IDs survive, but the
   text behind them moves. Treat it as a re-labelling exercise, not a tweak.
4. **Gold labels are chunk IDs, not text.** The eval harness never needs document content; keep
   it decoupled from the store.

## Commands

```bash
docker compose up -d                        # Qdrant
.venv/bin/python -m rag.build_index         # (re)build the index — needed after corpus changes
.venv/bin/python -m eval.run_eval           # score retrieval
.venv/bin/python -m pytest tests/ -q        # full suite (integration tests skip without Qdrant)
.venv/bin/python -m pytest tests/ -q -m "not integration"   # fast subset
.venv/bin/uvicorn api.main:app --reload     # API
.venv/bin/python scripts/fetch_corpus.py --dry-run          # corpus, without downloading
git config core.hooksPath .githooks          # enable the shared pre-commit hook (once per clone)
```

## Agent tooling

- `/eval` (`.claude/skills/eval/SKILL.md`) — runs the retrieval eval against the committed
  baseline in `eval/baseline.json` and reports the delta. Never update the baseline to turn a
  red run green.
- `.claude/hooks/protect-corpus.sh` — a PreToolUse hook that blocks edits to `data/*.pdf`,
  enforcing invariant 2 above. Adding files stays allowed.
- `.githooks/pre-commit` — fast tests plus a duplicate-dict-key check (which caught a real bug
  in `scripts/corpus_sources.py`). Enable with `git config core.hooksPath .githooks`.
- No subagents: nothing here justifies the coordination cost, and inventing a use would be
  decoration rather than engineering.

## Conventions

- One PR per step of the plan; squash merge, keeping the first commit message.
- Any non-obvious decision gets `docs/decisions/NNN-*.md`:
  observation → hypothesis → decision → trade-off → measured validation.
- Claims about quality come with a measurement. No number in the README that isn't reproducible
  by `eval/run_eval.py`.
- Imports that cost seconds belong inside the function that needs them — `langchain_text_splitters`
  alone costs ~11s and was being paid by the API, which never chunks anything.
