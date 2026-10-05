"""FastAPI service exposing the RAG pipeline."""

import os
import time
from collections import defaultdict, deque

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from rag.generate import answer_question

app = FastAPI(title="RAG Evaluation Pipeline")

# Per-IP limit. Best-effort and deliberately so: the window lives in process
# memory, so it resets on every cold start and is not shared between concurrent
# Lambda containers. The real ceiling on cost is the function's reserved
# concurrency (infra/main.tf); this only blunts a single impatient client.
RATE_LIMIT_PER_MIN = int(os.environ.get("RATE_LIMIT_PER_MIN", "10"))
_hits: dict[str, deque[float]] = defaultdict(deque)


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if request.url.path != "/ask":
        return await call_next(request)

    client = request.headers.get("x-forwarded-for", "").split(",")[0].strip() or (
        request.client.host if request.client else "unknown"
    )
    now = time.monotonic()
    window = _hits[client]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= RATE_LIMIT_PER_MIN:
        return JSONResponse(
            status_code=429,
            content={"detail": f"Rate limit is {RATE_LIMIT_PER_MIN} questions per minute."},
        )
    window.append(now)
    return await call_next(request)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=3, ge=1, le=10)


class Source(BaseModel):
    chunk_id: str
    citation: str
    section: str | None = None
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    answer, chunks = answer_question(request.question, request.k)
    return AskResponse(
        answer=answer,
        sources=[
            Source(
                chunk_id=c.chunk_id,
                citation=c.cite(),
                section=c.section,
                score=round(c.score, 3),
            )
            for c in chunks
        ],
    )


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _PAGE


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ask the papers</title>
<style>
  :root { color-scheme: light dark; --fg:#111; --muted:#666; --line:#ddd; --bg:#fff; --accent:#0b5; }
  @media (prefers-color-scheme: dark) {
    :root { --fg:#e8e8e8; --muted:#999; --line:#333; --bg:#111; --accent:#3d8; }
  }
  body { font: 16px/1.6 ui-sans-serif, system-ui, sans-serif; color: var(--fg); background: var(--bg);
         max-width: 46rem; margin: 0 auto; padding: 2rem 1rem 4rem; }
  h1 { font-size: 1.4rem; margin: 0 0 .25rem; }
  p.sub { color: var(--muted); margin: 0 0 1.5rem; }
  form { display: flex; gap: .5rem; }
  input { flex: 1; padding: .7rem .8rem; font: inherit; border: 1px solid var(--line);
          border-radius: 6px; background: transparent; color: inherit; }
  button { padding: .7rem 1.1rem; font: inherit; border: 0; border-radius: 6px;
           background: var(--accent); color: #fff; cursor: pointer; }
  button[disabled] { opacity: .5; cursor: progress; }
  .examples { margin: .75rem 0 0; font-size: .9rem; color: var(--muted); }
  .examples button { background: none; color: var(--accent); border: 0; padding: 0 .4rem 0 0;
                     font-size: inherit; text-decoration: underline; cursor: pointer; }
  #answer { margin-top: 2rem; white-space: pre-wrap; }
  ol { padding-left: 1.2rem; }
  li { margin: .35rem 0; font-size: .92rem; }
  .cite { font-variant-numeric: tabular-nums; }
  .sec { color: var(--muted); }
  footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--line);
           font-size: .85rem; color: var(--muted); }
  a { color: var(--accent); }
</style>
</head>
<body>
<h1>Ask the papers</h1>
<p class="sub">Grounded question answering over 48 NLP papers &mdash; the literature review behind a
master&rsquo;s thesis on Bengali coreference resolution.
Answers cite the document and page they came from.</p>

<form id="f">
  <input id="q" name="q" placeholder="How is coreference resolution evaluated?" autocomplete="off" required>
  <button id="go" type="submit">Ask</button>
</form>
<p class="examples">
  Try:
  <button type="button" onclick="fill(this)">What is the BenCoref dataset?</button>
  <button type="button" onclick="fill(this)">Why does back-translation hurt coreference?</button>
  <button type="button" onclick="fill(this)">What is embedding collapse?</button>
</p>

<div id="answer"></div>

<footer>
  Retrieval quality is measured, not assumed &mdash; recall@k, hit-rate@k and MRR on a hand-labelled set.
  <a href="https://github.com/zenith19/rag-eval-pipeline">Source and results on GitHub</a>.
</footer>

<script>
const f = document.getElementById('f'), q = document.getElementById('q'),
      go = document.getElementById('go'), out = document.getElementById('answer');

function fill(b) { q.value = b.textContent; q.focus(); }

f.onsubmit = async (e) => {
  e.preventDefault();
  go.disabled = true;
  out.textContent = 'Thinking\\u2026 (a cold start takes a few seconds)';
  try {
    const r = await fetch('ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q.value, k: 4 })
    });
    if (!r.ok) {
      const e = await r.json().catch(() => ({}));
      out.textContent = e.detail || ('Request failed: ' + r.status);
      return;
    }
    const d = await r.json();
    out.innerHTML = '';
    const p = document.createElement('p');
    p.textContent = d.answer;
    out.appendChild(p);
    const h = document.createElement('p');
    h.innerHTML = '<strong>Sources</strong>';
    out.appendChild(h);
    const ol = document.createElement('ol');
    for (const s of d.sources) {
      const li = document.createElement('li');
      const c = document.createElement('span');
      c.className = 'cite';
      c.textContent = s.citation;
      li.appendChild(c);
      if (s.section) {
        const sec = document.createElement('span');
        sec.className = 'sec';
        sec.textContent = ' \\u2014 ' + s.section;
        li.appendChild(sec);
      }
      ol.appendChild(li);
    }
    out.appendChild(ol);
  } catch (err) {
    out.textContent = 'Could not reach the service: ' + err.message;
  } finally {
    go.disabled = false;
  }
};
</script>
</body>
</html>
"""
