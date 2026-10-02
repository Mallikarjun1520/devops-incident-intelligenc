# DevOps Incident Intelligence Platform

An advisory tool that helps on-call engineers understand production incidents quickly. Paste logs or an
incident description and the platform extracts technical signals, retrieves similar **historical incidents**
from ChromaDB, asks Groq to reason about the *current* incident using only that evidence, and shows a
structured assessment in an SRE-style dashboard.

It is the production-oriented successor to the `tcs_hack` hackathon MVP and reuses its 390-incident
knowledge base (plus 10 additional curated incidents).

> **Advisory only.** The platform never executes shell commands, `kubectl`, cloud API calls, database
> changes, restarts, deployments or pipeline changes. Suggested commands are shown flagged as
> *"requires human review"*.

## Features

- **Ingestion**: paste text, or upload `.txt` / `.log` / `.json` (UI and API).
- **Demo workflow**: load the included production checkout incident from the dashboard, then analyze it.
- **Conservative preprocessing**: whitespace normalisation, stack traces/indentation/timestamps preserved
  for the LLM; secrets (passwords, tokens, AWS keys) masked before anything leaves your machine.
- **Signal extraction**: service, environment, category, error types (CrashLoopBackOff, OOMKilled, AccessDenied,
  ...), affected components (Kubernetes, EKS, IAM, PostgreSQL, Redis, ingress, ...), severity indicators.
- **Retrieval**: local `sentence-transformers` embeddings + ChromaDB (cosine). Results below a similarity
  threshold are discarded, so irrelevant history is never sent to the LLM.
- **Context construction**: the LLM sees only ID, title, severity, category, service, symptoms, root cause,
  resolution and preventive actions of each retrieved incident - never whole documents.
- **Grounded reasoning**: strict prompt (no invented facts/logs/commands, inference vs. observation, safe
  diagnostics first). Structured JSON output validated with Pydantic.
- **Severity signal index**: a deterministic 1-100 score based on explicit priority declarations,
  availability failures, production context and observed ERROR/FATAL lines. Each contribution is shown;
  no score is shown when the input contains no severity-driving evidence.
- **Server-side guarantees**: evidence IDs the model cites but that were not retrieved are removed;
  a human-review warning is always present; model scores are clamped to 0-100.
- **Graceful degradation**: LLM, ChromaDB or embedding failures never crash the app; evidence and the
  deterministic severity signal index are still returned when Groq fails.
- **Dashboard**: evidence-based severity index, reported severity and confidence, summary, root cause,
  components, symptoms, extracted facts, reasoning, resolution / prevention / verification steps,
  historical evidence with similarity, a read-only incident assistant, warnings, plus loading / error / empty states.
- **Guarded incident assistant**: answers from a signed, expiring snapshot of the current incident,
  its completed analysis and retrieved historical records. It has no tools, rejects operational requests,
  redacts credentials, and treats logs and chat history as untrusted data.

## Architecture

```
 Browser (HTML / CSS / vanilla JS, served by FastAPI at "/")
        |  POST /api/analyze  (text, or .txt/.log/.json upload)
        v
 +----------------------------- FastAPI (app/api/routes.py) ------------------------------+
 |                                                                                          |
 |  IncidentService (app/services/incident_service.py)  - orchestration only               |
 |    |                                                                                     |
 |    |-1-> parser.py            clean text, redact secrets, extract signals                |
 |    |                          -> cleaned_text (for LLM) + retrieval_query (for embedding)|
 |    |                                                                                     |
 |    |-2-> embedding_service.py  sentence-transformers (local, all-MiniLM-L6-v2)           |
 |    |-3-> retrieval_service.py  ChromaDB cosine search -> top_k above MIN_SIMILARITY      |
 |    |                          -> context construction (only useful fields)               |
 |    |                                                                                     |
 |     |-4-> llm_service.py       Groq (strict prompt) -> JSON -> Pydantic validation        |
 |    |                                                                                     |
 |    '-5-> grounding           drop un-retrieved evidence IDs, add advisory warnings       |
 +------------------------------------------------------------------------------------------+
        |
        v
 { success, analysis{...}, severity_assessment{...}, similar_incidents[...], extracted_signals{...} }
```

Retrieval and reasoning are separate services: retrieval decides *what evidence exists*, the LLM only
reasons over what it is handed.

## Tech stack

Python, FastAPI, Pydantic, Uvicorn - ChromaDB, sentence-transformers, Groq API - python-dotenv -
pytest - plain HTML/CSS/JavaScript (no frameworks).

## Directory structure

```
devops-incident-intelligence/
├── app/
│   ├── main.py                    # app factory, serves the frontend
│   ├── config.py                  # env-based settings
│   ├── api/routes.py              # HTTP routes
│   ├── models/schemas.py          # Pydantic models
│   ├── services/
│   │   ├── parser.py              # preprocessing + signal extraction
│   │   ├── embedding_service.py   # local embeddings
│   │   ├── retrieval_service.py   # ChromaDB + context construction
│   │   ├── llm_service.py         # Groq prompt, call, JSON validation
│   │   ├── severity_service.py    # deterministic evidence-based signal index
│   │   └── incident_service.py    # pipeline orchestration
│   └── utils/logger.py
├── frontend/                      # index.html, style.css, script.js
├── data/
│   ├── incidents/                 # historical incidents (*.json) loaded by the ingest script
│   └── sample_incident.txt        # try this one in the UI
├── scripts/ingest_incidents.py
├── tests/                         # pytest (no network, no API key)
├── chroma_db/                     # vector store (generated, git-ignored)
├── .env.example  requirements.txt  run.bat  FINAL_SETUP.md
```

## Installation

Python 3.10 - 3.12 is recommended. The first ingestion downloads the embedding model (~90 MB).

```bash
python -m venv venv
venv\Scripts\activate           # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env          # macOS/Linux: cp .env.example .env
# edit .env and set GROQ_API_KEY
python scripts/ingest_incidents.py
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>. Exact PowerShell commands are in [FINAL_SETUP.md](FINAL_SETUP.md).
Use **Load demo incident** in the dashboard to preview the included checkout outage scenario before analysis.

### Using VS Code Live Server

Keep the FastAPI process running as above. In VS Code, right-click `frontend/index.html` and choose
**Open with Live Server**. The page detects Live Server on localhost and sends API requests to
`http://127.0.0.1:8000`; local Live Server origins are enabled by the API's CORS configuration.
If the API runs on a different address, pass it as a query parameter, for example
`?api=http://127.0.0.1:8001`. Opening the HTML with `file://` alone will not start the API.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | *(empty)* | Groq API key. Keep it private; without it the deterministic severity index and retrieval still work. |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | Groq model ID used for reasoning. |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model. If you change it, re-run ingestion with `--reset`. |
| `CHROMA_PATH` | `./chroma_db` | ChromaDB directory (relative paths resolve against the project root). |
| `TOP_K` | `3` | Max similar incidents retrieved (1-10). |
| `MIN_SIMILARITY` | `0.30` | Minimum cosine similarity for an incident to count as relevant. |

### Severity signal index

The dashboard score is deterministic and separate from the LLM's qualitative severity. It starts at 1,
adds 75 for an explicit critical/SEV-1/P0/outage declaration (or 55 for SEV-2/P1/major impact), 35 for a
detected availability failure, 10 when that failure is explicitly in production, and 5 per observed
ERROR/FATAL log line (up to 20). The score is capped at 100. Bands are Low 1-24, Moderate 25-49, High
50-74 and Critical 75-100. With no matching evidence, the score is unavailable. These are transparent
text-signal rules, not a substitute for SLO burn, affected-user counts, or an organization's severity policy.

## ChromaDB setup and ingestion

ChromaDB runs embedded (no server). `scripts/ingest_incidents.py` reads every `*.json` file in
`data/incidents/`, validates records, embeds them and upserts them into the `incidents` collection
(cosine space). It is safe to re-run.

```bash
python scripts/ingest_incidents.py            # upsert all incidents
python scripts/ingest_incidents.py --reset    # rebuild the collection from scratch
python scripts/ingest_incidents.py --path my_incidents/
```

Each incident has: `incident_id`, `title`, `severity`, `category`, `service`, `environment`, `description`,
`root_cause`, `resolution`, `preventive_actions` (plus optional `symptoms`, `logs`, `affected_components`).
The embedded text describes *what happened* (symptoms, logs, cause), not the fix, so new incidents match
on symptoms. You can also add incidents at runtime through `POST /api/ingest`.

## API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Dashboard |
| GET | `/health` | Component status (never fails because Groq is unreachable) |
| POST | `/api/analyze` | Analyse an incident: JSON `{"text": "..."}` or multipart `file` (`.txt/.log/.json`) |
| POST | `/api/assistant` | Ask about the latest signed analysis context: `{"question": "...", "context_token": "...", "history": []}` |
| POST | `/api/ingest` | Add historical incidents (JSON object / list / `{"incidents": [...]}` or `.json` upload) |
| GET | `/api/incidents/{incident_id}` | Fetch a stored historical incident |

Status codes for `/api/analyze`: `200` success; `400` empty or malformed input; `413` too large; `415`
unsupported file type; `502`/`503`/`429` Groq problems (missing/invalid key, API failure, rate limit,
malformed model output) - these still include `similar_incidents`.
`POST /api/assistant` accepts only a signed analyzer context (valid for one hour); changed, expired, or
server-restarted contexts must be re-analyzed. Its replies are limited to the current incident, its
analysis, retrieved incidents and the assistant's read-only policy. It has no execution or external-system
tools; operational-action and credential-disclosure requests are refused.

### Example request

```bash
curl -X POST http://127.0.0.1:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "ERROR orders-api: PAYMENT_GATEWAY_KEY is not set\nBack-off restarting failed container\norders-api-6c9d7f8b5-k2m4n 0/1 CrashLoopBackOff 6 11m"}'

# or upload a file
curl -X POST http://127.0.0.1:8000/api/analyze -F "file=@data/sample_incident.txt"
```

### Example response (abridged; analysis values vary)

```json
{
  "success": true,
  "severity_assessment": {
    "score": 56,
    "level": "High",
    "evidence": ["Availability failure detected: CrashLoopBackOff (+35)", "Production environment explicitly detected (+10)", "2 ERROR/FATAL-level log line(s) (+10, capped at +20)"],
    "basis": "Rule-based signal index (1-100); not a measure of SLO burn or users affected."
  },
  "analysis": {
    "incident_summary": "orders-api pods fail at startup after release v2.8.0 ...",
    "severity": "High",
    "severity_score": 82,
    "confidence_score": 88,
    "affected_components": ["orders-api", "Kubernetes"],
    "observed_symptoms": ["CrashLoopBackOff", "503 errors on checkout"],
    "root_cause": "Inference: missing PAYMENT_GATEWAY_KEY environment variable ...",
    "reasoning": "Observed: the log states the variable is not set. Inference: matches CUR-001 ...",
    "recommended_resolution": ["Run `kubectl describe pod <pod> -n <namespace>` to confirm ..."],
    "preventive_actions": ["Validate required env vars/secrets in CI before deploy"],
    "verification_steps": ["Confirm all replicas are Ready and the error rate returns to baseline"],
    "warnings": ["Human review required. This analysis is AI-generated and advisory only; ..."],
    "evidence_incidents": ["CUR-001", "INC-001"]
  },
  "similar_incidents": [
    {"incident_id": "CUR-001", "title": "EKS Pod CrashLoopBackOff After Missing Secret Key",
     "severity": "High", "category": "Kubernetes", "service": "orders-api",
     "similarity": 0.81, "distance": 0.19, "symptoms": ["..."], "root_cause": "...",
     "resolution": ["..."], "preventive_actions": ["..."]}
  ],
  "extracted_signals": {"service": "orders-api", "environment": "Production", "category": "Kubernetes",
                        "error_types": ["CrashLoopBackOff"], "affected_components": ["Kubernetes"], "...": "..."}
}
```

On a Groq failure, the error response still includes `severity_assessment` and parsed signals when available.

## Testing

```bash
python -m pytest
```

Tests need no network, no real embedding model and no Groq key: embeddings use a small deterministic
stand-in encoder and the Groq client is replaced with a fake. They cover health, parsing, severity scoring, retrieval,
the analyze endpoint, empty/malformed input, file uploads, every LLM failure mode, malformed LLM JSON,
and ChromaDB failure.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Header shows "Degraded"; hover for details | Usually the knowledge base is empty (run the ingest script) or `GROQ_API_KEY` is unset. |
| "GROQ_API_KEY is not set" | Add the key to `.env` and restart uvicorn. |
| "Groq rejected the API key" | Check the key; remove stray quotes/spaces in `.env`. |
| "Groq model ... was not found" | Set `GROQ_MODEL` to a model your account can access. |
| Ingest fails while loading the embedding model | The first run needs internet access to download it (Hugging Face). Check proxy/firewall. |
| "No sufficiently similar historical incidents" | Nothing cleared `MIN_SIMILARITY`; lower it slightly in `.env` or add relevant incidents. |
| Changed `EMBEDDING_MODEL` and results look wrong | Re-run `python scripts/ingest_incidents.py --reset` (vectors from different models are not comparable). |
| `pip install` fails for chromadb/torch on very new Python | Use Python 3.11 or 3.12. |

## Safety considerations

- No code path executes commands or touches infrastructure; the LLM only produces text.
- The prompt forbids fabricated logs/commands, requires warnings for destructive steps, and treats the
  incident text as untrusted data (log lines that say "ignore your instructions" are not followed).
- Secrets in submitted text are masked before being sent to Groq. This is pattern-based and
  best-effort: **do not paste data you are not allowed to share with the Groq API.**
- The severity index is a transparent, deterministic signal score, not an objective measure of customer
  impact or SLO burn; it uses only indicators present in the submitted text. Reported severity and
  confidence are shown separately from that index.
- `/api/ingest` and `/api/analyze` are unauthenticated. Run on localhost or add authentication before
  exposing the service.

## Extension points (not implemented)

`IncidentService.analyze_text(text)` is the single entry point for any trigger, so these can be added
without restructuring: n8n / PagerDuty / Alertmanager / Slack / Teams webhooks (new routes that call it),
hybrid retrieval and reranking (inside `RetrievalService.retrieve_similar_incidents`), a local LLM
(a second class with the same `analyze(parsed, similar)` method next to `LLMService`), authentication
(FastAPI dependency on the routers), incident history and audit logs (wrap `analyze_text`), and a human
approval workflow (consume `recommended_resolution` downstream).

## Future improvements

Hybrid BM25 + vector retrieval, cross-encoder reranking, incident history and feedback ("was this
helpful?") to improve retrieval, streaming responses, auth and audit logging, chat/ticketing integrations,
and an approval workflow for remediation.
