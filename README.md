# v-call — AI Meeting Intelligence Platform

Production-grade pipeline that turns raw meeting recordings (Google Meet, Zoom) into
structured intelligence: diarized transcripts, action items, decisions, sentiment,
follow-up emails — pushed into your SaaS stack (Jira, Notion, Zoho CRM/Desk, Freshdesk)
via a resilient, multi-tenant, async-first architecture.

## Architecture at a glance

```
                ┌────────────┐   ┌──────────────┐   ┌─────────────────┐
  Meet/Zoom ───▶│  FastAPI    │──▶│  Celery       │──▶│  Transcription   │
  uploads/webhooks│  (async)   │   │  + DLQ/Redis  │   │  Faster-Whisper  │
                └─────┬──────┘   └──────┬───────┘   │  + pyannote      │
                      │ WebSocket       │ progress  └────────┬────────┘
                      ▼                 ▼                    ▼
                ┌───────────────────────────────────────────────────┐
                │        LangGraph stateful multi-agent graph        │
                │  extraction → sentiment → email → dispatcher       │
                └────────────────────────────┬──────────────────────┘
                                             ▼
                ┌───────────────────────────────────────────────────┐
                │ Integrations: MCP (Calendar/Gmail) + REST          │
                │ Jira · Notion · Zoho CRM/Desk · Freshdesk          │
                └───────────────────────────────────────────────────┘
```

## Tech stack

| Concern | Choice |
| --- | --- |
| API | FastAPI (async-native), Pydantic v2 strict |
| DB | PostgreSQL 15 + SQLAlchemy 2.0 async + Alembic |
| Cache / broker | Redis |
| Tasks | Celery + Flower + Dead-Letter queue |
| ASR / diarization | Faster-Whisper + pyannote.audio |
| Agents | LangGraph (typed state, conditional edges, retries) |
| Integrations | Async MCP clients + `tenacity`-guarded REST connectors |
| Compliance | DPDP Act 2023 / GDPR audio scrubber + PII redaction |
| Observability | OpenTelemetry + LangSmith |

## Quick start

```bash
# 1. Configure secrets
cp .env.example .env
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
python -c "import secrets; print(secrets.token_urlsafe(48))"   # JWT_SECRET

# 2. Boot the full stack (web, worker, beat, flower, redis, postgres)
./build.sh compose

# 3. Apply migrations
docker compose exec web alembic upgrade head

# 4. Open docs / monitoring
open http://localhost:8000/docs       # FastAPI Swagger
open http://localhost:5555            # Flower
```

### Local (non-Docker) dev

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,ai]"          # add ,full for torch-based transcription
alembic upgrade head
uvicorn app.main:app --reload
```

## Observability

Logging is powered by **structlog over stdlib**: every module logger
(`app/*`, `vcall.access`) emits the same record shape — `ts`, `level`,
`logger`, `event` — and `RequestContextMiddleware` binds the `request_id` so
every line during a request is correlated.

- Human-readable console output (default)
- `LOG_JSON=true` → single-line JSON for aggregators
- `LOG_LEVEL` controls verbosity (default `INFO`); OpenTelemetry tracing is
  available via the `observability` extra

## Module map


| Path | Responsibility |
| --- | --- |
| `app/core/` | Config, security (Fernet, JWT), rate-limiting, logging/OTel |
| `app/db/` | Async SQLAlchemy models, sessions, repositories |
| `app/services/` | Audio validation, chunking, transcription orchestration |
| `app/agents/` | LangGraph state, nodes (extraction/sentiment/email/dispatch) |
| `app/integrations/` | MCP clients + Jira/Notion/Zoho/Freshdesk connectors |
| `app/workers/` | Celery app, queues, beat schedule, DLQ |
| `app/api/` | Routes, deps, WebSocket progress streaming |
| `alembic/` | Versioned schema migrations |
| `tests/` | Unit + async integration suite |

## Security & compliance

- Multi-tenant row-level isolation via `tenant_id` on every domain table.
- Integration tokens encrypted at rest with Fernet symmetric encryption; masked in logs/APIs.
- OAuth2 (JWT bearer) auth with `slowapi` global rate limiting and strict CORS.
- Automated **DPDP/GDPR scrubber**: raw audio is securely deleted after transcription
  (configurable retention), and PII is redacted from transcripts when enabled.

## License

Proprietary — all rights reserved.
