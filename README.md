# RAG Backend (FastAPI)

Document ingestion + conversational RAG with interview booking.

## Stack
FastAPI · Qdrant (vectors) · SQLite/SQLAlchemy (metadata + bookings) · Redis (chat memory) · Sentence Transformers (local embeddings) · Ollama (local chat LLM; any OpenAI-compatible server works)

## Prerequisites
- Python 3.11+
- Docker Desktop (for Qdrant and Redis)
- [Ollama](https://ollama.com) with a chat model pulled: `ollama pull llama3.2:3b`

## Run
```bash
docker compose up -d            # Qdrant + Redis
ollama pull llama3.2:3b         # local chat model
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env            # Windows: copy .env.example .env (defaults use Ollama; no API key needed)
uvicorn app.main:app --reload
```
Docs: http://localhost:8000/docs

The first chat reply can be slow while the local model loads into memory.

## Configuration
Settings are read from `.env` (see `.env.example`):

| Variable | Purpose | Default |
|---|---|---|
| `LLM_BASE_URL` | OpenAI-compatible chat endpoint | `http://localhost:11434/v1` (Ollama) |
| `LLM_API_KEY` | Key for that endpoint (Ollama ignores it) | `ollama` |
| `LLM_CHAT_MODEL` | Chat model name | `llama3.2:3b` |
| `LOCAL_EMBEDDING_MODEL` | Sentence Transformers model | `sentence-transformers/all-MiniLM-L6-v2` |
| `EMBEDDING_DIM` | Must match the embedding model | `384` |
| `QDRANT_URL` / `REDIS_URL` / `DATABASE_URL` | Service locations | local defaults |

Changing the embedding model changes the vector size, so delete the Qdrant collection and re-ingest documents afterwards.

## APIs
**POST /api/v1/documents** (multipart): `file` (.pdf/.txt), `strategy` (`fixed` | `sentence`), `chunk_size`, `chunk_overlap`.

**POST /api/v1/chat** (JSON): `{"session_id": "abc", "message": "..."}`

```bash
curl -F file=@doc.pdf -F strategy=sentence localhost:8000/api/v1/documents
curl -X POST localhost:8000/api/v1/chat -H 'Content-Type: application/json' \
  -d '{"session_id":"u1","message":"What is the notice policy?"}'
curl -X POST localhost:8000/api/v1/chat -H 'Content-Type: application/json' \
  -d '{"session_id":"u1","message":"Book an interview for Sugam, sugam@example.com, tomorrow 3pm"}'
```

## Design
- **Chunking**: `fixed` (sliding char window) vs `sentence` (sentence-aware packing with overlap), selected via strategy map in `services/chunking.py`.
- **Custom RAG** (`services/rag.py`): condense follow-up into standalone query using Redis history -> embed -> Qdrant search -> grounded generation. No RetrievalQAChain.
- **Memory**: Redis list per session, trimmed to last N messages with TTL.
- **Booking** (`services/booking.py`, `services/chat.py`): the LLM extracts name/email/date/time as JSON each turn; slots accumulate in Redis across turns, are validated (email, future date, HH:MM), missing fields are asked for, and the completed booking is saved to SQL.
- **Providers**: embeddings run locally with all-MiniLM-L6-v2. Chat uses any OpenAI-compatible endpoint (Ollama by default), so switching to a hosted model is a config change, not a code change.
- **Layering**: `api` (HTTP) -> `services` (logic) -> `db`/vector store/Redis. Dependencies are wired in `core/deps.py`.

## Limitations
- Small local models can return malformed JSON during booking extraction. The code falls back safely, and the user can simply resend the message.
- Answer quality depends on the chosen chat model.

## Possible improvements
Alembic migrations, async clients, authentication, streaming responses, document deletion, and integration tests against real Qdrant and Redis.

## Tests
```bash
pip install -r requirements-dev.txt
pytest
```
Tests use in-memory SQLite and fakes for the LLM, Redis memory, RAG and Qdrant, so no external services are needed.

## CI
GitHub Actions (`.github/workflows/ci.yml`) compiles the app and runs the tests on every push and pull request.

## Trying the API
- **Postman**: import `postman/rag-backend.postman_collection.json` (set the `file` field on the upload requests).
- **Shell**: `./scripts/demo.sh path/to/sample.txt` runs upload -> question -> follow-up -> booking against a running server.
- **Swagger UI**: http://localhost:8000/docs