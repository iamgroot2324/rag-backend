# RAG Backend — FastAPI

A backend service for **document ingestion, conversational Retrieval-Augmented Generation (RAG), Redis-backed chat memory, and interview booking**.

## Stack

* **FastAPI** — REST API
* **Qdrant** — vector database
* **SQLite + SQLAlchemy** — document metadata and interview bookings
* **Redis** — conversational memory and booking state
* **Sentence Transformers** — local document embeddings
* **Ollama** — local chat LLM through an OpenAI-compatible API
* **Pytest** — automated tests
* **Docker Compose** — Qdrant and Redis services

---

## Features

### Document ingestion

* Supports `.pdf` and `.txt` files
* Extracts document text
* Two selectable chunking strategies:

  * `fixed` — sliding character window
  * `sentence` — sentence-aware chunking with overlap
* Generates local embeddings using Sentence Transformers
* Stores vectors in Qdrant
* Stores document metadata in SQLite

### Conversational RAG

* Custom RAG pipeline implemented without `RetrievalQAChain`
* Uses Redis for multi-turn conversation history
* Condenses follow-up questions into standalone search queries
* Retrieves relevant chunks from Qdrant
* Generates grounded answers using the retrieved context
* Returns source document information with answers

### Interview booking

* Detects interview-booking requests
* Extracts:

  * Name
  * Email
  * Date
  * Time
* Maintains incomplete booking information across turns using Redis
* Validates email, date, and time
* Rejects dates in the past
* Supports booking cancellation
* Stores completed bookings in SQLite

---

## Project Structure

```text
rag-backend/
├── app/
│   ├── api/
│   │   ├── chat.py
│   │   └── ingestion.py
│   ├── core/
│   │   ├── config.py
│   │   └── deps.py
│   ├── db/
│   │   ├── models.py
│   │   └── session.py
│   ├── schemas/
│   │   ├── chat.py
│   │   └── ingestion.py
│   ├── services/
│   │   ├── booking.py
│   │   ├── chat.py
│   │   ├── chunking.py
│   │   ├── extraction.py
│   │   ├── ingestion.py
│   │   ├── llm.py
│   │   ├── memory.py
│   │   ├── rag.py
│   │   └── vector_store.py
│   └── main.py
├── postman/
│   └── rag-backend.postman_collection.json
├── scripts/
│   └── demo.sh
├── tests/
├── .env.example
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
└── README.md
```

---

## Prerequisites

* Python **3.11+**
* Docker Desktop
* Ollama

Install Ollama and pull the chat model:

```bash
ollama pull llama3.2:3b
```

---

## Running the Project

### 1. Start Qdrant and Redis

From the project root:

```bash
docker compose up -d
```

This starts:

* Qdrant on `localhost:6333`
* Redis on `localhost:6379`

### 2. Start Ollama

Make sure Ollama is running and the model is available:

```bash
ollama pull llama3.2:3b
```

### 3. Create the Python virtual environment

#### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Create the environment file

#### Windows PowerShell

```powershell
Copy-Item .env.example .env
```

#### Linux/macOS

```bash
cp .env.example .env
```

The default configuration uses Ollama locally, so no external LLM API key is required.

### 6. Start the API

```bash
python -m uvicorn app.main:app
```

The API will be available at:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

> The first chat request can take longer while the local model loads into memory.

---

## Configuration

Configuration is loaded from `.env`. See `.env.example` for the available settings.

| Variable                | Purpose                         | Default                                  |
| ----------------------- | ------------------------------- | ---------------------------------------- |
| `LLM_BASE_URL`          | OpenAI-compatible chat endpoint | `http://localhost:11434/v1`              |
| `LLM_API_KEY`           | API key for the chat endpoint   | `ollama`                                 |
| `LLM_CHAT_MODEL`        | Chat model                      | `llama3.2:3b`                            |
| `LOCAL_EMBEDDING_MODEL` | Sentence Transformers model     | `sentence-transformers/all-MiniLM-L6-v2` |
| `EMBEDDING_DIM`         | Embedding vector dimension      | `384`                                    |
| `QDRANT_URL`            | Qdrant service URL              | `http://localhost:6333`                  |
| `QDRANT_COLLECTION`     | Qdrant collection name          | `documents`                              |
| `REDIS_URL`             | Redis service URL               | `redis://localhost:6379/0`               |
| `DATABASE_URL`          | SQLAlchemy database URL         | `sqlite:///./app.db`                     |

### Embedding model

The default embedding model is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

It produces 384-dimensional vectors.

If the embedding model or vector dimension is changed, the existing Qdrant collection must be recreated and the documents re-ingested.

---

## API Endpoints

### Health Check

```http
GET /health
```

Returns the application health status.

---

### Document Ingestion

```http
POST /api/v1/documents
```

Multipart form fields:

| Field           | Description               |
| --------------- | ------------------------- |
| `file`          | `.pdf` or `.txt` document |
| `strategy`      | `fixed` or `sentence`     |
| `chunk_size`    | Chunk size                |
| `chunk_overlap` | Chunk overlap             |

Example:

```bash
curl -F "file=@doc.pdf" \
     -F "strategy=sentence" \
     http://localhost:8000/api/v1/documents
```

Example using fixed-size chunking:

```bash
curl -F "file=@doc.txt" \
     -F "strategy=fixed" \
     -F "chunk_size=800" \
     -F "chunk_overlap=100" \
     http://localhost:8000/api/v1/documents
```

---

### Conversational Chat

```http
POST /api/v1/chat
```

Request:

```json
{
  "session_id": "user-1",
  "message": "What is the notice policy?"
}
```

Example:

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d "{\"session_id\":\"user-1\",\"message\":\"What is the notice policy?\"}"
```

The same `session_id` can be used for follow-up questions so Redis can maintain conversation context.

---

## RAG Flow

The conversational RAG pipeline follows this flow:

```text
User message
     ↓
Redis conversation history
     ↓
Follow-up query condensation
     ↓
Local embedding generation
     ↓
Qdrant similarity search
     ↓
Retrieved document chunks
     ↓
Context-grounded prompt
     ↓
Ollama chat model
     ↓
Answer + sourc
```
