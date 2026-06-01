# DocuQuery — Production RAG Pipeline

> Ask questions. Get answers grounded in real documents.

A production-grade **Retrieval-Augmented Generation (RAG)** pipeline built with FastAPI, PostgreSQL + pgvector, HuggingFace sentence-transformers, and Groq LLM. Features real-time streaming, PDF/TXT ingestion, session-based document isolation, and a polished two-mode chat interface.

**[Live Demo](https://balanced-manifestation-production-50d4.up.railway.app)** · **[API Docs](https://docuquery-production-872a.up.railway.app/docs)**

---

## Features

- **Two modes** — Personal document assistant (blue theme) + DC federal tech jobs explorer (amber theme)
- **Real-time streaming** — Token-by-token response streaming like ChatGPT
- **PDF & TXT upload** — Drag and drop documents directly from the UI
- **Session isolation** — Each browser session has its own private document space
- **Auto-cleanup** — User-uploaded documents are automatically deleted after 24 hours
- **Chat history** — Full conversational context across follow-up questions with per-mode state
- **pgvector search** — HNSW index for fast cosine similarity search
- **Airflow DAG** — Scheduled daily ingestion of federal tech jobs from USAJobs API
- **Rate limiting** — 10 queries/min, 5 uploads/min per IP
- **Docker** — Fully containerized, runs locally with one command

---

## Architecture

```
Documents (PDF / TXT / USAJobs API)
        |
   Chunking (500 words, 50 overlap)
        |
   Embeddings (all-MiniLM-L6-v2, 384-dim)
        |
   PostgreSQL + pgvector (HNSW index)
        |
   FastAPI /query endpoint
        |
   Groq LLM (llama-3.1-8b-instant)
        |
   Streaming response to Frontend
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Vector Store | PostgreSQL + pgvector (HNSW) |
| LLM | Groq API llama-3.1-8b-instant |
| Backend | FastAPI + Python 3.11 |
| Orchestration | Apache Airflow |
| Frontend | Vanilla HTML/CSS/JS |
| Containerization | Docker + Docker Compose |
| Deployment | Railway |

---

## Quick Start

### Prerequisites

- Docker Desktop
- Groq API key (free at console.groq.com)

### 1. Clone the repo

```bash
git clone https://github.com/michael-demissie/docuquery.git
cd docuquery
```

### 2. Set up environment variables

```bash
cp .env.example .env
```

Open .env and add your GROQ_API_KEY.

### 3. Start the stack

```bash
docker-compose up --build
```

### 4. Open the app

- Frontend: http://localhost:3000
- API docs: http://localhost:8000/docs

### 5. Ingest documents

```bash
# Ingest a text file (personal mode)
python3 ingestion/ingest.py --file your_document.txt --title "My Document"

# Ingest a URL
python3 ingestion/ingest.py --url https://example.com/page --title "Web Page"

# Ingest as jobs mode
python3 ingestion/ingest.py --file jobs.txt --title "Job Listings" --mode jobs
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | /health | Health check |
| POST | /query | RAG query with streaming |
| POST | /ingest | Ingest text content |
| POST | /upload | Upload PDF or TXT file |
| GET | /documents | List documents by mode and session |
| DELETE | /documents/{id} | Delete a document |
| POST | /cleanup-expired | Manually trigger session cleanup |

---

## Two Modes

### Personal Assistant (blue theme)
Upload your own PDF or TXT files and chat with them. Each browser session gets a unique session ID stored in localStorage. Documents are automatically deleted after 24 hours. Default demo documents (DC Wikipedia articles) are always available.

### DC Tech Jobs (amber theme)
Pre-loaded with real federal tech job postings from USAJobs.gov, updated daily via an Apache Airflow DAG. Ask questions like "What Python skills do federal agencies require?" or "Which agencies are hiring data engineers?"

---

## Airflow DAG

The airflow/dags/usajobs_ingest.py DAG runs daily at 6am and fetches federal tech job postings for keywords including data engineer, software engineer, machine learning, cybersecurity, and more — filtered to IT occupational series codes (2210, 1550, 0854, 1560).

To run locally:

```bash
export AIRFLOW_HOME=~/docuquery/airflow
python3 -m airflow db init
python3 -m airflow webserver --port 8080 -D
python3 -m airflow scheduler -D
```

---

## Project Structure

```
docuquery/
├── api/
│   ├── main.py              FastAPI app and all endpoints
│   ├── database.py          PostgreSQL connection pool
│   ├── embeddings.py        HuggingFace embedding model
│   ├── chunker.py           Text chunking logic
│   ├── Dockerfile
│   ├── railway.toml
│   └── requirements.txt
├── frontend/
│   ├── index.html           Single-page chat UI
│   └── Dockerfile
├── ingestion/
│   └── ingest.py            CLI ingestion script
├── airflow/
│   └── dags/
│       └── usajobs_ingest.py    Daily jobs DAG
├── scripts/
│   └── init.sql             DB schema and pgvector setup
├── docker-compose.yml
└── .env.example
```

---

## Environment Variables

| Variable | Description |
|---|---|
| DATABASE_URL | PostgreSQL connection string |
| GROQ_API_KEY | Groq API key for LLM |
| EMBEDDING_MODEL | HuggingFace model name |
| CHUNK_SIZE | Words per chunk (default 500) |
| CHUNK_OVERLAP | Overlap between chunks (default 50) |
| USAJOBS_API_KEY | USAJobs API key for job ingestion |

---

## Author

**Michael Mulugeta Demissie** — Data Engineer

- Portfolio: https://michael-demissie.github.io
- GitHub: https://github.com/michael-demissie
- LinkedIn: https://www.linkedin.com/in/michael-mulugeta-demissie/
