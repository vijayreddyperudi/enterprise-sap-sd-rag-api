# Enterprise SAP SD RAG API

Production-ready Retrieval-Augmented Generation (RAG) REST API built with FastAPI, HuggingFace Embeddings, FAISS Vector Search, Cross-Encoder Reranking, and Groq LLM inference.

---

## Features

- **FastAPI Core**: High-performance asynchronous REST API with automatic Swagger/OpenAPI documentation.
- **FAISS Vector Index**: Fast similarity search using `sentence-transformers/all-MiniLM-L6-v2`.
- **Cross-Encoder Reranker**: Precision document reranking using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **Groq LLM Acceleration**: Low-latency generation powered by `llama-3.3-70b-versatile` (or other Groq-supported models).
- **Docker Ready**: Self-contained multi-stage container build with built-in health checks and pre-baked index ingestion.
- **CORS & Validation**: Configured CORS middleware and strict Pydantic input validation.

---

## Architecture Overview

```
+-----------------------------------------------------------+
|                      Client Request                       |
+-----------------------------------------------------------+
                             | (POST /api/v1/chat)
                             v
+-----------------------------------------------------------+
|                   FastAPI Validation                     |
+-----------------------------------------------------------+
                             |
                             v
+-----------------------------------------------------------+
|            FAISS Vector Search (Top-10 Candidates)        |
+-----------------------------------------------------------+
                             |
                             v
+-----------------------------------------------------------+
|         Cross-Encoder Reranker (Score & Filter Top-3)     |
+-----------------------------------------------------------+
                             |
                             v
+-----------------------------------------------------------+
|           Groq LLM (Strict Grounded Answer Generation)    |
+-----------------------------------------------------------+
                             |
                             v
+-----------------------------------------------------------+
|                   JSON Response + Sources                 |
+-----------------------------------------------------------+
```

---

## Quick Start (Local)

### 1. Prerequisites
- Python 3.11 or 3.12
- Groq API Key ([Get an API Key](https://console.groq.com/keys))

### 2. Environment Setup
```bash
# Clone repository and create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env` and provide your `GROQ_API_KEY`:
```ini
GROQ_API_KEY=gsk_your_actual_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
```

### 4. Ingest Document (Generate Vector Index)
```bash
python app/services/ingestion_service.py
```

### 5. Run API Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API docs are available at `http://localhost:8000/docs`.

---

## Running with Docker

### Build Image
```bash
docker build -t enterprise-rag-api .
```

### Run Container
```bash
docker run -d \
  -p 8000:8000 \
  -e GROQ_API_KEY="gsk_your_actual_groq_api_key" \
  -e GROQ_MODEL="llama-3.3-70b-versatile" \
  --name rag-api \
  enterprise-rag-api
```

---

## API Endpoints

### 1. Health Check
```http
GET /health
```
**Response (200 OK):**
```json
{
  "status": "healthy",
  "model": "llama-3.3-70b-versatile",
  "index_ready": true
}
```

### 2. Chat / Query
```http
POST /api/v1/chat
Content-Type: application/json

{
  "question": "What are the key organizational units in SAP SD?"
}
```
**Response (200 OK):**
```json
{
  "answer": "The core organizational units in SAP SD include Sales Organization, Distribution Channel, Division, Sales Area, and Plant.",
  "sources": [12, 14],
  "model": "llama-3.3-70b-versatile"
}
```

---

## Running Tests
```bash
pytest tests/ -v
```
