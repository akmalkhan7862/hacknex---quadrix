# HNX26PSI01 – Multimodal Document Intelligence (MDI)

Source-of-truth implementation for HNX26PSI01 Multimodal Document Intelligence architecture specification (Phase 1 MVP).

## System Architecture

```
USER
  ↓
REACT + TYPESCRIPT FRONTEND (Vite)
  ↓
FASTAPI API LAYER
  ↓
QUERY ANALYZER
  ↓
QUERY ROUTER
  ↓
┌─────────────────────────────────────┐
│ Vector RAG │ BM25 │ Graph RAG (Ext) │
└─────────────────────────────────────┘
  ↓
RERANKER
  ↓
CONTEXT BUILDER
  ↓
LLM + VLM + PYTHON REASONING
  ↓
EVIDENCE ENGINE
  ↓
ANSWER + CLAIMS + SOURCES + CALCULATIONS
  ↓
REACT FRONTEND PDF VIEWER + EVIDENCE
```

---

## Directory Structure

```
mdi/
├── README.md
├── requirements.txt
├── .env
├── .env.example
├── data/
│   ├── raw/
│   ├── processed/
│   └── eval/
├── backend/
│   └── app/
│       ├── main.py
│       ├── db.py
│       ├── api/
│       │   ├── documents.py
│       │   ├── query.py
│       │   └── health.py
│       ├── ingest/
│       │   ├── classify.py
│       │   ├── layout.py
│       │   ├── tables.py
│       │   ├── sections.py
│       │   ├── provenance.py
│       │   └── pipeline.py
│       ├── index/
│       │   ├── schema.py
│       │   ├── embeddings.py
│       │   ├── vector_store.py
│       │   └── bm25.py
│       ├── query/
│       │   ├── analyzer.py
│       │   ├── router.py
│       │   ├── retriever.py
│       │   ├── reranker.py
│       │   └── context_builder.py
│       ├── reasoning/
│       │   ├── llm.py
│       │   └── calculator.py
│       └── evidence/
│           ├── engine.py
│           ├── citations.py
│           └── validator.py
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   └── src/
│       ├── App.tsx
│       ├── main.tsx
│       ├── components/
│       │   ├── UploadPanel.tsx
│       │   ├── DocumentList.tsx
│       │   ├── ChatPanel.tsx
│       │   ├── AnswerPanel.tsx
│       │   ├── EvidencePanel.tsx
│       │   └── PdfViewer.tsx
│       └── services/
│           └── api.ts
└── tests/
    └── test_all.py
```

---

## Installation & Environment Setup

### 1. Requirements
- Python 3.11+ (or Python 3.10)
- Node.js 18+ and npm

### 2. Backend Setup
```bash
cd mdi
python -m pip install -r requirements.txt
```

### 3. Environment Variables
Copy `.env.example` to `.env` and set your API keys if available:
```bash
cp .env.example .env
```
Key variables:
- `GEMINI_API_KEY`: API key for Gemini 2.5 models
- `OPENAI_API_KEY`: API key for OpenAI GPT-4o models
- `DATABASE_URL`: `sqlite:///./data/processed/mdi.db`

*(Note: If no API keys are provided, the backend seamlessly switches to the deterministic evidence extractor fallback guarantee).*

---

## Running the Application

### 1. Run Backend Server
```bash
cd mdi
$env:PYTHONPATH="."  # (PowerShell)
uvicorn backend.app.main:app --reload --port 8000
```

### 2. Run Frontend Server
```bash
cd mdi/frontend
npm install
npm run dev
```
Open browser at `http://localhost:3000`.

---

## Running Tests

Execute full test suite:
```bash
cd mdi
$env:PYTHONPATH="."
python -m pytest tests/test_all.py
```

---

## API Endpoints Summary

- `GET /health`: Health status
- `POST /documents/upload`: Upload one or multiple PDF documents
- `GET /documents`: List uploaded documents and status
- `GET /documents/{document_id}`: View document details, page classifications, evidence count
- `POST /query`: Execute hybrid RAG query, reranking, LLM reasoning, and evidence verification
- `GET /evidence/{element_id}`: Retrieve raw evidence unit details
