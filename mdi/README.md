# HNX26PSI01 – Multimodal Document Intelligence (MDI)

Source-of-truth implementation for HNX26PSI01 Multimodal Document Intelligence architecture specification.

---

## Assigned Module: Adaptive Router + Multimodal Retrieval

The **Adaptive Router + Multimodal Retrieval** module serves as the core intelligent information retrieval engine for VERITAS / MDI, sitting between the user's question and the downstream reasoning engine.

```
                    USER QUESTION
                          │
                          ▼
                   QUERY ANALYZER
                          │
                          ▼
                   ADAPTIVE ROUTER
                          │
             ┌────────────┼────────────┐
             ▼            ▼            ▼
         VECTOR RAG   HYBRID SEARCH   GRAPH RAG
             │            │            │
             └────────────┼────────────┘
                          ▼
                MULTIMODAL RETRIEVER
                          │
                 ┌────────┼────────┐
                 ▼        ▼        ▼
               TEXT     TABLE    VISUAL (Chart / Image)
                 │        │        │
                 └────────┼────────┘
                          ▼
                    EVIDENCE PACKET
                          │
                          ▼
                REASONING ENGINE
                (downstream teammate)
```

---

## Module Architecture & Components

### 1. Query Analyzer (`backend/app/query/analyzer.py`)
- **`build_plan(question: str) -> Plan`**: Rule-based, deterministic analyzer converting natural-language queries into structured `Plan` objects.
- Detects:
  - `intent`: `fact_lookup`, `comparison`, `calculation`, `visual_analysis`, `cross_doc_synthesis`, `summary`
  - `entities`: Named organizational, plant, and thematic entities (e.g., Dallas Plant, Austin Plant)
  - `periods`: Quarters (`Q1`-`Q4`), fiscal years (`FY2023`), standard years
  - `modalities`: Required document modalities (`text`, `table`, `chart`, `image`)
  - `needs_calculation`: Identifies numerical or mathematical questions
  - `cross_document`: Identifies queries requiring cross-document synthesis

### 2. Adaptive Router (`backend/app/query/router.py`)
- **`route_plan(plan: Plan) -> RetrievalStrategy`**: Selects the optimal retrieval strategy based on deterministic priority rules:
  1. `cross_document == True` $\rightarrow$ `RetrievalStrategy.CROSS_DOCUMENT`
  2. `chart` or `image` in modalities $\rightarrow$ `RetrievalStrategy.MULTIMODAL`
  3. `table` in modalities $\rightarrow$ `RetrievalStrategy.HYBRID`
  4. `needs_calculation == True` $\rightarrow$ `RetrievalStrategy.HYBRID`
  5. `intent == "comparison"` $\rightarrow$ `RetrievalStrategy.HYBRID`
  6. Exact terminology / numeric values $\rightarrow$ `RetrievalStrategy.HYBRID`
  7. Default (general text) $\rightarrow$ `RetrievalStrategy.VECTOR`

### 3. Retrieval Engines
- **Dense Vector Search (`backend/app/index/vector_store.py`, `backend/app/query/retriever.py`)**:
  - `VectorRetriever`: Embeds questions via `EmbeddingProvider`, executes cosine similarity search over `VectorStore`, and preserves all source metadata (`element_id`, `doc_id`, `page`, `section_path`, `element_type`).
- **Keyword BM25 Search (`backend/app/index/bm25.py`, `backend/app/query/retriever.py`)**:
  - `BM25Index`: BM25Okapi with native TF-IDF fallback, $O(1)$ reference resolution via `doc_map`, and reference-only retrieval (`search_ids`).
  - `BM25Retriever`: Keyword search with metadata filtering (`document_id`, `content_type`, `section`, `period`, `page`).
- **Hybrid Search & Fusion (`backend/app/query/retriever.py`)**:
  - `reciprocal_rank_fusion(ranked_lists, rrf_k=60)`: Reciprocal Rank Fusion (RRF) boosting dual-matched documents without score distortion.
  - `HybridRetriever`: Combines vector search + BM25 search with metadata filtering and provides `retrieve_vector_only()` and `retrieve_bm25_only()`.
- **Multimodal Retrieval (`backend/app/query/multimodal.py`)**:
  - `MultimodalRetriever`: Dispatches across specialized paths:
    - Text retrieval (`element_type="text"`)
    - Table retrieval (`element_type="table"`)
    - Visual retrieval (`element_type in ("chart", "image")` or `visual_ref`)
  - Assembles unified `EvidencePacket` preserving visual crop paths and tabular structures.
- **Graph RAG Extension Point (`backend/app/query/graph.py`)**:
  - `GraphRetriever`: Extension point interface with `status()` and `retrieve()`, returning entity-linked evidence units or safe empty lists without pipeline failures.

### 4. Reranking Layer (`backend/app/query/rerank.py`, `backend/app/query/reranker.py`)
- `Reranker`: Modular abstract base class for optional second-stage reranking.
- Implementations:
  - `DeterministicScoreFusionReranker`: Linear score fusion (60% vector + 40% BM25/RRF) preserving all source provenance.
  - `MockReranker`: Query token overlap scoring for zero-dependency offline testing.
  - `CrossEncoderReranker`: Neural cross-encoder hook with graceful fallback.
  - `NoOpReranker`: Pass-through mode when reranking is bypassed.

### 5. Modality Readers (`backend/app/query/readers/`)
- **`TextReader` (`readers/text.py`)**: Extracts `relevant_fact`, `supporting_text` context window, and overlap confidence.
- **`TableReader` (`readers/table.py`)**: Parses JSON and Markdown tables, extracts targeted cell values, row/column mappings, and units (`%`, `$`, etc.) without performing downstream arithmetic (preserving numbers for the reasoning teammate).
- **`ChartReader` (`readers/chart.py`)**: Preserves visual crop paths (`visual_ref`), detects chart types (line/trend, bar, pie), and delegates visual comprehension to the VLM layer.
- **`ImageReader` (`readers/image.py`)**: Preserves image crops and routes visual description prompts to the VLM layer.

### 6. Model Abstractions (`backend/app/models/`)
- **`EmbeddingProvider` (`models/embedder.py`)**:
  - `SentenceTransformerEmbedder`: Neural sentence embeddings (`all-MiniLM-L6-v2`).
  - `DeterministicFallbackEmbedder`: L2-normalized 384-dimensional hash vectors.
  - `MockEmbedder`: Controllable test embedder.
- **`VisionProvider` (`models/vlm.py`)**:
  - `MockVisionProvider`: Zero-dependency deterministic offline VLM.
  - `GeminiVisionProvider`: Google GenAI multimodal vision with exponential retry backoff.
  - `OpenAIVisionProvider`: OpenAI GPT-4o multimodal vision with retries.
  - In-memory response caching to prevent duplicate API requests.

---

## Public Module Interface

The public entry point consumed by the reasoning teammate is:

```python
from backend.app.query import fetch_evidence, build_plan

# 1. Build structured query plan
plan = build_plan("What was production efficiency in Q4?")

# 2. Fetch verified multimodal evidence packet
packet = fetch_evidence(plan=plan)

# 3. Inspect structured evidence packet
print(f"Strategy used: {packet.strategy}")
print(f"Total evidence units: {len(packet.evidence_units)}")
for unit in packet.evidence_units:
    print(f"[{unit.element_type}] {unit.element_id} (Page {unit.page}): {unit.content[:80]}")
    if unit.visual_ref:
        print(f"  Visual crop: {unit.visual_ref}")
```

### Signature:
```python
def fetch_evidence(
    plan: Plan,
    sub_question: Optional[str] = None,
    top_k: int = 10,
    vector_retriever: Optional[VectorRetriever] = None,
    hybrid_retriever: Optional[HybridRetriever] = None,
    multimodal_retriever: Optional[MultimodalRetriever] = None,
    graph_retriever: Optional[GraphRetriever] = None,
    reranker: Optional[Reranker] = None,
) -> EvidencePacket:
```

---

## Installation & Setup

```bash
cd mdi
python -m pip install -r requirements.txt
```

### Environment Variables (.env)
```bash
GEMINI_API_KEY=your_gemini_key   # Optional (falls back to MockVisionProvider)
OPENAI_API_KEY=your_openai_key   # Optional (falls back to MockVisionProvider)
DATABASE_URL=sqlite:///./data/processed/mdi.db
```

---

## Running Tests

Execute the complete test suite (251 passing tests across all module phases):

```bash
cd mdi
$env:PYTHONPATH="."  # PowerShell
python -m pytest tests/ -v
```

### Individual Test Suites:
- `tests/test_all.py`: Original baseline regression tests (12 tests)
- `tests/test_phase_b_analyzer.py`: Query Analyzer unit tests (41 tests)
- `tests/test_phase_c_router.py`: Adaptive Router unit tests (38 tests)
- `tests/test_phase_d_vector.py`: Vector Retriever & Embedding Provider tests (49 tests)
- `tests/test_phase_e_bm25.py`: BM25 Index & Retriever tests (18 tests)
- `tests/test_phase_f_hybrid.py`: Hybrid Search & RRF Fusion tests (13 tests)
- `tests/test_phase_g_rerank.py`: Reranker abstraction & fusion tests (10 tests)
- `tests/test_phase_h_multimodal.py`: Multimodal Retriever & EvidencePacket tests (9 tests)
- `tests/test_phase_i_readers.py`: Text & Table Readers tests (11 tests)
- `tests/test_phase_j_visual_readers.py`: Chart & Image Readers tests (9 tests)
- `tests/test_phase_k_vlm.py`: VLM abstraction & caching tests (10 tests)
- `tests/test_phase_l_embedder.py`: Embedding Provider abstraction tests (11 tests)
- `tests/test_phase_m_graph.py`: Graph RAG extension point tests (8 tests)
- `tests/test_phase_n_pipeline.py`: End-to-end `fetch_evidence` pipeline tests (6 tests)
- `tests/test_phase_o_scenarios.py`: Canonical end-to-end scenarios (6 tests)
