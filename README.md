# RAG Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot that enables intelligent question-answering over uploaded documents. This system combines advanced vector search, reranking, and LLM-based generation to provide accurate answers with proper source citations.

**Key Innovation**: Hybrid search architecture (BM25 + vector embeddings) with CrossEncoder reranking and comprehensive evaluation framework for continuous model improvement.

## 🎯 Overview

This RAG chatbot implements a complete document intelligence pipeline:

1. **Document Processing** → Documents (PDF, DOCX, TXT) are chunked and embedded into 768-dimensional vectors
2. **Hybrid Retrieval** → Questions trigger both semantic (vector) and keyword (BM25) search across 80+ candidates
3. **Reranking** → CrossEncoder model refines top 10 most relevant chunks
4. **Generation** → Google Gemini synthesizes answers from reranked context with citations
5. **Evaluation** → Comprehensive metrics track retrieval quality, generation accuracy, and citation precision

## ✨ Key Features

### Core RAG Pipeline
- **Multi-format Document Support**: PDF, DOCX, TXT with automatic text extraction
- **Semantic Chunking**: Respects document structure (chapters, sections) with configurable overlap
- **Hybrid Search**: Combines dense vectors (Vietnamese document embeddings) and sparse vectors (BM25)
- **CrossEncoder Reranking**: Fine-tuned PhoRanker model for Vietnamese text relevance
- **Source Citations**: Every answer includes chunk-level citations with document references
- **Query Expansion**: LLM-generated query variants for improved recall

### Vietnamese Language Optimization
- **Vietnamese Embeddings**: `dangvantuan/vietnamese-document-embedding` (768-dim)
- **Vietnamese Reranker**: `itdainb/PhoRanker` or custom fine-tuned model
- **Underthesea Tokenization**: Vietnamese-specific word segmentation for BM25
- **Language Detection**: Automatic Vietnamese/English detection with `langdetect`

### Advanced Development Tools
- **Evaluation Framework**: `eval_run.py` with Gemini-based automated ratings (1364 lines)
- **Training Pipeline**: Query generation, hard negative mining, reranker fine-tuning
- **Performance Metrics**: Recall@K, MRR, NDCG, precision tracking
- **GPU Acceleration**: CUDA support for embeddings and reranking
- **MongoDB Integration**: Persistent storage for evaluation data and training artifacts

## 🏗️ Architecture

```
┌─────────────┐
│   Frontend  │  Next.js 16 + React 19 + TypeScript
│ (Port 3000) │  • Document upload UI
└──────┬──────┘  • Chat interface with citations
       │         • Loading states & error handling
       ↓
┌─────────────┐
│   Backend   │  Django 5.2.5 + DRF 3.14.0
│ (Port 8000) │  • Document processing & chunking
└──────┬──────┘  • Embedding generation
       │         • API endpoints
       ↓
┌─────────────────────────────────────────┐
│         RAG Pipeline Components         │
├─────────────┬─────────────┬─────────────┤
│   Qdrant    │   Gemini    │   Models    │
│ Vector DB   │     LLM     │ CrossEncoder│
│  (6333)     │   (API)     │  Reranker   │
└─────────────┴─────────────┴─────────────┘
       │             │             │
       └─────────────┴─────────────┘
                   ↓
       ┌─────────────────────────┐
       │   Retrieval Strategy    │
       ├─────────────────────────┤
       │ 1. Hybrid Search (80+)  │
       │    • Dense: Vector      │
       │    • Sparse: BM25       │
       │    • Fusion: RRF        │
       ├─────────────────────────┤
       │ 2. Rerank (Top 10)      │
       │    • CrossEncoder       │
       │    • Score threshold    │
       ├─────────────────────────┤
       │ 3. Generate Answer      │
       │    • Context assembly   │
       │    • Citation tracking  │
       └─────────────────────────┘
```

## 🛠️ Tech Stack

### Backend (Django 5.2.5)
- **Framework**: Django 5.2.5 + Django REST Framework 3.14.0
- **Vector Database**: Qdrant (local or remote, port 6333)
- **Embeddings**: `dangvantuan/vietnamese-document-embedding` (768-dim, Sentence Transformers)
- **Reranker**: `itdainb/PhoRanker` or custom fine-tuned CrossEncoder
- **LLM**: Google Gemini (`gemini-2.5-flash` or `gemini-1.5-pro`)
- **Document Processing**: PyMuPDF (PDF), python-docx (DOCX)
- **NLP**: underthesea (Vietnamese tokenization), langdetect
- **Data**: SQLite (dev), MongoDB (optional for training/eval)

### Frontend (Next.js 16)
- **Framework**: Next.js 16.0.8 (App Router)
- **UI Library**: React 19.2.1 with TypeScript 5.x
- **Styling**: Tailwind CSS v4
- **Icons**: Lucide React
- **HTTP Client**: Fetch API with error handling

### Infrastructure
- **Containerization**: Docker + Docker Compose
- **Vector Search**: Qdrant (COSINE distance, 768-dim)
- **API Gateway**: Django CORS middleware
- **Environment**: python-dotenv for configuration

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+** with pip
- **Node.js 18+** with npm
- **Qdrant** server (Docker recommended)
- **Google Gemini API key** ([Get it here](https://aistudio.google.com/))

### 1. Clone Repository
```bash
git clone https://github.com/TQanh23/rag_chatbot.git
cd rag_chatbot
```

### 2. Backend Setup
```bash
cd chatbot
python -m venv .venv
.\.venv\Scripts\activate  # Windows: \.venv\Scripts\activate
pip install -r requirements.txt

# Configure environment (copy .env.example to .env)
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# Initialize database
python manage.py migrate

# Initialize Qdrant collection
python recreate_collection.py

# Start Django server
python manage.py runserver
```

### 3. Frontend Setup
```bash
cd ../frontend
npm install
npm run dev  # Starts on http://localhost:3000
```

### 4. Start Qdrant (Docker)
```bash
# Option 1: Docker Compose (includes backend + Qdrant)
docker-compose up

# Option 2: Standalone Qdrant
docker run -p 6333:6333 -v $(pwd)/qdrant_data:/qdrant/storage qdrant/qdrant
```

### 5. Verify Setup
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000/api/health/
- **Qdrant**: http://localhost:6333/dashboard


## 🔌 API Endpoints

**Base URL**: `http://localhost:8000/api/`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upload/` | POST | Upload document (PDF, DOCX, TXT) |
| `/ask/` | POST | Ask question with optional document filter |
| `/health/` | GET | System health check (Qdrant, Gemini, GPU) |
| `/init-qdrant/` | POST | Initialize vector collection |
| `/check-qdrant/` | GET | Collection statistics and health |
| `/delete-qdrant/` | POST | Delete collections |

### Example: Ask Question
```json
POST /api/ask/
{
  "question": "What are the key findings?",
  "document_id": "optional-doc-id",
  "top_k": 10
}

Response:
{
  "answer": "Based on the documents...",
  "sources": [
    {
      "document_name": "research.pdf",
      "chunk_id": "chunk_123",
      "relevance_score": 0.95,
      "text": "..."
    }
  ],
  "retrieval_time": 0.234,
  "generation_time": 1.567
}
```

## 🧪 Evaluation & Training

### Running Evaluation
The system includes a comprehensive evaluation framework for measuring RAG performance:

```bash
cd chatbot
python eval_run.py
```

**Metrics Collected**:
- **Retrieval**: Recall@K, MRR, NDCG
- **Generation**: Gemini-based answer quality ratings (1-5 scale)
- **Citations**: Precision, recall of cited sources
- **Performance**: Latency, token usage

Results are saved to `media/evaluation_results.json` with detailed breakdowns.

### Training Pipeline

**1. Generate Training Queries**
```bash
python generate_training_queries.py  # LLM-generated queries from chunks
```

**2. Mine Hard Negatives**
```bash
python mine_hard_negatives.py  # Find challenging negative examples
```

**3. Fine-tune Reranker**
```bash
cd reranker_finetune
# Follow reranker fine-tuning guide
```

**4. Evaluate Performance**
```bash
python eval_run.py  # Compare before/after metrics
python plot_recall_at_k.py  # Visualize improvements
```


## 🐛 Troubleshooting

### Common Issues

**1. Qdrant Connection Error**
```bash
# Verify Qdrant is running
python verify_qdrant.py

# Check collection health
curl http://localhost:6333/collections/test_collection
```

**2. Empty Search Results**
```bash
# Lower similarity threshold
SCORE_THRESHOLD=0.20  # in .env

# Enable fallback strategies
ENABLE_KEYWORD_FALLBACK=true
ENABLE_QUERY_EXPANSION=true
```

**3. GPU Not Detected**
```bash
python verify_gpu_setup.py

# Force CPU mode if needed
GPU_ENABLED=false
```

**4. Slow Embeddings**
- Enable GPU acceleration
- Use smaller batch sizes
- Cache BM25 vocabulary: `SPARSE_VOCAB_CACHE=media/bm25_cache.pkl`

**5. Poor Answer Quality**
- Adjust chunking parameters: `CHUNK_TARGET_TOKENS=600`
- Increase retrieval candidates: `RETRIEVAL_TOP_K=100`
- Run evaluation: `python eval_run.py` to identify issues

## 📊 Performance Benchmarks

Typical performance on Vietnamese legal documents (100+ pages):

| Metric | Value |
|--------|-------|
| Document Processing | ~2s per PDF page |
| Embedding Generation | ~50ms per chunk (GPU) |
| Retrieval (Hybrid) | ~200-300ms for 80 candidates |
| Reranking (Top 10) | ~100-150ms (GPU) |
| Answer Generation | ~1-2s (Gemini Flash) |
| **Total E2E Latency** | **~2-3s per question** |

**Retrieval Quality** (on gold standard test set):
- Recall@10: ~85-90%
- MRR: ~0.75
- NDCG@10: ~0.80

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- **Model Fine-tuning**: Improve Vietnamese embeddings and reranker
- **Chunking Strategies**: Experiment with semantic chunking algorithms
- **Evaluation Metrics**: Add domain-specific quality measures
- **Performance**: Optimize GPU utilization and caching
- **Multi-language**: Extend beyond Vietnamese

Please open issues for bugs or feature requests.
