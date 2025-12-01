# MongoDB Migration Summary

## ✅ Completed Tasks

### 1. ✓ Updated requirements.txt
- Added `pymongo==4.6.1`
- Added `dnspython==2.4.2`

### 2. ✓ Created MongoDB Client Utility
**File**: `backend/utils/mongo_client.py`
- Connection management with singleton pattern
- Index initialization for all collections
- Health check functionality
- Automatic retry on connection failures

### 3. ✓ Created MongoDB Models/Schemas
**File**: `backend/utils/mongo_models.py`
- `MongoDocument`: Document metadata schema
- `MongoChunk`: Chunk storage with embeddings
- `MongoQueryLog`: Query logging schema
- `MongoRetrievalRun`: Retrieval evaluation metrics
- `MongoGenerationRun`: Generation evaluation metrics
- `MongoEvalRun`: Aggregate evaluation results

### 4. ✓ Created MongoDB Repository Layer
**File**: `backend/utils/mongo_repository.py`
- Complete CRUD operations for all collections
- Batch operations for chunks
- Query optimization with indexes
- Cleanup utilities for old data
- Database statistics

### 5. ✓ Updated FileUploadView
**File**: `api/views/file_upload_view.py`
**Changes**:
- Saves document metadata to MongoDB
- Bulk saves chunks with embeddings to MongoDB
- Updates document status (processing → processed/failed)
- Maintains backward compatibility with Qdrant

### 6. ✓ Updated AskView
**File**: `api/views/ask_view.py`
**Changes**:
- Logs all queries to MongoDB with correlation IDs
- Tracks retrieval results and scores
- Records final answers and latency
- Maintains backward compatibility with CSV logging

### 7. ✓ Created MongoDB Initialization Script
**File**: `backend/init_mongodb.py`
- Automated setup of all collections
- Index creation with verification
- Health checks and diagnostics
- User-friendly output

### 8. ✓ Updated Django Settings
**File**: `backend/settings.py`
**Added**:
- `MONGO_URI`: MongoDB connection string
- `MONGO_DB_NAME`: Database name configuration

## 📁 Additional Files Created

### 1. `.env.example`
Template for environment configuration with MongoDB settings

### 2. `MONGODB_SETUP.md`
Comprehensive setup guide including:
- Installation steps
- Configuration instructions
- Testing procedures
- MongoDB queries and operations
- Troubleshooting guide
- Backup/restore instructions

### 3. `mongo_utils.py`
Utility script for MongoDB operations:
- `stats`: Show database statistics
- `health`: Check MongoDB health
- `cleanup`: Remove old query logs
- `list-docs`: List uploaded documents
- `list-queries`: View recent queries
- `export-logs`: Export logs to CSV
- `reset`: Reset all collections

### 4. Updated `health_check_view.py`
Added MongoDB health check to API endpoint

## 🗄️ MongoDB Collections

| Collection | Purpose | Indexes |
|------------|---------|---------|
| `documents` | File metadata | content_hash (unique), uploaded_at, status |
| `chunks` | Text chunks + embeddings | document_id, (document_id, order_index), text |
| `query_logs` | User questions & answers | ts (TTL 90d), question_id, correlation_id, user_id |
| `retrieval_runs` | Retrieval metrics | run_id, query_id, ts |
| `generation_runs` | Generation metrics | run_id, query_id, ts |
| `eval_runs` | Aggregate results | run_id (unique), created_at, backend |

## 🔧 How to Use

### Initial Setup
```bash
# 1. Install dependencies
cd backend
pip install -r requirements.txt

# 2. Configure .env
cp .env.example .env
# Edit .env with your MongoDB URI and other settings

# 3. Start MongoDB (if local)
mongod --dbpath D:\mongodb_data

# 4. Initialize MongoDB
python init_mongodb.py

# 5. Run Django
python manage.py runserver
```

### Daily Operations
```bash
# View statistics
python mongo_utils.py stats

# Check health
python mongo_utils.py health

# List recent queries
python mongo_utils.py list-queries --days 7 --limit 20

# Export logs
python mongo_utils.py export-logs --output logs.csv --days 30

# Cleanup old data
python mongo_utils.py cleanup --days 90
```

### API Health Check
```bash
curl http://localhost:8000/api/health/
```

Response includes MongoDB status:
```json
{
  "status": "healthy",
  "components": {
    "django": {...},
    "qdrant": {...},
    "mongodb": {
      "status": "healthy",
      "mongodb_version": "7.0.0",
      "collections": {
        "documents": 5,
        "chunks": 123,
        "query_logs": 45
      }
    }
  }
}
```

## 🔍 MongoDB Queries

### View Recent Documents
```javascript
db.documents.find().sort({uploaded_at: -1}).limit(10)
```

### View Chunks for a Document
```javascript
db.chunks.find({document_id: "your_doc_id"}).sort({order_index: 1})
```

### View Recent Queries
```javascript
db.query_logs.find().sort({ts: -1}).limit(20)
```

### Calculate Average Latency
```javascript
db.query_logs.aggregate([
  {$group: {_id: null, avg_latency: {$avg: "$latency_ms"}}}
])
```

### Count Documents by Status
```javascript
db.documents.aggregate([
  {$group: {_id: "$status", count: {$sum: 1}}}
])
```

## ⚙️ Configuration

### Environment Variables
```bash
# MongoDB (required)
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=rag_chatbot

# MongoDB Atlas (cloud alternative)
MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/

# TTL for query logs (default: 90 days)
QUERY_LOG_TTL_DAYS=90
```

## 🚀 Key Features

### 1. Dual Storage Strategy
- **Qdrant**: Vector similarity search (fast retrieval)
- **MongoDB**: All metadata, logs, and structured data (persistent, queryable)

### 2. Automatic Logging
- Every document upload logged to MongoDB
- Every query logged with full context
- Performance metrics tracked automatically

### 3. Evaluation Support
- Retrieval metrics per query
- Generation quality metrics
- Aggregate evaluation runs
- Historical comparison

### 4. Data Retention
- Automatic cleanup of old query logs (TTL index)
- Manual cleanup utilities
- Configurable retention periods

### 5. Production Ready
- Connection pooling
- Error handling and fallbacks
- Health monitoring
- Backup/restore support

## 📊 Benefits

1. **Persistent Storage**: Data survives server restarts
2. **Query Audit Trail**: Full history of all questions and answers
3. **Performance Tracking**: Latency and quality metrics
4. **Scalable**: Ready for MongoDB Atlas or replica sets
5. **Developer Friendly**: Easy to query and analyze data

## 🔄 Data Flow

### Document Upload
```
File → Django → MongoDB (metadata) → Chunking → MongoDB (chunks) → Qdrant (vectors)
```

### Question Answering
```
Question → Qdrant (vector search) → Reranking → MongoDB (log query) → 
Gemini (answer) → MongoDB (update log with answer) → Response
```

### Evaluation
```
Test queries → Retrieval → MongoDB (retrieval_runs) → 
Generation → MongoDB (generation_runs) → 
Aggregate → MongoDB (eval_runs)
```

## 🛠️ Maintenance

### Weekly
- Check database size: `python mongo_utils.py stats`
- Review query logs: `python mongo_utils.py list-queries --days 7`

### Monthly
- Export logs for analysis: `python mongo_utils.py export-logs`
- Run cleanup if needed: `python mongo_utils.py cleanup --days 90`

### Quarterly
- Backup database: `mongodump --db=rag_chatbot --out=backup/`
- Review indexes: Check query performance in MongoDB Compass

## 📝 Notes

- **Backward Compatibility**: All existing CSV logging still works
- **Qdrant Required**: Vector search still uses Qdrant
- **TTL Index**: Query logs auto-delete after 90 days
- **Embedding Storage**: 384-dim embeddings stored in MongoDB chunks
- **Connection Pooling**: Handled automatically by pymongo

## 🎯 Next Steps

1. **Install MongoDB** (local or Atlas)
2. **Run `init_mongodb.py`** to set up collections
3. **Test with document upload** and queries
4. **Monitor with `mongo_utils.py`** commands
5. **Integrate with evaluation scripts** (future work)

## 📚 Reference

- MongoDB docs: https://docs.mongodb.com/
- PyMongo docs: https://pymongo.readthedocs.io/
- Setup guide: `MONGODB_SETUP.md`
- Utility commands: `python mongo_utils.py --help`
