# MongoDB Migration - Setup Guide

## Overview
This migration adds MongoDB as the primary data store for:
- **Documents**: Uploaded file metadata
- **Chunks**: Text chunks with embeddings
- **Query Logs**: All user questions and answers
- **Retrieval Runs**: Per-query retrieval metrics
- **Generation Runs**: Per-query generation metrics
- **Eval Runs**: Aggregate evaluation results

**Note**: Qdrant is still used for vector similarity search. MongoDB complements it by storing all metadata and logs.

## Prerequisites

1. **MongoDB Server** (local or remote)
   - Download: https://www.mongodb.com/try/download/community
   - Or use MongoDB Atlas (cloud): https://www.mongodb.com/atlas

2. **Python Dependencies**
   ```bash
   pip install pymongo dnspython
   ```

## Installation Steps

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` with your settings:
```bash
# MongoDB Configuration
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=rag_chatbot

# Other required settings
GEMINI_API_KEY=your-actual-api-key-here
```

### 3. Start MongoDB Server

**Local MongoDB:**
```bash
# Windows
mongod --dbpath D:\mongodb_data

# Linux/Mac
mongod --dbpath /path/to/mongodb_data
```

**Or use MongoDB Atlas** (cloud):
- Update `MONGO_URI` in `.env`:
  ```
  MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/
  ```

### 4. Initialize MongoDB Collections and Indexes
```bash
python init_mongodb.py
```

Expected output:
```
============================================================
MongoDB Initialization for RAG Chatbot
============================================================

Step 1: Testing MongoDB connection...
✓ Connected to MongoDB
  Database: rag_chatbot

Step 2: Checking existing collections...
  Found 0 existing collections:

Step 3: Creating indexes...
...
✓ All MongoDB indexes initialized successfully
============================================================
✓ MongoDB initialization complete!
============================================================
```

### 5. Run Django Migrations (for SQLite compatibility)
```bash
python manage.py migrate
```

### 6. Start the Application
```bash
python manage.py runserver
```

## Testing the MongoDB Integration

### Test 1: Document Upload
Upload a document to verify MongoDB storage:

```bash
curl -X POST http://localhost:8000/api/upload/ \
  -F "file=@your-document.pdf"
```

Check MongoDB:
```javascript
// In mongosh
use rag_chatbot
db.documents.find().pretty()
db.chunks.find().limit(5).pretty()
```

### Test 2: Query Logging
Ask a question to verify query logs:

```bash
curl -X POST http://localhost:8000/api/ask/ \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'
```

Check query logs:
```javascript
db.query_logs.find().sort({ts: -1}).limit(5).pretty()
```

### Test 3: View Database Statistics
Use the Python shell:

```python
python manage.py shell

from backend.utils.mongo_repository import MongoRepository
repo = MongoRepository()
stats = repo.get_database_stats()
print(stats)
# Output: {'documents': 1, 'chunks': 42, 'query_logs': 5, ...}
```

## MongoDB Collections Schema

### documents
```javascript
{
  "_id": "hash_of_file",
  "document_id": "hash_of_file",
  "filename": "example.pdf",
  "content_hash": "hash_of_file",
  "mimetype": "application/pdf",
  "size_bytes": 123456,
  "uploaded_at": ISODate("2025-11-04T..."),
  "status": "processed",
  "token_count": 5000,
  "vector_dim": 384,
  "num_chunks": 42
}
```

### chunks
```javascript
{
  "_id": "hash::chunk_0",
  "chunk_id": "hash::chunk_0",
  "document_id": "hash_of_file",
  "text": "This is the chunk text...",
  "embedding": [0.123, -0.456, ...], // 384-dim array
  "page": 1,
  "section": "Introduction",
  "order_index": 0,
  "text_len": 350,
  "created_at": ISODate("2025-11-04T...")
}
```

### query_logs
```javascript
{
  "_id": "uuid-v4",
  "question_id": "uuid-v4",
  "question": "What is this about?",
  "correlation_id": "uuid-v4",
  "retrieved_chunk_ids": ["chunk1", "chunk2"],
  "retrieved_scores": [0.89, 0.76],
  "reranked_chunk_ids": ["chunk1", "chunk3"],
  "final_answer": "This document is about...",
  "latency_ms": 1250,
  "ts": ISODate("2025-11-04T..."),
  "document_id": "hash_of_file",
  "top_k": 5
}
```

## Useful MongoDB Queries

### Count documents by status
```javascript
db.documents.aggregate([
  {$group: {_id: "$status", count: {$sum: 1}}}
])
```

### Recent queries
```javascript
db.query_logs.find().sort({ts: -1}).limit(10)
```

### Average query latency
```javascript
db.query_logs.aggregate([
  {$group: {_id: null, avg_latency: {$avg: "$latency_ms"}}}
])
```

### Find all chunks for a document
```javascript
db.chunks.find({document_id: "your_doc_id"}).sort({order_index: 1})
```

### Search chunks by text
```javascript
db.chunks.find({$text: {$search: "keyword"}}).limit(5)
```

## Troubleshooting

### Connection Error
```
ServerSelectionTimeoutError: Failed to connect to MongoDB
```
**Solution:**
1. Verify MongoDB is running: `mongosh --eval "db.version()"`
2. Check `MONGO_URI` in `.env`
3. Check firewall/network settings

### Authentication Error
```
Authentication failed
```
**Solution:**
1. Update `MONGO_URI` with credentials:
   ```
   mongodb://username:password@localhost:27017/rag_chatbot
   ```

### Import Error
```
ModuleNotFoundError: No module named 'pymongo'
```
**Solution:**
```bash
pip install pymongo dnspython
```

## Backup and Restore

### Backup
```bash
# Backup entire database
mongodump --db=rag_chatbot --out=backup/

# Backup specific collection
mongodump --db=rag_chatbot --collection=documents --out=backup/
```

### Restore
```bash
# Restore entire database
mongorestore --db=rag_chatbot backup/rag_chatbot/

# Restore specific collection
mongorestore --db=rag_chatbot --collection=documents backup/rag_chatbot/documents.bson
```

## Performance Tips

1. **Indexes**: Already created by `init_mongodb.py`
2. **Connection Pooling**: Automatically handled by pymongo
3. **Query Optimization**: Use indexed fields in queries
4. **TTL**: Query logs auto-delete after 90 days (configurable)

## Migration from Existing Data

If you have existing data in SQLite/Qdrant:

1. Export from SQLite:
```python
from documents.models import Document
from backend.utils.mongo_repository import MongoRepository
from backend.utils.mongo_models import MongoDocument

repo = MongoRepository()
for doc in Document.objects.all():
    mongo_doc = MongoDocument(
        document_id=doc.document_id,
        filename=doc.filename,
        content_hash=doc.hash,
        mimetype=doc.mimetype,
        size_bytes=doc.size,
        token_count=doc.token_count or 0,
        vector_dim=doc.vector_dim
    )
    repo.save_document(mongo_doc)
```

2. Chunks are automatically saved when re-uploading documents with `force=true`

## Next Steps

1. **Monitor Logs**: Check `query_logs` collection regularly
2. **Run Evaluations**: Evaluation results now stored in MongoDB
3. **Analytics**: Use MongoDB aggregation for insights
4. **Scaling**: Consider MongoDB Atlas for production

## Support

For issues or questions:
1. Check the logs: `backend/logs/` (if configured)
2. Review MongoDB logs: `mongod.log`
3. Check application logs in terminal
