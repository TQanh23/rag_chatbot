# MongoDB Quick Reference

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install pymongo dnspython

# 2. Configure .env
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=rag_chatbot

# 3. Initialize MongoDB
python init_mongodb.py

# 4. Start application
python manage.py runserver
```

## 📊 Common Commands

### Check Status
```bash
python mongo_utils.py health
python mongo_utils.py stats
```

### View Data
```bash
python mongo_utils.py list-docs --limit 10
python mongo_utils.py list-queries --days 7
```

### Export & Cleanup
```bash
python mongo_utils.py export-logs --output queries.csv --days 30
python mongo_utils.py cleanup --days 90
```

## 🔍 MongoDB Queries

### Recent Queries
```javascript
db.query_logs.find().sort({ts: -1}).limit(10)
```

### Documents by Status
```javascript
db.documents.aggregate([
  {$group: {_id: "$status", count: {$sum: 1}}}
])
```

### Average Latency
```javascript
db.query_logs.aggregate([
  {$group: {_id: null, avg: {$avg: "$latency_ms"}}}
])
```

## 🏗️ Architecture

- **Qdrant**: Vector search (fast retrieval)
- **MongoDB**: Metadata + logs (persistent storage)
- **Django**: Application logic
- **Gemini**: Answer generation

## 📁 Collections

- `documents`: Uploaded file metadata
- `chunks`: Text chunks + embeddings
- `query_logs`: All questions & answers (TTL: 90 days)
- `retrieval_runs`: Retrieval metrics
- `generation_runs`: Generation metrics
- `eval_runs`: Evaluation summaries

## 🛠️ Troubleshooting

### Connection Error
```bash
# Check MongoDB is running
mongosh --eval "db.version()"

# Verify .env settings
cat .env | grep MONGO
```

### Missing Dependencies
```bash
pip install pymongo dnspython
```

## 📚 Documentation

- Full setup: `MONGODB_SETUP.md`
- Migration summary: `MONGODB_MIGRATION_SUMMARY.md`
- API health: `GET /api/health/`
