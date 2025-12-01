import os
import sys
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

import pandas as pd
from backend.utils.mongo_repository import MongoRepository
from backend.utils.qdrant_client import QdrantClient
from qdrant_client import models

# Load gold standard
gold_df = pd.read_csv('backend/media/qa_gold.csv')
print("=" * 80)
print("GOLD STANDARD CHUNK IDs (first 5 questions):")
print("=" * 80)
for idx, row in gold_df.head(5).iterrows():
    print(f"Q: {row['question_id']}")
    print(f"  Gold chunks: {row['gold_support_chunk_ids']}")
    print()

# Check retrieval run
if os.path.exists('backend/media/retrieval_run.csv'):
    ret_df = pd.read_csv('backend/media/retrieval_run.csv')
    print("=" * 80)
    print("RETRIEVED CHUNK IDs (first 5 entries):")
    print("=" * 80)
    for idx, row in ret_df.head(5).iterrows():
        print(f"Q: {row['question_id']}")
        print(f"  Retrieved: {row['retrieved_chunk_ids']}")
        print()

# Check Qdrant chunk IDs
print("=" * 80)
print("QDRANT CHUNK ID FORMAT (first 10):")
print("=" * 80)
qdrant_client = QdrantClient()
collection_name = os.getenv('QDRANT_COLLECTION', 'test_collection')

points, _ = qdrant_client._client.scroll(
    collection_name=collection_name,
    limit=10,
    with_payload=True,
    with_vectors=False
)

for p in points:
    chunk_id = p.payload.get('chunk_id', 'N/A') if p.payload else 'N/A'
    doc_id = p.payload.get('document_id', 'N/A') if p.payload else 'N/A'
    print(f"  Point ID: {p.id}")
    print(f"  chunk_id: {chunk_id}")
    print(f"  document_id: {doc_id}")
    print()

# Search for chunks from the test document
search_filter = models.Filter(
    must=[
        models.FieldCondition(
            key="document_id",
            match=models.MatchValue(value="1e8e6c622bfb571b0b783d8347182318")
        )
    ]
)

results = qdrant_client._client.scroll(
    collection_name=collection_name,
    scroll_filter=search_filter,
    limit=10
)

print("Sample chunk IDs from Qdrant:")
for point in results[0]:
    print(f"  chunk_id: {point.payload.get('chunk_id')}")
    print(f"  chunk_index: {point.payload.get('chunk_index')}")
    print()

# Check MongoDB retrieval logs
print("=" * 80)
print("MONGODB RETRIEVAL LOG CHUNK IDs (first entry):")
print("=" * 80)
mongo_repo = MongoRepository()
log = mongo_repo.db['retrieval_log'].find_one({})
if log:
    print(f"  Question: {log.get('question', '')[:60]}...")
    print(f"  retrieved_chunk_ids: {log.get('retrieved_chunk_ids', [])[:5]}")