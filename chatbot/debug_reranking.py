import pandas as pd
import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

from backend.utils.qdrant_client import QdrantClient
from backend.utils.embeddings import HuggingfaceEmbeddingsModel
from sentence_transformers import CrossEncoder

# Load test case
qa_gold = pd.read_csv('backend/media/qa_gold.csv')
test_row = qa_gold[qa_gold['question_id'] == 'q0002'].iloc[0]

question = test_row['question_text']
document_id = test_row['document_id']
gold_chunks = test_row['gold_support_chunk_ids'].split('|')

print("=" * 80)
print(f"Question: {question}")
print(f"Document: {document_id}")
print(f"Gold chunks: {gold_chunks}")
print("=" * 80)

# Get embedding
embeddings_model = HuggingfaceEmbeddingsModel()
query_embedding = embeddings_model.embed_texts([question])[0]

# Search WITHOUT reranking
qdrant_client = QdrantClient()
collection_name = os.getenv('QDRANT_COLLECTION', 'test_collection')

from qdrant_client import models

search_filter = models.Filter(
    must=[
        models.FieldCondition(
            key="document_id",
            match=models.MatchValue(value=document_id)
        )
    ]
)

initial_results = qdrant_client._client.search(
    collection_name=collection_name,
    query_vector=("default", query_embedding),
    limit=100,
    score_threshold=0.25,
    query_filter=search_filter
)

print(f"\nInitial retrieval (top 10 of {len(initial_results)}):")
for i, result in enumerate(initial_results[:10], 1):
    chunk_id = result.payload.get('chunk_id', 'N/A')
    score = result.score
    is_gold = '✓' if chunk_id in gold_chunks else ' '
    print(f"  {i}. [{is_gold}] {chunk_id} (score: {score:.3f})")

# Apply reranking
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

pairs = [(question, result.payload.get('text', result.payload.get('content', ''))) for result in initial_results]
rerank_scores = reranker.predict(pairs)

# Combine and sort
reranked = [(initial_results[i], rerank_scores[i]) for i in range(len(initial_results))]
reranked.sort(key=lambda x: x[1], reverse=True)

print(f"\nAfter reranking (top 10):")
for i, (result, rerank_score) in enumerate(reranked[:10], 1):
    chunk_id = result.payload.get('chunk_id', 'N/A')
    original_score = result.score
    is_gold = '✓' if chunk_id in gold_chunks else ' '
    print(f"  {i}. [{is_gold}] {chunk_id} (vec: {original_score:.3f}, rerank: {rerank_score:.3f})")

# Check if gold chunks are present
print(f"\nGold chunk positions:")
for gold_chunk in gold_chunks:
    # Find in initial results
    initial_pos = next((i+1 for i, r in enumerate(initial_results) if r.payload.get('chunk_id') == gold_chunk), None)
    # Find in reranked results
    reranked_pos = next((i+1 for i, (r, _) in enumerate(reranked) if r.payload.get('chunk_id') == gold_chunk), None)
    
    print(f"  {gold_chunk}:")
    print(f"    Initial position: {initial_pos if initial_pos else 'NOT FOUND'}")
    print(f"    Reranked position: {reranked_pos if reranked_pos else 'NOT FOUND'}")

print("=" * 80)