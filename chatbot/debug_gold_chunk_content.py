import pandas as pd
import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

from backend.utils.qdrant_client import QdrantClient
from qdrant_client import models

# Load test case
qa_gold = pd.read_csv('backend/media/qa_gold.csv')
test_row = qa_gold[qa_gold['question_id'] == 'q0002'].iloc[0]

question = test_row['question_text']
document_id = test_row['document_id']
gold_chunks = test_row['gold_support_chunk_ids'].split('|')

print("=" * 80)
print(f"Question: {question}")
print("=" * 80)

qdrant_client = QdrantClient()
collection_name = os.getenv('QDRANT_COLLECTION', 'test_collection')

# Get content of gold chunks
for chunk_id in gold_chunks:
    search_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="chunk_id",
                match=models.MatchValue(value=chunk_id)
            )
        ]
    )
    
    results = qdrant_client._client.scroll(
        collection_name=collection_name,
        scroll_filter=search_filter,
        limit=1,
        with_payload=True
    )
    
    if results[0]:
        point = results[0][0]
        chunk_text = point.payload.get('text', point.payload.get('content', 'N/A'))
        print(f"\n--- Gold Chunk: {chunk_id} ---")
        print(f"Content:\n{chunk_text}")
        print("\n" + "-" * 80)

# Also show top reranked chunks for comparison
from backend.utils.embeddings import HuggingfaceEmbeddingsModel
from sentence_transformers import CrossEncoder

embeddings_model = HuggingfaceEmbeddingsModel()
query_embedding = embeddings_model.embed_texts([question])[0]

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

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
pairs = [(question, r.payload.get('text', r.payload.get('content', ''))) for r in initial_results]
rerank_scores = reranker.predict(pairs)

reranked = [(initial_results[i], rerank_scores[i]) for i in range(len(initial_results))]
reranked.sort(key=lambda x: x[1], reverse=True)

print("\n" + "=" * 80)
print("TOP 3 RERANKED CHUNKS (for comparison):")
print("=" * 80)

for i, (result, rerank_score) in enumerate(reranked[:3], 1):
    chunk_id = result.payload.get('chunk_id', 'N/A')
    chunk_text = result.payload.get('text', result.payload.get('content', 'N/A'))
    is_gold = '✓' if chunk_id in gold_chunks else 'X'
    print(f"\n[{is_gold}] Rank {i}: {chunk_id} (rerank: {rerank_score:.3f})")
    print(f"Content:\n{chunk_text[:300]}...")
    print("-" * 80)