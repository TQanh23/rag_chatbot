import os
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

from backend.utils.embeddings import HuggingfaceEmbeddingsModel
from backend.utils.qdrant_client import get_qdrant_client
from sentence_transformers import CrossEncoder
import numpy as np

question = "Trong chiến lược phân trang, PTBR (Page-table base register) có chức năng gì?"

# Get all retrieved chunks
embeddings_model = HuggingfaceEmbeddingsModel()
embedding = embeddings_model.embed_texts([question])[0]

client = get_qdrant_client()
results = client.search(
    collection_name="test_collection",
    query_vector=("default", embedding),
    limit=20,
    score_threshold=0.25,
    with_payload=True
)

print(f"\n{'='*80}")
print(f"CHUNK CONTENT ANALYSIS FOR: {question[:60]}...")
print(f"{'='*80}\n")

# Rerank
reranker = CrossEncoder('itdainb/PhoRanker')
rerank_inputs = [[question, r.payload['text']] for r in results]
rerank_scores = reranker.predict(rerank_inputs, convert_to_numpy=True)

combined = [(results[i], rerank_scores[i]) for i in range(len(results))]
combined.sort(key=lambda x: x[1], reverse=True)

# Detailed analysis
print("ANALYSIS: Why is reranking failing?")
print("-" * 80)

for i, (r, rerank_score) in enumerate(combined[:5], 1):
    payload = r.payload
    text = payload.get('text', '')[:150].replace('\n', ' ')
    
    # Keywords in question
    keywords = ['phân trang', 'PTBR', 'page-table', 'base register', 'chiến lược']
    keyword_match = sum(1 for kw in keywords if kw.lower() in text.lower())
    
    print(f"\n#{i} [Rerank: {rerank_score:.4f}, Vector: {r.score:.4f}]")
    print(f"  Keywords matched: {keyword_match}/{len(keywords)}")
    print(f"  Text: {text}...")
    print(f"  Document: {payload.get('document_id', 'N/A')}")
    print(f"  Page: {payload.get('page', 'N/A')}")
    print(f"  Section: {payload.get('section_title', 'N/A')}")

print(f"\n{'='*80}")
print("DIAGNOSIS:")
print(f"{'='*80}")

# FIX: Simplified checks
mentions_ptbr = sum(1 for r, _ in combined if 'PTBR' in r.payload.get('text', ''))
print(f"✓ Chunks mentioning 'PTBR': {mentions_ptbr}")

high_vector_score = sum(1 for r, _ in combined if r.score >= 0.5)
print(f"✓ Chunks with vector score ≥ 0.5: {high_vector_score}")

high_rerank_score = sum(1 for _, score in combined if score >= 0.1)
print(f"✓ Chunks with rerank score ≥ 0.1: {high_rerank_score}")

if high_rerank_score <= 1:
    print("\n✗ PROBLEM: Reranker sees almost all chunks as irrelevant")
    print("  Possible causes:")
    print("  1. Chunks don't contain answer to the question")
    print("  2. Chunking strategy fragments related content")
    print("  3. Question is too specific, docs don't cover it")
    print("  4. Language/terminology mismatch")
else:
    print("\n✓ Reranker working correctly, threshold needs adjustment")

print(f"{'='*80}\n")