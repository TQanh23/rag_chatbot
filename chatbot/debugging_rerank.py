import os
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

from backend.utils.embeddings import HuggingfaceEmbeddingsModel
from backend.utils.qdrant_client import get_qdrant_client
from sentence_transformers import CrossEncoder
import numpy as np

question = "Trong chiến lược phân trang, PTBR (Page-table base register) có chức năng gì?"

# Step 1: Retrieve
embeddings_model = HuggingfaceEmbeddingsModel()
embedding = embeddings_model.embed_texts([question])[0]

client = get_qdrant_client()
results = client.search(
    collection_name="test_collection",
    query_vector=("default", embedding),
    limit=20,
    score_threshold=0.3,
    with_payload=True
)

print(f"\n{'='*80}")
print(f"Retrieved {len(results)} chunks from vector search")
print(f"{'='*80}\n")

# Step 2: Rerank with PROPER format
reranker = CrossEncoder('itdainb/PhoRanker')

# Format EXACTLY as expected
rerank_inputs = [[question, r.payload['text']] for r in results]

print(f"Reranking {len(rerank_inputs)} pairs...")
rerank_scores = reranker.predict(rerank_inputs, convert_to_numpy=True)

print(f"\nRerank score statistics:")
print(f"  Min: {rerank_scores.min():.4f}")
print(f"  Max: {rerank_scores.max():.4f}")
print(f"  Mean: {rerank_scores.mean():.4f}")
print(f"  Median: {np.median(rerank_scores):.4f}")
print(f"  Std: {rerank_scores.std():.4f}")

# Combine
combined = [(results[i], rerank_scores[i]) for i in range(len(results))]
combined.sort(key=lambda x: x[1], reverse=True)

print(f"\n{'='*80}")
print("Top 10 results:")
print(f"{'='*80}")
for i, (r, score) in enumerate(combined[:10], 1):
    text_preview = r.payload['text'][:70].replace('\n', ' ')
    status = "✓ GOOD" if score >= 0.5 else ("~ OK" if score >= 0.0 else "✗ BAD")
    print(f"{i:2d}. [{score:7.3f}] {status} | {text_preview}...")

# Count by threshold
print(f"\n{'='*80}")
print("Score distribution:")
print(f"{'='*80}")
for threshold in [0.5, 0.0, -0.5, -1.0]:
    count = sum(1 for s in rerank_scores if s >= threshold)
    pct = 100 * count / len(rerank_scores)
    print(f"  ≥ {threshold:5.1f}: {count:2d} chunks ({pct:5.1f}%)")

print(f"\n{'='*80}")
print("RECOMMENDATION:")
print(f"{'='*80}")
good_threshold = float(os.getenv("MIN_RERANK_SCORE", "0.1"))
good_count = sum(1 for s in rerank_scores if s >= good_threshold)
print(f"With MIN_RERANK_SCORE={good_threshold}: {good_count} chunks pass filter")
if good_count == 0 and len(rerank_scores) > 0:
    median = np.median(rerank_scores)
    print(f"✗ TOO STRICT! Median score is {median:.3f}")
    print(f"→ Try setting MIN_RERANK_SCORE={median:.2f}")
else:
    print(f"✓ Configuration looks reasonable")
print(f"{'='*80}\n")