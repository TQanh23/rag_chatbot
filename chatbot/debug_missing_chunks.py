import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

from backend.utils.qdrant_client import QdrantClient
from backend.utils.embeddings import HuggingfaceEmbeddingsModel
from qdrant_client import models

# Load test data
import pandas as pd
qa_gold = pd.read_csv('backend/media/qa_gold.csv')
test_row = qa_gold[qa_gold['question_id'] == 'q0002'].iloc[0]

question = test_row['question_text']
document_id = test_row['document_id']
gold_chunks = test_row['gold_support_chunk_ids'].split('|')

print("=" * 80)
print(f"Question: {question}")
print(f"Missing gold chunks: {gold_chunks}")
print("=" * 80)

# Initialize clients
qdrant_client = QdrantClient()
embeddings_model = HuggingfaceEmbeddingsModel()
collection_name = os.getenv('QDRANT_COLLECTION', 'test_collection')

# Check if missing chunks exist in Qdrant
for chunk_id in gold_chunks:
    print(f"\n--- Checking chunk: {chunk_id} ---")
    
    # Search by chunk_id in payload
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
        with_payload=True,
        with_vectors=True
    )
    
    if results[0]:
        point = results[0][0]
        chunk_text = point.payload.get('text', point.payload.get('content', 'N/A'))
        print(f"✓ Found in Qdrant")
        print(f"Text preview: {chunk_text[:200]}...")
        
        # Calculate similarity with question
        query_embedding = embeddings_model.embed_texts([question])[0]
        chunk_embedding = point.vector.get('default') if isinstance(point.vector, dict) else point.vector
        
        # Compute cosine similarity
        import numpy as np
        similarity = np.dot(query_embedding, chunk_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
        )
        print(f"Cosine similarity with question: {similarity:.3f}")
        
        # Test if it would pass threshold
        if similarity < 0.3:
            print(f"⚠ Below threshold (0.3) - Would be filtered out!")
        
    else:
        print(f"✗ NOT FOUND in Qdrant")
        print(f"⚠ This chunk may not have been uploaded or has wrong chunk_id")

print("\n" + "=" * 80)
print("RECOMMENDATION:")
print("1. If chunks don't exist → Re-upload document or check chunking logic")
print("2. If similarity < 0.3 → Gold standard may be incorrect or threshold too high")
print("3. If similarity is decent → Increase initial retrieval limit (top_k)")
print("=" * 80)