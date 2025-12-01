import os
import sys
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

import pandas as pd
from backend.utils.mongo_repository import MongoRepository
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

qa_gold_path = 'backend/media/qa_gold.csv'

def debug_generation_answers():
    """Compare generated vs gold standard answers."""
    
    # Load gold standard
    gold_df = pd.read_csv(qa_gold_path)
    
    # Get uploaded document
    from backend.utils.qdrant_client import QdrantClient
    qdrant_wrapper = QdrantClient()
    qdrant = qdrant_wrapper._client
    collection_name = os.getenv('QDRANT_COLLECTION', 'test_collection')
    
    points, _ = qdrant.scroll(collection_name=collection_name, limit=1, with_payload=True, with_vectors=False)
    uploaded_doc_id = points[0].payload.get('document_id')
    
    # Get questions for this document
    doc_gold = gold_df[gold_df['document_id'] == uploaded_doc_id].set_index('question_text')
    
    # Get generation logs from MongoDB
    mongo_repo = MongoRepository()
    gen_logs = list(mongo_repo.db['generation_log'].find({'document_id': uploaded_doc_id}))
    
    print("=" * 100)
    print("GENERATION ANSWER COMPARISON")
    print("=" * 100)
    print()
    
    for idx, log in enumerate(gen_logs[:5], 1):  # First 5
        question = log.get('question_text', '').strip()
        generated = log.get('final_answer_text', '').strip()
        
        if question in doc_gold.index:
            gold_answer = doc_gold.loc[question, 'gold_answer']
            
            print(f"\n{idx}. Question:")
            print(f"   {question[:80]}...")
            print()
            print(f"   Generated:")
            print(f"   {generated[:150]}...")
            print()
            print(f"   Gold Standard:")
            print(f"   {gold_answer[:150]}...")
            print()
            print(f"   Match: {'YES' if generated == gold_answer else 'NO'}")
            print("-" * 100)

if __name__ == "__main__":
    debug_generation_answers()