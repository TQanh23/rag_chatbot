import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

from backend.utils.mongo_repository import MongoRepository

repo = MongoRepository()

# List all collections
collections = repo.db.list_collection_names()
print("Available MongoDB collections:")
for col in collections:
    count = repo.db[col].count_documents({})
    print(f"  - {col}: {count} documents")

print("\n" + "="*80)
print("Checking generation_log schema:")
print("="*80)

if 'generation_log' in collections:
    sample = repo.db['generation_log'].find_one()
    if sample:
        print("\nSample generation_log document:")
        print(f"  question_id: {sample.get('question_id')}")
        print(f"  question_text: {sample.get('question_text', 'N/A')[:60]}...")
        print(f"  document_id: {sample.get('document_id')}")
        print(f"  generated_answer: {sample.get('generated_answer', 'N/A')[:80]}...")
        print(f"  model_name: {sample.get('model_name')}")
        
        citations = sample.get('citations', [])
        print(f"\n  citations ({len(citations)} items):")
        for i, cit in enumerate(citations[:3], 1):
            print(f"    [{i}] chunk_id={cit.get('chunk_id', 'N/A')}")
            print(f"        document_id={cit.get('document_id', 'N/A')}")
            print(f"        page={cit.get('page', 'N/A')}")
            print(f"        score={cit.get('score', 'N/A')}")
        
        print(f"\n  timestamp: {sample.get('ts')}")
        print(f"  latency_ms: {sample.get('latency_ms')}")
    else:
        print("  (No documents found)")
else:
    print("  generation_log collection does NOT exist!")

print("\n" + "="*80)
print("Checking if citation_log exists:")
print("="*80)
if 'citation_log' in collections:
    count = repo.db['citation_log'].count_documents({})
    print(f"  citation_log EXISTS with {count} documents")
    if count > 0:
        sample = repo.db['citation_log'].find_one()
        print(f"  Sample: {sample}")
else:
    print("  citation_log does NOT exist (expected - citations are in generation_log)")