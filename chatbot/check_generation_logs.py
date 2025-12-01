import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

from backend.utils.mongo_repository import MongoRepository
import pandas as pd

mongo_repo = MongoRepository()

# Get recent generation logs with detailed fields
gen_logs = list(mongo_repo.db['generation_log'].find({}).sort("ts", -1).limit(20))

print("=" * 80)
print("GENERATION LOG ANALYSIS")
print("=" * 80)

if not gen_logs:
    print("No generation logs found!")
else:
    print(f"\nTotal logs: {len(gen_logs)}")
    
    # Analyze answer content
    empty_count = 0
    error_count = 0
    content_count = 0
    
    for i, log in enumerate(gen_logs[:5], 1):
        print(f"\n--- Log {i} ---")
        print(f"Question: {log.get('question_text', 'N/A')[:60]}...")
        print(f"Answer: {log.get('generated_answer', 'N/A')[:80]}...")
        print(f"Status: {log.get('status', 'success')}")
        print(f"Error: {log.get('error', 'None')[:100] if log.get('error') else 'None'}")
        print(f"Finish reason: {log.get('finish_reason', 'N/A')}")
        print(f"Model: {log.get('model_name', 'N/A')}")
        print(f"Citations: {len(log.get('citations', []))}")
        print(f"Context length: {log.get('context_length', 'N/A')}")
        
        answer = log.get('generated_answer', '')
        if not answer or not answer.strip():
            empty_count += 1
        elif log.get('error'):
            error_count += 1
        else:
            content_count += 1
    
    print(f"\n\nSummary:")
    print(f"  Content: {content_count}")
    print(f"  Empty: {empty_count}")
    print(f"  Errors: {error_count}")
    
    # Check for common error patterns
    errors = [log.get('error') for log in gen_logs if log.get('error')]
    if errors:
        print(f"\n\nCommon errors:")
        for err in set(errors[:5]):
            print(f"  - {err[:150]}")