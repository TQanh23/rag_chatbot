import pandas as pd
import os

print("=" * 80)
print("DEBUGGING EVALUATION MISMATCH")
print("=" * 80)

# Load the files
qa_gold = pd.read_csv('backend/media/qa_gold.csv')
retrieval_run = pd.read_csv('backend/media/retrieval_run.csv')

print("\n1. Gold Standard Structure:")
print(qa_gold.head(2))
print(f"\nGold columns: {qa_gold.columns.tolist()}")

print("\n2. Retrieval Run Structure:")
print(retrieval_run.head(2))
print(f"\nRetrieval columns: {retrieval_run.columns.tolist()}")

# Check for matching question_ids
gold_qids = set(qa_gold['question_id'].unique())
run_qids = set(retrieval_run['question_id'].unique())

print(f"\n3. Question ID Overlap:")
print(f"   Gold questions: {len(gold_qids)}")
print(f"   Run questions: {len(run_qids)}")
print(f"   Matching: {len(gold_qids & run_qids)}")

if len(gold_qids & run_qids) > 0:
    # Check a sample question
    sample_qid = list(gold_qids & run_qids)[0]
    print(f"\n4. Sample Question Analysis (ID: {sample_qid}):")
    
    gold_sample = qa_gold[qa_gold['question_id'] == sample_qid].iloc[0]
    run_sample = retrieval_run[retrieval_run['question_id'] == sample_qid].iloc[0]
    
    print(f"\n   Gold relevant chunks (using 'gold_support_chunk_ids'):")
    # FIX: Use correct column name
    gold_chunks = gold_sample['gold_support_chunk_ids'].split('|') if pd.notna(gold_sample['gold_support_chunk_ids']) else []
    print(f"   {gold_chunks[:5]}")
    
    print(f"\n   Retrieved chunks:")
    retrieved_chunks = run_sample['retrieved_chunk_ids'].split('|') if pd.notna(run_sample['retrieved_chunk_ids']) else []
    print(f"   {retrieved_chunks[:5]}")
    
    overlap = set(gold_chunks) & set(retrieved_chunks)
    print(f"\n   Overlap: {len(overlap)} chunks")
    if overlap:
        print(f"   Matching: {list(overlap)[:3]}")
    
    # Check chunk ID formats
    print(f"\n5. Chunk ID Format Analysis:")
    print(f"   Gold format example: {gold_chunks[0] if gold_chunks else 'N/A'}")
    print(f"   Retrieved format example: {retrieved_chunks[0] if retrieved_chunks else 'N/A'}")
    
    # Show document IDs
    print(f"\n6. Document ID Check:")
    print(f"   Gold document: {gold_sample['document_id']}")
    print(f"   Run document: {run_sample['document_id']}")

print("\n" + "=" * 80)