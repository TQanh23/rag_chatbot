import pandas as pd

print("=" * 80)
print("VERIFYING CHUNK ID CONVERSION")
print("=" * 80)

# Load files
qa_gold = pd.read_csv('backend/media/qa_gold.csv')
retrieval_run = pd.read_csv('backend/media/retrieval_run.csv')

# Sample a few questions
sample_qids = retrieval_run['question_id'].head(3).tolist()

for qid in sample_qids:
    print(f"\n{'=' * 80}")
    print(f"Question ID: {qid}")
    print("=" * 80)
    
    gold_row = qa_gold[qa_gold['question_id'] == qid].iloc[0]
    run_row = retrieval_run[retrieval_run['question_id'] == qid].iloc[0]
    
    print(f"\nQuestion: {run_row['question_text'][:100]}...")
    
    gold_chunks = gold_row['gold_support_chunk_ids'].split('|') if pd.notna(gold_row['gold_support_chunk_ids']) else []
    retrieved_chunks = run_row['retrieved_chunk_ids'].split('|') if pd.notna(run_row['retrieved_chunk_ids']) else []
    
    print(f"\nGold Support Chunks ({len(gold_chunks)}):")
    for i, chunk in enumerate(gold_chunks[:5], 1):
        print(f"  {i}. {chunk}")
    
    print(f"\nRetrieved Chunks ({len(retrieved_chunks)}):")
    for i, chunk in enumerate(retrieved_chunks[:5], 1):
        print(f"  {i}. {chunk}")
    
    # Check overlap
    overlap = set(gold_chunks) & set(retrieved_chunks)
    print(f"\n✓ Matching chunks: {len(overlap)}/{len(gold_chunks)}")
    if overlap:
        print(f"  Matches: {list(overlap)[:3]}")
    
    # Analyze format differences
    if gold_chunks and retrieved_chunks:
        gold_sample = gold_chunks[0]
        retrieved_sample = retrieved_chunks[0]
        
        print(f"\nFormat Analysis:")
        print(f"  Gold format:      '{gold_sample}'")
        print(f"  Retrieved format: '{retrieved_sample}'")
        
        # Check if chunk numbers are close
        if '::chunk_' in gold_sample and '::chunk_' in retrieved_sample:
            try:
                gold_num = int(gold_sample.split('::chunk_')[1])
                retr_num = int(retrieved_sample.split('::chunk_')[1])
                print(f"  Chunk numbers: Gold={gold_num}, Retrieved={retr_num}, Diff={abs(gold_num - retr_num)}")
            except:
                pass

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

# Calculate overall match rate
total_matches = 0
total_gold = 0

for _, row in retrieval_run.iterrows():
    qid = row['question_id']
    gold_row = qa_gold[qa_gold['question_id'] == qid].iloc[0]
    
    gold_chunks = set(gold_row['gold_support_chunk_ids'].split('|')) if pd.notna(gold_row['gold_support_chunk_ids']) else set()
    retrieved_chunks = set(row['retrieved_chunk_ids'].split('|')) if pd.notna(row['retrieved_chunk_ids']) else set()
    
    total_gold += len(gold_chunks)
    total_matches += len(gold_chunks & retrieved_chunks)

print(f"\nOverall Match Rate: {total_matches}/{total_gold} = {total_matches/total_gold*100:.1f}%")
print(f"Mean Recall@10: 0.350 (from evaluation)")
print("\n" + "=" * 80)