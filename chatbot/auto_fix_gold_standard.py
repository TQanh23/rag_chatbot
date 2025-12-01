import os
import django
import pandas as pd
from datetime import datetime

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

from backend.utils.qdrant_client import QdrantClient
from backend.utils.embeddings import HuggingfaceEmbeddingsModel
from sentence_transformers import CrossEncoder
from qdrant_client import models

def find_best_chunks(question, document_id, qdrant_client, embeddings_model, reranker, collection_name, top_k=3):
    """Find most relevant chunks using retrieval + reranking."""
    query_embedding = embeddings_model.embed_texts([question])[0]
    
    search_filter = models.Filter(
        must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
    )
    
    # Retrieve top 100 candidates
    results = qdrant_client._client.search(
        collection_name=collection_name,
        query_vector=("default", query_embedding),
        limit=100,
        score_threshold=0.25,
        query_filter=search_filter
    )
    
    if not results:
        return []
    
    # Rerank candidates
    pairs = [(question, r.payload.get('text', r.payload.get('content', ''))) for r in results]
    rerank_scores = reranker.predict(pairs)
    
    # Combine and sort by rerank score
    reranked = [(results[i], rerank_scores[i]) for i in range(len(results))]
    reranked.sort(key=lambda x: x[1], reverse=True)
    
    # Return top_k chunk IDs with scores
    return [(r.payload.get('chunk_id'), score) for r, score in reranked[:top_k]]

def calculate_overlap(current_gold, suggested_chunks):
    """Calculate overlap between current and suggested gold chunks."""
    current_set = set(current_gold)
    suggested_set = set([chunk_id for chunk_id, _ in suggested_chunks])
    
    overlap = len(current_set & suggested_set)
    return overlap, len(current_set), len(suggested_set)

def auto_fix_gold_standard(
    min_rerank_score=0.5,
    top_k=3,
    min_overlap=1,
    dry_run=True,
    document_filter=None,
    question_filter=None
):
    """
    Automatically fix gold standard based on reranking scores.
    
    Args:
        min_rerank_score: Minimum rerank score to consider chunk relevant (default: 0.5)
        top_k: Number of top chunks to use as new gold standard (default: 3)
        min_overlap: Minimum overlap with current gold to apply change (default: 1)
        dry_run: If True, don't save changes, just show recommendations (default: True)
        document_filter: Only process questions for this document_id (optional)
        question_filter: Only process specific question_id (optional)
    """
    
    # Initialize clients
    print("Initializing models...")
    qdrant_client = QdrantClient()
    embeddings_model = HuggingfaceEmbeddingsModel()
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    collection_name = os.getenv('QDRANT_COLLECTION', 'test_collection')
    
    # Load current gold standard
    qa_gold = pd.read_csv('backend/qa_gold.csv')
    print(f"Loaded {len(qa_gold)} questions from qa_gold.csv")
    
    # Apply filters
    if document_filter:
        qa_gold = qa_gold[qa_gold['document_id'] == document_filter]
        print(f"Filtered to {len(qa_gold)} questions for document {document_filter}")
    
    if question_filter:
        qa_gold = qa_gold[qa_gold['question_id'] == question_filter]
        print(f"Filtered to question {question_filter}")
    
    if len(qa_gold) == 0:
        print("No questions to process after filtering")
        return
    
    # Process each question
    print("\n" + "=" * 100)
    print("PROCESSING QUESTIONS")
    print("=" * 100)
    
    corrections = []
    stats = {
        'total': len(qa_gold),
        'changed': 0,
        'kept': 0,
        'low_score': 0,
        'low_overlap': 0
    }
    
    for idx, row in qa_gold.iterrows():
        question_id = row['question_id']
        question = row['question_text']
        document_id = row['document_id']
        current_gold = row['gold_support_chunk_ids'].split('|')
        
        print(f"\n[{question_id}] {question[:60]}...")
        print(f"  Current gold ({len(current_gold)}): {', '.join([c.split(':')[-1] for c in current_gold])}")
        
        # Find best chunks using reranking
        best_chunks = find_best_chunks(
            question, document_id, qdrant_client, embeddings_model, 
            reranker, collection_name, top_k=top_k
        )
        
        if not best_chunks:
            print(f"  ⚠ No chunks found")
            stats['kept'] += 1
            continue
        
        # Show top suggestions
        print(f"  Suggested ({len(best_chunks)}):")
        for i, (chunk_id, score) in enumerate(best_chunks, 1):
            is_gold = '✓' if chunk_id in current_gold else ' '
            print(f"    [{is_gold}] {chunk_id.split(':')[-1]} (score: {score:.3f})")
        
        # Check if top chunk meets minimum score
        top_score = best_chunks[0][1]
        if top_score < min_rerank_score:
            print(f"  ⊘ Top score {top_score:.3f} below threshold {min_rerank_score}")
            stats['low_score'] += 1
            stats['kept'] += 1
            continue
        
        # Check overlap with current gold
        suggested_chunk_ids = [chunk_id for chunk_id, _ in best_chunks]
        overlap, current_count, suggested_count = calculate_overlap(current_gold, best_chunks)
        
        print(f"  Overlap: {overlap}/{current_count} chunks match")
        
        if overlap < min_overlap:
            print(f"  ⚠ Overlap {overlap} below minimum {min_overlap} - needs manual review")
            stats['low_overlap'] += 1
            stats['kept'] += 1
            continue
        
        # Check if change is needed
        if set(current_gold) == set(suggested_chunk_ids):
            print(f"  ✓ Already optimal")
            stats['kept'] += 1
            continue
        
        # Record correction
        new_gold = '|'.join(suggested_chunk_ids)
        corrections.append({
            'question_id': question_id,
            'question_text': question[:80],
            'old_gold': row['gold_support_chunk_ids'],
            'new_gold': new_gold,
            'top_score': top_score,
            'overlap': overlap
        })
        
        print(f"  ✓ Will update to: {', '.join([c.split(':')[-1] for c in suggested_chunk_ids])}")
        stats['changed'] += 1
    
    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Total questions processed: {stats['total']}")
    print(f"  Changed: {stats['changed']}")
    print(f"  Kept unchanged: {stats['kept']}")
    print(f"  Low rerank score: {stats['low_score']}")
    print(f"  Low overlap (needs review): {stats['low_overlap']}")
    
    if not corrections:
        print("\n✓ No changes needed")
        return
    
    # Show all corrections
    print("\n" + "=" * 100)
    print(f"PROPOSED CHANGES ({len(corrections)})")
    print("=" * 100)
    
    for i, corr in enumerate(corrections, 1):
        print(f"\n{i}. {corr['question_id']}: {corr['question_text']}...")
        print(f"   OLD: {corr['old_gold']}")
        print(f"   NEW: {corr['new_gold']}")
        print(f"   Top score: {corr['top_score']:.3f}, Overlap: {corr['overlap']}")
    
    # Apply changes if not dry run
    if dry_run:
        print("\n" + "=" * 100)
        print("DRY RUN MODE - No changes saved")
        print("Run with dry_run=False to apply changes")
        print("=" * 100)
    else:
        print("\n" + "=" * 100)
        print("APPLYING CHANGES")
        print("=" * 100)
        
        # Backup original
        backup_path = f'backend/media/qa_gold_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        qa_gold_original = pd.read_csv('backend/media/qa_gold.csv')
        qa_gold_original.to_csv(backup_path, index=False)
        print(f"✓ Backup saved to: {backup_path}")
        
        # Apply corrections
        qa_gold_updated = qa_gold_original.copy()
        for corr in corrections:
            qa_gold_updated.loc[
                qa_gold_updated['question_id'] == corr['question_id'], 
                'gold_support_chunk_ids'
            ] = corr['new_gold']
        
        # Save updated version
        qa_gold_updated.to_csv('backend/qa_gold.csv', index=False)
        print(f"✓ Applied {len(corrections)} changes to qa_gold.csv")
        
        # Save correction log
        log_path = f'backend/media/gold_standard_corrections_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        corrections_df = pd.DataFrame(corrections)
        corrections_df.to_csv(log_path, index=False)
        print(f"✓ Correction log saved to: {log_path}")
        
        print("\n" + "=" * 100)
        print("CHANGES APPLIED SUCCESSFULLY")
        print("=" * 100)

if __name__ == "__main__":
    import sys
    
    print("=" * 100)
    print("AUTO-FIX GOLD STANDARD BASED ON RERANKING SCORES")
    print("=" * 100)
    
    # Parse command line arguments
    dry_run = True
    document_filter = None
    question_filter = None
    min_rerank_score = 0.5
    top_k = 3
    min_overlap = 1
    
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg == '--apply':
                dry_run = False
            elif arg.startswith('--document='):
                document_filter = arg.split('=')[1]
            elif arg.startswith('--question='):
                question_filter = arg.split('=')[1]
            elif arg.startswith('--min-score='):
                min_rerank_score = float(arg.split('=')[1])
            elif arg.startswith('--top-k='):
                top_k = int(arg.split('=')[1])
            elif arg.startswith('--min-overlap='):
                min_overlap = int(arg.split('=')[1])
    
    print("\nConfiguration:")
    print(f"  Mode: {'APPLY CHANGES' if not dry_run else 'DRY RUN (preview only)'}")
    print(f"  Min rerank score: {min_rerank_score}")
    print(f"  Top K chunks: {top_k}")
    print(f"  Min overlap: {min_overlap}")
    if document_filter:
        print(f"  Document filter: {document_filter}")
    if question_filter:
        print(f"  Question filter: {question_filter}")
    print()
    
    # Run auto-fix
    auto_fix_gold_standard(
        min_rerank_score=min_rerank_score,
        top_k=top_k,
        min_overlap=min_overlap,
        dry_run=dry_run,
        document_filter=document_filter,
        question_filter=question_filter
    )