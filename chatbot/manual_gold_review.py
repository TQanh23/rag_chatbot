import os
import django
import pandas as pd

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

from backend.utils.qdrant_client import QdrantClient
from backend.utils.embeddings import HuggingfaceEmbeddingsModel
from sentence_transformers import CrossEncoder
from qdrant_client import models

def get_chunk_content(qdrant_client, collection_name, chunk_id):
    """Retrieve chunk content from Qdrant."""
    search_filter = models.Filter(
        must=[models.FieldCondition(key="chunk_id", match=models.MatchValue(value=chunk_id))]
    )
    results = qdrant_client._client.scroll(
        collection_name=collection_name,
        scroll_filter=search_filter,
        limit=1,
        with_payload=True
    )
    if results[0]:
        return results[0][0].payload.get('text', results[0][0].payload.get('content', 'N/A'))
    return 'NOT FOUND'

def find_best_chunks(question, document_id, qdrant_client, embeddings_model, reranker, collection_name, top_k=10):
    """Find most relevant chunks using retrieval + reranking."""
    query_embedding = embeddings_model.embed_texts([question])[0]
    
    search_filter = models.Filter(
        must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
    )
    
    results = qdrant_client._client.search(
        collection_name=collection_name,
        query_vector=("default", query_embedding),
        limit=100,
        score_threshold=0.25,
        query_filter=search_filter
    )
    
    pairs = [(question, r.payload.get('text', r.payload.get('content', ''))) for r in results]
    rerank_scores = reranker.predict(pairs)
    
    reranked = [(results[i], rerank_scores[i]) for i in range(len(results))]
    reranked.sort(key=lambda x: x[1], reverse=True)
    
    return reranked[:top_k]

def review_question(qa_row, qdrant_client, embeddings_model, reranker, collection_name):
    """Display question review interface."""
    question_id = qa_row['question_id']
    question = qa_row['question_text']
    document_id = qa_row['document_id']
    current_gold = qa_row['gold_support_chunk_ids'].split('|')
    
    print("\n" + "=" * 100)
    print(f"QUESTION ID: {question_id}")
    print("=" * 100)
    print(f"Question: {question}")
    print(f"Document: {document_id}")
    print()
    
    # Show current gold chunks
    print("-" * 100)
    print("CURRENT GOLD CHUNKS:")
    print("-" * 100)
    for i, chunk_id in enumerate(current_gold, 1):
        content = get_chunk_content(qdrant_client, collection_name, chunk_id)
        print(f"\n[{i}] {chunk_id}")
        print(f"Content: {content[:300]}{'...' if len(content) > 300 else ''}")
    
    # Get and show top reranked chunks
    best_chunks = find_best_chunks(question, document_id, qdrant_client, embeddings_model, reranker, collection_name, top_k=10)
    
    print("\n" + "-" * 100)
    print("TOP 10 CHUNKS BY RERANKING:")
    print("-" * 100)
    for i, (result, score) in enumerate(best_chunks, 1):
        chunk_id = result.payload.get('chunk_id')
        content = result.payload.get('text', result.payload.get('content', ''))
        is_current_gold = '✓ GOLD' if chunk_id in current_gold else ''
        print(f"\n[{i}] {chunk_id} (rerank: {score:.3f}) {is_current_gold}")
        print(f"Content: {content[:300]}{'...' if len(content) > 300 else ''}")
    
    print("\n" + "=" * 100)
    print("REVIEW INSTRUCTIONS:")
    print("  - Compare CURRENT GOLD chunks with TOP RERANKED chunks")
    print("  - Identify which chunks actually answer the question")
    print("  - Note chunk IDs that should be in the gold standard")
    print("=" * 100)
    
    return best_chunks

def main():
    # Initialize clients
    qdrant_client = QdrantClient()
    embeddings_model = HuggingfaceEmbeddingsModel()
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    collection_name = os.getenv('QDRANT_COLLECTION', 'test_collection')
    
    # Load gold standard
    qa_gold = pd.read_csv('backend/media/qa_gold.csv')
    
    print("=" * 100)
    print("MANUAL GOLD STANDARD REVIEW")
    print("=" * 100)
    print(f"Total questions to review: {len(qa_gold)}")
    print()
    print("This script will display each question with:")
    print("  1. Current gold chunk annotations")
    print("  2. Top 10 chunks by retrieval + reranking")
    print()
    print("Review each question and note corrections needed.")
    print("=" * 100)
    
    # Ask which questions to review
    print("\nReview options:")
    print("  1. Review all questions (1-50)")
    print("  2. Review specific question by ID (e.g., q0002)")
    print("  3. Review by document ID")
    print("  4. Review questions 1-10 only")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    questions_to_review = []
    
    if choice == '1':
        questions_to_review = qa_gold.to_dict('records')
    elif choice == '2':
        qid = input("Enter question ID (e.g., q0002): ").strip()
        questions_to_review = qa_gold[qa_gold['question_id'] == qid].to_dict('records')
    elif choice == '3':
        doc_id = input("Enter document ID: ").strip()
        questions_to_review = qa_gold[qa_gold['document_id'] == doc_id].to_dict('records')
    elif choice == '4':
        questions_to_review = qa_gold.head(10).to_dict('records')
    else:
        print("Invalid choice. Exiting.")
        return
    
    if not questions_to_review:
        print("No questions found matching criteria.")
        return
    
    print(f"\nReviewing {len(questions_to_review)} question(s)...")
    
    # Review each question
    corrections = []
    for qa_row in questions_to_review:
        best_chunks = review_question(qa_row, qdrant_client, embeddings_model, reranker, collection_name)
        
        print("\nSUGGESTED CORRECTIONS:")
        print("  Based on reranking, suggested gold chunks (top 3):")
        suggested = [best_chunks[i][0].payload.get('chunk_id') for i in range(min(3, len(best_chunks)))]
        print(f"  {' | '.join(suggested)}")
        
        # Ask for manual annotation
        print("\nACTION OPTIONS:")
        print("  1. Keep current gold chunks (no change)")
        print("  2. Use suggested top 3 chunks")
        print("  3. Enter custom chunk IDs (comma-separated)")
        print("  4. Skip this question")
        
        action = input("\nEnter choice (1-4): ").strip()
        
        new_gold = None
        if action == '1':
            print("✓ Keeping current gold chunks")
        elif action == '2':
            new_gold = '|'.join(suggested)
            print(f"✓ Updated to: {new_gold}")
        elif action == '3':
            custom = input("Enter chunk IDs (comma-separated, e.g., chunk_id1,chunk_id2): ").strip()
            if custom:
                new_gold = '|'.join([c.strip() for c in custom.split(',')])
                print(f"✓ Updated to: {new_gold}")
        elif action == '4':
            print("⊘ Skipped")
        else:
            print("Invalid choice, keeping current")
        
        if new_gold:
            corrections.append({
                'question_id': qa_row['question_id'],
                'old_gold': qa_row['gold_support_chunk_ids'],
                'new_gold': new_gold
            })
        
        # Ask to continue
        if len(questions_to_review) > 1:
            cont = input("\nContinue to next question? (y/n): ").strip().lower()
            if cont != 'y':
                break
    
    # Summary of corrections
    if corrections:
        print("\n" + "=" * 100)
        print("SUMMARY OF CORRECTIONS")
        print("=" * 100)
        for corr in corrections:
            print(f"\n{corr['question_id']}:")
            print(f"  OLD: {corr['old_gold']}")
            print(f"  NEW: {corr['new_gold']}")
        
        # Ask to apply corrections
        apply = input("\nApply these corrections to qa_gold.csv? (y/n): ").strip().lower()
        if apply == 'y':
            for corr in corrections:
                qa_gold.loc[qa_gold['question_id'] == corr['question_id'], 'gold_support_chunk_ids'] = corr['new_gold']
            
            qa_gold.to_csv('backend/media/qa_gold.csv', index=False)
            print(f"✓ Applied {len(corrections)} correction(s) to qa_gold.csv")
        else:
            print("⊘ Corrections not applied")
    else:
        print("\n✓ No corrections made")
    
    print("\n" + "=" * 100)
    print("REVIEW COMPLETE")
    print("=" * 100)

if __name__ == "__main__":
    main()