import django
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

from backend.utils.embeddings import HuggingfaceEmbeddingsModel
from backend.utils.qdrant_client import get_qdrant_client
from sentence_transformers import CrossEncoder
from django.conf import settings
import pandas as pd
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_reranking_impact():
    """
    Compare retrieval quality before and after reranking using PhoRanker.
    Uses qa_gold.csv as ground truth for evaluation.
    """
    print("=" * 60)
    print("RERANKING IMPACT EVALUATION")
    print("=" * 60)
    print()
    
    # Load gold standard
    qa_gold_path = os.path.join(settings.BASE_DIR, 'qa_gold.csv')
    if not os.path.exists(qa_gold_path):
        print(f"✗ Gold standard file not found: {qa_gold_path}")
        print("  Please create qa_gold.csv with test questions and gold chunk IDs")
        return
    
    qa_df = pd.read_csv(qa_gold_path)
    print(f"✓ Loaded {len(qa_df)} test questions from qa_gold.csv")
    print()
    
    # Check required columns (updated to match actual CSV structure)
    required_cols = ['question_text', 'gold_support_chunk_ids']
    if not all(col in qa_df.columns for col in required_cols):
        print(f"✗ Missing required columns: {required_cols}")
        print(f"Available columns: {list(qa_df.columns)}")
        return
    
    # Initialize components
    model = HuggingfaceEmbeddingsModel()
    client = get_qdrant_client()
    reranker = CrossEncoder('itdainb/PhoRanker')
    
    print("✓ Initialized embedding model and reranker")
    print(f"  Embedding model: {os.getenv('EMBEDDING_MODEL', 'dangvantuan/vietnamese-embedding')}")
    print(f"  Reranker: itdainb/PhoRanker")
    print()
    
    results = []
    
    print("Evaluating queries...")
    print("-" * 60)
    
    for idx, row in qa_df.iterrows():
        question = row['question_text']
        
        # Parse gold chunk IDs (format: "doc_id:chunk_num|doc_id:chunk_num")
        gold_chunk_ids_str = str(row['gold_support_chunk_ids'])
        if pd.isna(gold_chunk_ids_str) or not gold_chunk_ids_str:
            logger.warning(f"Skipping question {idx}: no gold chunk IDs")
            continue
        
        # Split by | to get individual chunk references
        gold_chunk_ids = set(gold_chunk_ids_str.split('|'))
        
        print(f"\n[{idx+1}/{len(qa_df)}] Question: {question[:60]}...")
        print(f"  Gold chunks: {len(gold_chunk_ids)}")
        
        try:
            # Embed and search
            embedding = model.embed_texts([question])[0]
            search_result = client.search(
                collection_name=settings.QDRANT_COLLECTION,
                query_vector=("default", embedding),
                limit=20,
                with_payload=True,
                score_threshold=0.0  # Get all results for fair comparison
            )
            
            if not search_result:
                print(f"  ✗ No results returned from Qdrant")
                continue
            
            # Before reranking - check if gold chunks in top-10
            before_ids = [r.payload.get('chunk_id', r.id) for r in search_result[:10]]
            before_hit = any(chunk_id in before_ids for chunk_id in gold_chunk_ids)
            before_recall = len([cid for cid in gold_chunk_ids if cid in before_ids]) / len(gold_chunk_ids)
            
            # Find rank of first gold chunk before reranking
            before_rank = None
            for i, r in enumerate(search_result):
                chunk_id = r.payload.get('chunk_id', r.id)
                if chunk_id in gold_chunk_ids:
                    before_rank = i + 1
                    break
            
            # After reranking
            rerank_inputs = [(question, r.payload['text']) for r in search_result]
            rerank_scores = reranker.predict(rerank_inputs)
            reranked = sorted(zip(search_result, rerank_scores), key=lambda x: x[1], reverse=True)
            
            after_ids = [r[0].payload.get('chunk_id', r[0].id) for r in reranked[:10]]
            after_hit = any(chunk_id in after_ids for chunk_id in gold_chunk_ids)
            after_recall = len([cid for cid in gold_chunk_ids if cid in after_ids]) / len(gold_chunk_ids)
            
            # Find rank of first gold chunk after reranking
            after_rank = None
            for i, (r, score) in enumerate(reranked):
                chunk_id = r.payload.get('chunk_id', r.id)
                if chunk_id in gold_chunk_ids:
                    after_rank = i + 1
                    break
            
            # Calculate MRR (Mean Reciprocal Rank)
            before_mrr = 1.0 / before_rank if before_rank else 0.0
            after_mrr = 1.0 / after_rank if after_rank else 0.0
            
            results.append({
                'question': question[:50],
                'gold_chunks': len(gold_chunk_ids),
                'before_hit': before_hit,
                'after_hit': after_hit,
                'before_recall': before_recall,
                'after_recall': after_recall,
                'before_rank': before_rank,
                'after_rank': after_rank,
                'before_mrr': before_mrr,
                'after_mrr': after_mrr,
                'improved': after_hit and not before_hit,
                'degraded': before_hit and not after_hit,
                'recall_improved': after_recall > before_recall,
                'rank_improved': (after_rank and before_rank and after_rank < before_rank)
            })
            
            # Print immediate feedback
            status = "✓" if after_hit else "✗"
            change = ""
            if after_hit and not before_hit:
                change = " (IMPROVED ⬆️)"
            elif before_hit and not after_hit:
                change = " (DEGRADED ⬇️)"
            elif after_rank and before_rank:
                if after_rank < before_rank:
                    change = f" (Rank: {before_rank}→{after_rank} ⬆️)"
                elif after_rank > before_rank:
                    change = f" (Rank: {before_rank}→{after_rank} ⬇️)"
            
            print(f"  {status} Recall: {before_recall:.2f}→{after_recall:.2f}{change}")
            
        except Exception as e:
            logger.exception(f"Error processing question {idx}: {e}")
            continue
    
    if not results:
        print("\n✗ No results to analyze")
        return
    
    df_results = pd.DataFrame(results)
    
    # Calculate aggregate metrics
    print("\n" + "=" * 60)
    print("RERANKING IMPACT ANALYSIS")
    print("=" * 60)
    print()
    
    print(f"Total questions evaluated: {len(df_results)}")
    print()
    
    print("RECALL METRICS:")
    print(f"  Before reranking: {df_results['before_recall'].mean()*100:.1f}%")
    print(f"  After reranking:  {df_results['after_recall'].mean()*100:.1f}%")
    recall_change = (df_results['after_recall'].mean() - df_results['before_recall'].mean()) * 100
    print(f"  Change: {recall_change:+.1f}%")
    print()
    
    print("HIT RATE (at least 1 gold chunk in top-10):")
    print(f"  Before reranking: {df_results['before_hit'].mean()*100:.1f}%")
    print(f"  After reranking:  {df_results['after_hit'].mean()*100:.1f}%")
    hit_change = (df_results['after_hit'].mean() - df_results['before_hit'].mean()) * 100
    print(f"  Change: {hit_change:+.1f}%")
    print()
    
    print("MEAN RECIPROCAL RANK (MRR):")
    print(f"  Before reranking: {df_results['before_mrr'].mean():.4f}")
    print(f"  After reranking:  {df_results['after_mrr'].mean():.4f}")
    mrr_change = df_results['after_mrr'].mean() - df_results['before_mrr'].mean()
    print(f"  Change: {mrr_change:+.4f}")
    print()
    
    print("QUERY-LEVEL CHANGES:")
    print(f"  Improved by reranking: {df_results['improved'].sum()} queries")
    print(f"  Degraded by reranking: {df_results['degraded'].sum()} queries")
    print(f"  Recall improved: {df_results['recall_improved'].sum()} queries")
    print(f"  Rank improved: {df_results['rank_improved'].sum()} queries")
    print()
    
    # Save detailed results
    output_path = os.path.join(settings.MEDIA_ROOT, 'reranking_impact.csv')
    df_results.to_csv(output_path, index=False)
    print(f"✓ Detailed results saved to: {output_path}")
    print()
    
    # Show top improvements and degradations
    print("=" * 60)
    print("TOP 5 IMPROVEMENTS")
    print("=" * 60)
    improvements = df_results[df_results['recall_improved']].sort_values('after_recall', ascending=False).head(5)
    if len(improvements) > 0:
        for idx, row in improvements.iterrows():
            print(f"\n  Question: {row['question']}")
            print(f"  Recall: {row['before_recall']:.2f} → {row['after_recall']:.2f}")
            if row['before_rank'] and row['after_rank']:
                print(f"  Rank: {row['before_rank']} → {row['after_rank']}")
    else:
        print("\n  No improvements found")
    
    print("\n" + "=" * 60)
    print("TOP 5 DEGRADATIONS")
    print("=" * 60)
    degradations = df_results[df_results['recall_improved'] == False].sort_values('after_recall').head(5)
    if len(degradations) > 0:
        for idx, row in degradations.iterrows():
            print(f"\n  Question: {row['question']}")
            print(f"  Recall: {row['before_recall']:.2f} → {row['after_recall']:.2f}")
            if row['before_rank'] and row['after_rank']:
                print(f"  Rank: {row['before_rank']} → {row['after_rank']}")
    else:
        print("\n  No degradations found")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    
    if recall_change > 2:
        print("\n✅ Reranking significantly improves recall")
        print("   → KEEP PhoRanker enabled in production")
    elif recall_change > 0:
        print("\n✓ Reranking provides marginal improvement")
        print("   → Consider keeping enabled if latency acceptable")
    else:
        print("\n⚠️  Reranking does not improve recall")
        print("   → Consider disabling to reduce latency")
        print("   → Or try different reranker model")
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    evaluate_reranking_impact()
