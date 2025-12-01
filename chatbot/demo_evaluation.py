#!/usr/bin/env python3
"""
Demonstration script for comprehensive retrieval evaluation system.
Shows how to use all the new retrieval metrics and analysis tools.
"""

import sys
from pathlib import Path
import logging

# Add backend to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backend.utils.retrieval_metrics import RetrievalMetrics, RetrievalEvaluator
from backend.utils.evaluation_runner import ComprehensiveRetrievalEvaluator
from backend.utils.retrieval_analyzer import RetrievalAnalyzer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_individual_metrics():
    """Demonstrate individual retrieval metrics calculation."""
    print("\n" + "=" * 60)
    print("DEMO: Individual Retrieval Metrics")
    print("=" * 60)
    
    # Example data
    relevant_items = {"chunk_1", "chunk_3", "chunk_7", "chunk_12"}
    retrieved_items = ["chunk_1", "chunk_2", "chunk_3", "chunk_4", "chunk_5", 
                      "chunk_6", "chunk_7", "chunk_8", "chunk_9", "chunk_10"]
    
    print(f"Relevant items: {relevant_items}")
    print(f"Retrieved items (top 10): {retrieved_items}")
    print()
    
    # Calculate metrics for different K values
    k_values = [1, 3, 5, 10]
    
    for k in k_values:
        print(f"Metrics @ K={k}:")
        print("-" * 15)
        
        recall = RetrievalMetrics.recall_at_k(relevant_items, retrieved_items, k)
        precision = RetrievalMetrics.precision_at_k(relevant_items, retrieved_items, k)
        mrr = RetrievalMetrics.mrr_at_k(relevant_items, retrieved_items, k)
        ndcg = RetrievalMetrics.ndcg_at_k(relevant_items, retrieved_items, k)
        hit_rate = RetrievalMetrics.hit_rate_at_k(relevant_items, retrieved_items, k)
        ap = RetrievalMetrics.average_precision_at_k(relevant_items, retrieved_items, k)
        
        print(f"  Recall@{k}:           {recall:.4f}")
        print(f"  Precision@{k}:        {precision:.4f}")
        print(f"  MRR@{k}:              {mrr:.4f}")
        print(f"  nDCG@{k}:             {ndcg:.4f}")
        print(f"  Hit Rate@{k}:         {hit_rate:.4f}")
        print(f"  Average Precision@{k}: {ap:.4f}")
        print()


def demo_metric_explanations():
    """Explain what each metric means and when to use it."""
    print("\n" + "=" * 60)
    print("RETRIEVAL METRICS EXPLAINED")
    print("=" * 60)
    
    explanations = {
        "Recall@K": {
            "formula": "% of ground-truth chunks retrieved in top K",
            "use_when": "Core metric for coverage - how much relevant content did we find?",
            "good_value": "> 0.7 for K=5",
            "interpretation": "High recall = not missing important information"
        },
        "Precision@K": {
            "formula": "% of top-K retrieved chunks that are actually relevant",
            "use_when": "To check noise - are we showing irrelevant content?",
            "good_value": "> 0.6 for K=5",
            "interpretation": "High precision = less noise, more focused results"
        },
        "MRR@K": {
            "formula": "Mean Reciprocal Rank (1/position of first relevant chunk)",
            "use_when": "Ranking quality - how quickly do we find relevant content?",
            "good_value": "> 0.5 for K=5",
            "interpretation": "High MRR = relevant content appears early in results"
        },
        "nDCG@K": {
            "formula": "Normalized Discounted Cumulative Gain (weighted ranking)",
            "use_when": "Weighted ranking usefulness with graded relevance",
            "good_value": "> 0.6 for K=5",
            "interpretation": "High nDCG = good ranking with early relevant items valued more"
        },
        "Hit Rate@K": {
            "formula": "Whether any relevant chunk appears in top-K (binary)",
            "use_when": "Simpler than recall - did we find anything useful?",
            "good_value": "> 0.8 for K=5",
            "interpretation": "High hit rate = rarely completely missing relevant content"
        },
        "Average Precision@K": {
            "formula": "Average precision at each relevant item position",
            "use_when": "Balances precision and recall with ranking consideration",
            "good_value": "> 0.5 for K=5",
            "interpretation": "High AP = consistently good precision throughout ranking"
        }
    }
    
    for metric, details in explanations.items():
        print(f"{metric}:")
        print(f"  Formula: {details['formula']}")
        print(f"  Use when: {details['use_when']}")
        print(f"  Good value: {details['good_value']}")
        print(f"  Interpretation: {details['interpretation']}")
        print()


def demo_performance_scenarios():
    """Show different performance scenarios and their metric patterns."""
    print("\n" + "=" * 60)
    print("PERFORMANCE SCENARIOS")
    print("=" * 60)
    
    scenarios = {
        "Excellent System": {
            "relevant": {"chunk_1", "chunk_2", "chunk_3"},
            "retrieved": ["chunk_1", "chunk_2", "chunk_3", "chunk_4", "chunk_5"],
            "description": "Finds all relevant items early with minimal noise"
        },
        "High Precision, Low Recall": {
            "relevant": {"chunk_1", "chunk_2", "chunk_3", "chunk_4", "chunk_5"},
            "retrieved": ["chunk_1", "chunk_2", "irrelevant_1", "irrelevant_2", "irrelevant_3"],
            "description": "Conservative retrieval - finds some relevant items but misses many"
        },
        "High Recall, Low Precision": {
            "relevant": {"chunk_1", "chunk_2", "chunk_3"},
            "retrieved": ["chunk_1", "chunk_2", "chunk_3", "irrelevant_1", "irrelevant_2"],
            "description": "Aggressive retrieval - finds relevant items but includes noise"
        },
        "Poor Ranking": {
            "relevant": {"chunk_1", "chunk_2", "chunk_3"},
            "retrieved": ["irrelevant_1", "irrelevant_2", "chunk_1", "chunk_2", "chunk_3"],
            "description": "Finds relevant items but ranks them poorly"
        },
        "Failed Retrieval": {
            "relevant": {"chunk_1", "chunk_2", "chunk_3"},
            "retrieved": ["irrelevant_1", "irrelevant_2", "irrelevant_3", "irrelevant_4", "irrelevant_5"],
            "description": "Complete failure - no relevant items found"
        }
    }
    
    for scenario_name, scenario in scenarios.items():
        print(f"{scenario_name}:")
        print(f"  Description: {scenario['description']}")
        
        # Calculate metrics
        metrics = RetrievalMetrics.compute_all_metrics(
            scenario['relevant'], 
            scenario['retrieved'], 
            k=5
        )
        
        print("  Metrics:")
        for metric, score in metrics.items():
            print(f"    {metric:15s}: {score:.3f}")
        print()


def demo_evaluation_workflow():
    """Demonstrate the complete evaluation workflow."""
    print("\n" + "=" * 60)
    print("COMPLETE EVALUATION WORKFLOW")
    print("=" * 60)
    
    print("Step 1: Check if evaluation files exist")
    
    # Check for required files
    qa_gold_path = ROOT / "qa_gold.csv"
    retrieval_run_path = ROOT / "backend" / "media" / "retrieval_run.csv"
    
    print(f"  QA Gold file: {qa_gold_path} - {'EXISTS' if qa_gold_path.exists() else 'MISSING'}")
    print(f"  Retrieval run: {retrieval_run_path} - {'EXISTS' if retrieval_run_path.exists() else 'MISSING'}")
    
    if not qa_gold_path.exists() or not retrieval_run_path.exists():
        print("\n⚠️  Required files missing. To run full evaluation:")
        print("  1. Create qa_gold.csv with columns: question_id, question_text, gold_support_chunk_ids, gold_answer, document_id")
        print("  2. Run your RAG system to generate retrieval_run.csv")
        print("  3. Then run: python run_eval.py --comprehensive --analyze")
        return
    
    print("\nStep 2: Run comprehensive evaluation")
    print("Command: python run_eval.py --comprehensive --analyze")
    print("\nThis will generate:")
    print("  - retrieval_evaluation_detailed.json (full results)")
    print("  - retrieval_evaluation_detailed.txt (human-readable summary)")
    print("  - retrieval_evaluation_detailed.csv (per-question results)")
    print("  - retrieval_analysis_*.json (detailed analysis)")
    print("  - retrieval_analysis_*_summary.txt (insights and recommendations)")
    
    print("\nStep 3: Interpret results")
    print("Key things to look for:")
    print("  - Overall system health (excellent/good/fair/poor)")
    print("  - Performance distribution (% with zero recall, perfect recall)")
    print("  - Document-specific performance patterns")
    print("  - Failure modes and recommendations")
    
    print("\nStep 4: Optimize based on insights")
    print("Common optimizations:")
    print("  - Adjust chunk size/overlap if poor recall")
    print("  - Add reranking if poor MRR/nDCG")
    print("  - Try different embeddings if overall performance is poor")
    print("  - Increase top_k if high precision but low recall")


def main():
    """Run all demonstrations."""
    print("RAG RETRIEVAL EVALUATION SYSTEM DEMO")
    print("=" * 60)
    print("This demo shows how to use the comprehensive retrieval evaluation system.")
    print("It covers individual metrics, explanations, scenarios, and complete workflow.")
    
    demo_individual_metrics()
    demo_metric_explanations()
    demo_performance_scenarios()
    demo_evaluation_workflow()
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Set up your evaluation data (qa_gold.csv)")
    print("2. Run your RAG system to generate retrieval logs")
    print("3. Run comprehensive evaluation: python run_eval.py --comprehensive --analyze")
    print("4. Review the generated reports and implement recommended optimizations")
    print("5. Repeat the evaluation cycle to measure improvements")
    print("\nFor more details, see the generated documentation and analysis reports.")


if __name__ == "__main__":
    main()