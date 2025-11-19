import os
import sys
import django

# Setup Django environment
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

from backend.utils.retrieval_metrics import RetrievalEvaluator
from backend.utils.generate_metrics import GenerationEvaluator
from backend.utils.mongo_repository import MongoRepository
from backend.utils.qdrant_client import QdrantClient
import pandas as pd
import logging

# New imports for Gemini rating
import google.generativeai as genai
import re
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

qa_gold_path = 'backend/media/qa_gold.csv'
retrieval_run_path = 'backend/media/retrieval_run.csv'
generation_run_path = 'backend/media/generation_run_aligned.csv'

# Initialize clients
mongo_repo = MongoRepository()
qdrant_wrapper = QdrantClient()
qdrant = qdrant_wrapper._client

def load_gold_standard_mapping():
    """Load gold standard and create mappings."""
    try:
        gold_df = pd.read_csv(qa_gold_path)
        
        # Create mappings
        question_to_id = dict(zip(gold_df['question_text'], gold_df['question_id']))
        doc_questions = {}
        question_to_doc = {}
        question_norm_map = {}
        for _, row in gold_df.iterrows():
            doc_id = row['document_id']
            qtext = row['question_text']
            qtext_norm = qtext.strip().lower()
            question_to_doc[qtext] = doc_id
            question_norm_map[qtext_norm] = qtext
            
            if doc_id not in doc_questions:
                doc_questions[doc_id] = []
            doc_questions[doc_id].append(qtext)
        
        logger.info(f"Loaded {len(question_to_id)} gold standard Q&A pairs")
        
        return gold_df, question_to_id, doc_questions, question_norm_map, question_to_doc
    except Exception as e:
        logger.error(f"Failed to load gold standard: {e}")
        return None, {}, {}, {}, {}

def get_chunk_metadata_map():
    """Build mapping from Qdrant point ID (UUID) to canonical chunk ID format."""
    try:
        chunk_map = {}
        collection_name = os.getenv('QDRANT_COLLECTION', 'test_collection')
        
        points, _ = qdrant.scroll(
            collection_name=collection_name,
            limit=10000,
            with_payload=True,
            with_vectors=False
        )
        
        for point in points:
            if not point.payload:
                continue

            point_uuid = str(point.id)
            chunk_id_canonical = point.payload.get('chunk_id')
            
            if chunk_id_canonical:
                chunk_map[point_uuid] = chunk_id_canonical
        
        logger.info(f"Built chunk metadata map with {len(chunk_map)} mappings")
        return chunk_map
    except Exception as e:
        logger.warning(f"Could not build chunk metadata map: {e}")
        return {}

def export_retrieval_logs_from_mongo(question_to_id, question_norm_map, question_to_doc, chunk_map):
    """Export ALL retrieval logs from MongoDB, matching against gold standard."""
    try:
        retrieval_logs = list(mongo_repo.db['retrieval_log'].find({}))
        
        if not retrieval_logs:
            logger.warning("No retrieval logs found in MongoDB")
            return None, 0
        
        rows = []
        mapped_count = 0
        skipped_count = 0
        conversion_stats = {"converted": 0, "not_converted": 0}
        
        for log in retrieval_logs:
            doc_id = log.get("document_id", "")
            question_text = log.get("question", "").strip()
            
            # Skip template variables and empty questions
            if not question_text or "{{" in question_text:
                skipped_count += 1
                continue
            
            # Match only if the question exists in the gold standard (normalized)
            q_norm = question_text.lower()
            if q_norm not in question_norm_map:
                skipped_count += 1
                continue
            
            gold_qtext = question_norm_map[q_norm]
            question_id = question_to_id[gold_qtext]
            mapped_count += 1
            
            retrieved_chunk_ids = log.get("retrieved_chunk_ids", [])
            converted_ids = []
            for chunk_uuid in retrieved_chunk_ids:
                chunk_uuid_str = str(chunk_uuid)
                if chunk_uuid_str in chunk_map:
                    converted_ids.append(chunk_map[chunk_uuid_str])
                    conversion_stats["converted"] += 1
                else:
                    converted_ids.append(chunk_uuid_str)
                    conversion_stats["not_converted"] += 1
            
            retrieved_scores = log.get("retrieved_scores", [])
            
            rows.append({
                "question_id": question_id,
                "document_id": doc_id,
                "question_text": gold_qtext,
                "retrieved_chunk_ids": "|".join(converted_ids),
                "retrieved_scores": "|".join([str(s) for s in retrieved_scores]),
                "top_k": log.get("top_k", len(retrieved_chunk_ids))
            })
        
        logger.info(f"Exported {mapped_count} retrieval logs from database")
        logger.info(f"  Skipped: {skipped_count} (template/unmapped questions)")
        logger.info(f"  Chunk conversions - Successful: {conversion_stats['converted']}, Fallback: {conversion_stats['not_converted']}")
        
        if mapped_count == 0:
            logger.warning("No matching retrieval logs found")
            return None, 0
        
        df = pd.DataFrame(rows)
        df.to_csv(retrieval_run_path, index=False)
        logger.info(f"Saved to {retrieval_run_path}")
        
        return retrieval_run_path, mapped_count
    except Exception as e:
        logger.error(f"Failed to export retrieval logs: {e}")
        import traceback
        traceback.print_exc()
        return None, 0

def export_generation_logs_from_mongo(question_to_id, gold_df, question_norm_map):
    """Export ALL generation logs from MongoDB, matching against gold standard."""
    try:
        generation_logs = list(mongo_repo.db['generation_log'].find({}))
        
        if not generation_logs:
            logger.warning("No generation logs found in MongoDB")
            return None, 0
        
        gold_answers = {}
        for _, row in gold_df.iterrows():
            gold_answers[row['question_text']] = row['gold_answer']
        
        rows = []
        mapped_count = 0
        skipped_count = 0
        answer_stats = {"has_content": 0, "not_found": 0, "empty": 0}
        
        for log in generation_logs:
            question_text = log.get("question_text", "").strip()
            
            if not question_text or "{{" in question_text:
                skipped_count += 1
                continue
            
            # Match only if the question exists in the gold standard (normalized)
            q_norm = question_text.lower()
            if q_norm not in question_norm_map:
                skipped_count += 1
                continue
            
            gold_qtext = question_norm_map[q_norm]
            question_id = question_to_id[gold_qtext]
            mapped_count += 1
            
            answer_text = log.get("final_answer_text", "").strip()
            
            if not answer_text:
                answer_stats["empty"] += 1
            elif "không tìm thấy" in answer_text.lower() or "không có" in answer_text.lower():
                answer_stats["not_found"] += 1
            else:
                answer_stats["has_content"] += 1
            
            rows.append({
                "question_id": question_id,
                "question_text": gold_qtext,
                "generated_answer": answer_text,
                "citations": ""
            })
        
        logger.info(f"Exported {mapped_count} generation logs from database")
        logger.info(f"  Skipped: {skipped_count} (template/unmapped questions)")
        logger.info(f"  Answer quality - Content: {answer_stats['has_content']}, Not found: {answer_stats['not_found']}, Empty: {answer_stats['empty']}")
        
        if mapped_count == 0:
            logger.warning("No matching generation logs found")
            return None, 0
        
        df = pd.DataFrame(rows)
        df.to_csv(generation_run_path, index=False)
        logger.info(f"Saved to {generation_run_path}")
        
        return generation_run_path, mapped_count
    except Exception as e:
        logger.error(f"Failed to export generation logs: {e}")
        import traceback
        traceback.print_exc()
        return None, 0

def rate_generation_with_gemini(gen_csv_path, gold_df, model_name=None, api_key=None, sleep_between_calls=0.15):
    """Rate generated answers with Gemini model (1-5 scale)."""
    try:
        if not api_key:
            logger.warning("GEMINI_API_KEY not set - skipping Gemini rating")
            return None, {}
        
        genai.configure(api_key=api_key)
        
        if not model_name:
            model_name = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
        
        model = genai.GenerativeModel(model_name)
        
        df = pd.read_csv(gen_csv_path)
        
        gold_map = {}
        if 'question_text' in gold_df.columns and 'gold_answer' in gold_df.columns:
            gold_map = dict(zip(gold_df['question_text'], gold_df['gold_answer']))
        
        ratings = []
        raw_outputs = []
        
        for idx, row in df.iterrows():
            qtext = str(row.get('question_text', '') or '').strip()
            gen_ans = str(row.get('generated_answer', '') or '').strip()
            gold_ans = str(gold_map.get(qtext, '') or '').strip()
            
            if not gen_ans:
                ratings.append(None)
                raw_outputs.append("EMPTY_GENERATED_ANSWER")
                logger.debug(f"  [{idx}] Skipped: empty generated answer")
                continue
            
            if not gold_ans:
                ratings.append(None)
                raw_outputs.append("GOLD_ANSWER_NOT_FOUND")
                logger.debug(f"  [{idx}] Skipped: no matching gold answer for '{qtext[:50]}...'")
                continue
            
            prompt = (
                "You are an objective evaluator. Compare the generated answer to the reference (gold) answer "
                "and assign a single integer rating from 1 to 5 where:\n"
                "5 = Generated answer fully matches the reference in correctness, completeness and relevance\n"
                "4 = Generated answer is mostly correct with minor omissions\n"
                "3 = Generated answer has moderate issues or partial correctness\n"
                "2 = Generated answer is mostly incorrect or incomplete\n"
                "1 = Generated answer is incorrect or irrelevant\n\n"
                f"Reference (gold): \"{gold_ans}\"\n\n"
                f"Generated answer: \"{gen_ans}\"\n\n"
                "Respond with ONLY a single digit (1-5) and nothing else."
            )
            
            try:
                response = model.generate_content(prompt)
                text = response.text.strip() if response.text else ""
                
                raw_outputs.append(text)
                
                m = re.search(r'[1-5]', text)
                if m:
                    rating = int(m.group(0))
                    ratings.append(rating)
                    logger.debug(f"  [{idx}] Rating: {rating}")
                else:
                    ratings.append(None)
                    logger.warning(f"  [{idx}] No valid rating extracted from: {text}")
                    
            except Exception as e:
                logger.warning(f"  [{idx}] Gemini call failed: {e}")
                ratings.append(None)
                raw_outputs.append(f"ERROR: {str(e)}")
            
            time.sleep(sleep_between_calls)
        
        df['gemini_rating'] = ratings
        df['gemini_raw'] = raw_outputs
        
        out_path = gen_csv_path.replace('.csv', '_gemini_rated.csv')
        df.to_csv(out_path, index=False)
        
        valid_ratings = [r for r in ratings if isinstance(r, int)]
        rating_dist = {}
        for i in range(1, 6):
            rating_dist[str(i)] = valid_ratings.count(i)
        
        summary = {
            "count": len(ratings),
            "rated": len(valid_ratings),
            "mean_rating": float(sum(valid_ratings) / len(valid_ratings)) if valid_ratings else None,
            "rating_distribution": rating_dist
        }
        
        logger.info(f"Gemini rating saved to {out_path}")
        logger.info(f"  Rated: {summary['rated']}/{summary['count']}")
        if summary['mean_rating']:
            logger.info(f"  Mean rating: {summary['mean_rating']:.2f}")
        logger.info(f"  Distribution: {rating_dist}")
        
        return out_path, summary
        
    except Exception as e:
        logger.error(f"Failed to rate with Gemini: {e}")
        import traceback
        traceback.print_exc()
        return None, {}

# Main execution - evaluate ALL logs in database
print("=" * 80)
print("RAG CHATBOT EVALUATION - ALL DATABASE LOGS")
print("=" * 80)
print()

# Load gold standard
print("=" * 80)
print("Loading gold standard Q&A pairs...")
print("=" * 80)
gold_df, question_to_id, doc_questions, question_norm_map, question_to_doc = load_gold_standard_mapping()

if gold_df is None:
    logger.error("Cannot proceed without gold standard data")
    sys.exit(1)

print(f"Gold standard contains {len(gold_df)} questions")
print()

# Build chunk map for all documents
print("Building chunk metadata map from Qdrant...")
chunk_map = get_chunk_metadata_map()
print()

# Export ALL logs
print("=" * 80)
print("Exporting ALL logs from MongoDB...")
print("=" * 80)
retrieval_path, retrieval_count = export_retrieval_logs_from_mongo(
    question_to_id, question_norm_map, question_to_doc, chunk_map
)
gen_path, gen_count = export_generation_logs_from_mongo(
    question_to_id, gold_df, question_norm_map
)
print()

results = {
    "total_retrieval_logs": retrieval_count,
    "total_generation_logs": gen_count,
    "retrieval_evaluation": None,
    "generation_evaluation": None,
    "gemini_rating": None
}

# Run retrieval evaluation
if retrieval_path and os.path.exists(retrieval_path):
    print("=" * 80)
    print("Running Retrieval Evaluation...")
    print("=" * 80)
    try:
        retrieval = RetrievalEvaluator.evaluate(qa_gold_path, retrieval_path)
        results['retrieval_evaluation'] = retrieval
        if retrieval and retrieval.get('summary'):
            print(f"Recall@10: {retrieval['summary']['recall@10']['mean']:.3f}")
            print(f"MRR@10: {retrieval['summary']['mrr@10']['mean']:.3f}")
            print(f"NDCG@10: {retrieval['summary']['ndcg@10']['mean']:.3f}")
        print("Retrieval evaluation complete")
    except Exception as e:
        logger.error(f"Retrieval evaluation failed: {e}")
else:
    print("No retrieval logs to evaluate")

print()

# Run generation evaluation
if gen_path and os.path.exists(gen_path):
    print("=" * 80)
    print("Running Gemini rating for generated answers...")
    print("=" * 80)
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    gemini_model = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
    try:
        gemini_out_path, gemini_summary = rate_generation_with_gemini(
            gen_path, gold_df, model_name=gemini_model, api_key=gemini_api_key
        )
        results['gemini_rating'] = gemini_summary
        if gemini_summary and gemini_summary.get('mean_rating') is not None:
            print(f"Gemini mean rating: {gemini_summary['mean_rating']:.3f} (rated {gemini_summary['rated']}/{gemini_summary['count']})")
    except Exception as e:
        logger.error(f"Gemini rating failed: {e}")
    
    print()
    print("=" * 80)
    print("Running Generation Evaluation...")
    print("=" * 80)
    try:
        generation = GenerationEvaluator.evaluate(qa_gold_path, gen_path)
        results['generation_evaluation'] = generation
        if generation and generation.get('summary'):
            print(f"ROUGE-1: {generation['summary']['rouge1']['mean']:.3f}")
            print(f"Token F1: {generation['summary']['token_f1']['mean']:.3f}")
            print(f"EM: {generation['summary']['em']['mean']:.3f}")
        print("Generation evaluation complete")
    except Exception as e:
        logger.error(f"Generation evaluation failed: {e}")
else:
    print("No generation logs to evaluate")

print()
print("=" * 80)
print("EVALUATION SUMMARY")
print("=" * 80)
print(f"Total Retrieval Logs Evaluated: {retrieval_count}")
print(f"Total Generation Logs Evaluated: {gen_count}")
print("=" * 80)