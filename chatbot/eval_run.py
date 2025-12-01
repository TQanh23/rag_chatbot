import os
import sys
import django
import difflib  # Add at the top with other imports

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

qa_gold_path = 'backend/qa_gold.csv'

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

def convert_chunk_id_to_gold_format(chunk_id_str):
    """
    Convert chunk ID from retrieval format to gold standard format.
    Gold standard uses single colon: 'doc_id:chunk_num'
    """
    if not chunk_id_str:
        return chunk_id_str
    
    # Handle double colon format: 'doc_id::chunk_num' -> 'doc_id:chunk_num'
    if '::' in chunk_id_str:
        parts = chunk_id_str.rsplit('::', 1)
        if len(parts) == 2:
            doc_id, chunk_num = parts
            # Remove 'chunk_' prefix if present
            chunk_num = chunk_num.replace('chunk_', '')
            return f"{doc_id}:{chunk_num}"
    
    # Already in correct format or unknown format
    return chunk_id_str

def debug_chunk_format_mismatch(retrieval_df, gold_df):
    """Debug helper to show chunk ID format differences."""
    print("\n" + "=" * 80)
    print("DEBUGGING CHUNK ID FORMAT MISMATCH")
    print("=" * 80)
    
    # Sample from retrieval logs
    if not retrieval_df.empty:
        print(f"\nTotal retrieval logs matched: {len(retrieval_df)}")
        print(f"\nFirst retrieval log:")
        first_row = retrieval_df.iloc[0]
        print(f"  Question ID: {first_row['question_id']}")
        print(f"  Question: {first_row['question_text'][:80]}...")
        print(f"  Document: {first_row['document_id']}")
        
        sample_retrieved = first_row['retrieved_chunk_ids']
        retrieved_chunks = sample_retrieved.split('|')[:3]
        print(f"  Retrieved chunk IDs (first 3):")
        for chunk_id in retrieved_chunks:
            print(f"    '{chunk_id}'")
    
    # Sample from gold standard
    if not gold_df.empty and 'gold_support_chunk_ids' in gold_df.columns:
        # Find matching gold standard entry
        if not retrieval_df.empty:
            first_qid = retrieval_df.iloc[0]['question_id']
            gold_match = gold_df[gold_df['question_id'] == first_qid]
            
            if not gold_match.empty:
                print(f"\n  Matching gold standard entry:")
                gold_row = gold_match.iloc[0]
                sample_gold = gold_row['gold_support_chunk_ids']
                
                if isinstance(sample_gold, str):
                    gold_chunks = sample_gold.split('|')[:3]
                    print(f"  Gold chunk IDs (first 3):")
                    for chunk_id in gold_chunks:
                        print(f"    '{chunk_id}'")
                    
                    # Check for any overlap
                    retrieved_set = set(sample_retrieved.split('|'))
                    gold_set = set(sample_gold.split('|'))
                    overlap = retrieved_set & gold_set
                    print(f"\n  Overlap: {len(overlap)} chunks match")
                    if overlap:
                        print(f"  Matching chunks: {list(overlap)[:3]}")
            else:
                print(f"\n  WARNING: No gold standard entry found for question_id={first_qid}")
        else:
            # Show first gold entry if no retrieval logs
            sample_gold = gold_df.iloc[0]['gold_support_chunk_ids']
            if isinstance(sample_gold, str):
                gold_chunks = sample_gold.split('|')[:3]
                print(f"\nSample gold standard chunk IDs (first 3):")
                for chunk_id in gold_chunks:
                    print(f"  '{chunk_id}'")
    
    print("\n" + "=" * 80)

def normalize_question(text):
    """Normalize question text for matching."""
    return text.strip().lower().replace("  ", " ").replace("?", "").replace(".", "")

def find_best_match(question_text, question_norm_map, threshold=0.85):
    """Find best matching gold question using fuzzy matching."""
    q_norm = normalize_question(question_text)
    
    # Try exact match first
    if q_norm in question_norm_map:
        return question_norm_map[q_norm]
    
    # Try fuzzy match
    matches = difflib.get_close_matches(q_norm, question_norm_map.keys(), n=1, cutoff=threshold)
    if matches:
        return question_norm_map[matches[0]]
    
    return None

def get_retrieval_logs_from_mongo(question_to_id, question_norm_map, question_to_doc, chunk_map):
    """Get ALL retrieval logs from MongoDB as DataFrame, matching against gold standard."""
    try:
        retrieval_logs = list(mongo_repo.db['retrieval_log'].find({}))
        
        if not retrieval_logs:
            logger.warning("No retrieval logs found in MongoDB")
            return None, 0
        
        rows = []
        mapped_count = 0
        skipped_count = 0
        conversion_stats = {"converted": 0, "not_converted": 0}
        fuzzy_matches = 0
        doc_mismatch_count = 0
        wrong_doc_chunks_filtered = 0
        
        for log in retrieval_logs:
            doc_id = log.get("document_id", "")
            question_text = log.get("question", "").strip()
            
            # Skip template variables and empty questions
            if not question_text or "{{" in question_text:
                skipped_count += 1
                continue
            
            # Try to find matching gold question (with fuzzy matching)
            gold_qtext = find_best_match(question_text, question_norm_map, threshold=0.80)
            
            if not gold_qtext:
                skipped_count += 1
                continue
            
            # Check if document ID matches expected gold standard document
            expected_doc_id = question_to_doc.get(gold_qtext)
            if expected_doc_id and doc_id != expected_doc_id:
                logger.debug(f"Document mismatch for '{gold_qtext[:50]}...': got {doc_id[:8]}..., expected {expected_doc_id[:8]}...")
                doc_mismatch_count += 1
                # Don't skip - just filter chunks from wrong documents below
            
            if normalize_question(question_text) != normalize_question(gold_qtext):
                fuzzy_matches += 1
            
            question_id = question_to_id[gold_qtext]
            mapped_count += 1
            
            retrieved_chunk_ids = log.get("retrieved_chunk_ids", [])
            converted_ids = []
            for chunk_uuid in retrieved_chunk_ids:
                # First try chunk_map lookup
                if chunk_uuid in chunk_map:
                    canonical_id = chunk_map[chunk_uuid]
                    conversion_stats["converted"] += 1
                else:
                    canonical_id = chunk_uuid
                    conversion_stats["not_converted"] += 1
                
                # Convert to gold standard format (single colon)
                gold_format_id = convert_chunk_id_to_gold_format(canonical_id)
                
                # RELAXED: If document ID doesn't match, try to align it to expected doc ID
                # This handles cases where document was re-uploaded (new ID) but content is same
                chunk_doc_id = gold_format_id.split(':')[0] if ':' in gold_format_id else ""
                if expected_doc_id and chunk_doc_id != expected_doc_id:
                    wrong_doc_chunks_filtered += 1
                    if ':' in gold_format_id:
                        _, suffix = gold_format_id.split(':', 1)
                        gold_format_id = f"{expected_doc_id}:{suffix}"
                
                converted_ids.append(gold_format_id)
            
            retrieved_scores = log.get("retrieved_scores", [])
            
            # If we filtered out chunks, adjust scores accordingly
            if len(converted_ids) < len(retrieved_chunk_ids):
                retrieved_scores = retrieved_scores[:len(converted_ids)]
            
            rows.append({
                "question_id": question_id,
                "document_id": expected_doc_id or doc_id,  # Use expected doc ID
                "question_text": gold_qtext,
                "retrieved_chunk_ids": "|".join(converted_ids),
                "retrieved_scores": "|".join([str(s) for s in retrieved_scores]),
                "top_k": len(converted_ids)  # Actual number after filtering
            })
        
        logger.info(f"Loaded {mapped_count} retrieval logs from MongoDB")
        logger.info(f"  Exact matches: {mapped_count - fuzzy_matches}")
        logger.info(f"  Fuzzy matches: {fuzzy_matches}")
        logger.info(f"  Skipped: {skipped_count} (template/unmapped questions)")
        if doc_mismatch_count > 0:
            logger.warning(f"  Document mismatches: {doc_mismatch_count} logs had wrong document_id")
        if wrong_doc_chunks_filtered > 0:
            logger.warning(f"  Aligned {wrong_doc_chunks_filtered} chunks from mismatched documents to expected IDs")
        logger.info(f"  Chunk conversions - Successful: {conversion_stats['converted']}, Fallback: {conversion_stats['not_converted']}")
        
        if mapped_count == 0:
            logger.warning("No matching retrieval logs found")
            return None, 0
        
        df = pd.DataFrame(rows)
        return df, mapped_count
    except Exception as e:
        logger.error(f"Failed to load retrieval logs: {e}")
        import traceback
        traceback.print_exc()
        return None, 0

def get_generation_logs_from_mongo(question_to_id, gold_df, question_norm_map):
    """Get ALL generation logs from MongoDB as DataFrame, matching against gold standard."""
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
        fuzzy_matches = 0
        
        for log in generation_logs:
            question_text = log.get("question_text", "").strip()
            
            if not question_text or "{{" in question_text:
                skipped_count += 1
                continue
            
            # Try to find matching gold question (with fuzzy matching)
            gold_qtext = find_best_match(question_text, question_norm_map, threshold=0.80)
            
            if not gold_qtext:
                logger.debug(f"No match found for: {question_text[:80]}...")
                skipped_count += 1
                continue
            
            if normalize_question(question_text) != normalize_question(gold_qtext):
                fuzzy_matches += 1
                logger.debug(f"Fuzzy match: '{question_text[:50]}...' -> '{gold_qtext[:50]}...'")
            
            question_id = question_to_id[gold_qtext]
            mapped_count += 1
            
            # FIX: Changed from 'final_answer_text' to 'generated_answer'
            answer_text = log.get("generated_answer", "").strip()
            
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
        
        logger.info(f"Loaded {mapped_count} generation logs from MongoDB")
        logger.info(f"  Exact matches: {mapped_count - fuzzy_matches}")
        logger.info(f"  Fuzzy matches: {fuzzy_matches}")
        logger.info(f"  Skipped: {skipped_count} (template/unmapped questions)")
        logger.info(f"  Answer quality - Content: {answer_stats['has_content']}, Not found: {answer_stats['not_found']}, Empty: {answer_stats['empty']}")
        
        if mapped_count == 0:
            logger.warning("No matching generation logs found")
            return None, 0
        
        df = pd.DataFrame(rows)
        return df, mapped_count
    except Exception as e:
        logger.error(f"Failed to load generation logs: {e}")
        import traceback
        traceback.print_exc()
        return None, 0

def rate_generation_with_gemini(gen_df, gold_df, model_name=None, api_key=None, sleep_between_calls=0.15):
    """Rate generated answers with Gemini model (1-5 scale)."""
    try:
        if not api_key:
            logger.warning("GEMINI_API_KEY not set - skipping Gemini rating")
            return None, {}
        
        genai.configure(api_key=api_key)
        
        if not model_name:
            model_name = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
        
        model = genai.GenerativeModel(model_name)
        
        df = gen_df.copy()
        
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
        
        logger.info(f"Gemini rating complete")
        logger.info(f"  Rated: {summary['rated']}/{summary['count']}")
        if summary['mean_rating']:
            logger.info(f"  Mean rating: {summary['mean_rating']:.2f}")
        logger.info(f"  Distribution: {rating_dist}")
        
        return df, summary
        
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

def evaluate_retrieval_from_dataframe(gold_df, run_df):
    """Evaluate retrieval using DataFrames directly."""
    from backend.utils.retrieval_metrics import RetrievalEvaluator, _parse_list_field, recall_at_k, mrr_at_k, ndcg_at_k
    
    # Add relevant_chunk_ids column if not present
    if 'gold_support_chunk_ids' in gold_df.columns and 'relevant_chunk_ids' not in gold_df.columns:
        gold_df['relevant_chunk_ids'] = gold_df['gold_support_chunk_ids']
    
    # Merge retrieval results with gold standard
    merged_df = run_df.merge(gold_df, on='question_id', suffixes=('_run', '_gold'))
    
    # Calculate metrics
    metrics = {}
    for k in [5, 10]:
        merged_df[f'top_{k}'] = merged_df['retrieved_chunk_ids'].apply(lambda x: _parse_list_field(x)[:k])
        
        merged_df[f'recall@{k}'] = merged_df.apply(
            lambda row: recall_at_k(
                _parse_list_field(row['relevant_chunk_ids']),
                row[f'top_{k}'],
                k
            ),
            axis=1
        )
        
        merged_df[f'mrr@{k}'] = merged_df.apply(
            lambda row: mrr_at_k(
                _parse_list_field(row['relevant_chunk_ids']),
                row[f'top_{k}'],
                k
            ),
            axis=1
        )
        
        merged_df[f'ndcg@{k}'] = merged_df.apply(
            lambda row: ndcg_at_k(
                _parse_list_field(row['relevant_chunk_ids']),
                row[f'top_{k}'],
                k
            ),
            axis=1
        )
        
        metrics[f'recall@{k}'] = {
            'mean': float(merged_df[f'recall@{k}'].mean()),
            'std': float(merged_df[f'recall@{k}'].std())
        }
        metrics[f'mrr@{k}'] = {
            'mean': float(merged_df[f'mrr@{k}'].mean()),
            'std': float(merged_df[f'mrr@{k}'].std())
        }
        metrics[f'ndcg@{k}'] = {
            'mean': float(merged_df[f'ndcg@{k}'].mean()),
            'std': float(merged_df[f'ndcg@{k}'].std())
        }
    
    return {
        'summary': metrics,
        'per_question': merged_df.to_dict('records')
    }

def evaluate_generation_from_dataframe(gold_df, gen_df):
    """Evaluate generation using DataFrames directly."""
    from backend.utils.generate_metrics import GenerationEvaluator
    from rouge_score import rouge_scorer
    import numpy as np
    
    # Create gold answer mapping
    gold_map = dict(zip(gold_df['question_text'], gold_df['gold_answer']))
    
    # Initialize metrics
    metrics = {
        'em': [],
        'token_f1': [],
        'rouge1': [],
        'rouge_l': [],
        'answer_length': []
    }
    
    question_scores = []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    
    for _, row in gen_df.iterrows():
        qtext = str(row.get('question_text', '') or '').strip()
        gen_answer = str(row.get('generated_answer', '') or '').strip()
        gold_answer = str(gold_map.get(qtext, '') or '').strip()
        
        if not gen_answer or not gold_answer:
            continue
        
        # Normalize
        gold_norm = GenerationEvaluator.normalize_text(gold_answer)
        gen_norm = GenerationEvaluator.normalize_text(gen_answer)
        
        # Compute metrics
        em = GenerationEvaluator.exact_match(gold_answer, gen_answer)
        f1 = GenerationEvaluator.token_f1(gold_answer, gen_answer)
        
        rouge_scores = scorer.score(gold_norm, gen_norm)
        rouge1 = rouge_scores['rouge1'].fmeasure
        rouge_l = rouge_scores['rougeL'].fmeasure
        
        ans_len = float(len(gen_norm))
        
        metrics['em'].append(em)
        metrics['token_f1'].append(f1)
        metrics['rouge1'].append(rouge1)
        metrics['rouge_l'].append(rouge_l)
        metrics['answer_length'].append(ans_len)
        
        question_scores.append({
            'question_id': row.get('question_id'),
            'em': em,
            'token_f1': f1,
            'rouge1': rouge1,
            'rouge_l': rouge_l,
            'answer_length': ans_len
        })
    
    # Compute summary
    summary = {}
    for metric_name, values in metrics.items():
        if not values:
            continue
        vals_arr = np.array(values, dtype=float)
        summary[metric_name] = {
            'mean': float(np.mean(vals_arr)),
            'std': float(np.std(vals_arr)),
            'min': float(np.min(vals_arr)),
            'max': float(np.max(vals_arr)),
            'count': int(len(vals_arr))
        }
    
    return {
        'summary': summary,
        'per_question': question_scores
    }

# Load logs from MongoDB
print("=" * 80)
print("Loading ALL logs from MongoDB...")
print("=" * 80)
retrieval_df, retrieval_count = get_retrieval_logs_from_mongo(
    question_to_id, question_norm_map, question_to_doc, chunk_map
)
gen_df, gen_count = get_generation_logs_from_mongo(
    question_to_id, gold_df, question_norm_map
)
print()

# DEBUG: Check chunk formats
debug_chunk_format_mismatch(retrieval_df, gold_df)
sys.stdout.flush()

results = {
    "total_retrieval_logs": retrieval_count,
    "total_generation_logs": gen_count,
    "retrieval_evaluation": None,
    "generation_evaluation": None,
    "gemini_rating": None
}

# Run retrieval evaluation
if retrieval_df is not None and not retrieval_df.empty:
    print("=" * 80)
    print("Running Retrieval Evaluation...")
    print("=" * 80)
    try:
        retrieval = evaluate_retrieval_from_dataframe(gold_df, retrieval_df)
        results['retrieval_evaluation'] = retrieval
        if retrieval and retrieval.get('summary'):
            print(f"Recall@10: {retrieval['summary']['recall@10']['mean']:.3f}")
            print(f"MRR@10: {retrieval['summary']['mrr@10']['mean']:.3f}")
            print(f"NDCG@10: {retrieval['summary']['ndcg@10']['mean']:.3f}")
        print("Retrieval evaluation complete")
    except Exception as e:
        logger.error(f"Retrieval evaluation failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("No retrieval logs to evaluate")

print()

print(f"DEBUG: Retrieval DF: {len(retrieval_df) if retrieval_df is not None else 0} rows")
print(f"DEBUG: Generation DF: {len(gen_df) if gen_df is not None else 0} rows")
sys.stdout.flush()

# Run generation evaluation
if gen_df is not None and not gen_df.empty:
    print("=" * 80)
    print("Running Gemini rating for generated answers...")
    print("=" * 80)
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    gemini_model = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
    try:
        rated_gen_df, gemini_summary = rate_generation_with_gemini(
            gen_df, gold_df, model_name=gemini_model, api_key=gemini_api_key
        )
        results['gemini_rating'] = gemini_summary
        if gemini_summary and gemini_summary.get('mean_rating') is not None:
            print(f"Gemini mean rating: {gemini_summary['mean_rating']:.3f} (rated {gemini_summary['rated']}/{gemini_summary['count']})")
    except Exception as e:
        logger.error(f"Gemini rating failed: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 80)
    print("Running Generation Evaluation...")
    print("=" * 80)
    try:
        generation = evaluate_generation_from_dataframe(gold_df, gen_df)
        results['generation_evaluation'] = generation
        if generation and generation.get('summary'):
            print(f"ROUGE-1: {generation['summary']['rouge1']['mean']:.3f}")
            print(f"Token F1: {generation['summary']['token_f1']['mean']:.3f}")
            print(f"EM: {generation['summary']['em']['mean']:.3f}")
        print("Generation evaluation complete")
    except Exception as e:
        logger.error(f"Generation evaluation failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("No generation logs to evaluate")

print()
print("=" * 80)
print("EVALUATION SUMMARY")
print("=" * 80)
print(f"Total Retrieval Logs Evaluated: {retrieval_count}")
print(f"Total Generation Logs Evaluated: {gen_count}")
print("=" * 80)