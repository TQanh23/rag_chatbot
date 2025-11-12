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
        
        logger.info(f"✓ Loaded {len(question_to_id)} gold standard Q&A pairs")
        
        # Return normalized mapping and question->doc map as well
        return gold_df, question_to_id, doc_questions, question_norm_map, question_to_doc
    except Exception as e:
        logger.error(f"Failed to load gold standard: {e}")
        return None, {}, {}, {}, {}

def get_chunk_metadata_map(document_id=None):
    """Build mapping from Qdrant point ID (UUID) to canonical chunk ID format.
    If document_id provided, only include points whose payload.document_id matches.
    """
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
            # If filtering by document, skip others
            payload_doc = point.payload.get('document_id')
            if document_id and payload_doc != document_id:
                continue

            point_uuid = str(point.id)
            chunk_id_canonical = point.payload.get('chunk_id')
            
            if chunk_id_canonical:
                chunk_map[point_uuid] = chunk_id_canonical
        
        logger.info(f"✓ Built chunk metadata map with {len(chunk_map)} mappings for document_id={document_id}")
        return chunk_map
    except Exception as e:
        logger.warning(f"Could not build chunk metadata map: {e}")
        return {}

def export_retrieval_logs_from_mongo(question_to_id, doc_questions, chunk_map, uploaded_doc_id, question_norm_map, question_to_doc):
    """Export retrieval logs matching a specific document only."""
    try:
        retrieval_logs = list(mongo_repo.db['retrieval_log'].find({}))
        
        if not retrieval_logs:
            logger.warning("No retrieval logs found in MongoDB")
            return None, 0
        
        rows = []
        mapped_count = 0
        conversion_stats = {"converted": 0, "not_converted": 0}
        
        for log in retrieval_logs:
            doc_id = log.get("document_id", "")
            question_text = log.get("question", "").strip()
            
            # Skip template variables and empty questions
            if not question_text or "{{" in question_text:
                continue
            
            # Only process logs from target document
            if doc_id != uploaded_doc_id:
                continue
            
            # Match only if the question exists in the gold standard (normalized)
            q_norm = question_text.lower()
            if q_norm in question_norm_map:
                gold_qtext = question_norm_map[q_norm]
                # Ensure the gold question belongs to the target document
                if question_to_doc.get(gold_qtext) != uploaded_doc_id:
                    continue
                
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
        
        logger.info(f"✓ Exported {mapped_count} retrieval logs for document {uploaded_doc_id}")
        logger.info(f"  Chunk conversions - Successful: {conversion_stats['converted']}, Fallback: {conversion_stats['not_converted']}")
        
        if mapped_count == 0:
            logger.warning(f"No matching retrieval logs found for document {uploaded_doc_id}")
            return None, 0
        
        # Use document-specific filename to avoid overwriting when evaluating multiple docs
        out_path = retrieval_run_path.replace('.csv', f'_{uploaded_doc_id}.csv')
        df = pd.DataFrame(rows)
        df.to_csv(out_path, index=False)
        logger.info(f"✓ Saved to {out_path}")
        
        return out_path, mapped_count
    except Exception as e:
        logger.error(f"Failed to export retrieval logs: {e}")
        import traceback
        traceback.print_exc()
        return None, 0

def export_generation_logs_from_mongo(question_to_id, doc_questions, gold_df, uploaded_doc_id, question_norm_map, question_to_doc):
    """Export generation logs matching a specific document only."""
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
        answer_stats = {"has_content": 0, "not_found": 0, "empty": 0}
        
        for log in generation_logs:
            doc_id = log.get("document_id", "")
            question_text = log.get("question_text", "").strip()
            
            if not question_text or "{{" in question_text:
                continue
            
            # Only process logs from target document
            if doc_id != uploaded_doc_id:
                continue
            
            # Match only if the question exists in the gold standard (normalized)
            q_norm = question_text.lower()
            if q_norm in question_norm_map:
                gold_qtext = question_norm_map[q_norm]
                # Ensure the gold question belongs to the target document
                if question_to_doc.get(gold_qtext) != uploaded_doc_id:
                    continue
                
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
        
        logger.info(f"✓ Exported {mapped_count} generation logs for document {uploaded_doc_id}")
        logger.info(f"  Answer quality - Content: {answer_stats['has_content']}, Not found: {answer_stats['not_found']}, Empty: {answer_stats['empty']}")
        
        if mapped_count == 0:
            logger.warning(f"No matching generation logs found for document {uploaded_doc_id}")
            return None, 0
        
        out_path = generation_run_path.replace('.csv', f'_{uploaded_doc_id}.csv')
        df = pd.DataFrame(rows)
        df.to_csv(out_path, index=False)
        logger.info(f"✓ Saved to {out_path}")
        return out_path, mapped_count
    except Exception as e:
        logger.error(f"Failed to export generation logs: {e}")
        import traceback
        traceback.print_exc()
        return None, 0

# Main execution - multi-document support
print("=" * 80)
print("RAG CHATBOT EVALUATION - Multi Document Mode")
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

gold_docs = list(gold_df['document_id'].unique())
print(f"Gold standard contains {len(gold_df)} questions from {len(gold_docs)} documents")
print()

overall = {
    "documents_evaluated": 0,
    "retrieval_logs_total": 0,
    "generation_logs_total": 0,
    "retrieval_results": {},
    "generation_results": {}
}

for doc_id in gold_docs:
    print("=" * 60)
    print(f"Evaluating document: {doc_id}")
    print("=" * 60)
    
    matching_questions = gold_df[gold_df['document_id'] == doc_id]
    print(f"Found {len(matching_questions)} gold questions for this document")
    
    # Build chunk map only for this document
    print("Building chunk metadata map from Qdrant...")
    chunk_map = get_chunk_metadata_map(document_id=doc_id)
    
    # Export logs for this document
    print("Exporting logs from MongoDB for this document...")
    retrieval_path, retrieval_count = export_retrieval_logs_from_mongo(
        question_to_id, doc_questions, chunk_map, doc_id, question_norm_map, question_to_doc
    )
    gen_path, gen_count = export_generation_logs_from_mongo(
        question_to_id, doc_questions, gold_df, doc_id, question_norm_map, question_to_doc
    )
    
    overall['documents_evaluated'] += 1
    overall['retrieval_logs_total'] += retrieval_count
    overall['generation_logs_total'] += gen_count
    
    # Run retrieval evaluation if we have exported logs
    if retrieval_path and os.path.exists(retrieval_path):
        print("Running Retrieval Evaluation...")
        try:
            retrieval = RetrievalEvaluator.evaluate(qa_gold_path, retrieval_path)
            overall['retrieval_results'][doc_id] = retrieval
            if retrieval and retrieval.get('summary'):
                print(f"  Recall@10: {retrieval['summary']['recall@10']['mean']:.3f}")
                print(f"  MRR@10: {retrieval['summary']['mrr@10']['mean']:.3f}")
                print(f"  NDCG@10: {retrieval['summary']['ndcg@10']['mean']:.3f}")
            print("  Retrieval evaluation complete")
        except Exception as e:
            logger.error(f"Retrieval evaluation failed for {doc_id}: {e}")
    else:
        print("  No retrieval logs to evaluate for this document")
    
    # Run generation evaluation if we have exported logs
    if gen_path and os.path.exists(gen_path):
        print("Running Generation Evaluation...")
        try:
            generation = GenerationEvaluator.evaluate(qa_gold_path, gen_path)
            overall['generation_results'][doc_id] = generation
            if generation and generation.get('summary'):
                print(f"  ROUGE-1: {generation['summary']['rouge1']['mean']:.3f}")
                print(f"  Token F1: {generation['summary']['token_f1']['mean']:.3f}")
                print(f"  EM: {generation['summary']['em']['mean']:.3f}")
            print("  Generation evaluation complete")
        except Exception as e:
            logger.error(f"Generation evaluation failed for {doc_id}: {e}")
    else:
        print("  No generation logs to evaluate for this document")
    
    print()

# Final summary
print("=" * 80)
print("EVALUATION SUMMARY - MULTI DOCUMENT")
print("=" * 80)
print(f"Documents Evaluated: {overall['documents_evaluated']}")
print(f"Total Retrieval Logs Mapped: {overall['retrieval_logs_total']}")
print(f"Total Generation Logs Mapped: {overall['generation_logs_total']}")
print()
print("Per-document retrieval/generation results available in logs and returned objects.")
print("=" * 80)