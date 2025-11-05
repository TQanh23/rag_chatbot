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
        for _, row in gold_df.iterrows():
            doc_id = row['document_id']
            if doc_id not in doc_questions:
                doc_questions[doc_id] = []
            doc_questions[doc_id].append(row['question_text'])
        
        logger.info(f"✓ Loaded {len(question_to_id)} gold standard Q&A pairs")
        
        return gold_df, question_to_id, doc_questions
    except Exception as e:
        logger.error(f"Failed to load gold standard: {e}")
        return None, {}, {}

def get_uploaded_document_id():
    """Get the document_id from Qdrant (assumes single document)."""
    try:
        collection_name = os.getenv('QDRANT_COLLECTION', 'test_collection')
        points, _ = qdrant.scroll(collection_name=collection_name, limit=1, with_payload=True, with_vectors=False)
        
        if points and points[0].payload:
            return points[0].payload.get('document_id')
        return None
    except Exception as e:
        logger.error(f"Failed to get document_id: {e}")
        return None

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
            if point.payload:
                point_uuid = str(point.id)
                chunk_id_canonical = point.payload.get('chunk_id')
                
                if chunk_id_canonical:
                    chunk_map[point_uuid] = chunk_id_canonical
        
        logger.info(f"✓ Built chunk metadata map with {len(chunk_map)} mappings")
        return chunk_map
    except Exception as e:
        logger.warning(f"Could not build chunk metadata map: {e}")
        return {}

def export_retrieval_logs_from_mongo(question_to_id, doc_questions, chunk_map, uploaded_doc_id):
    """Export retrieval logs matching the uploaded document only."""
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
            
            # Only process logs from uploaded document
            if doc_id != uploaded_doc_id:
                continue
            
            # Exact match only
            if doc_id in doc_questions and question_text in doc_questions[doc_id]:
                question_id = question_to_id[question_text]
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
                    "question_text": question_text,
                    "retrieved_chunk_ids": "|".join(converted_ids),
                    "retrieved_scores": "|".join([str(s) for s in retrieved_scores]),
                    "top_k": log.get("top_k", len(retrieved_chunk_ids))
                })
        
        logger.info(f"✓ Exported {mapped_count}/{len(retrieval_logs)} retrieval logs (from uploaded document)")
        logger.info(f"  Chunk conversions - Successful: {conversion_stats['converted']}, Fallback: {conversion_stats['not_converted']}")
        
        if mapped_count == 0:
            logger.error("✗ No matching questions found!")
            return None, 0
        
        df = pd.DataFrame(rows)
        df.to_csv(retrieval_run_path, index=False)
        logger.info(f"✓ Saved to {retrieval_run_path}")
        
        return retrieval_run_path, mapped_count
    except Exception as e:
        logger.error(f"Failed to export retrieval logs: {e}")
        import traceback
        traceback.print_exc()
        return None, 0

def export_generation_logs_from_mongo(question_to_id, doc_questions, gold_df, uploaded_doc_id):
    """Export generation logs matching the uploaded document only."""
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
            
            # Only process logs from uploaded document
            if doc_id != uploaded_doc_id:
                continue
            
            if doc_id in doc_questions and question_text in doc_questions[doc_id]:
                question_id = question_to_id[question_text]
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
                    "question_text": question_text,
                    "generated_answer": answer_text,
                    "citations": ""
                })
        
        logger.info(f"✓ Exported {mapped_count}/{len(generation_logs)} generation logs (from uploaded document)")
        logger.info(f"  Answer quality - Content: {answer_stats['has_content']}, Not found: {answer_stats['not_found']}, Empty: {answer_stats['empty']}")
        
        if mapped_count == 0:
            logger.error("✗ No matching questions found!")
            return None, 0
        
        df = pd.DataFrame(rows)
        df.to_csv(generation_run_path, index=False)
        logger.info(f"✓ Saved to {generation_run_path}")
        return generation_run_path, mapped_count
    except Exception as e:
        logger.error(f"Failed to export generation logs: {e}")
        import traceback
        traceback.print_exc()
        return None, 0

# Main execution
print("=" * 80)
print("RAG CHATBOT EVALUATION - Single Document Mode")
print("=" * 80)
print()

# Get uploaded document
print("Detecting uploaded document...")
uploaded_doc_id = get_uploaded_document_id()
if not uploaded_doc_id:
    print("✗ No documents found in Qdrant!")
    sys.exit(1)
print(f"✓ Found document: {uploaded_doc_id}")
print()

# Load gold standard
print("=" * 80)
print("Loading gold standard Q&A pairs...")
print("=" * 80)
gold_df, question_to_id, doc_questions = load_gold_standard_mapping()

if gold_df is not None:
    # Show which documents are in gold standard
    gold_docs = gold_df['document_id'].unique()
    print(f"\nGold standard contains {len(gold_df)} questions from {len(gold_docs)} documents")
    print(f"Evaluating only questions from: {uploaded_doc_id}")
    
    matching_questions = gold_df[gold_df['document_id'] == uploaded_doc_id]
    print(f"✓ Found {len(matching_questions)} matching questions for evaluation")
    print()
    
    print("=" * 80)
    print("Building chunk metadata map from Qdrant...")
    print("=" * 80)
    chunk_map = get_chunk_metadata_map()
    print()
    
    # Export logs from MongoDB
    print("=" * 80)
    print("Exporting logs from MongoDB...")
    print("=" * 80)
    
    retrieval_run_path, retrieval_count = export_retrieval_logs_from_mongo(question_to_id, doc_questions, chunk_map, uploaded_doc_id)
    generation_run_path, generation_count = export_generation_logs_from_mongo(question_to_id, doc_questions, gold_df, uploaded_doc_id)
    
    print()
    
    # Run evaluation if logs exist
    if retrieval_run_path and os.path.exists(retrieval_run_path):
        print("=" * 80)
        print("Running Retrieval Evaluation...")
        print("=" * 80)
        try:
            retrieval = RetrievalEvaluator.evaluate(qa_gold_path, retrieval_run_path)
            if retrieval and retrieval.get('summary'):
                print(f"✓ Recall@10: {retrieval['summary']['recall@10']['mean']:.3f}")
                print(f"✓ MRR@10: {retrieval['summary']['mrr@10']['mean']:.3f}")
                print(f"✓ NDCG@10: {retrieval['summary']['ndcg@10']['mean']:.3f}")
            print("✓ Retrieval evaluation complete")
        except Exception as e:
            logger.error(f"Retrieval evaluation failed: {e}")
        print()
    
    if generation_run_path and os.path.exists(generation_run_path):
        print("=" * 80)
        print("Running Generation Evaluation...")
        print("=" * 80)
        try:
            generation = GenerationEvaluator.evaluate(qa_gold_path, generation_run_path)
            if generation and generation.get('summary'):
                print(f"✓ ROUGE-1: {generation['summary']['rouge1']['mean']:.3f}")
                print(f"✓ Token F1: {generation['summary']['token_f1']['mean']:.3f}")
                print(f"✓ EM: {generation['summary']['em']['mean']:.3f}")
            print("✓ Generation evaluation complete")
        except Exception as e:
            logger.error(f"Generation evaluation failed: {e}")
        print()
    
    print("=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Document Evaluated: {uploaded_doc_id}")
    print(f"Questions Evaluated: {retrieval_count}")
    print(f"Coverage: {retrieval_count} from {len(gold_df)} gold standard questions")
    print()
    print("Retrieval Performance:")
    print("  ✓ Evaluates how well the system finds relevant document chunks")
    print()
    print("Generation Performance:")
    print("  ✓ Evaluates how well generated answers match gold standard answers")
    print()
    print("=" * 80)
else:
    logger.error("Cannot proceed without gold standard data")
    sys.exit(1)