from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from backend.utils.embeddings import HuggingfaceEmbeddingsModel
from backend.utils.qdrant_client import get_qdrant_client
from qdrant_client.http.models import Distance
from django.conf import settings
import logging
import os
import re
import numpy as np
from langdetect import detect
from google import generativeai
from sentence_transformers import CrossEncoder
from django.http import JsonResponse
from backend.utils.retrieval_metrics import recall_at_k, mrr_at_k, ndcg_at_k
# NEW: add robust error handling imports
from google.api_core.exceptions import GoogleAPICallError, RetryError, DeadlineExceeded, ServiceUnavailable
import time
import uuid
from typing import List
from datetime import datetime  # Add this import for datetime.utcnow()

from backend.utils.mongo_repository import MongoRepository
from backend.utils.mongo_models import MongoQueryLog
from api.utils.query_preprocessing import preprocess_query

logger = logging.getLogger(__name__)

class AskView(APIView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mongo_repo = MongoRepository()
        # Add connection check
        try:
            self.mongo_repo.db.command('ping')
            logger.info("MongoDB connection successful.")
        except Exception as e:
            logger.error(f"MongoDB connection failed: {e}")
    
    # NEW: Retrieve chunks from MongoDB when Qdrant fails
    def _retrieve_from_mongodb(self, question_embedding, top_k=20):
        """
        Fallback retrieval using MongoDB vector search via cosine similarity.
        Returns list of results in same format as Qdrant for compatibility.
        """
        try:
            logger.info(f"Attempting MongoDB vector search with top_k={top_k}")
            
            # Get all chunks from MongoDB
            chunks = list(self.mongo_repo.db['chunks'].find({}))
            
            if not chunks:
                logger.warning("No chunks found in MongoDB")
                return []
            
            # Compute cosine similarity manually
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            embeddings = []
            chunk_ids = []
            
            for chunk in chunks:
                emb = chunk.get('embedding')
                if emb and isinstance(emb, list):
                    embeddings.append(emb)
                    chunk_ids.append(chunk.get('_id'))
            
            if not embeddings:
                logger.warning("No valid embeddings found in MongoDB chunks")
                return []
            
            # Convert to numpy arrays
            embeddings_array = np.array(embeddings)
            question_emb_array = np.array(question_embedding).reshape(1, -1)
            
            # Compute similarities
            similarities = cosine_similarity(question_emb_array, embeddings_array)[0]
            
            # Get top_k
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Build results in Qdrant-compatible format
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.0:  # Filter out zero/negative scores
                    chunk = chunks[idx]
                    result = type('obj', (object,), {
                        'id': str(chunk.get('_id')),
                        'score': float(similarities[idx]),
                        'payload': {
                            'text': chunk.get('text', ''),
                            'document_id': chunk.get('document_id', ''),
                            'page': chunk.get('page'),
                            'chunk_id': chunk.get('chunk_id'),
                            'section': chunk.get('section'),
                            'metadata': chunk.get('metadata', {})
                        }
                    })()
                    results.append(result)
            
            logger.info(f"MongoDB returned {len(results)} results with similarity scores")
            return results
            
        except Exception as e:
            logger.exception(f"MongoDB fallback retrieval failed: {e}")
            return []
        
    def _extract_gemini_text(self, resp):
        """
        Safely extract text from a Gemini response without using resp.text,
        which raises when there are no Parts.
        """
        try:
            candidates = getattr(resp, "candidates", []) or []
            for cand in candidates:
                content = getattr(cand, "content", None)
                parts = getattr(content, "parts", []) if content else []
                texts = [getattr(p, "text", "") for p in parts if getattr(p, "text", "")]
                if texts:
                    return " ".join(texts).strip(), getattr(cand, "finish_reason", None)
            # No parts in any candidate
            finish_reason = getattr(candidates[0], "finish_reason", None) if candidates else None
            return "", finish_reason
        except Exception:
            return "", None

    # NEW: sanitize prompt to avoid safety-filter triggers
    def _sanitize_prompt_for_safety(self, question, context_text):
        """
        Return a sanitized, neutral prompt to reduce chance of triggering safety filters.
        - Replaces common trigger words with neutral alternatives.
        - Trims context to SAFE_CONTEXT_CHARS.
        """
        replacements = {
            "giết": "loại bỏ",
            "bắn": "sử dụng",
            "tấn công": "tiếp cận",
            "phá hủy": "thay đổi",
            "chết": "kết thúc",
            # English fallbacks
            "kill": "remove",
            "attack": "approach",
            "bomb": "device",
        }

        safe_context = context_text or ""
        for trg, rep in replacements.items():
            safe_context = safe_context.replace(trg, rep)

        try:
            SAFE_CONTEXT_CHARS = int(os.getenv("SAFE_CONTEXT_CHARS", "3000"))
        except Exception:
            SAFE_CONTEXT_CHARS = 3000

        if len(safe_context) > SAFE_CONTEXT_CHARS:
            safe_context = safe_context[:SAFE_CONTEXT_CHARS]

        neutral_system = (
            "Bạn là một trợ lý tư vấn trung lập. "
            "Dựa trên thông tin dưới đây, trả lời ngắn gọn và khách quan. "
            "Nếu không có thông tin rõ ràng, hãy trả lời: 'Tôi không tìm thấy thông tin trong tài liệu.'"
        )

        sanitized_prompt = (
            f"{neutral_system}\n\nTài liệu tham khảo (đã được làm sạch):\n{safe_context}\n\n"
            f"Câu hỏi: {question}\nTrả lời ngắn gọn:"
        )
        return sanitized_prompt

    def _build_context_text(self, results):
        context_blocks = []
        for result in results:
            ref = f"[{result['payload']['document_id']} tr.{result['payload'].get('page', '?')} #{result['id']}]"
            context_blocks.append(f"{ref}\n{result['payload']['text']}\n")
        return "\n".join(context_blocks)

    # NEW: thin wrapper to call Gemini with a configurable timeout
    def _call_gemini_once(self, model_name, prompt, temperature=0.2, max_output_tokens=512, timeout_s=30):
        model = generativeai.GenerativeModel(model_name)
        # NOTE: google-generativeai SDK does not support request_options in generate_content().
        # Timeout is handled at the underlying gRPC/HTTP layer with default/environment settings.
        # For strict timeout control, consider using asyncio or threading wrapper (future enhancement).
        return model.generate_content(
            prompt,
            generation_config=generativeai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                candidate_count=1,
            ),
        )

    # UPDATED: retry logic with handling for safety filter finish_reason=2, progressive token increase,
    # and exponential backoff for transient Gemini/network errors
    def _gemini_generate_with_retry(self, prompt, attempts=3, base_timeout=30, backoff=2.0,
                                    initial_max_tokens=512, max_tokens_cap=4096, context_text=None, question=None):
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        last_exc = None

        # Ensure sensible integers
        try:
            initial_max_tokens = int(os.getenv("GEMINI_INITIAL_MAX_TOKENS", str(initial_max_tokens)))
        except Exception:
            initial_max_tokens = initial_max_tokens
        try:
            max_tokens_cap = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", str(max_tokens_cap)))
        except Exception:
            max_tokens_cap = max_tokens_cap

        for i in range(attempts):
            timeout = int(base_timeout * (backoff ** i))
            # progressive token strategy: increase tokens on each retry (but cap)
            try:
                tokens = min(int(initial_max_tokens * (backoff ** i)), max_tokens_cap)
            except Exception:
                tokens = initial_max_tokens

            # lower temperature on retries to reduce unexpected outputs
            temperature = 0.2 if i == 0 else 0.05

            try:
                logger.info(f"Calling Gemini attempt {i+1}/{attempts} (timeout={timeout}s, tokens={tokens}, temp={temperature}) using model={model_name}")
                resp = self._call_gemini_once(
                    model_name=model_name,
                    prompt=prompt,
                    temperature=temperature,
                    max_output_tokens=tokens,
                    timeout_s=timeout,
                )
                text, finish_reason = self._extract_gemini_text(resp)

                # Interpret finish_reason numeric/string values
                try:
                    fr_val = int(finish_reason) if finish_reason is not None and str(finish_reason).isdigit() else None
                except Exception:
                    fr_val = None

                # If no text and finish_reason suggests token limit, try increasing tokens (if we haven't hit cap)
                if (not text) and (fr_val == 4 or (isinstance(finish_reason, str) and "max" in finish_reason.lower() or "token" in str(finish_reason).lower())):
                    logger.warning(f"Gemini likely hit max tokens (finish_reason={finish_reason}). Attempt {i+1}/{attempts}.")
                    # If we can increase tokens, continue to next attempt with larger tokens
                    # The loop already increases tokens by factor backoff, so just continue
                    continue

                # If safety filter triggered (finish_reason == 2), attempt sanitized prompt if available
                if (not text) and (fr_val == 2 or (isinstance(finish_reason, str) and "safety" in finish_reason.lower())):
                    logger.warning(f"Safety filters triggered (finish_reason={finish_reason}), attempt {i+1}/{attempts}.")
                    if context_text and question:
                        prompt = self._sanitize_prompt_for_safety(question, context_text)
                        # After sanitizing, try again (loop will continue)
                        continue
                    # if no context/question to sanitize, just continue retries
                    continue

                # If we have text or other finish reasons, return result
                return text, finish_reason, resp

            except (RetryError, DeadlineExceeded, ServiceUnavailable, GoogleAPICallError) as e:
                last_exc = e
                logger.warning(f"Gemini transient error (attempt {i+1}/{attempts}, timeout={timeout}s): {e}")
                time.sleep(min(1.5 * (i + 1), 5))
            except Exception as e:
                last_exc = e
                logger.exception(f"Unexpected error calling Gemini (attempt {i+1}/{attempts}): {e}")
                time.sleep(1)

        # Exhausted attempts
        # last_exc may be None or not an Exception subclass (e.g. a string). Raise a proper exception.
        if last_exc is None:
            raise RuntimeError("Gemini exhausted retries without an exception (likely repeated safety filters or empty responses).")
        if isinstance(last_exc, BaseException):
            raise last_exc
        # fallback for non-exception values
        raise RuntimeError(f"Gemini failed with non-exception value: {last_exc}")

    def _qdrant_search_with_fallback(self, client, collection_name, question_embedding, top_k, initial_threshold):
        """
        Try Qdrant search with descending thresholds until we get any hits.
        """
        thresholds = []
        if initial_threshold is not None:
            thresholds.append(initial_threshold)
        thresholds.extend([0.35, 0.2, None])

        for t in thresholds:
            try:
                kwargs = {
                    "collection_name": collection_name,
                    "query_vector": ("default", question_embedding),
                    "limit": top_k,
                    "with_payload": True,
                }
                if t is not None:
                    kwargs["score_threshold"] = float(t)
                search_result = client.search(**kwargs)
                logger.info(f"Qdrant returned {len(search_result)} results with score_threshold={t}.")
                if search_result:
                    return search_result, t
            except Exception as e:
                logger.exception(f"Error during Qdrant search at threshold {t}: {e}")
                continue
        return [], None

    def _qdrant_search_with_fallback_filtered(self, client, collection_name, question_embedding, top_k, initial_threshold, search_filter=None):
        """
        Try Qdrant search with descending thresholds until we get any hits.
        Supports optional filter for document-specific searches.
        """
        thresholds = []
        if initial_threshold is not None:
            thresholds.append(initial_threshold)
        thresholds.extend([0.35, 0.2, None])

        for t in thresholds:
            try:
                kwargs = {
                    "collection_name": collection_name,
                    "query_vector": ("default", question_embedding),
                    "limit": top_k,
                    "with_payload": True,
                }
                if t is not None:
                    kwargs["score_threshold"] = float(t)
                if search_filter is not None:
                    kwargs["query_filter"] = search_filter
                search_result = client.search(**kwargs)
                logger.info(f"Qdrant returned {len(search_result)} results with score_threshold={t}, filter={bool(search_filter)}.")
                if search_result:
                    return search_result, t
            except Exception as e:
                logger.exception(f"Error during Qdrant search at threshold {t}: {e}")
                continue
        return [], None

    def _expand_query(self, question, num_variants=3):
        """
        Generate query variants using Gemini for better recall.
        Returns list of query variants including the original question.
        
        Args:
            question: Original question string
            num_variants: Number of variants to generate (default 3)
            
        Returns:
            List of query strings (original + variants)
        """
        # Check if query expansion is enabled
        if os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() != "true":
            return [question]
        
        try:
            model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            model = generativeai.GenerativeModel(model_name)
            
            # More structured prompt with examples for consistency (P1 fix)
            expansion_prompt = f"""Nhiệm vụ: Tạo {num_variants} cách diễn đạt KHÁC NHAU cho câu hỏi, giữ NGUYÊN ý nghĩa.

Câu hỏi gốc: "{question}"

Quy tắc bắt buộc:
1. Mỗi phiên bản có CẤU TRÚC CÂU khác nhau
2. Dùng TỪ ĐỒNG NGHĨA khi có thể
3. Giữ nguyên THUẬT NGỮ KỸ THUẬT
4. KHÔNG thêm hoặc bớt thông tin
5. Mỗi dòng BẮT ĐẦU bằng số thứ tự

Ví dụ:
Câu hỏi: "Python là gì?"
1. Định nghĩa ngôn ngữ lập trình Python
2. Giải thích khái niệm Python trong lập trình
3. Python được hiểu như thế nào

Bây giờ tạo {num_variants} phiên bản:"""

            resp = model.generate_content(
                expansion_prompt,
                generation_config=generativeai.types.GenerationConfig(
                    temperature=0.3,  # Lower temperature for more consistency
                    max_output_tokens=256,
                    candidate_count=1,
                ),
            )
            
            text, _ = self._extract_gemini_text(resp)
            if not text:
                logger.warning("Query expansion returned empty response")
                return [question]
            
            # Parse variants with better error handling (P1 fix)
            variants = [question]  # Always include original
            lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
            
            for line in lines:
                # Match: "1. text" or "1) text" or "1: text"
                match = re.match(r'^\d+[\.\)\:]\s*(.+)$', line)
                if match:
                    variant = match.group(1).strip()
                    # Remove quotes if present
                    # Remove quotes and validate variant
                    variant = variant.strip('"').strip("'")
                    if (variant and 
                        variant != question and 
                        len(variant) >= 5 and 
                        len(variant) <= len(question) * 2):  # Sanity check
                        variants.append(variant)
            
            # Ensure we have at least original + 1 variant
            if len(variants) < 2:
                logger.warning(f"Query expansion produced insufficient variants: {len(variants)}")
                # Fallback: create simple variant by reordering key terms
                words = question.split()
                if len(words) > 3:
                    # Simple reordering as fallback
                    variants.append(' '.join(words[-2:] + words[:-2]))
            
            # Limit to requested + original
            variants = variants[:num_variants + 1]
            logger.info(f"Query expansion: {len(variants)-1} variants for '{question[:40]}...'")
            
            return variants
            
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return [question]

    def _search_with_expanded_queries(self, client, collection_name, question, embedding_model, top_k, initial_threshold):
        """
        Perform retrieval using expanded queries for better recall.
        Combines results from multiple query variants and deduplicates.
        
        Args:
            client: Qdrant client
            collection_name: Name of the collection
            question: Original question
            embedding_model: Model for embedding texts
            top_k: Number of results to retrieve per query
            initial_threshold: Score threshold for retrieval
            
        Returns:
            Tuple of (merged_results, used_threshold, expansion_stats)
        """
        # Get query variants
        query_variants = self._expand_query(question)
        
        all_results = []
        seen_ids = set()
        used_threshold = initial_threshold
        
        for i, query in enumerate(query_variants):
            # Embed the query variant
            query_embedding = embedding_model.embed_texts([query])[0]
            
            # Search with this variant
            results, threshold = self._qdrant_search_with_fallback(
                client, collection_name, query_embedding, top_k, initial_threshold
            )
            
            if threshold is not None:
                used_threshold = threshold
            
            # Add unique results
            for result in results:
                if result.id not in seen_ids:
                    all_results.append(result)
                    seen_ids.add(result.id)
            
            logger.debug(f"Query variant {i+1}/{len(query_variants)}: '{query[:30]}...' returned {len(results)} results")
        
        # Sort all results by score (descending)
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        # Return top_k unique results
        merged_results = all_results[:top_k]
        
        expansion_stats = {
            "num_variants": len(query_variants),
            "variants": query_variants[:3],  # Log first 3 for debugging
            "total_unique_results": len(all_results),
            "merged_to": len(merged_results)
        }
        
        logger.info(f"Query expansion: {len(query_variants)} variants → {len(all_results)} unique results → top {len(merged_results)}")
        
        return merged_results, used_threshold, expansion_stats

    def _search_with_expanded_queries_filtered(self, client, collection_name, question, embedding_model, top_k, initial_threshold, search_filter=None):
        """
        Perform retrieval using expanded queries for better recall with optional document filter.
        Combines results from multiple query variants and deduplicates.
        
        Args:
            client: Qdrant client
            collection_name: Name of the collection
            question: Original question
            embedding_model: Model for embedding texts
            top_k: Number of results to retrieve per query
            initial_threshold: Score threshold for retrieval
            search_filter: Optional Qdrant filter for document-specific searches
            
        Returns:
            Tuple of (merged_results, used_threshold, expansion_stats)
        """
        # Get query variants
        query_variants = self._expand_query(question)
        
        all_results = []
        seen_ids = set()
        used_threshold = initial_threshold
        
        for i, query in enumerate(query_variants):
            # Embed the query variant
            query_embedding = embedding_model.embed_texts([query])[0]
            
            # Search with this variant and filter
            results, threshold = self._qdrant_search_with_fallback_filtered(
                client, collection_name, query_embedding, top_k, initial_threshold, search_filter
            )
            
            if threshold is not None:
                used_threshold = threshold
            
            # Add unique results
            for result in results:
                if result.id not in seen_ids:
                    all_results.append(result)
                    seen_ids.add(result.id)
            
            logger.debug(f"Query variant {i+1}/{len(query_variants)}: '{query[:30]}...' returned {len(results)} results (filtered={bool(search_filter)})")
        
        # Sort all results by score (descending)
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        # Return top_k unique results
        merged_results = all_results[:top_k]
        
        expansion_stats = {
            "num_variants": len(query_variants),
            "variants": query_variants[:3],  # Log first 3 for debugging
            "total_unique_results": len(all_results),
            "merged_to": len(merged_results)
        }
        
        logger.info(f"Query expansion (filtered={bool(search_filter)}): {len(query_variants)} variants → {len(all_results)} unique results → top {len(merged_results)}")
        
        return merged_results, used_threshold, expansion_stats

    def post(self, request):
        try:
            start_time = time.time()
            
            # Step 1: Normalize question with preprocessing
            raw_question = request.data.get("question", "").strip()
            document_id = request.data.get("document_id")  # Document filter
            final_k = int(request.data.get("final_k", 10))
            # IMPROVEMENT 1: Increase initial retrieval to 4x final_k for better recall
            top_k = int(request.data.get("top_k", final_k * 4))  # 4x multiplier
            
            if not raw_question:
                logger.error("No question provided in the request.")
                return Response({"error": "Question is required."}, status=status.HTTP_400_BAD_REQUEST)
            
            # Apply consistent preprocessing to match document preprocessing
            question = preprocess_query(raw_question)
            logger.info(f"Query preprocessing: '{raw_question}' → '{question}'")
            logger.info(f"Document filter: {document_id if document_id else 'None (search all)'}")

            # Generate a stable question_id for logging
            question_id = str(uuid.uuid4())
            correlation_id = str(uuid.uuid4())

            # Optional: quick language hint for debugging
            try:
                lang = detect(question)
                logger.info(f"Detected question language: {lang}")
            except Exception:
                lang = "unknown"

            # Step 2: Embed the question
            embedding_model = HuggingfaceEmbeddingsModel()
            question_embedding = embedding_model.embed_texts([question])[0]
            logger.debug("Question embedding generated successfully.")

            # Step 3: Retrieve from Qdrant with document filter
            client = get_qdrant_client()
            collection_name = settings.QDRANT_COLLECTION

            score_threshold_param = request.data.get("score_threshold", None)
            # IMPROVEMENT 2: Lower threshold to 0.3 for more permissive initial retrieval
            initial_threshold = float(score_threshold_param) if score_threshold_param is not None else 0.3
            
            use_expansion = request.data.get("expand_query", os.getenv("ENABLE_QUERY_EXPANSION", "true")).lower() == "true"

            logger.info(f"Retrieving up to {top_k} chunks from Qdrant (threshold={initial_threshold}, expand={use_expansion}, doc_filter={bool(document_id)}).")

            # Build document filter if provided
            search_filter = None
            if document_id:
                from qdrant_client import models
                search_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id)
                        )
                    ]
                )
                logger.info(f"Applying document filter: document_id={document_id}")

            # Try Qdrant with optional query expansion and document filter
            search_result = None
            used_threshold = None
            source = "qdrant"
            expansion_stats = None
            
            try:
                if use_expansion:
                    # Modified search with filter support
                    search_result, used_threshold, expansion_stats = self._search_with_expanded_queries_filtered(
                        client, collection_name, question, embedding_model, top_k, initial_threshold, search_filter
                    )
                else:
                    # Standard single-query search with filter
                    search_result, used_threshold = self._qdrant_search_with_fallback_filtered(
                        client, collection_name, question_embedding, top_k, initial_threshold, search_filter
                    )
            except Exception as e:
                logger.warning(f"Qdrant search failed: {e}. Falling back to MongoDB.")
                source = "mongodb"
            
            # If Qdrant failed or returned no results, try MongoDB
            if not search_result:
                logger.info("Attempting MongoDB fallback retrieval...")
                search_result = self._retrieve_from_mongodb(question_embedding, top_k)
                source = "mongodb" if search_result else "none"
            
            if not search_result:
                logger.warning(f"No results found from either Qdrant or MongoDB. lang={lang}")
                return Response(
                    {
                        "answer": "Tôi không tìm thấy thông tin trong tài liệu.",
                        "citations": [],
                        "debug": {
                            "reason": "No retrieval results",
                            "lang": lang,
                            "used_threshold": used_threshold,
                            "retrieval_source": source,
                            "document_filter": document_id
                        }
                    },
                    status=status.HTTP_200_OK
                )

            # IMPROVEMENT 3: Strict document filtering validation
            initial_count = len(search_result)
            if document_id:
                # Filter out any chunks that don't match the specified document
                filtered_search_result = []
                wrong_doc_count = 0
                for result in search_result:
                    result_doc_id = result.payload.get('document_id', '')
                    if result_doc_id == document_id:
                        filtered_search_result.append(result)
                    else:
                        wrong_doc_count += 1
                        logger.warning(f"Filtered chunk {result.id} from wrong document: {result_doc_id} (expected: {document_id})")
                
                search_result = filtered_search_result
                
                # IMPROVEMENT 5: Enhanced logging with filtering metrics
                if wrong_doc_count > 0:
                    logger.warning(f"Document filtering: {initial_count} → {len(search_result)} chunks (removed {wrong_doc_count} from wrong documents)")
                else:
                    logger.info(f"Document filtering: All {len(search_result)} chunks matched document_id={document_id}")
            
            logger.info(f"Retrieved {len(search_result)} results from {source} (initial: {initial_count}, after filtering: {len(search_result)}).")
            
            # Check if filtering removed all results
            if not search_result:
                logger.warning(f"No results remaining after document filtering for document_id={document_id}")
                return Response(
                    {
                        "answer": "Không tìm thấy thông tin liên quan trong tài liệu được chỉ định.",
                        "citations": [],
                        "debug": {
                            "reason": "No results after document filtering",
                            "initial_count": initial_count,
                            "filtered_count": 0,
                            "document_id": document_id,
                            "retrieval_source": source
                        }
                    },
                    status=status.HTTP_200_OK
                )

            # Step 4: Rerank results
            reranker = CrossEncoder('itdainb/PhoRanker')
            rerank_inputs = [(question, result.payload['text']) for result in search_result]
            rerank_scores = reranker.predict(rerank_inputs)
            logger.debug("Reranking completed successfully.")

            # Combine and sort
            reranked_results = [
                {"id": result.id, "score": float(score), "payload": result.payload}
                for result, score in zip(search_result, rerank_scores)
            ]
            reranked_results.sort(key=lambda x: x['score'], reverse=True)

            # Apply minimum score threshold
            MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "-0.5"))
            filtered_results = [r for r in reranked_results if r['score'] >= MIN_RERANK_SCORE]
            
            if not filtered_results:
                logger.warning(f"All reranked results below threshold {MIN_RERANK_SCORE}. Using top result anyway.")
                filtered_results = reranked_results[:1] if reranked_results else []
            
            top_results = filtered_results[:final_k]
            logger.info(f"Reranking: {len(search_result)} → {len(reranked_results)} → {len(filtered_results)} (threshold={MIN_RERANK_SCORE}) → top {len(top_results)}")

            # Log retrieval
            try:
                retrieved_chunk_ids = [result.id for result in search_result]
                retrieved_scores = [result.score for result in search_result]
                
                self.mongo_repo.db['retrieval_log'].insert_one({
                    "question_id": question_id,
                    "correlation_id": correlation_id,
                    "question": question,
                    "retrieved_chunk_ids": retrieved_chunk_ids,
                    "retrieved_scores": retrieved_scores,
                    "top_k": final_k,
                    "document_id": document_id or (top_results[0]['payload'].get('document_id') if top_results else None),
                    "latency_ms": int((time.time() - start_time) * 1000),
                    "ts": datetime.now(),
                    "user_id": None,
                })
                logger.debug(f"Logged retrieval to MongoDB: {len(retrieved_chunk_ids)} chunks")
            except Exception as e:
                logger.exception(f"Failed to log retrieval to MongoDB: {e}")
            
            # Step 5: Generate answer with Gemini
            google_api_key = os.getenv("GEMINI_API_KEY")
            if not google_api_key:
                logger.error("Google API key is not configured.")
                return Response({"error": "Google API key is not configured."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            generativeai.configure(api_key=google_api_key)

            # IMPROVEMENT 4: Enhanced prompt with better Vietnamese instructions and citation guidance
            system_message = (
                "Bạn là trợ lý AI thông minh chuyên phân tích tài liệu kỹ thuật.\n\n"
                "QUY TẮC TRẢ LỜI:\n"
                "1. Trả lời TRỰC TIẾP và SÚC TÍCH dựa vào ngữ cảnh được cung cấp\n"
                "2. CHỈ sử dụng thông tin có trong tài liệu tham khảo\n"
                "3. LUÔN trích dẫn nguồn theo định dạng: [document_id tr.X]\n"
                "4. Nếu cần kết hợp nhiều đoạn, hãy tổng hợp rõ ràng và trích dẫn TẤT CẢ nguồn\n"
                "5. Nếu KHÔNG tìm thấy thông tin, trả lời: 'Không tìm thấy thông tin trong tài liệu.'\n"
                "6. Ưu tiên ĐỘ CHÍNH XÁC hơn độ dài câu trả lời\n"
                "7. Trả lời bằng tiếng Việt, ngắn gọn nhưng đầy đủ ý"
            )

            context_text = self._build_context_text(top_results)

            MAX_CONTEXT_CHARS = 12000
            if len(context_text) > MAX_CONTEXT_CHARS:
                logger.info(f"Trimming context from {len(context_text)} to {MAX_CONTEXT_CHARS} chars.")
                context_text = context_text[:MAX_CONTEXT_CHARS]

            prompt = f"{system_message}\n\nNGỮ CẢNH:\n{context_text}\n\nCâu hỏi: {question}\nCâu trả lời:"

            try:
                llm_timeout = int(os.getenv("GEMINI_TIMEOUT", "30"))
                llm_attempts = int(os.getenv("GEMINI_MAX_ATTEMPTS", "3"))
                initial_max_tokens = int(os.getenv("GEMINI_INITIAL_MAX_TOKENS", "512"))
                max_tokens_cap = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "4096"))
            except Exception:
                llm_timeout, llm_attempts, initial_max_tokens, max_tokens_cap = 30, 3, 512, 4096

            try:
                response_text, finish_reason, _ = self._gemini_generate_with_retry(
                    prompt,
                    attempts=llm_attempts,
                    base_timeout=llm_timeout,
                    backoff=2.0,
                    initial_max_tokens=initial_max_tokens,
                    max_tokens_cap=max_tokens_cap,
                    context_text=context_text,
                    question=question,
                )

                if not response_text:
                    logger.warning(f"Gemini returned no content. finish_reason={finish_reason}")
                    return Response(
                        {"error": "LLM did not return content.", "finish_reason": str(finish_reason)},
                        status=status.HTTP_502_BAD_GATEWAY
                    )

                response = response_text
                logger.info("Response generated successfully.")
                
                # ADD GENERATION LOGGING HERE
                try:
                    generation_latency = int((time.time() - start_time) * 1000)
                    
                    # Extract citations from top_results (already available)
                    citation_list = [
                        {
                            "chunk_id": result['id'],
                            "document_id": result['payload']['document_id'],
                            "page": result['payload'].get('page', 'N/A'),
                            "score": result['score']
                        }
                        for result in top_results
                    ]
                    
                    self.mongo_repo.db['generation_log'].insert_one({
                        "question_id": question_id,
                        "correlation_id": correlation_id,
                        "question_text": question,
                        "raw_question": raw_question,
                        "document_id": document_id or (top_results[0]['payload'].get('document_id') if top_results else None),
                        "generated_answer": response,
                        "citations": citation_list,
                        "model_name": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
                        "finish_reason": str(finish_reason) if finish_reason else None,
                        "context_length": len(context_text),
                        "num_chunks_used": len(top_results),
                        "latency_ms": generation_latency,
                        "ts": datetime.now(),
                        "user_id": None,
                    })
                    logger.info(f"Logged generation to MongoDB: answer_length={len(response)}, citations={len(citation_list)}")
                    
                    # ADD: Log each citation separately to citation_log collection
                    if citation_list:
                        citation_docs = []
                        for rank, citation in enumerate(citation_list, 1):
                            citation_docs.append({
                                "question_id": question_id,
                                "correlation_id": correlation_id,
                                "citation_rank": rank,
                                "chunk_id": citation['chunk_id'],
                                "document_id": citation['document_id'],
                                "page": citation['page'],
                                "rerank_score": citation['score'],
                                "ts": datetime.now(),
                                "used_in_generation": True
                            })
                        
                        if citation_docs:
                            self.mongo_repo.db['citation_log'].insert_many(citation_docs)
                            logger.info(f"Logged {len(citation_docs)} citations to citation_log")
                    
                except Exception as e:
                    logger.exception(f"Failed to log generation to MongoDB: {e}")
                
            except Exception as e:
                logger.exception("Gemini call failed after retries.")
                
                # LOG FAILED GENERATION TOO
                try:
                    self.mongo_repo.db['generation_log'].insert_one({
                        "question_id": question_id,
                        "correlation_id": correlation_id,
                        "question_text": question,
                        "raw_question": raw_question,
                        "document_id": document_id,
                        "generated_answer": None,
                        "citations": [],
                        "model_name": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
                        "error": str(e)[:500],
                        "status": "failed",
                        "latency_ms": int((time.time() - start_time) * 1000),
                        "ts": datetime.now(),
                        "user_id": None,
                    })
                except Exception as log_err:
                    logger.error(f"Failed to log generation error: {log_err}")
                
                citations = [
                    {"document_id": r['payload']['document_id'], "page": r['payload'].get('page', 'N/A'), "chunk_id": r['id']}
                    for r in top_results
                ]
                return Response(
                    {
                        "answer": "Xin lỗi, dịch vụ tạo câu trả lời đang tạm thời không khả dụng. Vui lòng thử lại sau.",
                        "citations": citations,
                        "debug": {"llm_error": str(e)[:200]}
                    },
                    status=status.HTTP_200_OK
                )

            # Step 6: Extract citations and return response
            citations = [
                {
                    "document_id": result['payload']['document_id'],
                    "page": result['payload'].get('page', 'N/A'),
                    "chunk_id": result['id']
                }
                for result in top_results
            ]
            logger.debug(f"Mapped {len(citations)} citations successfully.")

            # IMPROVEMENT 5: Add comprehensive metrics in response
            total_latency = int((time.time() - start_time) * 1000)
            
            return Response({
                "answer": response,
                "citations": citations,
                "debug": {
                    "retrieval_source": source,
                    "retrieval_metrics": {
                        "initial_count": initial_count,
                        "after_filtering": len(search_result),
                        "after_reranking": len(filtered_results),
                        "final_top_k": len(top_results),
                        "chunks_filtered_out": initial_count - len(search_result)
                    },
                    "used_threshold": used_threshold,
                    "document_filter": document_id,
                    "expansion_stats": expansion_stats,
                    "total_latency_ms": total_latency
                }
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.exception(f"Unexpected error in post method: {e}")
            return Response(
                {"error": f"An unexpected error occurred: {str(e)[:200]}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
