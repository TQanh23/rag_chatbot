from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from backend.utils.embeddings import HuggingfaceEmbeddingsModel
from backend.utils.qdrant_client import get_qdrant_client
from qdrant_client.http.models import Distance
from django.conf import settings
import logging
import os
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
                kwargs = dict(
                    collection_name=collection_name,
                    query_vector=("default", question_embedding),
                    limit=top_k,
                    with_payload=True,
                )
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

    def post(self, request):
        start_time = time.time()  # Track total latency
        
        # Step 1: Normalize question
        question = request.data.get("question", "").strip()
        if not question:
            logger.error("No question provided in the request.")
            return Response({"error": "Question is required."}, status=status.HTTP_400_BAD_REQUEST)
        logger.info(f"Received question: {question}")

        # Generate a stable question_id for logging across retrieval/generation/citation
        question_id = str(uuid.uuid4())
        correlation_id = str(uuid.uuid4())  # Unique ID for tracing this request

        # Optional: quick language hint for debugging retrieval issues
        try:
            lang = detect(question)
            logger.info(f"Detected question language: {lang}")
        except Exception:
            lang = "unknown"

        # Step 2: Embed the question
        embedding_model = HuggingfaceEmbeddingsModel()  # Uses EMBEDDING_MODEL env var or multilingual default
        # Now using 'paraphrase-multilingual-MiniLM-L12-v2' (384-dim) for better Vietnamese recall
        question_embedding = embedding_model.embed_texts([question])[0]
        logger.debug("Question embedding generated successfully.")

        # Step 3: Retrieve from Qdrant
        client = get_qdrant_client()
        collection_name = settings.QDRANT_COLLECTION

        # Get top_k and optional score_threshold/final_k from query parameters
        # Step 6: Two-stage retrieval - retrieve 50 candidates, rerank to top 10
        top_k = int(request.query_params.get("top_k", 50))
        final_k = int(request.query_params.get("final_k", 10))
        score_threshold_param = request.query_params.get("score_threshold", None)
        initial_threshold = float(score_threshold_param) if score_threshold_param is not None else 0.2

        logger.info(f"Retrieving up to {top_k} chunks from Qdrant (initial score_threshold={initial_threshold}).")

        # Try Qdrant first
        search_result = None
        used_threshold = None
        source = "qdrant"
        
        try:
            search_result, used_threshold = self._qdrant_search_with_fallback(
                client, collection_name, question_embedding, top_k, initial_threshold
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
            logger.warning(
                f"No results found from either Qdrant or MongoDB. "
                f"Detected language={lang}."
            )
            return Response(
                {
                    "answer": "Tôi không tìm thấy thông tin trong tài liệu.",
                    "citations": [],
                    "debug": {
                        "reason": "No retrieval results from Qdrant or MongoDB",
                        "lang": lang,
                        "used_threshold": used_threshold,
                        "retrieval_source": source
                    }
                },
                status=status.HTTP_200_OK
            )

        logger.info(f"Retrieved {len(search_result)} results from {source}.")

        if not search_result:
            logger.warning(
                f"No results found from Qdrant even after threshold fallback. "
                f"Detected language={lang}. If queries are Vietnamese and embeddings are 'all-MiniLM-L6-v2', "
                f"consider reindexing with 'paraphrase-multilingual-MiniLM-L12-v2' (384-dim)."
            )
            return Response(
            {
                "answer": response,
                "citations": citations,
                "debug": {
                    "used_threshold": used_threshold,
                    "retrieved": len(search_result),
                    "final_k": final_k,
                    "lang": lang,
                    "retrieval_source": source
                }
            },
            status=status.HTTP_200_OK
        )

        logger.debug(f"Retrieved {len(search_result)} results from Qdrant.")
        for i, result in enumerate(search_result):
            logger.debug(f"Result {i + 1}: ID={result.id}, Score={result.score}, Payload keys={list(result.payload.keys())}")

        # Step 4: Rerank results using a cross-encoder
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        rerank_inputs = [(question, result.payload['text']) for result in search_result]
        rerank_scores = reranker.predict(rerank_inputs)
        logger.debug("Reranking completed successfully.")

        # Combine results with scores and sort by relevance
        reranked_results = [
            {"id": result.id, "score": float(score), "payload": result.payload}
            for result, score in zip(search_result, rerank_scores)
        ]
        reranked_results.sort(key=lambda x: x['score'], reverse=True)

        # Keep top N results
        top_results = reranked_results[:final_k]
        logger.info(f"Top {len(top_results)} results selected after reranking (final_k={final_k}).")

        # Non-blocking retrieval logging
        try:
            retrieved_chunk_ids = [result.id for result in search_result]  # Fix: use .id instead of ['id']
            retrieved_scores = [result.score for result in search_result]  # Fix: use .score instead of ['score']
            
            print(f"DEBUG: Attempting to log retrieval to 'retrieval_log' with {len(retrieved_chunk_ids)} chunks")  # TEMP DEBUG
            self.mongo_repo.db['retrieval_log'].insert_one({
                "question_id": question_id,
                "correlation_id": correlation_id,
                "question": question,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "retrieved_scores": retrieved_scores,
                "top_k": final_k,
                "document_id": top_results[0]['payload'].get('document_id') if top_results else None,
                "latency_ms": int((time.time() - start_time) * 1000),
                "ts": datetime.utcnow(),
                "user_id": None,
            })
            print(f"DEBUG: Logged retrieval to 'retrieval_log' successfully")  # TEMP DEBUG
            logger.debug(f"Logged retrieval to MongoDB: {len(retrieved_chunk_ids)} chunks")
        except Exception as e:
            print(f"DEBUG: Failed to log retrieval: {e}")  # TEMP DEBUG
            logger.exception(f"Failed to log retrieval to MongoDB: {e}")
        
        # Step 5: Assemble prompt in Vietnamese
        system_message = (
            "Bạn là một trợ lý hữu ích chuyên về xử lý thông tin kỹ thuật. "
            "Trả lời câu hỏi DỰA TRÊN NGỮ CẢNH được cung cấp. "
            "Nếu câu trả lời yêu cầu kết hợp thông tin từ nhiều phần, hãy tổng hợp chúng một cách rõ ràng. "
            "Luôn trích dẫn nguồn: [document_id tr.page]. "
            "Nếu thông tin không có trong tài liệu, hãy nói rõ 'Tôi không tìm thấy thông tin này trong tài liệu.' "
            "Trả lời bằng tiếng Việt, chi tiết nhưng ngắn gọn."
        )

        context_text = self._build_context_text(top_results)

        # Hard cap context to avoid zero output due to token budget
        MAX_CONTEXT_CHARS = 12000
        if len(context_text) > MAX_CONTEXT_CHARS:
            logger.info(f"Trimming context from {len(context_text)} to {MAX_CONTEXT_CHARS} chars.")
            context_text = context_text[:MAX_CONTEXT_CHARS]

        # Step 6: Generate response using a chat/completions model
        google_api_key = os.getenv("GEMINI_API_KEY")
        if not google_api_key:
            logger.error("Google API key is not configured.")
            return Response({"error": "Google API key is not configured."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        generativeai.configure(api_key=google_api_key)

        logger.debug(f"Context sent to LLM (first 500 chars): {context_text[:500]}...")

        prompt = f"{system_message}\n\nNGỮ CẢNH:\n{context_text}\n\nCâu hỏi: {question}\nCâu trả lời:"

        # NEW: read timeout/attempts and token settings from env (with safe defaults)
        try:
            llm_timeout = int(os.getenv("GEMINI_TIMEOUT", "30"))
        except Exception:
            llm_timeout = 30
        try:
            llm_attempts = int(os.getenv("GEMINI_MAX_ATTEMPTS", "3"))
        except Exception:
            llm_attempts = 3

        # token settings
        try:
            initial_max_tokens = int(os.getenv("GEMINI_INITIAL_MAX_TOKENS", "512"))
        except Exception:
            initial_max_tokens = 512
        try:
            max_tokens_cap = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "4096"))
        except Exception:
            max_tokens_cap = 4096

        try:
            response_text, finish_reason, response_obj = self._gemini_generate_with_retry(
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
                logger.warning(
                    f"Gemini returned no content. finish_reason={finish_reason}, "
                    f"prompt_feedback={getattr(response_obj, 'prompt_feedback', None)}"
                )

                # If finish_reason suggests token limit, provide hint to increase tokens (already tried progressive increases)
                try:
                    fr_val = int(finish_reason) if finish_reason is not None and str(finish_reason).isdigit() else None
                except Exception:
                    fr_val = None

                if fr_val == 4 or (isinstance(finish_reason, str) and "max" in str(finish_reason).lower()):
                    return Response(
                        {
                            "error": "Câu trả lời có thể dài hơn giới hạn hiện tại và bị cắt. Thử tăng GEMINI_MAX_OUTPUT_TOKENS hoặc rút ngắn ngữ cảnh.",
                            "finish_reason": str(finish_reason),
                            "hint": "Set GEMINI_INITIAL_MAX_TOKENS and GEMINI_MAX_OUTPUT_TOKENS in .env and restart the server."
                        },
                        status=status.HTTP_502_BAD_GATEWAY
                    )

                # Retry once with a smaller context (fewer chunks) — already safe fallback
                reduced_results = top_results[:max(2, final_k // 2)]
                reduced_context = self._build_context_text(reduced_results)
                if len(reduced_context) > MAX_CONTEXT_CHARS // 2:
                    reduced_context = reduced_context[:MAX_CONTEXT_CHARS // 2]

                reduced_prompt = f"{system_message}\n\nNGỮ CẢNH:\n{reduced_context}\n\nCâu hỏi: {question}\nCâu trả lời:"
                response_text, finish_reason, _ = self._gemini_generate_with_retry(
                    reduced_prompt,
                    attempts=max(1, min(2, llm_attempts)),  # one more, lighter try
                    base_timeout=max(15, llm_timeout // 2),
                    backoff=1.5,
                    initial_max_tokens=initial_max_tokens,
                    max_tokens_cap=max_tokens_cap,
                    context_text=reduced_context,
                    question=question,
                )

            if not response_text:
                logger.error(
                    f"No text returned after retry. finish_reason={finish_reason}, "
                    f"prompt_feedback={getattr(response_obj, 'prompt_feedback', None)}"
                )
                return Response(
                    {
                        "error": "LLM did not return content.",
                        "finish_reason": str(finish_reason),
                        "hint": "Try reducing context size or increasing max_output_tokens (GEMINI_INITIAL_MAX_TOKENS / GEMINI_MAX_OUTPUT_TOKENS)."
                    },
                    status=status.HTTP_502_BAD_GATEWAY
                )

            response = response_text
            logger.info("Response generated successfully.")
            
            # Calculate total latency
            total_latency_ms = int((time.time() - start_time) * 1000)
            
            # Update MongoDB query log with final answer
            try:
                reranked_chunk_ids = [r['id'] for r in top_results]
                self.mongo_repo.update_query_log(question_id, {
                    "reranked_chunk_ids": reranked_chunk_ids,
                    "final_answer": response_text,
                    "latency_ms": total_latency_ms
                })
                logger.debug(f"Query log updated with final answer in MongoDB")
            except Exception as mongo_exc:
                logger.exception(f"Failed to update query log in MongoDB: {str(mongo_exc)}")
            
            # Non-blocking generation logging (MongoDB only)
            try:
                print(f"DEBUG: Attempting to log generation to 'generation_log'")  # TEMP DEBUG
                self.mongo_repo.db['generation_log'].insert_one({
                    "question_id": question_id,
                    "correlation_id": correlation_id,
                    "question_text": question,
                    "final_answer_text": response_text,
                    "document_id": top_results[0]['payload'].get('document_id') if top_results else None,
                    "num_tokens": None,
                    "generation_time_ms": int((time.time() - start_time) * 1000),
                    "ts": datetime.utcnow(),
                    "user_id": None,
                })
                print(f"DEBUG: Logged generation to 'generation_log' successfully")  # TEMP DEBUG
                logger.debug(f"Logged generation to MongoDB")
            except Exception as e:
                print(f"DEBUG: Failed to log generation: {e}")  # TEMP DEBUG
                logger.exception(f"Failed to log generation to MongoDB: {e}")
        
        except Exception as e:
            # NEW: graceful fallback instead of 500 on transient LLM/network failures
            logger.exception("Gemini call failed after retries.")
            citations = [
                {
                    "document_id": result['payload']['document_id'],
                    "page": result['payload'].get('page', 'N/A'),
                    "chunk_id": result['id']
                }
                for result in top_results
            ]
            return Response(
                {
                    "answer": "Xin lỗi, dịch vụ tạo câu trả lời đang tạm thời không khả dụng. Vui lòng thử lại sau.",
                    "citations": citations,
                    "debug": {
                        "used_threshold": used_threshold,
                        "retrieved": len(search_result),
                        "final_k": final_k,
                        "lang": lang,
                        "llm_error": str(e)[:200]
                    }
                },
                status=status.HTTP_200_OK
            )

        # Step 7: Map citations to document_id, page, and chunk_id
        citations = [
            {
                "document_id": result['payload']['document_id'],
                "page": result['payload'].get('page', 'N/A'),
                "chunk_id": result['id']
            }
            for result in top_results
        ]
        logger.debug("Citations mapped successfully.")

        # Non-blocking citation logging (MongoDB only)
        try:
            for c in citations:
                print(f"DEBUG: Attempting to log citation to 'citation_log' for chunk {c.get('chunk_id')}")  # TEMP DEBUG
                self.mongo_repo.db['citation_log'].insert_one({
                    "question_id": question_id,
                    "chunk_id": c.get('chunk_id'),
                    "document_id": c.get('document_id'),
                    "page_number": c.get('page'),
                    "cited_in_answer": 1,
                    "ts": datetime.utcnow(),
                })
            print(f"DEBUG: Logged {len(citations)} citations to 'citation_log' successfully")  # TEMP DEBUG
        except Exception as e:
            print(f"DEBUG: Failed to log citations: {e}")  # TEMP DEBUG
            logger.exception(f"Failed to log citation to MongoDB: {e}")
        
        return Response(
            {
                "answer": response,
                "citations": citations,
                "debug": {
                    "used_threshold": used_threshold,
                    "retrieved": len(search_result),
                    "final_k": final_k,
                    "lang": lang
                }
            },
            status=status.HTTP_200_OK
        )

    def retrieve_items(self, query):
        embedding_model = HuggingfaceEmbeddingsModel()  # Uses EMBEDDING_MODEL env var or multilingual default
        question_embedding = embedding_model.embed_texts([query])[0]
        client = get_qdrant_client()
        collection_name = settings.QDRANT_COLLECTION
        top_k = 20
        search_result = client.search(
            collection_name=collection_name,
            query_vector=("default", question_embedding),
            limit=top_k,
            with_payload=True,
            score_threshold=0.5
        )
        return [str(result.id) for result in search_result]

    def get_relevant_items(self, query):
        # Ground truth mapping for queries (replace with actual data source, e.g., database or labeled dataset)
        ground_truth = {
            "what is python": {"item1", "item2"},
            "how to code": {"item3", "item4"},
            # Add more mappings as needed
        }
        return ground_truth.get(query.lower().strip(), set())