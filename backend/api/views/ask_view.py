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

logger = logging.getLogger(__name__)

class AskView(APIView):
    def post(self, request):
        # Step 1: Normalize question
        question = request.data.get("question", "").strip()
        if not question:
            logger.error("No question provided in the request.")
            return Response({"error": "Question is required."}, status=status.HTTP_400_BAD_REQUEST)
        logger.info(f"Received question: {question}")

        # Step 2: Embed the question
        embedding_model = HuggingfaceEmbeddingsModel('all-MiniLM-L6-v2')
        question_embedding = embedding_model.embed_texts([question])[0]
        logger.debug("Question embedding generated successfully.")

        # Step 3: Retrieve from Qdrant
        client = get_qdrant_client()
        collection_name = settings.QDRANT_COLLECTION

        # Get top_k from query parameters or default to 20
        top_k = int(request.query_params.get("top_k", 20))
        logger.info(f"Retrieving top {top_k} chunks from Qdrant.")

        try:
            search_result = client.search(
                collection_name=collection_name,
                query_vector=("default", question_embedding),  # Use named vector format
                limit=top_k,
                with_payload=True,
                score_threshold=0.5
            )
            logger.debug(f"Retrieved {len(search_result)} results from Qdrant.")
            # Log the retrieved results
            for i, result in enumerate(search_result):
                logger.debug(f"Result {i + 1}: ID={result.id}, Score={result.score}, Payload={result.payload}")
        except Exception as e:
            logger.exception("Error during Qdrant search.")
            return Response({"error": f"Error during Qdrant search: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # DEBUG: show search results
        logger.debug(f"Search results: {[r.payload['text'][:100] + '...' for r in search_result]}")

        # Step 4: Rerank results using a cross-encoder
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        rerank_inputs = [(question, result.payload['text']) for result in search_result]
        rerank_scores = reranker.predict(rerank_inputs)
        logger.debug("Reranking completed successfully.")

        # Log the rerank inputs
        logger.debug(f"Rerank inputs: {[(question, result.payload['text'][:100]) for result in search_result]}")

        # Log the rerank scores
        logger.debug(f"Rerank scores: {rerank_scores}")

        # Combine results with scores and sort by relevance
        reranked_results = [
            {
                "id": result.id,
                "score": rerank_score,
                "payload": result.payload
            }
            for result, rerank_score in zip(search_result, rerank_scores)
        ]
        reranked_results.sort(key=lambda x: x['score'], reverse=True)

        # Keep top 5–8 results
        top_results = reranked_results[:8]
        logger.info(f"Top {len(top_results)} results selected after reranking.")

        # DEBUG: show reranked results
        logger.debug(f"Reranked results: {[r['payload']['text'][:100] + '...' for r in top_results]}")

        # Log the reranked results
        for i, result in enumerate(reranked_results):
            logger.debug(f"Reranked Result {i + 1}: ID={result['id']}, Score={result['score']}, Payload={result['payload']}")

        # Step 5: Assemble prompt in Vietnamese
        system_message = (
            "Bạn là một trợ lý hữu ích. "
            "Hãy trả lời câu hỏi DỰA TRÊN ngữ cảnh được cung cấp bên dưới. "
            "Nếu ngữ cảnh chứa thông tin liên quan, hãy sử dụng nó để trả lời câu hỏi. "
            "Nếu không tìm thấy câu trả lời trong ngữ cảnh, hãy nói rõ 'Tôi không tìm thấy thông tin trong tài liệu'. "
            "Khi trả lời, hãy trích dẫn nguồn bằng mã tài liệu và số trang (ví dụ: [doc1 tr.12])."
        )

        context_blocks = []
        for result in top_results:
            ref = f"[{result['payload']['document_id']} tr.{result['payload'].get('page', '?')} #{result['id']}]"
            context_blocks.append(f"{ref}\n{result['payload']['text']}\n")
        context_text = "\n".join(context_blocks)

        # Step 6: Generate response using a chat/completions model
        google_api_key = os.getenv("GEMINI_API_KEY")  # Retrieve API key from environment variables
        if not google_api_key:
            logger.error("Google API key is not configured.")
            return Response({"error": "Google API key is not configured."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        generativeai.configure(api_key=google_api_key)

        # DEBUG: show context sent to LLM
        logger.debug(f"Context sent to LLM: {context_text[:500]}...")

        try:
            model = generativeai.GenerativeModel('gemini-1.5-flash')
            response_obj = model.generate_content(
                f"{system_message}\n\nNGỮ CẢNH:\n{context_text}\n\nCâu hỏi: {question}\nCâu trả lời:",
                generation_config=generativeai.types.GenerationConfig(
                    temperature=0.2,  # Optional, adjust as needed
                    max_output_tokens=512  # Optional, adjust as needed
                )
            )
            # Validate response
            if not response_obj or not hasattr(response_obj, 'text') or not response_obj.text.strip():
                logger.error("Invalid response from Gemini API.")
                return Response({"error": "Invalid response from Gemini API."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            response = response_obj.text.strip()
            logger.info("Response generated successfully.")
        except Exception as e:
            logger.exception("Error during response generation.")
            return Response({"error": f"An error occurred while generating the response: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

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

        return Response({"answer": response, "citations": citations}, status=status.HTTP_200_OK)

    def retrieve_items(self, query):
        embedding_model = HuggingfaceEmbeddingsModel('all-MiniLM-L6-v2')
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