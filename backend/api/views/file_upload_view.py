from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from backend.utils.compute_file_hash import compute_file_hash
from documents.models import Document
from django.core.files.storage import default_storage
from backend.utils.qdrant_client import get_qdrant_client
from backend.utils.embeddings import HuggingfaceEmbeddingsModel
from qdrant_client.http.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue
from transformers import AutoTokenizer
import os
import uuid
import nltk
import fitz
from django.conf import settings
import logging

# new import
from ..tools.chunk_eval import eval_chunks
from ..tools.export_chunks import write_chunks_csv

logger = logging.getLogger(__name__)

class FileUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    embeddings_model = HuggingfaceEmbeddingsModel('all-MiniLM-L6-v2')  # Initialize once
    
    def delete_points_for_document(self, client, collection, document_id):
        """Delete all vector points for a specific document_id"""
        flt = Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])
        client.delete(collection_name=collection, points_selector=flt)
        logger.info(f"Deleted existing vector points for document_id: {document_id}")

    def post(self, request, *args, **kwargs):
        # Ensure NLTK 'punkt' is available
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        file = request.FILES.get('file')
        if not file:
            return Response(
                {"error": "No file uploaded. Make sure you use Body -> form-data and key name 'file' (type: File)."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Log file details
        logger.debug(f"Received file: name={file.name}, content_type={getattr(file, 'content_type', None)}, size={file.size}")

        # Basic validation
        allowed = [
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
        ]
        if file.content_type not in allowed:
            return Response(
                {"error": "Unsupported file type", "content_type": file.content_type},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Save the file temporarily
        saved_path = default_storage.save(f"uploads/{file.name}", file)
        logger.debug(f"Saved uploaded file to: {saved_path}")

        # Compute hash from the saved file
        saved_file_path = default_storage.path(saved_path) if hasattr(default_storage, 'path') else os.path.join(settings.MEDIA_ROOT, saved_path)
        with open(saved_file_path, "rb") as f:
            file_hash = compute_file_hash(f)

        # Check for force parameter - allows reingestion of the same file
        force = request.query_params.get("force") == "true"
        
        # Check for duplicates
        existing_doc = Document.objects.filter(hash=file_hash).first()
        if existing_doc and not force:
            default_storage.delete(saved_path)  # Clean up the temporary file
            return Response({
                'document_id': existing_doc.document_id,
                'filename': existing_doc.filename,
                'mimetype': existing_doc.mimetype,
                'size': existing_doc.size,
                'hash': existing_doc.hash,
            })

        # If forcing, reuse the same document_id (hash) and delete existing points
        if existing_doc and force:
            client = get_qdrant_client()
            collection_name = settings.QDRANT_COLLECTION
            # Remove any leftover points (safe even if collection doesn't exist)
            try:
                self.delete_points_for_document(client, collection_name, existing_doc.document_id)
            except Exception as e:
                logger.warning(f"Could not delete existing points: {str(e)}")

            # Reset counters on the existing document row
            existing_doc.token_count = 0
            existing_doc.vector_dim = None
            existing_doc.save()
            document = existing_doc
            logger.info(f"Force reingestion of document: {document.document_id}")
        else:
            # Save metadata (new document)
            document = Document.objects.create(
                document_id=file_hash,
                filename=file.name,
                mimetype=file.content_type,
                size=file.size,
                hash=file_hash,
            )

        try:
            # Extract text and metadata based on file type
            if file.content_type == 'application/pdf':
                raw_blocks = self.extract_text_and_metadata_from_pdf(saved_file_path, document.document_id)
            elif file.content_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                raw_blocks = self.extract_text_and_metadata_from_docx(saved_file_path, document.document_id)
            elif file.content_type == 'text/plain':
                raw_blocks = self.extract_text_and_metadata_from_txt(saved_file_path, document.document_id)
            else:
                raise ValueError("Unsupported file type")

            # Clean and chunk text
            from ..tools.chunking import clean_and_chunk_text
            chunks = clean_and_chunk_text(raw_blocks, self.tokenizer)
            logger.debug(f"Chunks generated: {chunks}")
            logger.debug(f"Number of chunks: {len(chunks)}")

            # Evaluate chunking quality before upload
            try:
                chunk_stats = eval_chunks(chunks, self.tokenizer, max_tokens=512)
                logger.info(f"Chunk evaluation: {chunk_stats}")
            except Exception as eval_exc:
                logger.exception("Chunk evaluation failed")
                chunk_stats = {"error": str(eval_exc)}

            # Generate embeddings using Hugging Face model
            chunk_texts = [chunk['text'] for chunk in chunks]
            embeddings = self.embeddings_model.embed_texts(chunk_texts)  # Batch embedding
            logger.debug(f"Embeddings generated: {embeddings}")
            logger.debug(f"Number of embeddings: {len(embeddings)}")

            # Ensure Qdrant collection exists
            client = get_qdrant_client()
            collection_name = settings.QDRANT_COLLECTION
            try:
                client.get_collection(collection_name)
            except Exception:
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "default": VectorParams(
                            size=len(embeddings[0]),
                            distance=Distance.COSINE
                        )
                    }
                )

            # Upload chunks and embeddings to Qdrant
            points = [
                PointStruct(
                    id=str(uuid.uuid4()),  # Generate a valid UUID for each point
                    vector={"default": embedding},  # Use named vector with "default"
                    payload=chunk
                )
                for embedding, chunk in zip(embeddings, chunks)
            ]
            logger.debug(f"Points being upserted: {points}")
            logger.debug(f"Number of points: {len(points)}")

            response = client.upsert(
                collection_name=collection_name,
                points=points
            )
            logger.debug(f"Upsert response: {response}")

            # Update document status and store counts
            document.token_count = sum(len(self.tokenizer.encode(chunk['text'], add_special_tokens=False)) for chunk in chunks)
            document.vector_dim = len(embeddings[0])
            document.save()

            # Write per-document CSV of chunks to MEDIA_ROOT/exports/{document_id}_chunks.csv
            try:
                csv_rel = f"exports/{document.document_id}_chunks.csv"
                csv_abs = os.path.join(settings.MEDIA_ROOT, csv_rel)
                # ensure exports directory exists
                os.makedirs(os.path.dirname(csv_abs), exist_ok=True)
                write_chunks_csv(chunks, csv_abs)
                logger.info(f"Wrote per-document chunks CSV: {csv_rel}")
            except Exception as csv_exc:
                logger.exception(f"Failed to write per-document CSV for {document.document_id}: {str(csv_exc)}")

            # Clean up the temporary file
            default_storage.delete(saved_path)

            # return chunk evaluation stats for quick validation (remove in production)
            return Response({"message": "File uploaded and processed successfully", "document_id": document.document_id, "chunk_eval": chunk_stats}, status=status.HTTP_201_CREATED)

        except Exception as e:
            # Clean up the temporary file in case of an error
            if default_storage.exists(saved_path):
                default_storage.delete(saved_path)
            return Response({"error": str(e)}, status=500)

    def extract_text_and_metadata_from_pdf(self, file_path, document_id):
        """Extract text and metadata from a PDF."""
        raw_blocks = []
        with fitz.open(file_path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text()
                raw_blocks.append({
                    "text": text,
                    "page": page_number,
                    "document_id": document_id
                })
        return raw_blocks

    def extract_text_and_metadata_from_docx(self, file_path, document_id):
        """Extract text and metadata from a DOCX file."""
        from docx import Document as DocxDocument
        doc = DocxDocument(file_path)
        raw_blocks = []
        for i, paragraph in enumerate(doc.paragraphs, start=1):
            if paragraph.text.strip():  # Skip empty paragraphs
                raw_blocks.append({
                    "text": paragraph.text,
                    "section": f"Paragraph {i}",
                    "document_id": document_id
                })
        return raw_blocks

    def extract_text_and_metadata_from_txt(self, file_path, document_id):
        """Extract text and metadata from a TXT file."""
        raw_blocks = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file, start=1):
                if line.strip():  # Skip empty lines
                    raw_blocks.append({
                        "text": line.strip(),
                        "section": f"Line {i}",
                        "document_id": document_id
                    })
        return raw_blocks

