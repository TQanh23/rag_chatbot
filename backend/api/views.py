from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Distance, VectorParams
from django.conf import settings
import os
from backend.utils.qdrant_client import get_qdrant_client

class AskView(APIView):
    def post(self, request):
        question = request.data.get("question")
        return Response({"answer": f"Echo: {question}"})

class FileUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        file = request.FILES.get('file')
        if not file:
            return Response({"error": "No file provided"}, status=status.HTTP_400_BAD_REQUEST)
        
        # Save the file to the media directory
        with open(f'media/{file.name}', 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)
        
        return Response({"message": "File uploaded successfully"}, status=status.HTTP_201_CREATED)

class PDFUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        file = request.FILES.get('file')
        if not file:
            return Response({"error": "No file provided"}, status=status.HTTP_400_BAD_REQUEST)

        # Save the file temporarily
        file_path = os.path.join(settings.MEDIA_ROOT, file.name)
        with open(file_path, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)

        try:
            # Extract text from the PDF
            text = self.extract_text_from_pdf(file_path)
            print(f"Extracted text: {text[:500]}")  # Print the first 500 characters of the text

            # Chunk the text
            chunks = self.chunk_text(text)
            print(f"Chunks: {chunks}")  # Print the chunks

            # Generate embeddings
            model = SentenceTransformer('all-MiniLM-L6-v2')  # Use a lightweight embedding model
            embeddings = model.encode(chunks)
            print(f"Embeddings shape: {len(embeddings)} vectors of size {len(embeddings[0])}")  # Print embedding info

            # Upsert into Qdrant Embedded
            client = get_qdrant_client()
            points = [
                PointStruct(id=i, vector={"default": embedding.tolist()}, payload={"text": chunk})
                for i, (embedding, chunk) in enumerate(zip(embeddings, chunks))
            ]
            print(f"Points to upsert: {points}")  # Print the points being upserted

            client.upsert(collection_name=settings.QDRANT_COLLECTION, points=points)

            # Clean up the temporary file
            os.remove(file_path)

            return Response({"message": "PDF uploaded and processed successfully"}, status=status.HTTP_201_CREATED)

        except Exception as e:
            # Clean up the temporary file in case of an error
            if os.path.exists(file_path):
                os.remove(file_path)
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def extract_text_from_pdf(self, file_path):
        """Extract text from a PDF using PyMuPDF."""
        text = ""
        with fitz.open(file_path) as pdf:
            for page in pdf:
                text += page.get_text()
        return text

    def chunk_text(self, text, chunk_size=500):
        """Split text into smaller chunks."""
        words = text.split()
        chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        return chunks
class InitQdrantView(APIView):
    def get(self, request):
        client = get_qdrant_client()

        try:
            client.get_collection(settings.QDRANT_COLLECTION)
            return Response({"message": f"Qdrant collection '{settings.QDRANT_COLLECTION}' already exists."})
        except Exception:
            client.recreate_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config={
                    "default": VectorParams(
                        size=384,  # Adjust based on your embedding model
                        distance=Distance.COSINE  # Use "COSINE", "EUCLID", or "DOT" based on your use case
                    )
                }
            )
            return Response({"message": f"Qdrant collection '{settings.QDRANT_COLLECTION}' created successfully."})
class DeleteQdrantCollectionView(APIView):
    def post(self, request):
        client = get_qdrant_client()
        try:
            client.delete_collection(settings.QDRANT_COLLECTION)
            return Response({"message": "Collection deleted."})
        except Exception as e:
            return Response({"error": str(e)}, status=500)