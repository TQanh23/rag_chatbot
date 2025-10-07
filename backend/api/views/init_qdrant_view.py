from rest_framework.views import APIView
from rest_framework.response import Response
from backend.utils.qdrant_client import get_qdrant_client
from django.conf import settings
from qdrant_client.http.models import VectorParams, Distance

class InitQdrantView(APIView):
    def get(self, request):
        client = get_qdrant_client()
        try:
            client.get_collection(settings.QDRANT_COLLECTION)
            return Response({"message": f"Collection '{settings.QDRANT_COLLECTION}' already exists."})
        except Exception:
            client.recreate_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config={
                    "default": VectorParams(size=384, distance=Distance.COSINE)
                }
            )
            return Response({"message": f"Collection '{settings.QDRANT_COLLECTION}' created successfully."})