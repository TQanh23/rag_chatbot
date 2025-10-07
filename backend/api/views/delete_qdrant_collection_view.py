from rest_framework.views import APIView
from rest_framework.response import Response
from backend.utils.qdrant_client import get_qdrant_client
from django.conf import settings

class DeleteQdrantCollectionView(APIView):
    def post(self, request):
        client = get_qdrant_client()
        try:
            client.delete_collection(settings.QDRANT_COLLECTION)
            return Response({"message": "Collection deleted."})
        except Exception as e:
            return Response({"error": str(e)}, status=500)