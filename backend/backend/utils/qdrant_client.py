from qdrant_client import QdrantClient
from django.conf import settings

_qdrant_client = None

def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        # Use Qdrant embedded instance with the specified storage path
        _qdrant_client = QdrantClient(path=settings.QDRANT_EMBEDDED_PATH)
         # Connect to Qdrant server running in Docker
        # _qdrant_client = QdrantClient(
        #     url=settings.QDRANT_URL,
        #     api_key=settings.QDRANT_API_KEY if settings.QDRANT_API_KEY else None
        # )
    return _qdrant_client