from qdrant_client import QdrantClient
from django.conf import settings

_qdrant_client = None

def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(path=settings.QDRANT_EMBEDDED_PATH)
    return _qdrant_client