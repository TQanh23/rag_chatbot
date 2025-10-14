from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

# Qdrant server URL
QDRANT_SERVER_URL = "http://localhost:6333"

# Initialize Qdrant client in server mode
client = QdrantClient(url=QDRANT_SERVER_URL)

def get_qdrant_client():
    """
    Returns the Qdrant client instance.
    """
    return client

def collection_exists(collection_name: str) -> bool:
    """
    Return True if collection exists, False otherwise.
    Non-destructive: does not create the collection.
    """
    try:
        collections = client.get_collections()
        return any(c.name == collection_name for c in collections.collections)
    except Exception:
        # Fallback to single-collection check if get_collections fails
        try:
            client.get_collection(collection_name=collection_name)
            return True
        except Exception:
            return False

def get_collection_info(collection_name: str):
    """
    Return collection info dict or None if not found.
    """
    try:
        info = client.get_collection(collection_name=collection_name)
        return info.dict()
    except Exception:
        return None

def create_collection(collection_name, vector_size=384, distance=Distance.COSINE):
    """
    Creates a collection in Qdrant if it doesn't already exist.
    """
    client = get_qdrant_client()
    collections = client.get_collections()
    collection_exists = any(collection.name == collection_name for collection in collections.collections)

    if not collection_exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance)
        )
        print(f"Collection '{collection_name}' created successfully.")
    else:
        print(f"Collection '{collection_name}' already exists.")