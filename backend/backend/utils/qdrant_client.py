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