import os
import django
from qdrant_client.models import VectorParams, Distance
from backend.utils.qdrant_client import get_qdrant_client
from django.conf import settings

# Set up Django environment
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

def recreate_collection():
    client = get_qdrant_client()
    collection_name = settings.QDRANT_COLLECTION

    # Delete the existing collection if it exists
    try:
        client.delete_collection(collection_name=collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except Exception as e:
        print(f"Error deleting collection (if it existed): {e}")

    # Create the collection with the correct vector configuration
    # Using 768 dimensions for dangvantuan/vietnamese-embedding model
    vector_size = int(os.getenv("EMBEDDING_DIM", "768"))
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "default": VectorParams(
                    size=vector_size,  # 768 for dangvantuan/vietnamese-embedding
                    distance=Distance.COSINE  # Use cosine similarity
                )
            }
        )
        print(f"Created collection '{collection_name}' with {vector_size}-dim vectors (named vector field 'default').")
    except Exception as e:
        print(f"Error creating collection: {e}")

if __name__ == "__main__":
    recreate_collection()