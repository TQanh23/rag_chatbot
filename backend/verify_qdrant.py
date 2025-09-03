import os
import django
from qdrant_client.models import PointStruct

# Set up Django environment
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

from backend.utils.qdrant_client import get_qdrant_client
from backend.utils.embeddings import HuggingfaceEmbeddingsModel
from django.conf import settings

client = get_qdrant_client()
collection_name = settings.QDRANT_COLLECTION

def verify_qdrant():
    model = HuggingfaceEmbeddingsModel('all-MiniLM-L6-v2')

    # 1. Verify collection
    try:
        collection_info = client.get_collection(collection_name)
        print(f"Collection '{collection_name}' exists:")
        print(collection_info)
    except Exception as e:
        print(f"Error: {e}")
        return

    # 2. Verify points
    try:
        points = client.scroll(
            collection_name=collection_name,
            limit=10,
            with_vectors=True  # Make sure to request vectors
        )
        
        print(f"\nPoints in collection '{collection_name}':")
        if points[0]:
            for point in points[0]:
                vector_info = "None" if point.vector is None else f"Type: {type(point.vector)}, Length: {len(point.vector) if isinstance(point.vector, (list, dict)) else 'N/A'}"
                print(f"Point ID: {point.id}, Vector: {vector_info}")
                if point.vector is None:
                    print(f"Point {point.id} has no vector!")
        else:
            print("No points found in collection")
    except Exception as e:
        print(f"Error retrieving points: {e}")

    # 3. Test similarity search
    try:
        print("\nTesting similarity search:")
        test_query = "This is a test document"
        test_embedding = model.embed_texts([test_query])[0]
        print(f"Test embedding shape: {len(test_embedding)}")

        search_result = client.search(
            collection_name=collection_name,
            query_vector=("default", test_embedding),  # Use named vector format
            limit=10
        )
        
        print("Similarity search results:")
        if search_result:
            for result in search_result:
                print(f"Point ID: {result.id}, Score: {result.score}")
        else:
            print("No search results found")
    except Exception as e:
        print(f"Error during similarity search: {e}")

if __name__ == "__main__":
    verify_qdrant()