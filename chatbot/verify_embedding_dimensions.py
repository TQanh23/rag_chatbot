import django
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

from backend.utils.embeddings import HuggingfaceEmbeddingsModel
from backend.utils.qdrant_client import get_qdrant_client
from django.conf import settings

def verify_dimensions():
    # Check embedding model dimension
    model = HuggingfaceEmbeddingsModel()
    test_embedding = model.embed_texts(["test"])[0]
    embedding_dim = len(test_embedding)
    
    # Check Qdrant collection dimension
    client = get_qdrant_client()
    try:
        collection_info = client.get_collection(settings.QDRANT_COLLECTION)
        
        # Debug: print the structure to see what we're working with
        print(f"Collection info type: {type(collection_info)}")
        print(f"Collection config type: {type(collection_info.config)}")
        
        # Handle different Qdrant API versions
        if hasattr(collection_info.config, 'params'):
            # Newer API structure
            vectors_config = collection_info.config.params.vectors
            
            # Check if it's a dict (named vectors) or VectorParams object
            if isinstance(vectors_config, dict):
                # Named vectors - get the 'default' vector config
                qdrant_dim = vectors_config.get('default', {}).get('size') or vectors_config.get('default').size
            else:
                # Single vector config
                qdrant_dim = vectors_config.size
        else:
            # Older API structure
            qdrant_dim = collection_info.config.vector_size
        
        print(f"\n=== Dimension Check ===")
        print(f"Embedding model: {os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')}")
        print(f"Embedding dimension: {embedding_dim}")
        print(f"Qdrant collection: {settings.QDRANT_COLLECTION}")
        print(f"Qdrant dimension: {qdrant_dim}")
        
        if embedding_dim != qdrant_dim:
            print(f"\n⚠️ DIMENSION MISMATCH DETECTED!")
            print(f"Embedding produces {embedding_dim}-dim vectors")
            print(f"Qdrant expects {qdrant_dim}-dim vectors")
            print(f"\nFix options:")
            print(f"1. Recreate Qdrant collection: python recreate_collection.py")
            print(f"2. Change EMBEDDING_MODEL in .env to match {qdrant_dim} dimensions")
            return False
        else:
            print("\n✓ Dimensions match correctly")
            return True
            
    except Exception as e:
        print(f"Error checking Qdrant: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    verify_dimensions()