import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

# Specify the collection name
collection_name = "test_collection"

try:
    # Initialize Qdrant client in server mode
    client = QdrantClient(url="http://localhost:6333")
    print("Connected to Qdrant server.")

    # Check if the collection exists
    collections = client.get_collections()
    collection_exists = any(collection.name == collection_name for collection in collections.collections)

    if not collection_exists:
        print(f"Collection '{collection_name}' does not exist.")
    else:
        print(f"Collection '{collection_name}' already exists.")

    # List all collections
    print("Available collections:")
    for collection in collections.collections:
        print(f"- {collection.name}")

    # Fetch all points from the collection
    response = client.scroll(
        collection_name=collection_name,
        with_payload=True,  # Include payload in the response
        limit=100           # Limit the number of points retrieved
    )

    # Print the retrieved points
    if response[0]:
        print(f"Found {len(response[0])} points in the collection:")
        for point in response[0]:
            print(f"ID: {point.id}")
            print(f"Payload: {point.payload}")
            print("-" * 50)
    else:
        print("No points found in the collection.")

except Exception as e:
    print(f"Error: {str(e)}")