import requests
import os
from sentence_transformers import SentenceTransformer

class GeminiEmbeddingsModel:
    def __init__(self, model_name, api_url=None, api_key=None):
        """
        Initialize the Gemini Embeddings Model.
        :param model_name: Name of the embedding model (e.g., 'text-embedding-004').
        :param api_url: Base URL of the Gemini API.
        :param api_key: API key for authentication.
        """
        self.model_name = model_name
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        self.api_url = api_url or "https://api.gemini.com/v1/embeddings"
    def embed_texts(self, texts):
        """
        Generate embeddings for a list of texts.
        :param texts: List of strings to embed.
        :return: List of embeddings (List[List[float]]).
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "inputs": texts
        }

        response = requests.post(self.api_url, json=payload, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Failed to generate embeddings: {response.text}")

        data = response.json()
        return data.get("embeddings", [])
class HuggingfaceEmbeddingsModel:
    def __init__(self, model_name):
        """
        Initialize the Hugging Face Embeddings Model.
        :param model_name: Name of the embedding model (e.g., 'all-MiniLM-L6-v2').
        """
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts):
        """
        Generate embeddings for a list of texts.
        :param texts: List of strings to embed.
        :return: List of embeddings (List[List[float]]).
        """
        return self.model.encode(texts, convert_to_numpy=True).tolist()