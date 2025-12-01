import os
from pathlib import Path
from dotenv import load_dotenv

# load .env next to this file
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

from google import generativeai
generativeai.configure(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
print([m.name for m in generativeai.list_models()])