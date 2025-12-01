"""
Test script to debug the chunking tokenizer issue.
Run: python test_chunking.py
"""
import os
import sys

# Setup Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")

import django
django.setup()

from transformers import AutoTokenizer
from api.tools.chunking import clean_and_chunk_text

def test_chunking():
    model_name = os.getenv("EMBEDDING_MODEL", "dangvantuan/vietnamese-embedding")
    print(f"Loading tokenizer: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Tokenizer loaded: {type(tokenizer)}")
    
    # Test tokenizer directly
    print("\nTesting tokenizer.encode()...")
    test_texts = [
        "Hello world",
        "Xin chào",
        "",  # Empty
        "   ",  # Whitespace
        "Test với tiếng Việt và số 123",
    ]
    
    for text in test_texts:
        try:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            print(f"  '{text[:30]}...' -> {len(tokens)} tokens")
        except Exception as e:
            print(f"  '{text[:30]}...' -> ERROR: {e}")
    
    # Simulate raw blocks from PDF
    print("\nTesting chunking with sample blocks...")
    raw_blocks = [
        {
            "text": "Đây là đoạn văn bản thử nghiệm. Nó chứa nhiều câu để kiểm tra.",
            "page": 1,
            "document_id": "test_doc_123"
        },
        {
            "text": "Another paragraph with English text. This should work fine.",
            "page": 2,
            "document_id": "test_doc_123"
        },
        {
            "text": "",  # Empty block
            "page": 3,
            "document_id": "test_doc_123"
        },
    ]
    
    try:
        chunks = clean_and_chunk_text(raw_blocks, tokenizer)
        print(f"\n✓ Generated {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i}: {len(chunk['text'])} chars, preview: '{chunk['text'][:50]}...'")
        return True
    except Exception as e:
        print(f"\n✗ Chunking failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_chunking()
    sys.exit(0 if success else 1)