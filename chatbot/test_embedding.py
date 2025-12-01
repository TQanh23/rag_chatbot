"""
Test script to verify the embedding model works correctly.
Run: python test_embedding.py
"""
import os
import sys

# Setup Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")

import django
django.setup()

from backend.utils.embeddings import HuggingfaceEmbeddingsModel

def test_embedding():
    print("Loading model...")
    model = HuggingfaceEmbeddingsModel()
    print(f"Model loaded: {model.model_name}")
    print(f"Device: {model.device}")
    
    # Test texts
    test_texts = [
        "Xin chào, đây là một câu tiếng Việt.",
        "Hello, this is an English sentence.",
        "Tài liệu hướng dẫn sử dụng phần mềm.",
    ]
    
    print("\nTesting with sample texts:")
    for i, text in enumerate(test_texts):
        print(f"  {i+1}. {text[:50]}...")
    
    try:
        embeddings = model.encode(test_texts, convert_to_numpy=True)
        print(f"\n✓ Success! Generated embeddings with shape: {embeddings.shape}")
        print(f"  Embedding dimension: {embeddings.shape[1]}")
        
        # Test single text
        print("\nTesting single text embedding...")
        single_emb = model.encode(["Test single"], convert_to_numpy=True)
        print(f"✓ Single embedding shape: {single_emb.shape}")
        
        # Test with edge cases
        print("\nTesting edge cases...")
        edge_cases = [
            "a",  # Very short
            "   spaces   ",  # Whitespace
            "Special chars: @#$%^&*()",  # Special characters
            "Số 123 và ký tự đặc biệt: áàảãạ",  # Vietnamese with numbers
        ]
        edge_emb = model.encode(edge_cases, convert_to_numpy=True)
        print(f"✓ Edge cases embeddings shape: {edge_emb.shape}")
        
        # Test with a longer text (simulating a chunk)
        print("\nTesting with longer text (chunk simulation)...")
        long_text = """Đây là một đoạn văn bản dài hơn để kiểm tra khả năng xử lý của mô hình embedding.
        Văn bản này chứa nhiều câu và có thể có các ký tự đặc biệt như: dấu ngoặc (), dấu phẩy, dấu chấm.
        Mục đích là để đảm bảo rằng mô hình có thể xử lý các đoạn văn bản thực tế từ tài liệu."""
        long_emb = model.encode([long_text], convert_to_numpy=True)
        print(f"✓ Long text embedding shape: {long_emb.shape}")
        
        # Test tokenizer directly
        print("\nTesting tokenizer directly...")
        tokens = model.tokenizer("Test text", return_tensors='pt')
        print(f"✓ Tokenizer output keys: {tokens.keys()}")
        print(f"  Input IDs shape: {tokens['input_ids'].shape}")
        
        print("\n✓ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_embedding()
    sys.exit(0 if success else 1)