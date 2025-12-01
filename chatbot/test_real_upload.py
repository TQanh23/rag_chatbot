"""
Test with actual file content to find the problematic text.
"""
import os
import sys
import glob

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")

import django
django.setup()

from transformers import AutoTokenizer
from backend.utils.embeddings import HuggingfaceEmbeddingsModel
from api.tools.chunking import clean_and_chunk_text
import fitz

def test_with_file(file_path):
    model_name = os.getenv("EMBEDDING_MODEL", "dangvantuan/vietnamese-embedding")
    
    print(f"Testing with file: {file_path}")
    print(f"Model: {model_name}")
    print("=" * 60)
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    embeddings_model = HuggingfaceEmbeddingsModel(model_name)
    
    # Extract text from PDF
    raw_blocks = []
    print("\n1. Extracting text from PDF...")
    
    with fitz.open(file_path) as pdf:
        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text()
            if text.strip():
                raw_blocks.append({
                    "text": text,
                    "page": page_num,
                    "document_id": "test_doc"
                })
                print(f"   Page {page_num}: {len(text)} chars")
    
    print(f"   Total: {len(raw_blocks)} pages with text")
    
    # Chunk
    print("\n2. Chunking...")
    try:
        chunks = clean_and_chunk_text(raw_blocks, tokenizer)
        print(f"   Generated {len(chunks)} chunks")
    except Exception as e:
        print(f"   ERROR during chunking: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Filter
    print("\n3. Filtering chunks...")
    chunks = [c for c in chunks if c.get('text') and c['text'].strip()]
    print(f"   {len(chunks)} valid chunks after filtering")
    
    if not chunks:
        print("   ERROR: No valid chunks!")
        return False
    
    # Test embedding each chunk
    print("\n4. Testing embedding for each chunk...")
    failed_chunks = []
    
    for i, chunk in enumerate(chunks):
        text = chunk['text']
        try:
            # Test tokenization first
            tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=256)
            
            # Test embedding
            emb = embeddings_model.encode([text], convert_to_numpy=True)
            
            if i < 3 or i == len(chunks) - 1:
                print(f"   Chunk {i}: OK ({len(text)} chars, {emb.shape[1]} dim)")
        except Exception as e:
            print(f"   Chunk {i}: FAILED - {e}")
            print(f"      Text preview: {repr(text[:100])}")
            failed_chunks.append((i, text, str(e)))
    
    if failed_chunks:
        print(f"\n   {len(failed_chunks)} chunks failed!")
        for i, text, err in failed_chunks[:5]:
            print(f"\n   Failed chunk {i}:")
            print(f"      Error: {err}")
            print(f"      Text ({len(text)} chars): {repr(text[:200])}")
        return False
    
    # Test batch embedding
    print("\n5. Testing batch embedding...")
    try:
        chunk_texts = [c['text'] for c in chunks]
        embeddings = embeddings_model.encode(chunk_texts, convert_to_numpy=True)
        print(f"   SUCCESS: {embeddings.shape}")
    except Exception as e:
        print(f"   FAILED: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    return True

if __name__ == "__main__":
    # Find a PDF file to test with
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Look for uploaded PDFs
        upload_dir = os.path.join(os.path.dirname(__file__), "media", "uploads")
        pdfs = glob.glob(os.path.join(upload_dir, "*.pdf"))
        if pdfs:
            file_path = pdfs[0]
            print(f"Using first PDF found: {file_path}")
        else:
            print("No PDF file specified and none found in media/uploads/")
            print("Usage: python test_real_upload.py <path_to_pdf>")
            sys.exit(1)
    
    success = test_with_file(file_path)
    sys.exit(0 if success else 1)