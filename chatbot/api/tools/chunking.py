import nltk
import re
import os
import logging

logger = logging.getLogger(__name__)


def detect_semantic_boundaries(text):
    """
    Detect semantic boundaries in text based on structural markers.
    Returns a list of (position, boundary_type, weight) tuples.
    Higher weight = stronger boundary (better place to split).
    """
    boundaries = []
    
    # Vietnamese and English section markers with weights
    patterns = [
        # Strong boundaries (weight 1.0) - Major sections
        (r'\n\s*(?:Chương|CHƯƠNG|Chapter|CHAPTER)\s+\d+', 'chapter', 1.0),
        (r'\n\s*(?:Phần|PHẦN|Part|PART)\s+\d+', 'part', 1.0),
        (r'\n\s*(?:Mục|MỤC|Section|SECTION)\s+\d+', 'section_num', 0.95),
        
        # Medium-strong boundaries (weight 0.8) - Subsections
        (r'\n\s*\d+\.\d+\.?\s+[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬ]', 'numbered_subsection', 0.8),
        (r'\n\s*[IVX]+\.\s+[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬ]', 'roman_section', 0.8),
        (r'\n\s*[a-z]\)\s+', 'lettered_item', 0.6),
        
        # Medium boundaries (weight 0.6) - Paragraph breaks
        (r'\n\s*\n', 'paragraph_break', 0.6),
        (r'\n\s*[-•●○]\s+', 'bullet_point', 0.5),
        (r'\n\s*\d+\)\s+', 'numbered_list', 0.5),
        
        # Weak boundaries (weight 0.3) - Sentence-like breaks
        (r'[.!?]\s+(?=[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỬỮỰ])', 'sentence_end', 0.3),
    ]
    
    for pattern, boundary_type, weight in patterns:
        for match in re.finditer(pattern, text):
            boundaries.append((match.start(), boundary_type, weight))
    
    # Sort by position
    boundaries.sort(key=lambda x: x[0])
    return boundaries


def find_best_split_point(text, target_pos, window=200, boundaries=None):
    """
    Find the best split point near target_pos considering semantic boundaries.
    Returns the best position to split the text.
    """
    if boundaries is None:
        boundaries = detect_semantic_boundaries(text)
    
    # Look for boundaries within the window
    best_pos = target_pos
    best_score = 0
    
    window_start = max(0, target_pos - window)
    window_end = min(len(text), target_pos + window)
    
    for pos, boundary_type, weight in boundaries:
        if window_start <= pos <= window_end:
            # Score based on weight and distance from target
            distance_penalty = abs(pos - target_pos) / window
            score = weight * (1 - distance_penalty * 0.5)
            
            if score > best_score:
                best_score = score
                best_pos = pos
    
    # If no good boundary found, fall back to sentence boundary
    if best_score < 0.2:
        # Look for sentence end near target
        sentence_end = text.rfind('. ', window_start, target_pos)
        if sentence_end > window_start:
            best_pos = sentence_end + 2
    
    return best_pos


def semantic_chunk_text(raw_blocks, tokenizer, target_tokens=400, max_tokens=512, overlap_tokens=100):
    """
    Enhanced semantic chunking that respects document structure.
    - Detects section boundaries (chapters, paragraphs, lists)
    - Prefers splitting at semantic boundaries over arbitrary token limits
    - Preserves context with smart overlap
    - Better for Vietnamese and multilingual documents
    """
    def token_len(s: str) -> int:
        return len(tokenizer.encode(s, add_special_tokens=False))
    
    def make_chunk_id(document_id: str, order_index: int) -> str:
        return f"{document_id}:{order_index}"
    
    chunks = []
    current_doc = None
    order_index = 0
    
    for block in raw_blocks:
        doc_id = block["document_id"]
        
        if doc_id != current_doc:
            current_doc = doc_id
            order_index = 0
        
        text = " ".join((block.get("text") or "").split())
        if not text:
            continue
        
        # Detect semantic boundaries in the block
        boundaries = detect_semantic_boundaries(text)
        
        # If block is small enough, keep as single chunk
        block_tokens = token_len(text)
        if block_tokens <= max_tokens:
            chunks.append({
                "chunk_id": make_chunk_id(doc_id, order_index),
                "text": text,
                "document_id": doc_id,
                "page": block.get("page"),
                "end_page": block.get("end_page"),
                "section": block.get("section"),
                "order_index": order_index,
                "text_len": block_tokens,
            })
            order_index += 1
            continue
        
        # Need to split - use semantic boundaries
        current_start = 0
        current_tokens = 0
        
        # Use sentences as base units
        sentences = nltk.sent_tokenize(text)
        sentence_positions = []
        pos = 0
        for sent in sentences:
            sent_start = text.find(sent, pos)
            sentence_positions.append((sent_start, sent_start + len(sent), sent))
            pos = sent_start + len(sent)
        
        current_sentences = []
        
        for sent_start, sent_end, sent in sentence_positions:
            sent_tokens = token_len(sent)
            
            # Check if adding this sentence exceeds target
            if current_tokens + sent_tokens > target_tokens and current_sentences:
                # Check if we're near a semantic boundary
                should_split = False
                
                for bpos, btype, bweight in boundaries:
                    if current_start <= bpos <= sent_start and bweight >= 0.5:
                        should_split = True
                        break
                
                # Split if we hit target or found a good boundary
                if should_split or current_tokens >= target_tokens:
                    chunk_text = " ".join(current_sentences)
                    chunks.append({
                        "chunk_id": make_chunk_id(doc_id, order_index),
                        "text": chunk_text,
                        "document_id": doc_id,
                        "page": block.get("page"),
                        "end_page": block.get("end_page"),
                        "section": block.get("section"),
                        "order_index": order_index,
                        "text_len": token_len(chunk_text),
                    })
                    order_index += 1
                    
                    # Calculate overlap - keep last few sentences for context
                    overlap_sents = []
                    overlap_tok = 0
                    for s in reversed(current_sentences):
                        s_tok = token_len(s)
                        if overlap_tok + s_tok > overlap_tokens:
                            break
                        overlap_sents.insert(0, s)
                        overlap_tok += s_tok
                    
                    current_sentences = overlap_sents + [sent]
                    current_tokens = overlap_tok + sent_tokens
                    current_start = sent_start
                    continue
            
            # Check if sentence alone exceeds max tokens (needs hard split)
            if sent_tokens > max_tokens:
                # First, flush current chunk if exists
                if current_sentences:
                    chunk_text = " ".join(current_sentences)
                    chunks.append({
                        "chunk_id": make_chunk_id(doc_id, order_index),
                        "text": chunk_text,
                        "document_id": doc_id,
                        "page": block.get("page"),
                        "end_page": block.get("end_page"),
                        "section": block.get("section"),
                        "order_index": order_index,
                        "text_len": token_len(chunk_text),
                    })
                    order_index += 1
                    current_sentences = []
                    current_tokens = 0
                
                # Hard split the long sentence
                sent_toks = tokenizer.encode(sent, add_special_tokens=False)
                i = 0
                while i < len(sent_toks):
                    end_i = min(i + max_tokens, len(sent_toks))
                    part_ids = sent_toks[i:end_i]
                    part_text = tokenizer.decode(part_ids, clean_up_tokenization_spaces=True)
                    
                    chunks.append({
                        "chunk_id": make_chunk_id(doc_id, order_index),
                        "text": part_text,
                        "document_id": doc_id,
                        "page": block.get("page"),
                        "end_page": block.get("end_page"),
                        "section": block.get("section"),
                        "order_index": order_index,
                        "text_len": len(part_ids),
                    })
                    order_index += 1
                    
                    # Overlap for next part
                    overlap_start = max(0, end_i - overlap_tokens)
                    i = overlap_start if end_i < len(sent_toks) else len(sent_toks)
                
                continue
            
            # Normal case: add sentence to current chunk
            current_sentences.append(sent)
            current_tokens += sent_tokens
        
        # Flush remaining sentences
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append({
                "chunk_id": make_chunk_id(doc_id, order_index),
                "text": chunk_text,
                "document_id": doc_id,
                "page": block.get("page"),
                "end_page": block.get("end_page"),
                "section": block.get("section"),
                "order_index": order_index,
                "text_len": token_len(chunk_text),
            })
            order_index += 1
    
    return chunks


def clean_and_chunk_text(raw_blocks, tokenizer, max_tokens=512, overlap_tokens=50):
    """
    Clean raw text blocks and create semantic chunks.
    
    Args:
        raw_blocks: List of dicts with 'text', 'page', 'document_id', etc.
        tokenizer: HuggingFace tokenizer for token counting
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Token overlap between chunks
    
    Returns:
        List of chunk dictionaries
    """
    chunks = []
    chunk_idx = 0
    
    for block in raw_blocks:
        text = block.get('text', '')
        
        # Skip empty or whitespace-only blocks
        if not text or not text.strip():
            logger.debug(f"Skipping empty block")
            continue
        
        # Clean the text
        text = clean_text(text)
        
        # Skip if text is empty after cleaning
        if not text or not text.strip():
            logger.debug(f"Skipping block after cleaning - empty")
            continue
        
        document_id = block.get('document_id', '')
        page = block.get('page')
        section = block.get('section')
        
        # Tokenize to check length
        try:
            tokens = tokenizer.encode(text, add_special_tokens=False)
        except Exception as e:
            logger.warning(f"Tokenization failed for block: {e}")
            continue
        
        # If text fits in one chunk, add it directly
        if len(tokens) <= max_tokens:
            if text.strip():  # Final check
                chunks.append({
                    'chunk_id': f"{document_id}::chunk_{chunk_idx}",
                    'document_id': document_id,
                    'text': text.strip(),
                    'page': page,
                    'section': section,
                    'order_index': chunk_idx,
                })
                chunk_idx += 1
        else:
            # Split into smaller chunks with overlap
            sub_chunks = split_text_into_chunks(text, tokenizer, max_tokens, overlap_tokens)
            for sub_text in sub_chunks:
                if sub_text and sub_text.strip():  # Only add non-empty chunks
                    chunks.append({
                        'chunk_id': f"{document_id}::chunk_{chunk_idx}",
                        'document_id': document_id,
                        'text': sub_text.strip(),
                        'page': page,
                        'section': section,
                        'order_index': chunk_idx,
                    })
                    chunk_idx += 1
    
    logger.info(f"Generated {len(chunks)} chunks from {len(raw_blocks)} raw blocks")
    return chunks


def clean_text(text):
    """Clean text by removing extra whitespace and normalizing."""
    if not text:
        return ''
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text


def split_text_into_chunks(text, tokenizer, max_tokens, overlap_tokens):
    """
    Split text into chunks based on token count with overlap.
    
    Args:
        text: Text to split
        tokenizer: Tokenizer for encoding
        max_tokens: Max tokens per chunk
        overlap_tokens: Overlap between chunks
    
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []
    
    # Split by sentences first
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s and s.strip()]
    
    if not sentences:
        return [text.strip()] if text.strip() else []
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        if not sentence or not sentence.strip():
            continue
            
        try:
            sentence_tokens = len(tokenizer.encode(sentence, add_special_tokens=False))
        except Exception:
            # If tokenization fails, estimate tokens
            sentence_tokens = len(sentence.split())
        
        # If single sentence exceeds max, split by words
        if sentence_tokens > max_tokens:
            # Save current chunk first
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                if chunk_text.strip():
                    chunks.append(chunk_text.strip())
                current_chunk = []
                current_tokens = 0
            
            # Split long sentence by words
            words = sentence.split()
            word_chunk = []
            word_tokens = 0
            
            for word in words:
                word_token_count = len(tokenizer.encode(word, add_special_tokens=False)) if word else 1
                if word_tokens + word_token_count > max_tokens and word_chunk:
                    chunk_text = ' '.join(word_chunk)
                    if chunk_text.strip():
                        chunks.append(chunk_text.strip())
                    word_chunk = []
                    word_tokens = 0
                word_chunk.append(word)
                word_tokens += word_token_count
            
            if word_chunk:
                chunk_text = ' '.join(word_chunk)
                if chunk_text.strip():
                    chunks.append(chunk_text.strip())
            continue
        
        # Check if adding this sentence exceeds max
        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            chunk_text = ' '.join(current_chunk)
            if chunk_text.strip():
                chunks.append(chunk_text.strip())
            
            # Start new chunk with overlap (take last few sentences)
            overlap_sentences = []
            overlap_count = 0
            for s in reversed(current_chunk):
                s_tokens = len(tokenizer.encode(s, add_special_tokens=False))
                if overlap_count + s_tokens <= overlap_tokens:
                    overlap_sentences.insert(0, s)
                    overlap_count += s_tokens
                else:
                    break
            
            current_chunk = overlap_sentences
            current_tokens = overlap_count
        
        current_chunk.append(sentence)
        current_tokens += sentence_tokens
    
    # Don't forget last chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        if chunk_text.strip():
            chunks.append(chunk_text.strip())
    
    return chunks
