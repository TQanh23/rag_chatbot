import nltk

def clean_and_chunk_text(raw_blocks, tokenizer, target_tokens=400, max_tokens=512, overlap_tokens=100):
    """
    Clean and chunk text into overlapping segments using NLTK for sentence tokenization.
    - Sentence-aware splitting for natural boundaries.
    - Token-based splitting to enforce token limits.
    - Enforces a hard cap of `max_tokens` per chunk.
    - Ensures overlap is measured in *tokens*, not sentences.
    - Generates a stable chunk_id: "<document_id>:<order_index>" with order_index reset per doc.
    """
    def token_len(s: str) -> int:
        return len(tokenizer.encode(s, add_special_tokens=False))

    def make_chunk_id(document_id: str, order_index: int) -> str:
        return f"{document_id}:{order_index}"

    chunks = []
    current_doc = None
    order_index = 0  # per-document counter

    for block in raw_blocks:
        doc_id = block["document_id"]

        # reset order index when document changes
        if doc_id != current_doc:
            current_doc = doc_id
            order_index = 0

        text = " ".join((block.get("text") or "").split())
        if not text:
            continue

        # sentence split (ensure punkt is installed for your language)
        sentences = nltk.sent_tokenize(text)

        current = []           # list[str] sentences in the working chunk
        current_tokens = 0     # token count for `current`

        for sent in sentences:
            s_tokens = token_len(sent)

            if current_tokens + s_tokens > max_tokens:
                # finalize current chunk
                if current:
                    chunk_text = " ".join(current)
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

                # start next chunk with token-accurate overlap
                tail, tail_tokens = [], 0
                for prev in reversed(current):
                    t = token_len(prev)
                    if tail_tokens + t > overlap_tokens:
                        break
                    tail.append(prev)
                    tail_tokens += t
                tail.reverse()

                current = tail + [sent]
                current_tokens = sum(token_len(x) for x in current)

                # if a single sentence is longer than max_tokens, hard-split it
                while current_tokens > max_tokens and len(current) == 1:
                    stoks = tokenizer.encode(current[0], add_special_tokens=False)
                    head_ids = stoks[:max_tokens]
                    # overlap inside a long sentence split
                    start_tail = max_tokens - overlap_tokens if overlap_tokens < max_tokens else max_tokens
                    tail_ids = stoks[start_tail:]

                    head = tokenizer.decode(head_ids, clean_up_tokenization_spaces=True)
                    tail_str = tokenizer.decode(tail_ids, clean_up_tokenization_spaces=True)

                    chunks.append({
                        "chunk_id": make_chunk_id(doc_id, order_index),
                        "text": head,
                        "document_id": doc_id,
                        "page": block.get("page"),
                        "end_page": block.get("end_page"),
                        "section": block.get("section"),
                        "order_index": order_index,
                        "text_len": len(head_ids),
                    })
                    order_index += 1

                    current = [tail_str] if tail_str.strip() else []
                    current_tokens = token_len(tail_str) if tail_str.strip() else 0

            else:
                # add sentence
                current.append(sent)
                current_tokens += s_tokens

                # proactive cut at target_tokens (sentence boundary)
                if current_tokens >= target_tokens:
                    chunk_text = " ".join(current)
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

                    # seed next chunk with token-accurate overlap from the tail
                    tail, tail_tokens = [], 0
                    for prev in reversed(current):
                        t = token_len(prev)
                        if tail_tokens + t > overlap_tokens:
                            break
                        tail.append(prev)
                        tail_tokens += t
                    tail.reverse()

                    current = tail
                    current_tokens = tail_tokens

        # flush last chunk
        if current:
            chunk_text = " ".join(current)
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
