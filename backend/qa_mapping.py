import json
import re
from pathlib import Path
from difflib import SequenceMatcher
import pandas as pd

# --- paths (relative to this script) ---
BASE = Path(__file__).resolve().parent
CHUNKS_CSV = BASE / "qa_candidates.csv"
QA_JSON = BASE / "qa_by_doc.json"
OUT_CSV = BASE / "qa_gold.csv"

# --- helpers ---
def strip_cites(s: str) -> str:
    if s is None: 
        return ""
    #s = re.sub(r"\[cite_start\]", "", s)
    s = re.sub(r"\[cite:\s*[^]]+\]", "", s)
    return " ".join(s.split()).lower()

def extract_pages(s: str):
    if s is None:
        return []
    hits = re.findall(r"\[cite:\s*([0-9,\s]+)\]", s)
    pages = []
    for h in hits:
        pages += [int(x) for x in re.findall(r"\d+", h)]
    return pages

def token_set(text):
    toks = re.findall(r"\w+", (text or "").lower())
    return set(toks)

def jaccard(a, b):
    A, B = token_set(a), token_set(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def best_matches(df_cands, answer_clean, topn=3):
    rows = []
    for _, r in df_cands.iterrows():
        t = (r.get("text") or "").lower()
        sub = 1.0 if answer_clean and answer_clean in t else 0.0
        jac = jaccard(answer_clean, t)
        dif = SequenceMatcher(None, answer_clean, t[:min(1500, len(t))]).ratio()
        score = (sub * 2.0) + (jac * 1.0) + (dif * 0.5)
        rows.append((score, r["chunk_id"]))
    rows.sort(reverse=True)
    return [cid for score, cid in rows[:topn] if score > 0.0]

def main():
    # load
    chunks = pd.read_csv(CHUNKS_CSV, dtype={"page": "Int64"})
    with open(QA_JSON, "r", encoding="utf-8") as f:
        qa_by_doc = json.load(f)

    out_rows = []
    qid = 1

    for doc_id, items in qa_by_doc.items():
        df_doc = chunks[chunks["document_id"] == doc_id].copy()
        for item in items:
            q = item.get("question", "").strip()
            a = item.get("answer", "").strip()
            pages = extract_pages(a)
            a_clean = strip_cites(a)

            # candidate pool
            if pages:
                df_cand = df_doc[df_doc["page"].isin(pages)]
                page_hint = "|".join(str(p) for p in pages)
            else:
                df_cand = df_doc
                page_hint = ""

            chunk_ids = best_matches(df_cand, a_clean, topn=3)

            # fallback: if none found with page filter, widen to full doc
            if not chunk_ids and pages:
                chunk_ids = best_matches(df_doc, a_clean, topn=3)

            out_rows.append({
                "question_id": f"q{qid:04d}",
                "document_id": doc_id,
                "question_text": q,
                "gold_answer": a,
                "gold_support_chunk_ids": "|".join(chunk_ids),
                "page_hint": page_hint
            })
            qid += 1

    qa_gold = pd.DataFrame(out_rows)
    qa_gold.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} with {len(qa_gold)} rows")

if __name__ == "__main__":
    main()