import os
import csv
from collections import defaultdict
import math
import numpy as np


def _parse_list_field(s):
    """Parse a list-like field which may be pipe-separated or comma-separated."""
    if s is None:
        return []
    if isinstance(s, (list, tuple)):
        return list(s)
    s = str(s).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    if "|" in s:
        parts = [p.strip() for p in s.split("|") if p.strip()]
    else:
        parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts


def recall_at_k(relevant_items, retrieved_items, k=5):
    """Compute Recall@k."""
    retrieved_at_k = retrieved_items[:k]
    relevant_retrieved = len(set(relevant_items) & set(retrieved_at_k))
    return relevant_retrieved / len(relevant_items) if relevant_items else 0.0


def mrr_at_k(relevant_items, retrieved_items, k=5):
    """Compute MRR@k."""
    for rank, item in enumerate(retrieved_items[:k], start=1):
        if item in relevant_items:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevant_items, retrieved_items, k=5):
    """Compute nDCG@k."""
    # relevance list for retrieved items
    rels = [1 if item in relevant_items else 0 for item in retrieved_items[:k]]

    def dcg(relevance_list):
        return sum((rel / math.log2(idx + 2)) for idx, rel in enumerate(relevance_list))

    ideal_rels = sorted([1] * len(relevant_items) + [0] * max(0, k - len(relevant_items)), reverse=True)[:k]
    idcg = dcg(ideal_rels)
    actual = dcg(rels)
    return actual / idcg if idcg > 0 else 0.0


class RetrievalEvaluator:
    """Evaluate retrieval logs against a gold QA file.

    Methods
    -------
    evaluate(qa_gold_path, retrieval_run_path, k_values=[5,10]) -> dict
        Returns a dict with 'per_question' list and 'summary' stats for each metric.
    """

    @staticmethod
    def _load_qa_gold(path):
        qa = {}
        if not os.path.exists(path):
            return qa
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                qid = str(row.get('question_id', '')).strip()
                qa[qid] = {
                    'gold_chunks': _parse_list_field(row.get('gold_support_chunk_ids', ''))
                }
        return qa

    @staticmethod
    def _load_retrieval(path):
        rows = []
        if not os.path.exists(path):
            return rows
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows

    @staticmethod
    def evaluate(qa_gold_path, retrieval_run_path, k_values=[5, 10]):
        qa_gold = RetrievalEvaluator._load_qa_gold(qa_gold_path)
        retrieval_rows = RetrievalEvaluator._load_retrieval(retrieval_run_path)

        per_q = []
        # accumulate lists for summary
        accum = defaultdict(list)

        for row in retrieval_rows:
            qid = str(row.get('question_id', '')).strip()
            retrieved = row.get('retrieved_chunk_ids', '')
            retrieved_list = _parse_list_field(retrieved)

            gold_chunks = qa_gold.get(qid, {}).get('gold_chunks', [])
            gold_set = set(gold_chunks)

            q_metrics = {'question_id': qid}

            for k in k_values:
                r_at_k = recall_at_k(gold_set, retrieved_list, k)
                mrr = mrr_at_k(gold_set, retrieved_list, k)
                ndcg = ndcg_at_k(gold_set, retrieved_list, k)

                q_metrics[f'recall@{k}'] = r_at_k
                q_metrics[f'mrr@{k}'] = mrr
                q_metrics[f'ndcg@{k}'] = ndcg

                accum[f'recall@{k}'].append(r_at_k)
                accum[f'mrr@{k}'].append(mrr)
                accum[f'ndcg@{k}'].append(ndcg)

            per_q.append(q_metrics)

        # build summary
        summary = {}
        for metric, vals in accum.items():
            if vals:
                summary[metric] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals)),
                    'count': len(vals)
                }

        return {'per_question': per_q, 'summary': summary}
