import numpy as np

def recall_at_k(relevant_items, retrieved_items, k=5):
    """
    Compute Recall@k.
    :param relevant_items: Set of relevant item IDs.
    :param retrieved_items: List of retrieved item IDs.
    :param k: Number of top items to consider.
    :return: Recall@k score.
    """
    retrieved_at_k = retrieved_items[:k]
    relevant_retrieved = len(set(relevant_items) & set(retrieved_at_k))
    return relevant_retrieved / len(relevant_items) if relevant_items else 0.0

def mrr_at_k(relevant_items, retrieved_items, k=5):
    """
    Compute Mean Reciprocal Rank (MRR)@k.
    :param relevant_items: Set of relevant item IDs.
    :param retrieved_items: List of retrieved item IDs.
    :param k: Number of top items to consider.
    :return: MRR@k score.
    """
    for rank, item in enumerate(retrieved_items[:k], start=1):
        if item in relevant_items:
            return 1 / rank
    return 0.0

def ndcg_at_k(relevant_items, retrieved_items, k=5):
    """
    Compute Normalized Discounted Cumulative Gain (nDCG)@k.
    :param relevant_items: Set of relevant item IDs.
    :param retrieved_items: List of retrieved item IDs.
    :param k: Number of top items to consider.
    :return: nDCG@k score.
    """
    def dcg(items):
        return sum((1 / np.log2(idx + 2)) for idx, item in enumerate(items) if item in relevant_items)

    ideal_dcg = dcg(sorted(relevant_items, reverse=True)[:k])
    actual_dcg = dcg(retrieved_items[:k])
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0