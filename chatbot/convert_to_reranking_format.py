"""
Convert training queries to reranking model training format.

This script transforms the generated query-chunk pairs into the format
needed for CrossEncoder/reranking model finetuning:
  - Query: Generated search query
  - Positive: Source chunk text
  - Negative: Random chunk from different document

Output format: JSONL with {query, positive, negative} per line

Usage:
    python convert_to_reranking_format.py [options]
"""

import os
import sys
import django
import argparse
import json
import random
from typing import List, Dict, Tuple

# Setup Django environment
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

from backend.utils.mongo_repository import MongoRepository
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RerankingFormatConverter:
    """Convert training queries to reranking model format."""
    
    def __init__(self, negatives_per_query: int = 1, use_hard_negatives: bool = False):
        self.mongo_repo = MongoRepository()
        self.negatives_per_query = negatives_per_query
        self.use_hard_negatives = use_hard_negatives
        self.training_collection = self.mongo_repo.db["training_set"]
        self.chunks_collection = self.mongo_repo.db["chunks"]
    
    def get_all_training_entries(self) -> List[Dict]:
        """Load all training entries from MongoDB."""
        return list(self.training_collection.find({}))
    
    def get_negative_chunks(self, positive_doc_id: str, count: int = 10) -> List[Dict]:
        """
        Get random chunks from different documents for negative examples.
        """
        # Get chunks from other documents
        negatives = list(
            self.chunks_collection.aggregate([
                {"$match": {"document_id": {"$ne": positive_doc_id}}},
                {"$sample": {"size": count * 2}}  # Get extra for filtering
            ])
        )
        
        # Filter out very short chunks
        negatives = [n for n in negatives if len(n.get("text", "")) > 50]
        
        return negatives[:count]
    
    def get_hard_negative_chunks(
        self, 
        positive_chunk: Dict, 
        positive_doc_id: str, 
        count: int = 5
    ) -> List[Dict]:
        """
        Get "hard" negative chunks that are similar in structure/length
        but from different documents (more challenging for reranker).
        """
        # Get chunks with similar length from other documents
        target_len = len(positive_chunk.get("text", ""))
        min_len = int(target_len * 0.7)
        max_len = int(target_len * 1.3)
        
        negatives = list(
            self.chunks_collection.aggregate([
                {
                    "$match": {
                        "document_id": {"$ne": positive_doc_id},
                        "$expr": {
                            "$and": [
                                {"$gte": [{"$strLenCP": "$text"}, min_len]},
                                {"$lte": [{"$strLenCP": "$text"}, max_len]}
                            ]
                        }
                    }
                },
                {"$sample": {"size": count * 2}}
            ])
        )
        
        return negatives[:count]
    
    def create_training_examples(self) -> List[Dict]:
        """
        Create training examples in reranking format.
        Each entry in training_set has 3 queries, so we create 3 examples per entry.
        """
        training_entries = self.get_all_training_entries()
        
        if not training_entries:
            logger.warning("No training entries found in MongoDB!")
            return []
        
        logger.info(f"Loaded {len(training_entries)} training entries")
        logger.info(f"Will create {len(training_entries) * 3} query-chunk pairs")
        logger.info(f"With {self.negatives_per_query} negative(s) per query")
        logger.info(f"Total examples: {len(training_entries) * 3 * (1 + self.negatives_per_query)}")
        
        examples = []
        
        for i, entry in enumerate(training_entries):
            chunk_text = entry.get("chunk_text", "")
            document_id = entry.get("document_id")
            queries = entry.get("queries", [])
            
            if not queries or not chunk_text:
                logger.warning(f"Skipping entry {entry.get('chunk_id')}: missing queries or text")
                continue
            
            # Get negative examples for this entry
            if self.use_hard_negatives:
                negatives = self.get_hard_negative_chunks(
                    entry, 
                    document_id, 
                    count=self.negatives_per_query * 3
                )
            else:
                negatives = self.get_negative_chunks(
                    document_id, 
                    count=self.negatives_per_query * 3
                )
            
            if not negatives:
                logger.warning(f"No negative chunks found for {entry.get('chunk_id')}")
                continue
            
            # Create examples for each of the 3 queries
            for query_idx, query in enumerate(queries):
                # Positive example (correct chunk)
                examples.append({
                    "query": query,
                    "chunk": chunk_text,
                    "label": 1,  # Relevant
                    "query_type": ["keyword", "natural_language", "scenario"][query_idx],
                    "chunk_id": entry.get("chunk_id"),
                    "document_id": document_id
                })
                
                # Negative examples (wrong chunks)
                for neg_idx in range(self.negatives_per_query):
                    negative = negatives[query_idx * self.negatives_per_query + neg_idx]
                    examples.append({
                        "query": query,
                        "chunk": negative.get("text", ""),
                        "label": 0,  # Not relevant
                        "query_type": ["keyword", "natural_language", "scenario"][query_idx],
                        "chunk_id": entry.get("chunk_id"),
                        "negative_chunk_id": negative.get("_id"),
                        "document_id": document_id
                    })
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(training_entries)} entries...")
        
        logger.info(f"Created {len(examples)} total examples")
        return examples
    
    def save_to_jsonl(self, examples: List[Dict], output_file: str):
        """Save examples to JSONL format (one JSON object per line)."""
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(examples)} examples to {output_file}")
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        logger.info(f"File size: {file_size_mb:.2f} MB")
    
    def save_to_json(self, examples: List[Dict], output_file: str):
        """Save examples to JSON format (single array)."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(examples)} examples to {output_file}")
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        logger.info(f"File size: {file_size_mb:.2f} MB")
    
    def split_train_val_test(
        self, 
        examples: List[Dict], 
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split examples into train/validation/test sets."""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001
        
        # Shuffle examples
        random.shuffle(examples)
        
        n = len(examples)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_set = examples[:train_end]
        val_set = examples[train_end:val_end]
        test_set = examples[val_end:]
        
        logger.info(f"Split into train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")
        return train_set, val_set, test_set


def main():
    parser = argparse.ArgumentParser(
        description="Convert training queries to reranking model format"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="reranking_training_data.jsonl",
        help="Output filename (default: reranking_training_data.jsonl)"
    )
    
    parser.add_argument(
        "--format",
        choices=["jsonl", "json"],
        default="jsonl",
        help="Output format: jsonl (one per line) or json (single array)"
    )
    
    parser.add_argument(
        "--negatives",
        type=int,
        default=1,
        help="Number of negative examples per query (default: 1)"
    )
    
    parser.add_argument(
        "--hard-negatives",
        action="store_true",
        help="Use hard negatives (similar length/structure from other docs)"
    )
    
    parser.add_argument(
        "--split",
        action="store_true",
        help="Split into train/val/test sets (80/10/10)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    try:
        converter = RerankingFormatConverter(
            negatives_per_query=args.negatives,
            use_hard_negatives=args.hard_negatives
        )
        
        logger.info("=" * 80)
        logger.info("Creating reranking training examples...")
        logger.info("=" * 80)
        
        examples = converter.create_training_examples()
        
        if not examples:
            logger.error("No examples created! Check training_set collection.")
            return
        
        # Show statistics
        logger.info("\n" + "=" * 80)
        logger.info("Dataset Statistics:")
        logger.info("=" * 80)
        positive_count = sum(1 for e in examples if e["label"] == 1)
        negative_count = sum(1 for e in examples if e["label"] == 0)
        logger.info(f"Positive examples: {positive_count}")
        logger.info(f"Negative examples: {negative_count}")
        logger.info(f"Total examples: {len(examples)}")
        logger.info(f"Positive ratio: {positive_count / len(examples):.2%}")
        
        # Count by query type
        from collections import Counter
        query_types = Counter(e["query_type"] for e in examples)
        logger.info(f"\nBy query type:")
        for qtype, count in query_types.items():
            logger.info(f"  {qtype}: {count}")
        
        # Split into sets if requested
        if args.split:
            train, val, test = converter.split_train_val_test(examples)
            
            # Save each set
            base_name = args.output.rsplit('.', 1)[0]
            ext = args.format
            
            if args.format == "jsonl":
                converter.save_to_jsonl(train, f"{base_name}_train.{ext}")
                converter.save_to_jsonl(val, f"{base_name}_val.{ext}")
                converter.save_to_jsonl(test, f"{base_name}_test.{ext}")
            else:
                converter.save_to_json(train, f"{base_name}_train.{ext}")
                converter.save_to_json(val, f"{base_name}_val.{ext}")
                converter.save_to_json(test, f"{base_name}_test.{ext}")
        else:
            # Save all examples to single file
            if args.format == "jsonl":
                converter.save_to_jsonl(examples, args.output)
            else:
                converter.save_to_json(examples, args.output)
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ Conversion complete!")
        logger.info("=" * 80)
        
        # Show sample
        logger.info("\nSample example:")
        sample = examples[0]
        logger.info(f"  Query: {sample['query']}")
        logger.info(f"  Chunk (first 100 chars): {sample['chunk'][:100]}...")
        logger.info(f"  Label: {sample['label']} ({'relevant' if sample['label'] == 1 else 'not relevant'})")
        logger.info(f"  Query type: {sample['query_type']}")
        
    except Exception as e:
        logger.exception(f"Error during conversion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
