import argparse
import csv
from pathlib import Path


def normalize(text: str) -> str:
    return " ".join((text or "").split()).lower()


def load_gold(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not path.exists():
        raise FileNotFoundError(f"Missing qa_gold file: {path}")

    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key = normalize(row.get("question_text", ""))
            if key:
                mapping[key] = row.get("question_id", "")
    return mapping


def align_ids(
    gold_path: Path,
    generation_path: Path,
    output_path: Path | None = None,
) -> tuple[int, int]:
    if not generation_path.exists():
        raise FileNotFoundError(f"Missing generation file: {generation_path}")

    gold_lookup = load_gold(gold_path)

    with generation_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    matched = 0
    for row in rows:
        key = normalize(row.get("question_text", ""))
        if key in gold_lookup:
            row["question_id"] = gold_lookup[key]
            matched += 1

    if output_path is None:
        output_path = generation_path.with_name("generation_run_aligned.csv")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return matched, len(rows) - matched


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align generation_run question IDs to match qa_gold."
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("backend/media/qa_gold_backup_20251130_214225.csv"),
        help="Path to qa_gold.csv",
    )
    parser.add_argument(
        "--generation",
        type=Path,
        default=Path("media/generation_run.csv"),
        help="Path to generation_run.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: generation_run_aligned.csv)",
    )
    args = parser.parse_args()

    matched, unmatched = align_ids(args.gold, args.generation, args.output)
    print(f"Aligned rows: {matched}")
    print(f"Unmatched rows: {unmatched}")
    if args.output is None:
        print("Wrote media/generation_run_aligned.csv")


if __name__ == "__main__":
    main()