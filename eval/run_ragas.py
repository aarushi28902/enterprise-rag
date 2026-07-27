import json
import sys
from pathlib import Path

import numpy as np
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from eval.ragas_config import get_ragas_llm, get_ragas_embeddings

THRESHOLDS = {
    "faithfulness": 0.75,
    "answer_relevancy": 0.70,
    "context_precision": 0.65,
    "context_recall": 0.60,
}


def load_dataset(path: Path) -> Dataset:
    records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return Dataset.from_list(records)


def main():
    dataset_path = Path("eval/datasets/sample_qa.jsonl")
    output_path = Path("eval/results.json")

    print("Loading evaluation dataset...")
    ds = load_dataset(dataset_path)
    print(f"Loaded {len(ds)} samples")

    print("Configuring RAGAS with Ollama...")
    llm = get_ragas_llm()
    embeddings = get_ragas_embeddings()

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    for metric in metrics:
        metric.llm = llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = embeddings

    print("Running RAGAS evaluation...")
    results = evaluate(ds, metrics=metrics, raise_exceptions=False)

    # only take numeric columns
    df = results.to_pandas()
    numeric_cols = [c for c in THRESHOLDS.keys() if c in df.columns]
    scores = {}
    for col in numeric_cols:
        vals = df[col].dropna()
        scores[col] = float(np.nanmean(vals)) if len(vals) > 0 else 0.0

    print("\n=== RAGAS Evaluation Results ===")
    failed = []
    for metric, threshold in THRESHOLDS.items():
        score = scores.get(metric, 0.0)
        status = "PASS" if score >= threshold else "FAIL"
        print(f"  [{status}] {metric}: {score:.3f} (threshold: {threshold})")
        if score < threshold:
            failed.append(metric)

    output_path.write_text(json.dumps(scores, indent=2))
    print(f"\nResults saved to {output_path}")

    if failed:
        print(f"\nFailed metrics: {failed}")
        sys.exit(1)
    print("\nAll metrics passed.")


if __name__ == "__main__":
    main()
