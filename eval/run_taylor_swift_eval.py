"""RAGAS evaluation on Taylor Swift Folklore/Evermore dataset."""
import json
import sys
from pathlib import Path
import numpy as np
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall
from eval.ragas_config import get_ragas_llm, get_ragas_embeddings

THRESHOLDS = {
    "context_precision": 0.65,
    "context_recall": 0.60,
}


def load_dataset(path: Path) -> Dataset:
    records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return Dataset.from_list(records)


def main():
    dataset_path = Path("eval/datasets/taylor_swift_qa.jsonl")
    output_path = Path("eval/taylor_swift_results.json")

    print("Loading Taylor Swift evaluation dataset...")
    ds = load_dataset(dataset_path)
    print(f"Loaded {len(ds)} samples")

    print("Configuring RAGAS with Ollama...")
    llm = get_ragas_llm()
    embeddings = get_ragas_embeddings()

    metrics = [context_precision, context_recall]
    for metric in metrics:
        metric.llm = llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = embeddings

    print("Running evaluation (this will take several minutes on CPU)...")
    results = evaluate(ds, metrics=metrics, raise_exceptions=False)

    df = results.to_pandas()
    scores = {}
    for col in THRESHOLDS.keys():
        if col in df.columns:
            vals = df[col].dropna()
            scores[col] = float(np.nanmean(vals)) if len(vals) > 0 else 0.0

    print("\n=== Taylor Swift Folklore/Evermore Evaluation ===")
    print(f"Dataset: {len(ds)} questions across Folklore and Evermore Wikipedia corpus\n")

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
        print(f"\nFailed: {failed}")
        sys.exit(1)
    print("\nAll thresholds passed.")


if __name__ == "__main__":
    main()
