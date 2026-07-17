#!/usr/bin/env python3
"""Print Markdown tables from the saved model evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def percent(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def collect_rows() -> list[dict]:
    rows = []
    for metrics_path in sorted(ROOT.glob("*/metrics*.json")):
        payload = load_json(metrics_path)
        test_metrics = payload["test_metrics"]
        config = payload["config"]

        evaluation_files = sorted(metrics_path.parent.glob("evaluation*.json"))
        test_records = []
        if evaluation_files:
            test_records = load_json(evaluation_files[0]).get("test", [])

        answers = [str(record.get("generated_answer", "")) for record in test_records]
        insufficient_count = sum(answer == "INSUFFICIENT_CONTEXT" for answer in answers)

        rows.append(
            {
                "model": payload["model_id"],
                "sequence_length": int(config["MAX_SEQ_LENGTH"]),
                "exact_match": float(test_metrics["exact_match"]),
                "token_f1": float(test_metrics["token_f1"]),
                "bleu": float(test_metrics["bleu"]),
                "rouge_l": float(test_metrics["rouge_l"]),
                "bertscore_f1": float(test_metrics["bertscore_f1"]),
                "grounding_rate": float(test_metrics["grounding_rate"]),
                "abstention_accuracy": float(
                    test_metrics["unanswerable_abstention_accuracy"]
                ),
                "critical_error_rate": float(test_metrics["critical_error_rate"]),
                "insufficient_rate": (
                    insufficient_count / len(test_records) if test_records else 0.0
                ),
                "unique_answers": len(set(answers)),
            }
        )
    return sorted(rows, key=lambda row: row["token_f1"], reverse=True)


def print_quality_table(rows: list[dict]) -> None:
    print("| Model | Seq. | Exact match | Token F1 | BLEU | ROUGE-L | BERTScore F1 |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        print(
            f"| `{row['model']}` | {row['sequence_length']} "
            f"| {percent(row['exact_match'])} | {percent(row['token_f1'])} "
            f"| {row['bleu']:.2f} | {percent(row['rouge_l'])} "
            f"| {percent(row['bertscore_f1'])} |"
        )


def print_behavior_table(rows: list[dict]) -> None:
    print("| Model | Grounding | Abstention accuracy | Critical error | Insufficient outputs | Unique answers |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        print(
            f"| `{row['model']}` | {percent(row['grounding_rate'])} "
            f"| {percent(row['abstention_accuracy'])} "
            f"| {percent(row['critical_error_rate'])} "
            f"| {percent(row['insufficient_rate'])} "
            f"| {row['unique_answers']} |"
        )


if __name__ == "__main__":
    results = collect_rows()
    print_quality_table(results)
    print()
    print_behavior_table(results)
