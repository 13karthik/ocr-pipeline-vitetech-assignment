from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from jiwer import cer, wer

from common import ensure_parent_dir


SPACE_RE = re.compile(r"\s+")
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")


PROMPT_TEMPLATE = """You are correcting OCR output from a handwritten document.

Rules:
- Fix only obvious OCR mistakes.
- Preserve the original meaning.
- Do not invent missing facts.
- Keep punctuation conservative.
- Return only the corrected text.

Raw OCR:
{raw_text}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a simple OCR post-processing comparison.")
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--examples-md", type=Path, required=True)
    return parser.parse_args()


def cleanup_prediction(text: str) -> str:
    cleaned = SPACE_RE.sub(" ", text).strip()
    cleaned = SPACE_BEFORE_PUNCT_RE.sub(r"\1", cleaned)
    return cleaned


def load_predictions(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_output_csv(path: Path, rows: list[dict[str, str]]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "split",
                "image_path",
                "reference",
                "prediction",
                "postprocessed_prediction",
                "llm_prompt",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_examples(path: Path, rows: list[dict[str, str]], metrics: dict[str, float], max_examples: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Post-Processing Examples",
        "",
        "## Summary",
        f"- CER before: {metrics['cer_before']:.4f}",
        f"- CER after: {metrics['cer_after']:.4f}",
        f"- WER before: {metrics['wer_before']:.4f}",
        f"- WER after: {metrics['wer_after']:.4f}",
        "",
    ]

    for row in rows[:max_examples]:
        lines.extend(
            [
                f"## {row['sample_id']}",
                f"- Reference: {row['reference']}",
                f"- Raw OCR: {row['prediction']}",
                f"- Postprocessed: {row['postprocessed_prediction']}",
                "",
                "Prompt used:",
                "```text",
                row["llm_prompt"],
                "```",
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_predictions(args.predictions_csv)
    if not rows:
        raise ValueError("Predictions CSV is empty.")

    updated_rows: list[dict[str, str]] = []
    references: list[str] = []
    raw_predictions: list[str] = []
    cleaned_predictions: list[str] = []

    for row in rows:
        cleaned = cleanup_prediction(row["prediction"])
        prompt = PROMPT_TEMPLATE.format(raw_text=row["prediction"])
        updated_row = dict(row)
        updated_row["postprocessed_prediction"] = cleaned
        updated_row["llm_prompt"] = prompt
        updated_rows.append(updated_row)

        references.append(row["reference"])
        raw_predictions.append(row["prediction"])
        cleaned_predictions.append(cleaned)

    metrics = {
        "num_samples": len(updated_rows),
        "cer_before": cer(references, raw_predictions),
        "cer_after": cer(references, cleaned_predictions),
        "wer_before": wer(references, raw_predictions),
        "wer_after": wer(references, cleaned_predictions),
    }

    write_output_csv(args.output_csv, updated_rows)
    ensure_parent_dir(args.metrics_json)
    args.metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_examples(args.examples_md, updated_rows, metrics)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
