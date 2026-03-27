from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from jiwer import cer, wer
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from common import ensure_parent_dir, normalize_text, read_csv, rows_for_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a TrOCR model on a manifest split.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", required=True, help="HF model id or local checkpoint folder.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means use all rows.")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--samples-md", type=Path, required=True)
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--strip-punctuation", action="store_true")
    return parser.parse_args()


class OCRInferenceDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        processor: TrOCRProcessor,
        lowercase: bool,
        strip_punctuation: bool,
    ) -> None:
        self.rows = rows
        self.processor = processor
        self.lowercase = lowercase
        self.strip_punctuation = strip_punctuation

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        image = Image.open(row["image_path"]).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        reference = normalize_text(
            row["text"],
            lowercase=self.lowercase,
            strip_punctuation=self.strip_punctuation,
        )
        return {
            "sample_id": row["sample_id"],
            "image_path": row["image_path"],
            "pixel_values": pixel_values,
            "reference": reference,
        }


def collate_fn(batch: list[dict]) -> dict:
    return {
        "sample_id": [item["sample_id"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "reference": [item["reference"] for item in batch],
    }


def write_samples_markdown(path: Path, predictions: list[dict], max_examples: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Sample Predictions", ""]
    for row in predictions[:max_examples]:
        lines.extend(
            [
                f"## {row['sample_id']}",
                f"- Reference: {row['reference']}",
                f"- Prediction: {row['prediction']}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def trim_rows(rows: list[dict[str, str]], max_samples: int) -> list[dict[str, str]]:
    if max_samples <= 0:
        return rows
    return rows[:max_samples]


def main() -> None:
    args = parse_args()

    rows = trim_rows(rows_for_split(read_csv(args.manifest), args.split), args.max_samples)
    if not rows:
        raise ValueError(f"No rows found for split: {args.split}")

    processor = TrOCRProcessor.from_pretrained(args.model_path)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataset = OCRInferenceDataset(
        rows,
        processor,
        lowercase=args.lowercase,
        strip_punctuation=args.strip_punctuation,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    predictions: list[dict[str, str]] = []
    references: list[str] = []
    generated_texts: list[str] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {args.split}"):
            pixel_values = batch["pixel_values"].to(device)
            generated_ids = model.generate(
                pixel_values,
                max_length=args.max_target_length,
                num_beams=args.num_beams,
            )
            decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
            decoded = [
                normalize_text(text, lowercase=args.lowercase, strip_punctuation=args.strip_punctuation)
                for text in decoded
            ]

            for sample_id, image_path, reference, prediction in zip(
                batch["sample_id"],
                batch["image_path"],
                batch["reference"],
                decoded,
                strict=True,
            ):
                references.append(reference)
                generated_texts.append(prediction)
                predictions.append(
                    {
                        "sample_id": sample_id,
                        "split": args.split,
                        "image_path": image_path,
                        "reference": reference,
                        "prediction": prediction,
                    }
                )

    metrics = {
        "num_samples": len(predictions),
        "cer": cer(references, generated_texts),
        "wer": wer(references, generated_texts),
        "model_path": args.model_path,
        "split": args.split,
    }

    ensure_parent_dir(args.output_csv)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        handle.write("sample_id,split,image_path,reference,prediction\n")
        for row in predictions:
            safe_row = {
                key: str(value).replace('"', '""')
                for key, value in row.items()
            }
            handle.write(
                f"\"{safe_row['sample_id']}\",\"{safe_row['split']}\",\"{safe_row['image_path']}\","
                f"\"{safe_row['reference']}\",\"{safe_row['prediction']}\"\n"
            )

    ensure_parent_dir(args.metrics_json)
    args.metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_samples_markdown(args.samples_md, predictions)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
