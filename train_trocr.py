from __future__ import annotations

import argparse
import csv
import logging
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from PIL import Image
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers.utils import logging as transformers_logging


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
transformers_logging.set_verbosity_error()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a pretrained TrOCR model on local images.")
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Path to an image file or a folder of images.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to save OCR predictions as CSV.",
    )
    parser.add_argument(
        "--model-name",
        default="microsoft/trocr-small-handwritten",
        help="Hugging Face model id or local checkpoint path.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--recursive", action="store_true", help="Search subfolders for images.")
    parser.add_argument(
        "--crop-left",
        type=int,
        default=0,
        help="Crop this many pixels from the left before OCR.",
    )
    parser.add_argument(
        "--uppercase",
        action="store_true",
        help="Convert the final OCR text to uppercase before printing.",
    )
    return parser.parse_args()


def list_images(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {input_path}")
        return [input_path]

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    pattern = "**/*" if recursive else "*"
    images = [
        path
        for path in input_path.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not images:
        raise ValueError(f"No supported images found in: {input_path}")
    return sorted(images)


def batched(items: list[Path], batch_size: int) -> list[list[Path]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def load_image(path: Path, crop_left: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if crop_left > 0:
        crop_left = min(crop_left, max(0, image.width - 1))
        image = image.crop((crop_left, 0, image.width, image.height))
    return image


def resolve_cached_model_snapshot(model_name: str) -> Path | None:
    if "\\" in model_name or ":" in model_name or Path(model_name).exists():
        return None

    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir = cache_root / f"models--{model_name.replace('/', '--')}" / "snapshots"
    if not repo_dir.exists():
        return None

    snapshots = [path for path in repo_dir.iterdir() if path.is_dir()]
    if not snapshots:
        return None

    def is_usable_snapshot(path: Path) -> bool:
        required = {
            "config.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
        }
        present = {child.name for child in path.iterdir() if child.is_file()}
        return required.issubset(present)

    usable_snapshots = [path for path in snapshots if is_usable_snapshot(path)]
    candidates = usable_snapshots or snapshots
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_processor_and_model(model_name: str) -> tuple[TrOCRProcessor, VisionEncoderDecoderModel]:
    try:
        processor = TrOCRProcessor.from_pretrained(model_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        return processor, model
    except OSError:
        cached_snapshot = resolve_cached_model_snapshot(model_name)
        if cached_snapshot is None:
            raise

        processor = TrOCRProcessor.from_pretrained(str(cached_snapshot), local_files_only=True)
        model = VisionEncoderDecoderModel.from_pretrained(str(cached_snapshot), local_files_only=True)
        return processor, model


def main() -> None:
    args = parse_args()

    image_paths = list_images(args.input_path, args.recursive)
    processor, model = load_processor_and_model(args.model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    rows: list[dict[str, str]] = []
    with torch.no_grad():
        for batch_paths in tqdm(
            batched(image_paths, args.batch_size),
            desc="Running OCR",
            disable=len(image_paths) == 1,
        ):
            images = [load_image(path, args.crop_left) for path in batch_paths]
            pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(device)
            generated_ids = model.generate(
                pixel_values,
                num_beams=args.num_beams,
                max_length=args.max_length,
            )
            predictions = processor.batch_decode(generated_ids, skip_special_tokens=True)

            for path, text in zip(batch_paths, predictions, strict=True):
                prediction = text.strip()
                if args.uppercase:
                    prediction = prediction.upper()
                rows.append({"image_path": str(path), "prediction": prediction})

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image_path", "prediction"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Predictions saved to: {args.output_csv}")

    if len(rows) == 1 and args.output_csv is None:
        print(rows[0]["prediction"])
        return

    for row in rows:
        print(f"{row['image_path']} -> {row['prediction']}")


if __name__ == "__main__":
    main()
