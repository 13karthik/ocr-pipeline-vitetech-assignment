from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from common import deterministic_split, normalize_text, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a clean IAM line-level manifest.")
    parser.add_argument("--iam-root", type=Path, required=True, help="Path to the IAM dataset root.")
    parser.add_argument("--manifest", type=Path, required=True, help="Output CSV manifest path.")
    parser.add_argument("--audit", type=Path, required=True, help="Output markdown audit path.")
    parser.add_argument("--lowercase", action="store_true", help="Lowercase text during normalization.")
    parser.add_argument(
        "--strip-punctuation",
        action="store_true",
        help="Remove punctuation during normalization.",
    )
    parser.add_argument(
        "--keep-non-ok",
        action="store_true",
        help="Keep IAM rows whose status is not 'ok'.",
    )
    return parser.parse_args()


def build_image_path(iam_root: Path, sample_id: str) -> Path:
    first, second, _ = sample_id.split("-")
    form_id = f"{first}-{second}"
    return iam_root / "lines" / first / form_id / f"{sample_id}.png"


def write_audit(
    audit_path: Path,
    stats: Counter,
    split_counts: Counter,
    lowercase: bool,
    strip_punctuation: bool,
) -> None:
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Data Audit",
        "",
        "## Findings",
        f"- Metadata rows parsed: {stats['metadata_rows']}",
        f"- Rows kept: {stats['rows_kept']}",
        f"- Missing image files skipped: {stats['missing_images']}",
        f"- Non-ok status rows skipped: {stats['non_ok_skipped']}",
        f"- Empty text rows skipped: {stats['empty_text_skipped']}",
        f"- Duplicate sample ids skipped: {stats['duplicate_sample_ids']}",
        "",
        "## Split Summary",
        f"- Train rows: {split_counts['train']}",
        f"- Validation rows: {split_counts['val']}",
        f"- Test rows: {split_counts['test']}",
        "",
        "## Fixes Applied",
        "- Collapsed repeated whitespace",
        f"- Lowercased text: {'yes' if lowercase else 'no'}",
        f"- Stripped punctuation: {'yes' if strip_punctuation else 'no'}",
        "- Removed rows with missing image files",
        "- Removed rows with empty normalized text",
        "",
        "## Ignored On Purpose",
        "- Repeated transcriptions were kept because different writers can produce the same line.",
        "- No image denoising or augmentation was applied at this stage to keep the baseline simple.",
    ]
    audit_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    lines_txt = args.iam_root / "ascii" / "lines.txt"
    if not lines_txt.exists():
        raise FileNotFoundError(f"Could not find IAM metadata file: {lines_txt}")

    rows: list[dict] = []
    stats: Counter = Counter()
    split_counts: Counter = Counter()
    seen_sample_ids: set[str] = set()

    for raw_line in lines_txt.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        stats["metadata_rows"] += 1
        parts = line.split()
        if len(parts) < 9:
            continue

        sample_id = parts[0]
        status = parts[1]
        transcription = " ".join(parts[8:])

        if sample_id in seen_sample_ids:
            stats["duplicate_sample_ids"] += 1
            continue

        if status != "ok" and not args.keep_non_ok:
            stats["non_ok_skipped"] += 1
            continue

        image_path = build_image_path(args.iam_root, sample_id)
        if not image_path.exists():
            stats["missing_images"] += 1
            continue

        normalized = normalize_text(
            transcription,
            lowercase=args.lowercase,
            strip_punctuation=args.strip_punctuation,
        )
        if not normalized:
            stats["empty_text_skipped"] += 1
            continue

        first, second, _ = sample_id.split("-")
        form_id = f"{first}-{second}"
        split = deterministic_split(form_id)

        rows.append(
            {
                "sample_id": sample_id,
                "form_id": form_id,
                "status": status,
                "split": split,
                "image_path": str(image_path),
                "raw_text": transcription.replace("|", " "),
                "text": normalized,
            }
        )
        seen_sample_ids.add(sample_id)
        split_counts[split] += 1
        stats["rows_kept"] += 1

    write_csv(
        args.manifest,
        rows,
        fieldnames=["sample_id", "form_id", "status", "split", "image_path", "raw_text", "text"],
    )
    write_audit(args.audit, stats, split_counts, args.lowercase, args.strip_punctuation)

    print(f"Manifest written to: {args.manifest}")
    print(f"Audit written to: {args.audit}")
    print(f"Rows kept: {stats['rows_kept']}")
    print(f"Split counts: train={split_counts['train']}, val={split_counts['val']}, test={split_counts['test']}")


if __name__ == "__main__":
    main()
