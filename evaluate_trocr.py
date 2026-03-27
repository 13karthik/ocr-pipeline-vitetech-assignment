from __future__ import annotations

import csv
import hashlib
import re
import string
from pathlib import Path


SPACE_RE = re.compile(r"\s+")
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def normalize_text(
    text: str,
    lowercase: bool = False,
    strip_punctuation: bool = False,
) -> str:
    cleaned = text.replace("|", " ")
    cleaned = SPACE_RE.sub(" ", cleaned).strip()

    if lowercase:
        cleaned = cleaned.lower()

    if strip_punctuation:
        cleaned = cleaned.translate(PUNCT_TABLE)

    return SPACE_RE.sub(" ", cleaned).strip()


def deterministic_split(group_id: str, train_ratio: float = 0.8, val_ratio: float = 0.1) -> str:
    bucket = int(hashlib.md5(group_id.encode("utf-8")).hexdigest(), 16) % 100
    train_cutoff = int(train_ratio * 100)
    val_cutoff = int((train_ratio + val_ratio) * 100)

    if bucket < train_cutoff:
        return "train"
    if bucket < val_cutoff:
        return "val"
    return "test"


def write_csv(path: str | Path, rows: list[dict], fieldnames: list[str]) -> None:
    ensure_parent_dir(path)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def rows_for_split(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    return [row for row in rows if row["split"] == split]
