from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a YOLO detection dataset for T-55 training.")
    parser.add_argument(
        "--raw-dir",
        default=r"C:\Users\popod\Desktop\TankProject\dataset_raw\Final_Dataset",
        help="Directory containing PNG/JPG images and matching YOLO txt labels.",
    )
    parser.add_argument(
        "--out-dir",
        default=r"C:\Users\popod\Desktop\TankProject\yolo_dataset",
        help="Output YOLO dataset directory.",
    )
    parser.add_argument("--train", type=float, default=0.7, help="Train split ratio.")
    parser.add_argument("--val", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--test", type=float, default=0.1, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits.")
    parser.add_argument("--clean", action="store_true", help="Delete output directory before writing.")
    return parser.parse_args()


def label_is_positive(label_path: Path) -> bool:
    return label_path.exists() and label_path.read_text(encoding="utf-8").strip() != ""


def validate_label_file(label_path: Path) -> list[str]:
    warnings: list[str] = []
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return warnings

    for line_number, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            warnings.append(f"{label_path.name}:{line_number} expected 5 fields, got {len(parts)}")
            continue
        try:
            class_id = int(parts[0])
            values = [float(value) for value in parts[1:]]
        except ValueError:
            warnings.append(f"{label_path.name}:{line_number} contains non-numeric values")
            continue
        if class_id != 0:
            warnings.append(f"{label_path.name}:{line_number} class id should be 0 for T-55, got {class_id}")
        if any(value < 0 or value > 1 for value in values):
            warnings.append(f"{label_path.name}:{line_number} bbox values should be normalized 0..1: {values}")
    return warnings


def split_items(items: list[Path], train_ratio: float, val_ratio: float) -> dict[str, list[Path]]:
    train_end = round(len(items) * train_ratio)
    val_end = train_end + round(len(items) * val_ratio)
    return {
        "train": items[:train_end],
        "val": items[train_end:val_end],
        "test": items[val_end:],
    }


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    ratio_sum = args.train + args.val + args.test
    if abs(ratio_sum - 1.0) > 1e-6:
        raise SystemExit(f"Split ratios must sum to 1.0, got {ratio_sum}")

    if not raw_dir.exists():
        raise SystemExit(f"Raw dataset directory does not exist: {raw_dir}")

    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)

    image_paths = sorted(
        path for path in raw_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS and not path.name.endswith(".meta")
    )

    pairs: list[Path] = []
    warnings: list[str] = []
    for image_path in image_paths:
        label_path = image_path.with_suffix(".txt")
        if not label_path.exists():
            warnings.append(f"Missing label for {image_path.name}")
            continue
        warnings.extend(validate_label_file(label_path))
        pairs.append(image_path)

    positives = [path for path in pairs if label_is_positive(path.with_suffix(".txt"))]
    negatives = [path for path in pairs if not label_is_positive(path.with_suffix(".txt"))]

    rng = random.Random(args.seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    split_map = {"train": [], "val": [], "test": []}
    for group in (positives, negatives):
        group_splits = split_items(group, args.train, args.val)
        for split_name, split_items_list in group_splits.items():
            split_map[split_name].extend(split_items_list)
            rng.shuffle(split_map[split_name])

    for split_name in split_map:
        (out_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)

        for image_path in split_map[split_name]:
            label_path = image_path.with_suffix(".txt")
            shutil.copy2(image_path, out_dir / "images" / split_name / image_path.name)
            shutil.copy2(label_path, out_dir / "labels" / split_name / label_path.name)

    yaml_text = "\n".join(
        [
            f"path: {out_dir.as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            "  0: T-55",
            "",
        ]
    )
    (out_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")

    stats = {
        split: {
            "images": len(paths),
            "positive": sum(label_is_positive(path.with_suffix(".txt")) for path in paths),
            "negative": sum(not label_is_positive(path.with_suffix(".txt")) for path in paths),
        }
        for split, paths in split_map.items()
    }
    stats["total"] = {
        "images": len(pairs),
        "positive": len(positives),
        "negative": len(negatives),
    }
    stats["warnings"] = warnings
    (out_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Prepared dataset: {out_dir}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
