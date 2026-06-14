from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw YOLO labels on sample images for visual inspection.")
    parser.add_argument(
        "--dataset-dir",
        default=r"C:\Users\popod\Desktop\TankProject\yolo_dataset",
        help="Prepared YOLO dataset root containing images/ and labels/.",
    )
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def draw_one(image_path: Path, label_path: Path, out_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    label_text = label_path.read_text(encoding="utf-8").strip() if label_path.exists() else ""
    if label_text:
        for line in label_text.splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            class_id, xc, yc, bw, bh = parts
            xc, yc, bw, bh = map(float, (xc, yc, bw, bh))
            x1 = (xc - bw / 2) * width
            y1 = (yc - bh / 2) * height
            x2 = (xc + bw / 2) * width
            y2 = (yc + bh / 2) * height
            draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=3)
            draw.text((x1 + 4, max(0, y1 - 18)), f"T-55 class {class_id}", fill=(255, 0, 0))
    else:
        draw.text((10, 10), "NEGATIVE / no object", fill=(255, 255, 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    image_dir = dataset_dir / "images" / args.split
    label_dir = dataset_dir / "labels" / args.split
    out_dir = dataset_dir / "preview_labels" / args.split

    image_paths = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.jpeg"))
    positives = [path for path in image_paths if (label_dir / f"{path.stem}.txt").read_text(encoding="utf-8").strip()]
    negatives = [path for path in image_paths if not (label_dir / f"{path.stem}.txt").read_text(encoding="utf-8").strip()]

    rng = random.Random(args.seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    selected = positives[: args.count] + negatives[: max(5, args.count // 5)]

    for image_path in selected:
        label_path = label_dir / f"{image_path.stem}.txt"
        draw_one(image_path, label_path, out_dir / image_path.name)

    print(f"Wrote previews to: {out_dir}")
    print(f"Positive previews: {min(args.count, len(positives))}, negative previews: {min(max(5, args.count // 5), len(negatives))}")


if __name__ == "__main__":
    main()
