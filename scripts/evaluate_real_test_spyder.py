from pathlib import Path

from ultralytics import YOLO


# Step 4 quantitative check: Evaluate on manually labeled real photos.
#
# Required folder structure:
#   real_test_dataset/images/test/*.jpg or *.png
#   real_test_dataset/labels/test/*.txt
#
# For T-55 images, draw a YOLO box in the matching .txt file.
# For non-T-55 or background images, create an empty matching .txt file.

PROJECT_DIR = r"C:\Users\popod\Desktop\TankProject\runs"
RUN_NAME = "t55_yolo-2"
REAL_TEST_DIR = r"C:\Users\popod\Desktop\TankProject\real_test_dataset"
DATA_YAML = r"C:\Users\popod\Desktop\TankProject\real_test_dataset\data.yaml"

IMG_SIZE = 640
BATCH_SIZE = 4
WORKERS = 0
DEVICE = "cpu"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def print_metrics(results):
    box = results.box
    print("=== Metrics ===")
    print(f"Precision:   {box.mp:.4f}")
    print(f"Recall:      {box.mr:.4f}")
    print(f"mAP50:       {box.map50:.4f}")
    print(f"mAP50-95:    {box.map:.4f}")


def validate_real_test_dataset():
    root = Path(REAL_TEST_DIR)
    image_dir = root / "images" / "test"
    label_dir = root / "labels" / "test"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        file for file in image_dir.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No real test images found: {image_dir}")

    missing_labels = [
        image.name for image in images
        if not (label_dir / f"{image.stem}.txt").exists()
    ]
    if missing_labels:
        joined = "\n".join(missing_labels[:20])
        raise FileNotFoundError(
            "Every real test image needs a matching label file. "
            "Use an empty .txt file for non-T-55/background images.\n"
            f"Missing labels:\n{joined}"
        )


def evaluate_real_test():
    validate_real_test_dataset()

    best_weight = Path(PROJECT_DIR) / RUN_NAME / "weights" / "best.pt"
    if not best_weight.exists():
        raise FileNotFoundError(f"best.pt was not found: {best_weight}")

    model = YOLO(str(best_weight))
    results = model.val(
        data=DATA_YAML,
        split="test",
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        device=DEVICE,
        project=PROJECT_DIR,
        name=f"{RUN_NAME}_real_test_eval",
        exist_ok=True,
        plots=True,
    )
    print_metrics(results)
    return results


if __name__ == "__main__":
    print("=== Step 4: Real test evaluation ===")
    evaluate_real_test()
    print(f"Result folder: {Path(PROJECT_DIR) / (RUN_NAME + '_real_test_eval')}")
    print("=== Done ===")
