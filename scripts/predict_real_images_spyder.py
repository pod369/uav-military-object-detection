from pathlib import Path

from ultralytics import YOLO


# Step 4 visual check: Predict on real photos without manual labels.
# Put real T-55 photos in REAL_IMAGE_DIR, then run this file in Spyder.

PROJECT_DIR = r"C:\Users\popod\Desktop\TankProject\runs"
RUN_NAME = "t55_yolo-2"
REAL_IMAGE_DIR = r"C:\Users\popod\Desktop\TankProject\real_images"

IMG_SIZE = 640
CONFIDENCE = 0.25
DEVICE = "cpu"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder):
    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)
    return sorted(
        file for file in path.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    )


def predict_real_images():
    best_weight = Path(PROJECT_DIR) / RUN_NAME / "weights" / "best.pt"
    if not best_weight.exists():
        raise FileNotFoundError(f"best.pt was not found: {best_weight}")

    images = list_images(REAL_IMAGE_DIR)
    if not images:
        print(f"No images found in: {REAL_IMAGE_DIR}")
        print("Add real photos first, then run this script again.")
        return None

    model = YOLO(str(best_weight))
    return model.predict(
        source=REAL_IMAGE_DIR,
        imgsz=IMG_SIZE,
        conf=CONFIDENCE,
        device=DEVICE,
        project=PROJECT_DIR,
        name=f"{RUN_NAME}_real_predict",
        exist_ok=True,
        save=True,
    )


if __name__ == "__main__":
    print("=== Step 4: Real image visual prediction ===")
    results = predict_real_images()
    if results is not None:
        print(f"Result folder: {Path(PROJECT_DIR) / (RUN_NAME + '_real_predict')}")
    print("=== Done ===")
