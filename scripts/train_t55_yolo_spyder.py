from pathlib import Path

from ultralytics import YOLO


# Step 2: Train a YOLO model with the Unity synthetic dataset.
# Run this file in Spyder when you want to train or resume training.

DATA_YAML = r"C:\Users\popod\Desktop\TankProject\yolo_dataset\data.yaml"
PROJECT_DIR = r"C:\Users\popod\Desktop\TankProject\runs"

# "resume" continues an interrupted run from weights/last.pt.
# "train_new" starts a new run from yolo11n.pt.
MODE = "resume"

EXISTING_RUN_NAME = "t55_yolo-2"
NEW_RUN_NAME = "t55_yolo_final"

MODEL = "yolo11n.pt"
EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 4
WORKERS = 0
TRAIN_DEVICE = 0


def train_new():
    model = YOLO(MODEL)
    return model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        project=PROJECT_DIR,
        name=NEW_RUN_NAME,
        device=TRAIN_DEVICE,
        plots=True,
    )


def resume_training():
    last_weight = Path(PROJECT_DIR) / EXISTING_RUN_NAME / "weights" / "last.pt"
    if not last_weight.exists():
        raise FileNotFoundError(f"last.pt was not found: {last_weight}")

    model = YOLO(str(last_weight))
    return model.train(
        resume=True,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        device=TRAIN_DEVICE,
        plots=True,
    )


if __name__ == "__main__":
    if MODE == "resume":
        print("=== Step 2: Resume training ===")
        resume_training()
        print(f"Training folder: {Path(PROJECT_DIR) / EXISTING_RUN_NAME}")
    elif MODE == "train_new":
        print("=== Step 2: New training ===")
        train_new()
        print(f"Training folder: {Path(PROJECT_DIR) / NEW_RUN_NAME}")
    else:
        raise ValueError('MODE must be "resume" or "train_new".')

    print("=== Done ===")
