from pathlib import Path

from ultralytics import YOLO


# Step 2 result check: Evaluate the trained model on the Unity test split.
# This uses yolo_dataset/images/test and yolo_dataset/labels/test.

DATA_YAML = r"C:\Users\popod\Desktop\TankProject\yolo_dataset\data.yaml"
PROJECT_DIR = r"C:\Users\popod\Desktop\TankProject\runs"
RUN_NAME = "t55_yolo-2"

IMG_SIZE = 640
BATCH_SIZE = 4
WORKERS = 0
DEVICE = "cpu"


def print_metrics(results):
    box = results.box
    print("=== Metrics ===")
    print(f"Precision:   {box.mp:.4f}")
    print(f"Recall:      {box.mr:.4f}")
    print(f"mAP50:       {box.map50:.4f}")
    print(f"mAP50-95:    {box.map:.4f}")


def evaluate_unity_test():
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
        name=f"{RUN_NAME}_unity_test_eval",
        exist_ok=True,
        plots=True,
    )
    print_metrics(results)
    return results


if __name__ == "__main__":
    print("=== Step 2 result check: Unity test evaluation ===")
    evaluate_unity_test()
    print(f"Result folder: {Path(PROJECT_DIR) / (RUN_NAME + '_unity_test_eval')}")
    print("=== Done ===")
