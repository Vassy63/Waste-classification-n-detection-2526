from pathlib import Path
from ultralytics import YOLO
import torch

# =====================================
# CẤU HÌNH
# =====================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "data" / "Detection_dataset_processed"
DATA_YAML = DATASET_DIR / "data.yaml"

EPOCHS = 20
IMGSZ = 640
BATCH = 8

PROJECT = "runs/yolo"
NAME = "yolo_trash_20epoch"

# =====================================
# KIỂM TRA DATASET
# =====================================

def check_dataset():
    required_dirs = [
        DATASET_DIR / "images/train",
        DATASET_DIR / "images/val",
        DATASET_DIR / "images/test",
        DATASET_DIR / "labels/train",
        DATASET_DIR / "labels/val",
        DATASET_DIR / "labels/test",
    ]

    for d in required_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Thiếu thư mục: {d}")

    print("✅ Dataset OK")


# =====================================
# TẠO LẠI data.yaml
# =====================================

def write_yaml():
    yaml_text = f"""
path: {DATASET_DIR.resolve().as_posix()}

train: images/train
val: images/val
test: images/test

names:
  0: trash
"""

    with open(DATA_YAML, "w", encoding="utf-8") as f:
        f.write(yaml_text)

    print("✅ Đã cập nhật data.yaml")


# =====================================
# THỐNG KÊ DATASET
# =====================================

def count_images(folder):
    exts = [".jpg", ".jpeg", ".png", ".bmp"]
    total = 0

    for ext in exts:
        total += len(list(folder.glob(f"*{ext}")))

    return total


def dataset_info():
    train_count = count_images(DATASET_DIR / "images/train")
    val_count = count_images(DATASET_DIR / "images/val")
    test_count = count_images(DATASET_DIR / "images/test")

    print("\n========== DATASET ==========")
    print(f"Train : {train_count}")
    print(f"Val   : {val_count}")
    print(f"Test  : {test_count}")
    print("=============================\n")


# =====================================
# TRAIN
# =====================================

def train():
    if torch.cuda.is_available():
        device = 0
        print("🚀 GPU:", torch.cuda.get_device_name(0))
    else:
        device = "cpu"
        print("⚠️ Không tìm thấy GPU, dùng CPU")

    model = YOLO("yolov8n.pt")

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=device,
        patience=10,
        project=PROJECT,
        name=NAME,
        save=True
    )

    print("\n✅ TRAIN HOÀN THÀNH")
    print(f"Model tốt nhất:")
    print(f"{PROJECT}/{NAME}/weights/best.pt")


# =====================================
# MAIN
# =====================================

if __name__ == "__main__":
    check_dataset()
    write_yaml()
    dataset_info()
    train()