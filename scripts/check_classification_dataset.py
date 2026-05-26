from pathlib import Path
from PIL import Image


BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = BASE_DIR / "data" / "Dataset_classification_processed" / "fine"

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

CLASSES = [
    "battery",
    "biological",
    "cardboard",
    "clothes",
    "fruit_waste",
    "glass",
    "meat_waste",
    "metal",
    "mixed_food_waste",
    "paper",
    "plant_waste",
    "plastic",
    "shoes",
    "starch_waste",
    "trash",
    "vegetable_waste",
]


def is_valid_image(image_path: Path) -> bool:
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Không tìm thấy dataset: {DATASET_DIR}")

    total_all = 0
    invalid_images = []
    missing_class_dirs = []

    for split in ["train", "val", "test"]:
        print(f"\n===== {split.upper()} =====")

        split_dir = DATASET_DIR / split

        if not split_dir.exists():
            print(f"[ERROR] Thiếu thư mục split: {split_dir}")
            continue

        split_total = 0

        for cls in CLASSES:
            class_dir = split_dir / cls

            if not class_dir.exists():
                missing_class_dirs.append(str(class_dir))
                print(f"{cls}: thiếu thư mục")
                continue

            images = []

            for ext in IMAGE_EXTS:
                images.extend(class_dir.rglob(f"*{ext}"))

            count = len(images)
            split_total += count

            print(f"{cls}: {count}")

            for image_path in images:
                if not is_valid_image(image_path):
                    invalid_images.append(str(image_path))

        print(f"Tổng {split}: {split_total}")
        total_all += split_total

    print("\n===== TỔNG KẾT CLASSIFICATION =====")
    print(f"Tổng ảnh: {total_all}")
    print(f"Thư mục class bị thiếu: {len(missing_class_dirs)}")
    print(f"Ảnh lỗi: {len(invalid_images)}")

    if missing_class_dirs:
        print("\nVí dụ thư mục thiếu:")
        for path in missing_class_dirs[:10]:
            print(path)

    if invalid_images:
        print("\nVí dụ ảnh lỗi:")
        for path in invalid_images[:10]:
            print(path)

    if not missing_class_dirs and not invalid_images:
        print("\nDataset classification hợp lệ.")


if __name__ == "__main__":
    main()