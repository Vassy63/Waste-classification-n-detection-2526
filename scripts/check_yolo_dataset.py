from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
YOLO_DIR = BASE_DIR / "data" / "Detection_dataset_processed"

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def check_dataset():
    total_images = 0
    total_labels = 0

    for split in ["train", "val", "test"]:
        image_dir = YOLO_DIR / "images" / split
        label_dir = YOLO_DIR / "labels" / split

        images = []

        if image_dir.exists():
            for ext in IMAGE_EXTS:
                images.extend(image_dir.glob(f"*{ext}"))

        labels = list(label_dir.glob("*.txt")) if label_dir.exists() else []

        missing_labels = []
        empty_labels = []
        invalid_lines = []

        for img_path in images:
            label_path = label_dir / f"{img_path.stem}.txt"

            if not label_path.exists():
                missing_labels.append(img_path.name)
                continue

            if label_path.stat().st_size == 0:
                empty_labels.append(img_path.name)
                continue

            with open(label_path, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, start=1):
                    parts = line.strip().split()

                    if len(parts) != 5:
                        invalid_lines.append((label_path.name, line_idx, line.strip()))
                        continue

                    try:
                        cls, x, y, w, h = parts
                        cls = int(cls)
                        x, y, w, h = map(float, [x, y, w, h])

                        if cls != 0:
                            invalid_lines.append((label_path.name, line_idx, "class_id khác 0"))

                        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                            invalid_lines.append((label_path.name, line_idx, "bbox không nằm trong [0,1]"))

                    except Exception:
                        invalid_lines.append((label_path.name, line_idx, line.strip()))

        total_images += len(images)
        total_labels += len(labels)

        print(f"\n===== {split.upper()} =====")
        print(f"Số ảnh: {len(images)}")
        print(f"Số label: {len(labels)}")
        print(f"Thiếu label: {len(missing_labels)}")
        print(f"Label rỗng: {len(empty_labels)}")
        print(f"Dòng label lỗi: {len(invalid_lines)}")

        if missing_labels:
            print("Ví dụ ảnh thiếu label:", missing_labels[:5])

        if invalid_lines:
            print("Ví dụ dòng lỗi:", invalid_lines[:5])

    print("\n===== TỔNG =====")
    print(f"Tổng ảnh: {total_images}")
    print(f"Tổng label: {total_labels}")


if __name__ == "__main__":
    check_dataset()