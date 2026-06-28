# Waste Classification and Detection Project

## 1. Mục tiêu đề tài

Đề tài xây dựng hệ thống nhận diện và phân loại rác thải dựa trên Deep Learning, gồm hai bài toán chính:

1. **Classification**: phân loại ảnh rác thành 16 lớp chi tiết như `plastic`, `paper`, `glass`, `meat_waste`, `starch_waste`, `mixed_food_waste`,...
2. **Detection**: phát hiện vùng chứa rác trong ảnh bằng YOLO.
3. **Mapping nhãn**: sau khi phân loại nhãn chi tiết, hệ thống ánh xạ kết quả sang nhóm lớn `organic` hoặc `inorganic`.

Pipeline cuối cùng hướng tới:

```text
Ảnh đầu vào
    ↓
YOLOv8 detect vùng rác
    ↓
Crop từng object
    ↓
Classifier dự đoán 16 lớp chi tiết
    ↓
Mapping fine label → organic/inorganic
    ↓
Hiển thị kết quả
```

---

## 2. Pipeline tổng quát

![Pipeline tổng quát](assets/pipeline_overview.png)

---

## 3. Cấu trúc thư mục

```text
source/
├── data/
│   ├── Dataset_classification_processed/
│   │   └── fine/
│   │       ├── train/
│   │       ├── val/
│   │       └── test/
│   │
│   └── Detection_dataset_processed/
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       ├── labels/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── data.yaml
│
├── scripts/
│   ├── check_classification_dataset.py
│   ├── check_yolo_dataset.py
│   ├── train_classification_baseline.py
│   └── app_classification_test.py
│
├── requirements.txt
├── requirements_gpu.txt
├── README.md
└── README_TRAINING.md
```


## 4. Dataset classification/detection

### 4.1. Classification dataset

Classification dataset sau xử lý nằm tại:
```text
data/Dataset_classification_processed/fine/
```

Bên trong gồm:
```text
fine/
├── train/
├── val/
└── test/
```

Bộ classification hiện tại:
```text
Tổng ảnh: 17,887
Số lớp: 16
Ảnh lỗi: 0
Thư mục class bị thiếu: 0
```

16 lớp classification:
```text
battery
biological
cardboard
clothes
fruit_waste
glass
meat_waste
metal
mixed_food_waste
paper
plant_waste
plastic
shoes
starch_waste
trash
vegetable_waste
```

### 4.2. Detection dataset

Detection dataset sau xử lý nằm tại:

```text
data/Detection_dataset_processed/
```

Cấu trúc YOLO:

```text
Detection_dataset_processed/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

Kết quả kiểm tra detection dataset:

```text
Tổng ảnh: 9,132
Tổng label: 9,132
```

Dataset detection dùng để train YOLOv8 phát hiện vùng chứa rác.

---

## 5. Mapping 16 class → organic/inorganic

Hệ thống không chỉ dự đoán `organic/inorganic` trực tiếp. Thay vào đó, mô hình classification dự đoán nhãn chi tiết trước, sau đó ánh xạ sang nhóm lớn.

### Organic

```text
fruit_waste       → organic
vegetable_waste   → organic
meat_waste        → organic
starch_waste      → organic
plant_waste       → organic
mixed_food_waste  → organic
biological        → organic
```

### Inorganic

```text
battery    → inorganic
cardboard  → inorganic
clothes    → inorganic
glass      → inorganic
metal      → inorganic
paper      → inorganic
plastic    → inorganic
shoes      → inorganic
trash      → inorganic
```

---


## 6. Cài đặt nhanh

Mở terminal tại thư mục dự án:

```powershell
cd <project-root>
```

Tạo và kích hoạt môi trường:

```powershell
python -m venv DeepL
.\DeepL\Scripts\Activate.ps1
```

Cài thư viện:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements_gpu.txt
```

Kiểm tra GPU:

```powershell
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 7. Check dataset nhanh

Check classification dataset:

```powershell
python scripts\check_classification_dataset.py
```

Check detection dataset:

```powershell
python scripts\check_yolo_dataset.py
```

Nếu kết quả classification không có ảnh lỗi và detection có số ảnh bằng số label thì dữ liệu đã sẵn sàng.

---

## 8. Chạy Demo Ứng Dụng

Hệ thống cung cấp sẵn hai file giao diện Web UI (sử dụng Gradio) để chạy thử nghiệm:

### 8.1. Demo tích hợp toàn bộ Pipeline (Khuyên dùng)
File này thực hiện trọn vẹn quy trình: **Ảnh đầu vào → YOLOv8 phát hiện vùng rác → Crop vật thể → Phân loại 16 lớp → Ánh xạ Organic/Inorganic**.

**Lưu ý đặc biệt:** Repo này **đã đính kèm sẵn (bundle)** 2 file model weight cần thiết nhất, chỉ cần clone repo và chạy mà không cần tải thêm weights thủ công:
1. Classification model: `runs/classification/efficientnet_b0_baseline_20260525_135202/best_model.pth`
2. YOLO model: `scripts/runs/detect/runs/yolo/yolo_trash_20epoch-2/weights/best.pt`

Chạy ứng dụng:
```powershell
python scripts\app_pipeline_demo.py
```

### 8.2. Demo Phân Loại Rác riêng lẻ (Classification Test)
File giao diện đơn giản dùng để kiểm tra riêng model phân loại 16 lớp trên ảnh đã crop sẵn:
```powershell
python scripts\app_classification_test.py
```

Khi chạy thành công, terminal sẽ hiển thị địa chỉ truy cập:
```text
Running on local URL: http://127.0.0.1:7860
```
Mở trình duyệt truy cập vào địa chỉ trên để sử dụng giao diện UI.

---

## 9. Link Google Drive dataset

Do dataset có dung lượng lớn, không lưu trực tiếp trên GitHub.
Google Drive: https://drive.google.com/drive/u/0/folders/1Jo81xS7--e9uifOLvG3EbcCAwwmeXu9V
