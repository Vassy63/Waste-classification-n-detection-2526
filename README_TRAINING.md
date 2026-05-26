# README_TRAINING.md

## 1. Mục đích

File này hướng dẫn cách kiểm tra dataset và train 2 mô hình baseline cho bài toán classification rác thải:

- EfficientNet-B0 baseline
- ResNet50 baseline

Hai baseline này dùng để so sánh backbone, chọn mô hình chính và làm mốc đối chứng trước khi áp dụng các cải tiến như class weight, label smoothing, augmentation và hard example mining.

---

## 2. Cấu trúc thư mục cần có

Tất cả lệnh bên dưới chạy tại thư mục:

```powershell
E:\Deep_learning\đồ án\source
```

Cấu trúc dữ liệu classification cần có:

```text
source/
├── data/
│   └── Dataset_classification_processed/
│       └── fine/
│           ├── train/
│           ├── val/
│           └── test/
├── scripts/
│   ├── check_classification_dataset.py
│   ├── check_yolo_dataset.py
│   └── train_classification_baseline.py
├── requirements.txt
├── requirements_gpu.txt
└── README_TRAINING.md
```

Bộ classification hiện tại gồm 16 lớp:

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

---

## 3. Kích hoạt môi trường ảo

Mở terminal tại thư mục `source`:

```powershell
cd "E:\Deep_learning\đồ án\source"
```

Kích hoạt môi trường `DeepL`:

```powershell
.\DeepL\Scripts\Activate.ps1
```

Nếu terminal hiện như sau là đúng:

```powershell
(DeepL) PS E:\Deep_learning\đồ án\source>
```

---

## 4. Cài thư viện

Cài toàn bộ thư viện trong `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements_gpu.txt
```

Kiểm tra nhanh các thư viện chính:

```powershell
python -c "import torch, torchvision, pandas, sklearn, matplotlib, tqdm; print('OK')"
```

Nếu hiện:

```text
OK
```

là môi trường đã sẵn sàng.

Kiểm tra PyTorch và GPU:

```powershell
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

Nếu `CUDA: True` thì máy có thể train bằng GPU. Nếu `CUDA: False` thì vẫn train được bằng CPU nhưng sẽ chậm hơn.

---

## 5. Kiểm tra dataset trước khi train

### 5.1. Check classification dataset

Chạy:

```powershell
python scripts\check_classification_dataset.py
```

Kết quả hợp lệ hiện tại:

```text
Tổng ảnh: 17887
Thư mục class bị thiếu: 0
Ảnh lỗi: 0
```

Nếu `Ảnh lỗi = 0` và `Thư mục class bị thiếu = 0` thì có thể train classification.

### 5.2. Check detection dataset

Chạy:

```powershell
python scripts\check_yolo_dataset.py
```

Kết quả hợp lệ hiện tại:

```text
Tổng ảnh: 9132
Tổng label: 9132
```

Nếu số ảnh bằng số label và không có label lỗi thì detection dataset đã sẵn sàng cho YOLO.

---

## 6. Train EfficientNet-B0 baseline

Lệnh train mặc định:

```powershell
python scripts\train_classification_baseline.py --model efficientnet_b0
```

Lệnh trên sử dụng các siêu tham số mặc định trong code:

```text
image_size = 224
epochs = 25
batch_size = 16
learning_rate = 0.0001
```

Nếu máy yếu hoặc bị lỗi bộ nhớ, giảm batch size:

```powershell
python scripts\train_classification_baseline.py --model efficientnet_b0 --batch 8
```

Có thể tự chỉnh số epoch và learning rate:

```powershell
python scripts\train_classification_baseline.py --model efficientnet_b0 --epochs 30 --batch 8 --lr 0.0001
```

---

## 7. Train ResNet50 baseline

Sau khi train EfficientNet-B0 xong, train tiếp ResNet50:

```powershell
python scripts\train_classification_baseline.py --model resnet50
```

Nếu máy yếu hoặc bị lỗi bộ nhớ, giảm batch size:

```powershell
python scripts\train_classification_baseline.py --model resnet50 --batch 8
```

Có thể tự chỉnh số epoch và learning rate:

```powershell
python scripts\train_classification_baseline.py --model resnet50 --epochs 30 --batch 8 --lr 0.0001
```

---

## 8. Fine-tuning v3

Nếu muốn chạy lại thử nghiệm fine-tuning v3, dùng script riêng hoặc phiên bản có hỗ trợ `--freeze_epochs`.

Ý tưởng của v3:

```text
5 epoch đầu: freeze backbone, chỉ train classifier head.
Từ epoch 6: unfreeze toàn bộ model và fine-tune với learning rate nhỏ hơn.
```

Lệnh tham khảo cho EfficientNet-B0:

```powershell
python scripts\train_classification_baseline.py --model efficientnet_b0 --epochs 20 --batch 16 --lr 0.00005 --freeze_epochs 5
```

Lệnh tham khảo cho ResNet50:

```powershell
python scripts\train_classification_baseline.py --model resnet50 --epochs 20 --batch 8 --lr 0.00005 --freeze_epochs 5
```

Lưu ý: nếu file `train_classification_baseline.py` đã được khôi phục về bản baseline gốc thì sẽ không còn tham số `--freeze_epochs`. Khi đó chỉ dùng file này để train baseline.

---

## 9. Lưu ý khi chạy script train

Tham số `--model` là bắt buộc vì script cần biết train mô hình nào.

Các model được hỗ trợ:

```text
efficientnet_b0
resnet50
```

Các tham số tùy chọn:

```text
--epochs    số epoch train
--batch     batch size
--lr        learning rate
```

Ví dụ:

```powershell
python scripts\train_classification_baseline.py --model efficientnet_b0 --epochs 25 --batch 16 --lr 0.0001
```

Không nên bấm nút Run trực tiếp trong VS Code nếu chưa cấu hình tham số, vì script sẽ báo lỗi thiếu `--model`.

---

## 10. Kết quả sau khi train

Sau khi train xong, kết quả được lưu trong:

```text
runs/classification/
```

Ví dụ:

```text
runs/classification/efficientnet_b0_baseline_YYYYMMDD_HHMMSS/
runs/classification/resnet50_baseline_YYYYMMDD_HHMMSS/
```

Mỗi thư mục kết quả gồm:

```text
best_model.pth
training_history.csv
loss_curve.png
accuracy_curve.png
confusion_matrix.png
classification_report.csv
classification_report.txt
test_metrics.json
class_names.json
train_config.json
```

Ý nghĩa các file quan trọng:

| File | Ý nghĩa |
|---|---|
| `best_model.pth` | Model tốt nhất theo validation accuracy |
| `training_history.csv` | Lịch sử train loss/accuracy và val loss/accuracy |
| `loss_curve.png` | Biểu đồ loss |
| `accuracy_curve.png` | Biểu đồ accuracy |
| `confusion_matrix.png` | Ma trận nhầm lẫn |
| `classification_report.csv` | Precision, recall, F1-score từng class |
| `test_metrics.json` | Accuracy, macro F1, weighted F1 trên test set |
| `class_names.json` | Danh sách class theo thứ tự model |
| `train_config.json` | Cấu hình train đã dùng |

---

## 11. So sánh 2 baseline

Sau khi train xong cả EfficientNet-B0 và ResNet50, mở file:

```text
test_metrics.json
```

trong từng thư mục model và lấy các chỉ số:

```text
test_accuracy
macro_f1
weighted_f1
```

Lập bảng so sánh:

| Model | Test Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| EfficientNet-B0 baseline | ... | ... | ... |
| ResNet50 baseline | ... | ... | ... |

Nếu EfficientNet-B0 có kết quả gần bằng ResNet50, nên ưu tiên EfficientNet-B0 vì nhẹ và phù hợp demo hơn. Nếu ResNet50 tốt hơn rõ rệt, có thể chọn ResNet50 làm model chính.

---

## 12. Vai trò của 2 baseline trong đề tài

Hai baseline này chưa phải là phần cải tiến chính. Chúng dùng để:

```text
1. So sánh hai backbone pretrained: EfficientNet-B0 và ResNet50.
2. Chọn mô hình chính để cải tiến tiếp.
3. Làm mốc đối chứng trước khi áp dụng improved training và hard example mining.
```

Phần cải tiến chính của đề tài nằm ở các bước sau:

```text
Baseline model
    ↓
Improved training
    ├── Data augmentation
    ├── Class weight
    ├── Label smoothing
    ├── Learning rate scheduler
    └── Early stopping
    ↓
Hard Example Mining
    ├── Lấy ảnh dự đoán sai
    ├── Lấy ảnh confidence thấp
    └── Fine-tune lại model
    ↓
YOLO + classifier pipeline
```

---

## 13. Câu giải thích trong báo cáo

Có thể ghi:

```text
Đề tài huấn luyện hai mô hình baseline sử dụng học chuyển tiếp là EfficientNet-B0 và ResNet50 trên cùng bộ dữ liệu classification 16 lớp. Cả hai mô hình được khởi tạo từ trọng số pretrained ImageNet và thay thế lớp phân loại cuối để phù hợp với số lớp rác của đề tài. Kết quả baseline được sử dụng làm mốc đối chứng để lựa chọn backbone phù hợp trước khi áp dụng các cải tiến như class weighting, label smoothing, data augmentation và hard example mining.
```

---

## 14. Lỗi thường gặp

### Lỗi thiếu `--model`

Nếu chạy:

```powershell
python scripts\train_classification_baseline.py
```

sẽ báo lỗi:

```text
error: the following arguments are required: --model
```

Cách sửa:

```powershell
python scripts\train_classification_baseline.py --model efficientnet_b0
```

hoặc:

```powershell
python scripts\train_classification_baseline.py --model resnet50
```

### Lỗi hết bộ nhớ

Giảm batch size:

```powershell
python scripts\train_classification_baseline.py --model efficientnet_b0 --batch 8
```

hoặc:

```powershell
python scripts\train_classification_baseline.py --model resnet50 --batch 8
```

### Train quá chậm

Kiểm tra GPU:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

Nếu kết quả là `False`, máy đang train bằng CPU nên sẽ chậm.

---

## 15. Lệnh chạy nhanh

Chạy EfficientNet-B0 baseline:

```powershell
python scripts\train_classification_baseline.py --model efficientnet_b0
```

Chạy ResNet50 baseline:

```powershell
python scripts\train_classification_baseline.py --model resnet50
```

Check classification dataset:

```powershell
python scripts\check_classification_dataset.py
```

Check detection dataset:

```powershell
python scripts\check_yolo_dataset.py
```
