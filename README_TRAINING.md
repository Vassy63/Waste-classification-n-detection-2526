# README_TRAINING.md

> [!IMPORTANT]
> **Lưu ý quan trọng:** Dữ liệu (dataset) có dung lượng rất lớn nên nhóm không push trực tiếp lên GitHub và cũng không nén lại để gửi (tránh giới hạn dung lượng và lỗi nén).

## 1. Mục đích

File này hướng dẫn cách kiểm tra dataset và train mô hình baseline cho bài toán classification rác thải.

## 2. Cấu trúc thư mục cần có

Tất cả lệnh bên dưới chạy tại thư mục gốc của dự án:

```powershell
<project-root>
```

## 3. Kích hoạt môi trường ảo

Mở terminal tại thư mục dự án:

```powershell
cd <project-root>
```

Kích hoạt môi trường `DeepL`:

```powershell
.\DeepL\Scripts\Activate.ps1
```

Nếu terminal hiện như sau là đúng:

```powershell
(DeepL) PS <project-root>>
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


## 11. 2 baseline và các cải tiến khác trong đề tài

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
Hierarchical Multi-Task Learning
    ├── Shared EfficientNet-B0 backbone
    ├── Fine head: phân loại 16 lớp rác chi tiết
    ├── Coarse head: phân loại Organic / Inorganic
    ├── Joint loss = Fine loss + alpha × Coarse loss
    └── Đánh giá đồng thời fine metrics và coarse metrics
    
YOLO + classifier pipeline
```

Lệnh huấn luyện Multi-Task:
```powershell
python scripts/train_classification_multitask.py --epochs 50 --batch 16 --lr 0.00005 --alpha 0.3 --dropout 0.4 --weight_decay 0.0005 --patience 8
```                              

---

## 12. Lỗi thường gặp

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

## 13. Lệnh chạy nhanh

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
