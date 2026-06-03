import os
import sys
import time
import cv2
import torch
import torch.nn as nn
import numpy as np
import gradio as gr
from pathlib import Path
from PIL import Image
from torchvision import transforms, models
from ultralytics import YOLO

# ==========================================
# PATH CONFIGURATION & DYNAMIC WEIGHT FINDING
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[1]

# Dynamic search list for YOLO weights
YOLO_SEARCH_PATHS = [
    BASE_DIR / "scripts" / "runs" / "detect" / "runs" / "yolo" / "yolo_trash_20epoch-2" / "weights" / "best.pt",
    BASE_DIR / "runs" / "detect" / "runs" / "yolo" / "yolo_trash_20epoch-2" / "weights" / "best.pt",
    BASE_DIR / "scripts" / "runs" / "yolo" / "yolo_trash_20epoch-2" / "weights" / "best.pt",
    BASE_DIR / "runs" / "yolo" / "yolo_trash_20epoch-2" / "weights" / "best.pt",
    BASE_DIR / "scripts" / "yolo26n.pt",
    BASE_DIR / "scripts" / "yolov8n.pt",
]

CLASSIFIER_PATH = BASE_DIR / "runs" / "classification" / "efficientnet_b0_baseline_20260525_135202" / "best_model.pth"

# ==========================================
# CLASS CONFIGURATIONS
# ==========================================
DEFAULT_CLASS_NAMES = [
    "battery", "biological", "cardboard", "clothes", "fruit_waste", 
    "glass", "meat_waste", "metal", "mixed_food_waste", "paper", 
    "plant_waste", "plastic", "shoes", "starch_waste", "trash", "vegetable_waste"
]

FINE_TO_COARSE = {
    "fruit_waste": "organic",
    "vegetable_waste": "organic",
    "meat_waste": "organic",
    "starch_waste": "organic",
    "plant_waste": "organic",
    "mixed_food_waste": "organic",
    "biological": "organic",

    "battery": "inorganic",
    "cardboard": "inorganic",
    "clothes": "inorganic",
    "glass": "inorganic",
    "metal": "inorganic",
    "paper": "inorganic",
    "plastic": "inorganic",
    "shoes": "inorganic",
    "trash": "inorganic",
}

# ==========================================
# INITIALIZATION & LOADING FUNCTIONS
# ==========================================
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find_yolo_model():
    """Dynamically search for YOLO best.pt weights in typical project paths."""
    for path in YOLO_SEARCH_PATHS:
        if path.exists():
            print(f"Found YOLO weights at: {path}")
            return path
    raise FileNotFoundError(
        "Không tìm thấy file weights tốt nhất của YOLO (best.pt).\n"
        "Vui lòng chạy train YOLO trước hoặc kiểm tra cấu trúc thư mục runs."
    )

def load_models(device):
    # 1. Load YOLO
    yolo_path = find_yolo_model()
    yolo_model = YOLO(str(yolo_path))
    
    # 2. Load Classifier Checkpoint
    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy model classification tại: {CLASSIFIER_PATH}")
    
    checkpoint = torch.load(CLASSIFIER_PATH, map_location="cpu")
    class_names = checkpoint.get("class_names", DEFAULT_CLASS_NAMES)
    num_classes = checkpoint.get("num_classes", len(class_names))
    model_name = checkpoint.get("model_name", "efficientnet_b0")
    
    # 3. Build Classifier (Matching baseline v0 architecture exactly)
    if model_name == "efficientnet_b0":
        classifier_model = models.efficientnet_b0(weights=None)
        in_features = classifier_model.classifier[1].in_features
        classifier_model.classifier[1] = nn.Linear(in_features, num_classes)
    elif model_name == "resnet50":
        classifier_model = models.resnet50(weights=None)
        in_features = classifier_model.fc.in_features
        classifier_model.fc = nn.Linear(in_features, num_classes)
    else:
         raise ValueError(f"Không hỗ trợ model name: {model_name}")
         
    classifier_model.load_state_dict(checkpoint["model_state_dict"])
    classifier_model = classifier_model.to(device)
    classifier_model.eval()
    
    return yolo_model, classifier_model, class_names, yolo_path

# Initialize models
device = get_device()
yolo_model, classifier_model, class_names, yolo_path_used = load_models(device)

# Image classification transforms
IMAGE_SIZE = 224
classifier_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# ==========================================
# PIPELINE INFERENCE
# ==========================================
def process_pipeline(input_image, conf_threshold=0.25, iou_threshold=0.45):
    if input_image is None:
        return None, "Vui lòng tải ảnh lên để bắt đầu.", []
    
    # Convert input to numpy array & PIL Image
    if isinstance(input_image, Image.Image):
        pil_img = input_image
        img_np = np.array(input_image)
    else:
        img_np = input_image.copy()
        pil_img = Image.fromarray(input_image)
        
    img_h, img_w = img_np.shape[:2]
    
    # Run YOLO detection
    results = yolo_model.predict(
        source=pil_img,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False
    )
    
    detections = results[0].boxes
    annotated_img = img_np.copy()
    objects_list = []
    
    # Fallback Mechanism: No objects detected
    if len(detections) == 0:
        fallback_msg = "⚠️ YOLO không phát hiện vùng rác, hệ thống tự động phân loại toàn ảnh."
        
        # Crop whole image
        crop_pil = pil_img.convert("RGB")
        input_tensor = classifier_transforms(crop_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = classifier_model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            
        top_probs, top_indices = torch.topk(probs, k=3)
        best_idx = top_indices[0].item()
        best_label = class_names[best_idx]
        best_conf = top_probs[0].item()
        coarse_label = FINE_TO_COARSE.get(best_label, "unknown")
        
        # Color based on category (Green for organic, Red/Indigo for inorganic)
        color = (34, 197, 94) if coarse_label == "organic" else (59, 130, 246) # RGB (Green / Blue)
        cv2.rectangle(annotated_img, (10, 10), (img_w - 10, img_h - 10), color[::-1], 3)
        
        # Label text
        label_text = f"ALL: {best_label} ({coarse_label.upper()}) | cls_conf: {best_conf:.2%}"
        cv2.putText(
            annotated_img, label_text, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color[::-1], 2, cv2.LINE_AA
        )
        
        # Build Top-3 predictions string
        top3_info = []
        for prob, idx in zip(top_probs, top_indices):
            c_name = class_names[idx.item()]
            c_group = FINE_TO_COARSE.get(c_name, "unknown")
            top3_info.append(f"{c_name} ({c_group}): {prob.item():.2%}")
            
        objects_list.append({
            "ID": "Toàn bộ ảnh (Fallback)",
            "YOLO Conf": "N/A",
            "Fine Class": best_label,
            "Coarse Class": coarse_label.upper(),
            "Classifier Conf": f"{best_conf:.2%}",
            "Top 3 Dự Đoán": ", ".join(top3_info)
        })
        
        return annotated_img, fallback_msg, objects_list
        
    # Main Pipeline: Bounding boxes detected
    for idx, box in enumerate(detections):
        # Extract bbox coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        yolo_conf = float(box.conf[0].cpu().numpy())
        
        # 10% padding
        w = x2 - x1
        h = y2 - y1
        pad_w = int(w * 0.1)
        pad_h = int(h * 0.1)
        
        x1_pad = max(0, x1 - pad_w)
        y1_pad = max(0, y1 - pad_h)
        x2_pad = min(img_w, x2 + pad_w)
        y2_pad = min(img_h, y2 + pad_h)
        
        # Crop & preprocess for classifier
        crop_np = img_np[y1_pad:y2_pad, x1_pad:x2_pad]
        if crop_np.size == 0:
            continue
            
        crop_pil = Image.fromarray(crop_np).convert("RGB")
        input_tensor = classifier_transforms(crop_pil).unsqueeze(0).to(device)
        
        # Classifier Inference
        with torch.no_grad():
            outputs = classifier_model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            
        top_probs, top_indices = torch.topk(probs, k=3)
        best_idx = top_indices[0].item()
        best_label = class_names[best_idx]
        cls_conf = top_probs[0].item()
        coarse_label = FINE_TO_COARSE.get(best_label, "unknown")
        
        # Draw bounding boxes and text
        # Green for organic, Blue for inorganic
        color = (34, 197, 94) if coarse_label == "organic" else (59, 130, 246)
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color[::-1], 3)
        
        # Draw background label strip
        label_str = f"Obj {idx+1}: {best_label} ({coarse_label.upper()}) | cls: {cls_conf:.1%}"
        (w_label, h_label), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Text position: top of bounding box (fallback to inside if near the top edge)
        text_y = max(y1 - 5, h_label + 5)
        cv2.rectangle(annotated_img, (x1, text_y - h_label - 4), (x1 + w_label, text_y + 4), color[::-1], -1)
        cv2.putText(
            annotated_img, label_str, (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )
        
        # Build Top-3 predictions
        top3_info = []
        for prob, p_idx in zip(top_probs, top_indices):
            c_name = class_names[p_idx.item()]
            c_group = FINE_TO_COARSE.get(c_name, "unknown")
            top3_info.append(f"{c_name} ({c_group}): {prob.item():.1%}")
            
        objects_list.append({
            "ID": f"Vật thể {idx+1}",
            "YOLO Conf": f"{yolo_conf:.2%}",
            "Fine Class": best_label,
            "Coarse Class": coarse_label.upper(),
            "Classifier Conf": f"{cls_conf:.2%}",
            "Top 3 Dự Đoán": ", ".join(top3_info)
        })
        
    status_msg = f"🎉 Đã phát hiện {len(detections)} vùng chứa rác và phân loại thành công."
    return annotated_img, status_msg, objects_list

# ==========================================
# GRADIO INTERFACE (Premium styling with Blocks)
# ==========================================
custom_css = """
body, .gradio-container {
    font-family: 'Inter', -apple-system, sans-serif;
    background-color: #0f172a;
    color: #f1f5f9;
}
.title-container {
    text-align: center;
    padding: 1.5rem 0;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid #1e293b;
}
.title-container h1 {
    font-size: 2.25rem;
    font-weight: 800;
    background: linear-gradient(to right, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.title-container p {
    color: #94a3b8;
    margin-top: 0.5rem;
    font-size: 1rem;
}
.info-box {
    background-color: #1e293b;
    border-left: 4px solid #818cf8;
    padding: 1rem;
    border-radius: 0.375rem;
    margin-bottom: 1rem;
}
.submit-btn {
    background: linear-gradient(to right, #6366f1, #4f46e5) !important;
    color: white !important;
    border: none !important;
}
.submit-btn:hover {
    background: linear-gradient(to right, #4f46e5, #4338ca) !important;
}
footer {
    display: none !important;
}
"""

with gr.Blocks(css=custom_css, title="Waste Detection & Classification Pipeline") as demo:
    with gr.Row(elem_classes="title-container"):
        gr.HTML(
            "<h1>Waste Detection and Classification Demo</h1>"
            "<p>Hệ thống tích hợp hai giai đoạn: YOLOv8 phát hiện vùng rác & EfficientNet-B0 phân loại chi tiết 16 lớp.</p>"
        )
        
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="numpy", label="Ảnh rác thải đầu vào")
            
            with gr.Accordion("Siêu tham số YOLO Detection", open=False):
                conf_slider = gr.Slider(
                    minimum=0.05, maximum=0.95, value=0.25, step=0.05, 
                    label="Confidence Threshold (YOLO)"
                )
                iou_slider = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.45, step=0.05, 
                    label="IoU Threshold (YOLO)"
                )
                
            run_btn = gr.Button("Phân Tích Ảnh", variant="primary", elem_classes="submit-btn")
            
            # Info box displaying model checkpoint parameters
            gr.HTML(
                f"<div class='info-box'>"
                f"<b>Thông tin mô hình đang load:</b><br>"
                f"• <b>YOLO Weights:</b> <code>{yolo_path_used.name}</code><br>"
                f"• <b>Classifier Checkpoint:</b> <code>best_model.pth (EfficientNet-B0)</code><br>"
                f"• <b>Số lớp phân loại:</b> 16 lớp chi tiết → Ánh xạ Organic/Inorganic"
                f"</div>"
            )
            
        with gr.Column(scale=1.5):
            output_img = gr.Image(type="numpy", label="Ảnh kết quả phân tích (Annotated)")
            status_output = gr.Textbox(label="Trạng thái hệ thống", interactive=False)
            
            results_table = gr.JSON(label="Báo cáo phân tích chi tiết vật thể")
            
    run_btn.click(
        fn=process_pipeline,
        inputs=[input_img, conf_slider, iou_slider],
        outputs=[output_img, status_output, results_table]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
