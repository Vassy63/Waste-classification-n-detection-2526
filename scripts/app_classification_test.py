from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import gradio as gr


# =========================
# CONFIG
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

# Sửa đường dẫn này tới model tốt nhất của bạn
MODEL_PATH = BASE_DIR / "runs" / "classification" / "efficientnet_b0_baseline_20260526_090305" / "best_model.pth"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Không tìm thấy model tại: {MODEL_PATH}")
IMAGE_SIZE = 224

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


# =========================
# LOAD MODEL
# =========================

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(model_name, num_classes):
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes)
        )

    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes)
        )

    else:
        raise ValueError(f"Không hỗ trợ model: {model_name}")

    return model


def load_checkpoint(model_path):
    checkpoint = torch.load(model_path, map_location="cpu")

    model_name = checkpoint["model_name"]
    class_names = checkpoint["class_names"]
    num_classes = checkpoint["num_classes"]

    model = build_model(model_name, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model, class_names, model_name


device = get_device()

model, class_names, model_name = load_checkpoint(MODEL_PATH)
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# =========================
# PREDICT
# =========================

def predict(image):
    if image is None:
        return "Chưa có ảnh.", {}

    image = image.convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    top_probs, top_indices = torch.topk(probs, k=5)

    top_results = {}

    for prob, idx in zip(top_probs, top_indices):
        label = class_names[idx.item()]
        top_results[label] = float(prob.item())

    best_idx = top_indices[0].item()
    best_label = class_names[best_idx]
    confidence = float(top_probs[0].item())
    coarse_label = FINE_TO_COARSE.get(best_label, "unknown")

    result_text = (
        f"Model: {model_name}\n"
        f"Fine label: {best_label}\n"
        f"Coarse label: {coarse_label}\n"
        f"Confidence: {confidence:.4f}"
    )

    return result_text, top_results


# =========================
# GRADIO UI
# =========================

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload ảnh rác"),
    outputs=[
        gr.Textbox(label="Kết quả dự đoán"),
        gr.Label(label="Top 5 dự đoán")
    ],
    title="Waste Classification Test",
    description="Giao diện test nhanh mô hình phân loại rác 16 lớp và ánh xạ organic/inorganic."
)


if __name__ == "__main__":
    print(f"Using device: {device}")
    print(f"Loaded model: {MODEL_PATH}")
    demo.launch()