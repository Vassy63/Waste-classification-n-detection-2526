import argparse
import shutil
import csv
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, choices=['val', 'test'], required=True)
    parser.add_argument('--confidence_threshold', type=float, default=0.70)
    parser.add_argument('--data_root', type=str, default='Dataset_classification_processed/fine')
    parser.add_argument('--output_dir', type=str, default='data/hard_examples')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    return parser.parse_args()

def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'num_classes' in checkpoint:
        num_classes = checkpoint['num_classes']
    elif 'class_names' in checkpoint:
        num_classes = len(checkpoint['class_names'])
    else:
        num_classes = 16
        print(f"⚠️ Không tìm thấy num_classes, dùng mặc định: {num_classes}")
    
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, num_classes

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model, num_classes = load_model(args.checkpoint, device)
    print(f"Loaded model with {num_classes} classes")
    
    data_path = Path(args.data_root) / args.split
    out_dir = Path(args.output_dir)
    wrong_dir = out_dir / 'wrong'
    low_conf_dir = out_dir / 'low_confidence'
    wrong_dir.mkdir(parents=True, exist_ok=True)
    low_conf_dir.mkdir(parents=True, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(data_path, transform=transform)
    # Lưu danh sách đường dẫn ảnh và nhãn
    samples = dataset.samples  # list of (path, label)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device.type=='cuda'))
    
    class_names = dataset.classes
    csv_rows = []
    
    with torch.no_grad():
        batch_idx = 0
        for images, labels in tqdm(loader, desc=f'Evaluating on {args.split}'):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            for i in range(images.size(0)):
                global_idx = batch_idx * args.batch_size + i
                if global_idx >= len(samples):
                    break
                img_path, true_label = samples[global_idx]
                pred_label = preds[i].item()
                confidence = probs[i, pred_label].item()
                is_correct = (true_label == pred_label)
                
                if (not is_correct) or (confidence < args.confidence_threshold):
                    img_name = Path(img_path).name
                    if not is_correct:
                        dest_dir = wrong_dir / class_names[true_label]
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copy(img_path, dest_dir / img_name)
                    else:
                        dest_dir = low_conf_dir / class_names[true_label]
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copy(img_path, dest_dir / img_name)
                    
                    csv_rows.append({
                        'image_path': str(img_path),
                        'true_class': class_names[true_label],
                        'pred_class': class_names[pred_label],
                        'confidence': confidence,
                        'is_correct': is_correct,
                        'split': args.split
                    })
            batch_idx += 1
    
    csv_path = out_dir / 'hard_examples.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['image_path', 'true_class', 'pred_class', 'confidence', 'is_correct', 'split'])
        writer.writeheader()
        writer.writerows(csv_rows)
    
    print(f"✅ Đã trích xuất {len(csv_rows)} hard examples từ {args.split}")
    print(f"   - Wrong: {len([r for r in csv_rows if not r['is_correct']])}")
    print(f"   - Low confidence: {len([r for r in csv_rows if r['is_correct'] and r['confidence'] < args.confidence_threshold])}")
    print(f"   - CSV: {csv_path}")

if __name__ == '__main__':
    main()