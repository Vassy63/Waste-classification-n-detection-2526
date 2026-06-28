"""
Fine-tune baseline với hard examples - Chỉ train classifier, giữ nguyên backbone
Usage:
    python scripts/finetune_hard_examples.py --checkpoint runs/.../best_model.pth
"""
import argparse
import json
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms, datasets, models
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr_classifier', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--hard_weight', type=float, default=2.0, help='Nhan trong so cho hard examples (repeat)')
    parser.add_argument('--data_root', type=str, default='Dataset_classification_processed/fine')
    parser.add_argument('--hard_examples_csv', type=str, default='data/hard_examples/hard_examples.csv')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='runs/classification')
    return parser.parse_args()

class HardExamplesDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, transform, class_to_idx, repeat=1):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.repeat = repeat
    def __len__(self):
        return len(self.df) * self.repeat
    def __getitem__(self, idx):
        orig_idx = idx % len(self.df)
        row = self.df.iloc[orig_idx]
        img = Image.open(row['image_path']).convert('RGB')
        img = self.transform(img)
        label = self.class_to_idx[row['true_class']]
        return img, label

def load_model(checkpoint_path, num_classes, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    return model

def freeze_all_except_classifier(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    print("✅ Frozen entire backbone, only classifier is trainable.")

def plot_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(train_losses)+1)
    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-s', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    ax2.plot(epochs, train_accs, 'b-o', label='Train Accuracy')
    ax2.plot(epochs, val_accs, 'r-s', label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_confusion_matrix(labels, preds, class_names, title, save_path):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, annot_kws={'size': 8})
    plt.title(title, fontsize=14)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_per_class_f1(classes, f1_baseline, f1_hem, save_path):
    x = np.arange(len(classes))
    width = 0.35
    plt.figure(figsize=(16, 6))
    plt.bar(x - width/2, f1_baseline, width, label='Baseline', color='skyblue')
    plt.bar(x + width/2, f1_hem, width, label='HEM (classifier-only)', color='salmon')
    plt.xticks(x, classes, rotation=45, ha='right', fontsize=9)
    plt.ylabel('F1 Score')
    plt.title('Per-class F1 Score Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def evaluate(model, loader, device):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Transform
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_path = Path(args.data_root) / 'train'
    train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
    num_classes = len(train_dataset.classes)
    class_to_idx = train_dataset.class_to_idx
    class_names = train_dataset.classes
    
    # Load hard examples (với repeat để tăng trọng số)
    hard_dataset = HardExamplesDataset(args.hard_examples_csv, train_transform, class_to_idx, repeat=int(args.hard_weight))
    combined_dataset = ConcatDataset([train_dataset, hard_dataset])
    
    train_loader = DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    
    # Validation loader
    val_path = Path(args.data_root) / 'val'
    val_dataset = datasets.ImageFolder(val_path, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    
    # Test loader
    test_path = Path(args.data_root) / 'test'
    test_dataset = datasets.ImageFolder(test_path, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    
    # Load baseline model (for comparison)
    baseline_model = load_model(args.checkpoint, num_classes, device)
    
    # Load model for fine-tuning and freeze backbone
    model = load_model(args.checkpoint, num_classes, device)
    freeze_all_except_classifier(model)
    
    # Optimizer chỉ cho classifier
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr_classifier, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss()
    
    # Training history
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 5
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f'efficientnet_b0_HEM_classifier_{timestamp}'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}'):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), out_dir / 'best_model.pth')
            patience_counter = 0
            print("  -> Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Draw curves
    plot_curves(train_losses, val_losses, train_accs, val_accs, out_dir / 'training_curves.png')
    
    # Evaluate on test
    print("\nLoading best model for test evaluation...")
    model.load_state_dict(torch.load(out_dir / 'best_model.pth'))
    model.eval()
    
    test_labels_hem, test_preds_hem = evaluate(model, test_loader, device)
    test_acc_hem = accuracy_score(test_labels_hem, test_preds_hem)
    macro_f1_hem = f1_score(test_labels_hem, test_preds_hem, average='macro')
    weighted_f1_hem = f1_score(test_labels_hem, test_preds_hem, average='weighted')
    
    baseline_model.eval()
    test_labels_base, test_preds_base = evaluate(baseline_model, test_loader, device)
    test_acc_base = accuracy_score(test_labels_base, test_preds_base)
    macro_f1_base = f1_score(test_labels_base, test_preds_base, average='macro')
    weighted_f1_base = f1_score(test_labels_base, test_preds_base, average='weighted')
    
    print(f"\nBaseline: Acc={test_acc_base:.4f}, Macro F1={macro_f1_base:.4f}, Weighted F1={weighted_f1_base:.4f}")
    print(f"HEM (classifier-only): Acc={test_acc_hem:.4f}, Macro F1={macro_f1_hem:.4f}, Weighted F1={weighted_f1_hem:.4f}")
    
    metrics = {
        'baseline': {'accuracy': test_acc_base, 'macro_f1': macro_f1_base, 'weighted_f1': weighted_f1_base},
        'hem': {'accuracy': test_acc_hem, 'macro_f1': macro_f1_hem, 'weighted_f1': weighted_f1_hem},
        'improvement': {
            'accuracy': test_acc_hem - test_acc_base,
            'macro_f1': macro_f1_hem - macro_f1_base,
            'weighted_f1': weighted_f1_hem - weighted_f1_base
        }
    }
    with open(out_dir / 'test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Confusion matrices and per-class F1
    plot_confusion_matrix(test_labels_base, test_preds_base, class_names, 
                          'Baseline Confusion Matrix', out_dir / 'confusion_baseline.png')
    plot_confusion_matrix(test_labels_hem, test_preds_hem, class_names,
                          'HEM Classifier-Only Confusion Matrix', out_dir / 'confusion_hem.png')
    
    report_base = classification_report(test_labels_base, test_preds_base, target_names=class_names, output_dict=True)
    report_hem = classification_report(test_labels_hem, test_preds_hem, target_names=class_names, output_dict=True)
    classes = [c for c in class_names if c in report_base and c in report_hem]
    f1_base = [report_base[c]['f1-score'] for c in classes]
    f1_hem = [report_hem[c]['f1-score'] for c in classes]
    plot_per_class_f1(classes, f1_base, f1_hem, out_dir / 'per_class_f1.png')
    
    # Save history
    pd.DataFrame({
        'epoch': range(1, len(train_losses)+1),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs
    }).to_csv(out_dir / 'training_history.csv', index=False)
    
    with open(out_dir / 'report.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("HEM (CLASSIFIER-ONLY) - SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Test Accuracy:  Baseline = {test_acc_base:.4f}, HEM = {test_acc_hem:.4f}, Change = {test_acc_hem-test_acc_base:+.4f}\n")
        f.write(f"Macro F1:      Baseline = {macro_f1_base:.4f}, HEM = {macro_f1_hem:.4f}, Change = {macro_f1_hem-macro_f1_base:+.4f}\n")
        f.write(f"Weighted F1:   Baseline = {weighted_f1_base:.4f}, HEM = {weighted_f1_hem:.4f}, Change = {weighted_f1_hem-weighted_f1_base:+.4f}\n\n")
        f.write("Per-class F1 changes:\n")
        for c in classes:
            diff = report_hem[c]['f1-score'] - report_base[c]['f1-score']
            f.write(f"  {c:20s}: {report_base[c]['f1-score']:.3f} -> {report_hem[c]['f1-score']:.3f} (change={diff:+.3f})\n")
        f.write("\n" + "="*60 + "\n")
        if macro_f1_hem > macro_f1_base:
            f.write("SUCCESS: HEM improved macro F1!\n")
        else:
            f.write("WARNING: HEM did not improve macro F1. This may be due to insufficient hard examples or too aggressive repetition.\n")
    
    print(f"\nAll results saved to {out_dir}")

if __name__ == '__main__':
    main()