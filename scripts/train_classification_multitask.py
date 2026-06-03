from pathlib import Path
import argparse
import json
import time
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ==========================================
# PATH CONFIG
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATASET_DIR = DATA_DIR / "Dataset_classification_processed" / "fine"
RUNS_DIR = BASE_DIR / "runs" / "classification"

IMAGE_SIZE = 224
NUM_WORKERS = 2
RANDOM_SEED = 42

# Define exact class divisions
ORGANIC_CLASSES = {
    "fruit_waste",
    "vegetable_waste",
    "meat_waste",
    "starch_waste",
    "plant_waste",
    "mixed_food_waste",
    "biological"
}

INORGANIC_CLASSES = {
    "battery",
    "cardboard",
    "clothes",
    "glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash"
}


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# DATASETS & DATA LOADERS
# ==========================================
class MultiTaskDataset(Dataset):
    """
    Wraps standard ImageFolder to return fine_label and coarse_label.
    coarse_label is dynamically mapped from fine class names:
      - 0: organic
      - 1: inorganic
    """
    def __init__(self, image_folder_dataset, class_to_coarse):
        self.dataset = image_folder_dataset
        self.class_to_coarse = class_to_coarse
        self.classes = image_folder_dataset.classes

    def __getitem__(self, index):
        img, fine_label = self.dataset[index]
        coarse_label = self.class_to_coarse[fine_label]
        return img, fine_label, coarse_label

    def __len__(self):
        return len(self.dataset)


def get_transforms():
    train_tfms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_tfms, eval_tfms


def get_dataloaders(batch_size: int):
    train_tfms, eval_tfms = get_transforms()

    train_dir = DATASET_DIR / "train"
    val_dir = DATASET_DIR / "val"
    test_dir = DATASET_DIR / "test"

    if not train_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy train dir: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy val dir: {val_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy test dir: {test_dir}")

    # Standard ImageFolder datasets
    train_base = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_base = datasets.ImageFolder(val_dir, transform=eval_tfms)
    test_base = datasets.ImageFolder(test_dir, transform=eval_tfms)

    fine_class_names = train_base.classes
    num_fine_classes = len(fine_class_names)

    # Dynamic mapping: map fine classes to coarse categories based on name
    class_to_coarse = {}
    for i, name in enumerate(fine_class_names):
        if name in ORGANIC_CLASSES:
            class_to_coarse[i] = 0  # organic
        elif name in INORGANIC_CLASSES:
            class_to_coarse[i] = 1  # inorganic
        else:
            raise ValueError(f"Tên class '{name}' không nằm trong tập organic/inorganic được cấu hình.")

    # Wrap inside multi-task datasets
    train_dataset = MultiTaskDataset(train_base, class_to_coarse)
    val_dataset = MultiTaskDataset(val_base, class_to_coarse)
    test_dataset = MultiTaskDataset(test_base, class_to_coarse)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, fine_class_names, num_fine_classes


# ==========================================
# MODEL DEFINITION
# ==========================================
class MultiTaskEfficientNet(nn.Module):
    def __init__(self, dropout: float = 0.4):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = models.efficientnet_b0(weights=weights)

        # Replace default classifier with identity, so self.backbone(x) flattens to (batch, 1280)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        # Fine classification head (16 classes)
        self.fine_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 16)
        )

        # Coarse classification head (2 classes: organic/inorganic)
        self.coarse_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 2)
        )

    def forward(self, x):
        features = self.backbone(x)
        fine_logits = self.fine_head(features)
        coarse_logits = self.coarse_head(features)
        return fine_logits, coarse_logits


# ==========================================
# TRAINING & EVALUATION FUNCTIONS
# ==========================================
def train_one_epoch(model, dataloader, optimizer, criterion_fine, criterion_coarse, alpha, device):
    model.train()

    running_total_loss = 0.0
    running_fine_loss = 0.0
    running_coarse_loss = 0.0

    all_fine_preds = []
    all_fine_labels = []
    all_coarse_preds = []
    all_coarse_labels = []

    for images, fine_labels, coarse_labels in tqdm(dataloader, desc="Train", leave=False):
        images = images.to(device)
        fine_labels = fine_labels.to(device)
        coarse_labels = coarse_labels.to(device)

        optimizer.zero_grad()

        fine_logits, coarse_logits = model(images)

        fine_loss = criterion_fine(fine_logits, fine_labels)
        coarse_loss = criterion_coarse(coarse_logits, coarse_labels)
        total_loss = fine_loss + alpha * coarse_loss

        total_loss.backward()
        optimizer.step()

        bs = images.size(0)
        running_total_loss += total_loss.item() * bs
        running_fine_loss += fine_loss.item() * bs
        running_coarse_loss += coarse_loss.item() * bs

        fine_preds = torch.argmax(fine_logits, dim=1)
        coarse_preds = torch.argmax(coarse_logits, dim=1)

        all_fine_preds.extend(fine_preds.detach().cpu().numpy())
        all_fine_labels.extend(fine_labels.detach().cpu().numpy())
        all_coarse_preds.extend(coarse_preds.detach().cpu().numpy())
        all_coarse_labels.extend(coarse_labels.detach().cpu().numpy())

    dataset_len = len(dataloader.dataset)
    epoch_total_loss = running_total_loss / dataset_len
    epoch_fine_loss = running_fine_loss / dataset_len
    epoch_coarse_loss = running_coarse_loss / dataset_len

    epoch_fine_acc = accuracy_score(all_fine_labels, all_fine_preds)
    _, _, epoch_fine_macro_f1, _ = precision_recall_fscore_support(
        all_fine_labels, all_fine_preds, average="macro", zero_division=0
    )
    _, _, epoch_fine_weighted_f1, _ = precision_recall_fscore_support(
        all_fine_labels, all_fine_preds, average="weighted", zero_division=0
    )

    epoch_coarse_acc = accuracy_score(all_coarse_labels, all_coarse_preds)
    _, _, epoch_coarse_macro_f1, _ = precision_recall_fscore_support(
        all_coarse_labels, all_coarse_preds, average="macro", zero_division=0
    )
    _, _, epoch_coarse_weighted_f1, _ = precision_recall_fscore_support(
        all_coarse_labels, all_coarse_preds, average="weighted", zero_division=0
    )

    return (
        epoch_total_loss,
        epoch_fine_loss,
        epoch_coarse_loss,
        epoch_fine_acc,
        epoch_fine_macro_f1,
        epoch_fine_weighted_f1,
        epoch_coarse_acc,
        epoch_coarse_macro_f1,
        epoch_coarse_weighted_f1,
    )


def evaluate(model, dataloader, criterion_fine, criterion_coarse, alpha, device, desc="Val"):
    model.eval()

    running_total_loss = 0.0
    running_fine_loss = 0.0
    running_coarse_loss = 0.0

    all_fine_preds = []
    all_fine_labels = []
    all_coarse_preds = []
    all_coarse_labels = []

    with torch.no_grad():
        for images, fine_labels, coarse_labels in tqdm(dataloader, desc=desc, leave=False):
            images = images.to(device)
            fine_labels = fine_labels.to(device)
            coarse_labels = coarse_labels.to(device)

            fine_logits, coarse_logits = model(images)

            fine_loss = criterion_fine(fine_logits, fine_labels)
            coarse_loss = criterion_coarse(coarse_logits, coarse_labels)
            total_loss = fine_loss + alpha * coarse_loss

            bs = images.size(0)
            running_total_loss += total_loss.item() * bs
            running_fine_loss += fine_loss.item() * bs
            running_coarse_loss += coarse_loss.item() * bs

            fine_preds = torch.argmax(fine_logits, dim=1)
            coarse_preds = torch.argmax(coarse_logits, dim=1)

            all_fine_preds.extend(fine_preds.detach().cpu().numpy())
            all_fine_labels.extend(fine_labels.detach().cpu().numpy())
            all_coarse_preds.extend(coarse_preds.detach().cpu().numpy())
            all_coarse_labels.extend(coarse_labels.detach().cpu().numpy())

    dataset_len = len(dataloader.dataset)
    epoch_total_loss = running_total_loss / dataset_len
    epoch_fine_loss = running_fine_loss / dataset_len
    epoch_coarse_loss = running_coarse_loss / dataset_len

    epoch_fine_acc = accuracy_score(all_fine_labels, all_fine_preds)
    _, _, epoch_fine_macro_f1, _ = precision_recall_fscore_support(
        all_fine_labels, all_fine_preds, average="macro", zero_division=0
    )
    _, _, epoch_fine_weighted_f1, _ = precision_recall_fscore_support(
        all_fine_labels, all_fine_preds, average="weighted", zero_division=0
    )

    epoch_coarse_acc = accuracy_score(all_coarse_labels, all_coarse_preds)
    _, _, epoch_coarse_macro_f1, _ = precision_recall_fscore_support(
        all_coarse_labels, all_coarse_preds, average="macro", zero_division=0
    )
    _, _, epoch_coarse_weighted_f1, _ = precision_recall_fscore_support(
        all_coarse_labels, all_coarse_preds, average="weighted", zero_division=0
    )

    return (
        epoch_total_loss,
        epoch_fine_loss,
        epoch_coarse_loss,
        epoch_fine_acc,
        epoch_fine_macro_f1,
        epoch_fine_weighted_f1,
        epoch_coarse_acc,
        epoch_coarse_macro_f1,
        epoch_coarse_weighted_f1,
        all_fine_labels,
        all_fine_preds,
        all_coarse_labels,
        all_coarse_preds,
    )


# ==========================================
# REPORT & PLOT SAVING FUNCTIONS
# ==========================================
def save_curves(history, output_dir: Path):
    df = pd.DataFrame(history)
    df.to_csv(output_dir / "training_history.csv", index=False)

    # 1. Loss curves (Total, Fine, Coarse)
    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["train_total_loss"], label="Train Total Loss", linestyle="-", marker="o")
    plt.plot(df["epoch"], df["val_total_loss"], label="Val Total Loss", linestyle="-", marker="s")
    plt.plot(df["epoch"], df["train_fine_loss"], label="Train Fine Loss", linestyle="--")
    plt.plot(df["epoch"], df["val_fine_loss"], label="Val Fine Loss", linestyle="--")
    plt.plot(df["epoch"], df["train_coarse_loss"], label="Train Coarse Loss", linestyle=":")
    plt.plot(df["epoch"], df["val_coarse_loss"], label="Val Coarse Loss", linestyle=":")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Multi-Task Loss Curves")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=200)
    plt.close()

    # 2. Fine Accuracy Curve
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_fine_acc"], label="Train Fine Acc", marker="o")
    plt.plot(df["epoch"], df["val_fine_acc"], label="Val Fine Acc", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Fine Classification Accuracy")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / "fine_accuracy_curve.png", dpi=200)
    plt.close()

    # 3. Coarse Accuracy Curve
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_coarse_acc"], label="Train Coarse Acc", marker="o")
    plt.plot(df["epoch"], df["val_coarse_acc"], label="Val Coarse Acc", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Coarse Classification Accuracy")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / "coarse_accuracy_curve.png", dpi=200)
    plt.close()


def save_reports(y_true, y_pred, class_names, prefix, output_dir: Path):
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(output_dir / f"{prefix}_classification_report.csv")

    with open(output_dir / f"{prefix}_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8) if len(class_names) < 5 else (14, 12))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )
    disp.plot(
        ax=ax,
        xticks_rotation=90 if len(class_names) > 5 else 0,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )
    plt.title(f"{prefix.capitalize()} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_confusion_matrix.png", dpi=200)
    plt.close()


# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Train a Hierarchical Multi-Task EfficientNet-B0")

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.00005)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--patience", type=int, default=8)

    args = parser.parse_args()

    set_seed(RANDOM_SEED)

    device = get_device()
    print(f"Using device: {device}")

    # Load dataloaders
    train_loader, val_loader, test_loader, fine_class_names, num_fine_classes = get_dataloaders(args.batch)

    coarse_class_names = ["organic", "inorganic"]

    print(f"Dataset location: {DATASET_DIR}")
    print(f"Fine Classes ({num_fine_classes}): {fine_class_names}")
    print(f"Coarse Classes (2): {coarse_class_names}")

    # Setup saving directory
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    run_name = f"efficientnet_b0_multitask_{timestamp}"
    output_dir = RUNS_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save fine/coarse class mappings
    with open(output_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(fine_class_names, f, indent=4, ensure_ascii=False)

    # 4. Save coarse classes as dictionary mapping index to name
    coarse_mapping_json = {"0": "organic", "1": "inorganic"}
    with open(output_dir / "coarse_class_names.json", "w", encoding="utf-8") as f:
        json.dump(coarse_mapping_json, f, indent=4, ensure_ascii=False)

    # Save training configuration parameters
    config = {
        "model": "efficientnet_b0_multitask",
        "epochs": args.epochs,
        "batch_size": args.batch,
        "learning_rate": args.lr,
        "alpha": args.alpha,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "image_size": IMAGE_SIZE,
        "dataset": str(DATASET_DIR),
        "num_fine_classes": num_fine_classes,
        "num_coarse_classes": 2,
        "method": "hierarchical_multi_task_learning",
        "optimizer": "AdamW",
        "loss": "CrossEntropyLoss(Fine) + alpha * CrossEntropyLoss(Coarse)",
    }

    with open(output_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    # Initialize model
    model = MultiTaskEfficientNet(dropout=args.dropout)
    model = model.to(device)

    # Initialize loss functions and optimizer
    criterion_fine = nn.CrossEntropyLoss()
    criterion_coarse = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    best_val_fine_macro_f1 = 0.0
    best_val_coarse_macro_f1 = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    early_stop_counter = 0

    history = []

    print("\n===== START MULTI-TASK TRAINING =====")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Learning rate: {args.lr}")
    print(f"Alpha weight: {args.alpha}")
    print(f"Dropout: {args.dropout}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Early stop patience: {args.patience}")
    print(f"Output folder: {output_dir}")

    # Training loop
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        (
            train_loss, train_fine_loss, train_coarse_loss,
            train_fine_acc, train_fine_macro_f1, train_fine_weighted_f1,
            train_coarse_acc, train_coarse_macro_f1, train_coarse_weighted_f1
        ) = train_one_epoch(
            model, train_loader, optimizer,
            criterion_fine, criterion_coarse, args.alpha, device
        )

        (
            val_loss, val_fine_loss, val_coarse_loss,
            val_fine_acc, val_fine_macro_f1, val_fine_weighted_f1,
            val_coarse_acc, val_coarse_macro_f1, val_coarse_weighted_f1,
            _, _, _, _
        ) = evaluate(
            model, val_loader,
            criterion_fine, criterion_coarse, args.alpha, device, desc="Val"
        )

        scheduler.step(val_fine_macro_f1)

        print(
            f"Train Loss Total: {train_loss:.4f} | "
            f"Fine Loss: {train_fine_loss:.4f} (Acc: {train_fine_acc:.4f}, MacroF1: {train_fine_macro_f1:.4f}) | "
            f"Coarse Loss: {train_coarse_loss:.4f} (Acc: {train_coarse_acc:.4f}, MacroF1: {train_coarse_macro_f1:.4f})"
        )
        print(
            f"Val Loss Total: {val_loss:.4f} | "
            f"Fine Loss: {val_fine_loss:.4f} (Acc: {val_fine_acc:.4f}, MacroF1: {val_fine_macro_f1:.4f}) | "
            f"Coarse Loss: {val_coarse_loss:.4f} (Acc: {val_coarse_acc:.4f}, MacroF1: {val_coarse_macro_f1:.4f})"
        )

        history.append({
            "epoch": epoch,
            "train_total_loss": train_loss,
            "train_fine_loss": train_fine_loss,
            "train_coarse_loss": train_coarse_loss,
            "train_fine_acc": train_fine_acc,
            "train_fine_macro_f1": train_fine_macro_f1,
            "train_fine_weighted_f1": train_fine_weighted_f1,
            "train_coarse_acc": train_coarse_acc,
            "train_coarse_macro_f1": train_coarse_macro_f1,
            "train_coarse_weighted_f1": train_coarse_weighted_f1,
            "val_total_loss": val_loss,
            "val_fine_loss": val_fine_loss,
            "val_coarse_loss": val_coarse_loss,
            "val_fine_acc": val_fine_acc,
            "val_fine_macro_f1": val_fine_macro_f1,
            "val_fine_weighted_f1": val_fine_weighted_f1,
            "val_coarse_acc": val_coarse_acc,
            "val_coarse_macro_f1": val_coarse_macro_f1,
            "val_coarse_weighted_f1": val_coarse_weighted_f1,
        })

        # Selection logic: highest val_fine_macro_f1. If tied, highest val_coarse_macro_f1 is preferred.
        is_best = False
        if val_fine_macro_f1 > best_val_fine_macro_f1:
            is_best = True
        elif abs(val_fine_macro_f1 - best_val_fine_macro_f1) < 1e-7:
            if val_coarse_macro_f1 > best_val_coarse_macro_f1:
                is_best = True

        if is_best:
            best_val_fine_macro_f1 = val_fine_macro_f1
            best_val_coarse_macro_f1 = val_coarse_macro_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            early_stop_counter = 0

            torch.save({
                "model_name": "efficientnet_b0_multitask",
                "model_state_dict": best_model_wts,
                "class_names": fine_class_names,
                "coarse_class_names": coarse_class_names,
                "val_fine_macro_f1": val_fine_macro_f1,
                "val_coarse_macro_f1": val_coarse_macro_f1,
                "config": config,
            }, output_dir / "best_model.pth")
            print(f"Saved best model with val_fine_macro_f1 = {best_val_fine_macro_f1:.4f} and val_coarse_macro_f1 = {best_val_coarse_macro_f1:.4f}")
        else:
            early_stop_counter += 1
            print(f"Early stop counter: {early_stop_counter}/{args.patience}")

        if early_stop_counter >= args.patience:
            print("\nEarly stopping triggered.")
            break

    # Save training curves plots
    save_curves(history, output_dir)

    print("\n===== TESTING BEST MULTI-TASK MODEL =====")
    model.load_state_dict(best_model_wts)

    (
        test_loss, test_fine_loss, test_coarse_loss,
        test_fine_acc, test_fine_macro_f1, test_fine_weighted_f1,
        test_coarse_acc, test_coarse_macro_f1, test_coarse_weighted_f1,
        y_fine_true, y_fine_pred,
        y_coarse_true, y_coarse_pred
    ) = evaluate(
        model, test_loader,
        criterion_fine, criterion_coarse, args.alpha, device, desc="Test"
    )

    # Save reports and confusion matrices for both levels
    save_reports(y_fine_true, y_fine_pred, fine_class_names, "fine", output_dir)
    save_reports(y_coarse_true, y_coarse_pred, coarse_class_names, "coarse", output_dir)

    # Save test metrics json containing exact required fields
    test_metrics = {
        "fine_test_accuracy": test_fine_acc,
        "fine_macro_f1": test_fine_macro_f1,
        "fine_weighted_f1": test_fine_weighted_f1,
        "coarse_test_accuracy": test_coarse_acc,
        "coarse_macro_f1": test_coarse_macro_f1,
        "coarse_weighted_f1": test_coarse_weighted_f1,
    }

    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=4)

    print(f"Test Fine Loss: {test_fine_loss:.4f} | Test Fine Acc: {test_fine_acc:.4f} | Test Fine Macro F1: {test_fine_macro_f1:.4f}")
    print(f"Test Coarse Loss: {test_coarse_loss:.4f} | Test Coarse Acc: {test_coarse_acc:.4f} | Test Coarse Macro F1: {test_coarse_macro_f1:.4f}")
    print("\n===== DONE =====")
    print(f"Result folder: {output_dir}")


if __name__ == "__main__":
    main()
