from pathlib import Path
import argparse
import json
import time
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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


# =========================
# PATH CONFIG
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

DATASET_DIR = DATA_DIR / "Dataset_classification_processed" / "fine"
RUNS_DIR = BASE_DIR / "runs" / "classification"


# =========================
# DEFAULT CONFIG
# =========================

IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 25
DEFAULT_LR = 0.0001
NUM_WORKERS = 2
RANDOM_SEED = 42


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    train_dataset = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_dataset = datasets.ImageFolder(val_dir, transform=eval_tfms)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_tfms)

    class_names = train_dataset.classes
    num_classes = len(class_names)

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

    return train_loader, val_loader, test_loader, class_names, num_classes


def build_model(model_name: str, num_classes: int, freeze_backbone: bool = False):
    model_name = model_name.lower()

    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)

        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False

        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes)
        )

    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)

        if freeze_backbone:
            for name, param in model.named_parameters():
                if not name.startswith("fc"):
                    param.requires_grad = False

        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes)
        )

    else:
        raise ValueError("model_name chỉ nhận: efficientnet_b0 hoặc resnet50")

    return model

def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device, desc="Val"):
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=desc, leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            preds = torch.argmax(outputs, dim=1)

            running_loss += loss.item() * images.size(0)

            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc, all_labels, all_preds


def save_curves(history, output_dir: Path):
    df = pd.DataFrame(history)
    df.to_csv(output_dir / "training_history.csv", index=False)

    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    plt.plot(df["epoch"], df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(df["epoch"], df["train_acc"], label="train_acc")
    plt.plot(df["epoch"], df["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training / Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curve.png", dpi=200)
    plt.close()


def save_test_reports(y_true, y_pred, class_names, output_dir: Path):
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
    report_df.to_csv(output_dir / "classification_report.csv")

    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(14, 12))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )
    disp.plot(
        ax=ax,
        xticks_rotation=90,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    acc = accuracy_score(y_true, y_pred)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    metrics = {
        "test_accuracy": acc,
        "macro_precision": precision_macro,
        "macro_recall": recall_macro,
        "macro_f1": f1_macro,
        "weighted_precision": precision_weighted,
        "weighted_recall": recall_weighted,
        "weighted_f1": f1_weighted,
    }

    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    return metrics


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["efficientnet_b0", "resnet50"],
        help="Model cần train baseline",
    )

    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--freeze_epochs", type=int, default=5)

    args = parser.parse_args()

    set_seed(RANDOM_SEED)

    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, class_names, num_classes = get_dataloaders(args.batch)

    print(f"Dataset: {DATASET_DIR}")
    print(f"Num classes: {num_classes}")
    print(f"Classes: {class_names}")

    run_name = f"{args.model}_baseline_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir = RUNS_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=4, ensure_ascii=False)

    config = {
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch,
        "learning_rate": args.lr,
        "image_size": IMAGE_SIZE,
        "dataset": str(DATASET_DIR),
        "num_classes": num_classes,
        "freeze_epochs": args.freeze_epochs,
        "initial_weight_decay": 5e-4,
        "finetune_weight_decay": 1e-3,
        "dropout": 0.4,
        "method": "freeze_backbone_then_unfreeze",
    }

    with open(output_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    model = build_model(args.model, num_classes, freeze_backbone=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    history = []

    print("\n===== START TRAINING =====")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Learning rate: {args.lr}")
    print(f"Output: {output_dir}")

    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_epochs + 1:
            print("\nUnfreezing full model for fine-tuning...")
            unfreeze_model(model)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.lr * 0.5,
                weight_decay=1e-3,
            )
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss, val_acc, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            device,
            desc="Val",
        )

        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss  : {val_loss:.4f} | Val Acc  : {val_acc:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

            torch.save({
                "model_name": args.model,
                "model_state_dict": best_model_wts,
                "class_names": class_names,
                "num_classes": num_classes,
                "val_acc": best_val_acc,
            }, output_dir / "best_model.pth")

            print(f"Saved best model with val_acc = {best_val_acc:.4f}")

    save_curves(history, output_dir)

    print("\n===== TEST BEST MODEL =====")

    model.load_state_dict(best_model_wts)

    test_loss, test_acc, y_true, y_pred = evaluate(
        model,
        test_loader,
        criterion,
        device,
        desc="Test",
    )

    metrics = save_test_reports(
        y_true,
        y_pred,
        class_names,
        output_dir,
    )

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc : {test_acc:.4f}")
    print(f"Macro F1 : {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")

    print("\n===== DONE =====")
    print(f"Result folder: {output_dir}")


if __name__ == "__main__":
    main()