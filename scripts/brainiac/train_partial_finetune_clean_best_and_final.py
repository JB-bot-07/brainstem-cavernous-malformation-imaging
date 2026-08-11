import os
import sys
import argparse
import random
import json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Resized,
    NormalizeIntensityd,
    ToTensord,
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Allow import from BrainIAC repo
sys.path.append("/scratch/jbayasi/BrainIAC/src")
from model import ViTBackboneNet, SingleScanModel


# ============================================================
# Constants
# ============================================================
SEED = 42

ROOT_DIR = "/scratch/jbayasi/Cavmalproject1/processednifti"
CKPT_PATH = "/scratch/jbayasi/BrainIAC/src/checkpoints/BrainIAC.ckpt"

NUM_CLASSES = 7
IMAGE_SIZE = (96, 96, 96)

BATCH_SIZE = 4
NUM_WORKERS = 4
MAX_EPOCHS = 150

CLASSIFIER_LR = 1e-3
WEIGHT_DECAY = 1e-3
USE_AMP = True


# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Dataset and transforms
# ============================================================
def get_cavmal_transform(image_size=(96, 96, 96)):
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=image_size, mode="trilinear"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ToTensord(keys=["image"]),
    ])


class CavMalDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        self.df = pd.read_csv(csv_path, dtype={"pat_id": str})
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pat_id = str(self.df.loc[idx, "pat_id"])
        label = int(self.df.loc[idx, "label"])

        img_path = os.path.join(self.root_dir, pat_id + ".nii.gz")

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing image: {img_path}")

        sample = {"image": img_path}
        sample = self.transform(sample)

        return {
            "image": sample["image"],
            "label": torch.tensor(label, dtype=torch.long),
            "pat_id": pat_id,
        }


# ============================================================
# Model
# ============================================================
class MLPClassifier(nn.Module):
    def __init__(self, d_model=768, hidden_dim=128, num_classes=7, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def build_model():
    backbone = ViTBackboneNet(simclr_ckpt_path=CKPT_PATH)
    classifier = MLPClassifier(
        d_model=768,
        hidden_dim=128,
        num_classes=NUM_CLASSES,
        dropout=0.3,
    )
    model = SingleScanModel(backbone, classifier)
    return model


def apply_partial_finetuning(model, unfreeze_last_n_blocks):
    # Freeze entire BrainIAC backbone
    for p in model.backbone.parameters():
        p.requires_grad = False

    # Unfreeze last N transformer blocks
    vit = model.backbone.backbone

    if not hasattr(vit, "blocks"):
        raise AttributeError("Expected MONAI ViT to have attribute 'blocks', but it does not.")

    n_blocks = len(vit.blocks)
    print(f"MONAI ViT has {n_blocks} transformer blocks.")
    print(f"Unfreezing last {unfreeze_last_n_blocks} blocks.")

    for block in vit.blocks[-unfreeze_last_n_blocks:]:
        for p in block.parameters():
            p.requires_grad = True

    # Unfreeze final norm if present
    if hasattr(vit, "norm"):
        print("Unfreezing final ViT norm layer.")
        for p in vit.norm.parameters():
            p.requires_grad = True

    # Classifier trainable
    for p in model.classifier.parameters():
        p.requires_grad = True

    return model


def print_trainable_parameters(model):
    total = 0
    trainable = 0

    print("\nTrainable parameter groups:")
    for name, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
            print(f"TRAINABLE: {name} | shape={tuple(p.shape)} | n={p.numel()}")

    print("\nParameter count:")
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters:    {total - trainable:,}")
    print(f"Trainable percent:    {100 * trainable / total:.2f}%")


# ============================================================
# Evaluation and saving
# ============================================================
def evaluate(model, loader, device, criterion=None):
    model.eval()

    all_pat_ids = []
    all_true = []
    all_pred = []
    all_probs = []

    total_loss = 0.0
    n_batches = 0

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)

            logits = model(x)
            loss = criterion(logits, y)

            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1)

            total_loss += loss.item()
            n_batches += 1

            all_pat_ids.extend(batch["pat_id"])
            all_true.extend(y.detach().cpu().numpy().tolist())
            all_pred.extend(pred.detach().cpu().numpy().tolist())
            all_probs.extend(probs.detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(n_batches, 1)

    acc = accuracy_score(all_true, all_pred)
    bal_acc = balanced_accuracy_score(all_true, all_pred)
    macro_f1 = f1_score(
        all_true,
        all_pred,
        labels=list(range(NUM_CLASSES)),
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        all_true,
        all_pred,
        labels=list(range(NUM_CLASSES)),
        average="weighted",
        zero_division=0,
    )

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "pat_ids": all_pat_ids,
        "y_true": all_true,
        "y_pred": all_pred,
        "probs": all_probs,
    }


def save_eval_outputs(eval_result, output_dir, prefix, fold, config):
    os.makedirs(output_dir, exist_ok=True)

    # Predictions
    pred_rows = []
    for i, pid in enumerate(eval_result["pat_ids"]):
        row = {
            "pat_id": pid,
            "true_label": int(eval_result["y_true"][i]),
            "pred_label": int(eval_result["y_pred"][i]),
            "fold": int(fold),
        }
        for c in range(NUM_CLASSES):
            row[f"prob_{c}"] = float(eval_result["probs"][i][c])
        pred_rows.append(row)

    pred_df = pd.DataFrame(pred_rows)
    pred_path = os.path.join(output_dir, f"{prefix}_predictions.csv")
    pred_df.to_csv(pred_path, index=False)

    # Confusion matrix
    cm = confusion_matrix(
        eval_result["y_true"],
        eval_result["y_pred"],
        labels=list(range(NUM_CLASSES)),
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{i}" for i in range(NUM_CLASSES)],
        columns=[f"pred_{i}" for i in range(NUM_CLASSES)],
    )

    cm_path = os.path.join(output_dir, f"{prefix}_confusion_matrix.csv")
    cm_df.to_csv(cm_path)

    # Metrics
    metrics = {
        "fold": int(fold),
        "selection": prefix,
        "epoch_used": int(config["epoch_used"]),
        "val_loss": eval_result["loss"],
        "val_accuracy": eval_result["accuracy"],
        "val_balanced_accuracy": eval_result["balanced_accuracy"],
        "val_macro_f1": eval_result["macro_f1"],
        "val_weighted_f1": eval_result["weighted_f1"],
        "unfreeze_last_n_blocks": config["unfreeze_last_n_blocks"],
        "backbone_lr": config["backbone_lr"],
        "classifier_lr": config["classifier_lr"],
        "weight_decay": config["weight_decay"],
        "batch_size": config["batch_size"],
        "max_epochs": config["max_epochs"],
    }

    metrics_path = os.path.join(output_dir, f"{prefix}_metrics.csv")
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    # Classification report
    report_path = os.path.join(output_dir, f"{prefix}_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(classification_report(
            eval_result["y_true"],
            eval_result["y_pred"],
            labels=list(range(NUM_CLASSES)),
            zero_division=0,
        ))

    print(f"Saved {prefix} predictions:", pred_path)
    print(f"Saved {prefix} confusion matrix:", cm_path)
    print(f"Saved {prefix} metrics:", metrics_path)
    print(f"Saved {prefix} classification report:", report_path)

    return metrics


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--unfreeze_last_n_blocks", type=int, required=True)
    parser.add_argument("--backbone_lr", type=float, required=True)
    parser.add_argument("--output_base", type=str, required=True)
    args = parser.parse_args()

    set_seed(SEED + args.fold)

    fold = args.fold
    unfreeze_last_n_blocks = args.unfreeze_last_n_blocks
    backbone_lr = args.backbone_lr
    output_dir = os.path.join(args.output_base, f"fold{fold}")
    os.makedirs(output_dir, exist_ok=True)

    train_csv = f"/scratch/jbayasi/Cavmalproject1/csvs/partial_finetune_5fold/train_fold{fold}.csv"
    val_csv = f"/scratch/jbayasi/Cavmalproject1/csvs/partial_finetune_5fold/val_fold{fold}.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("PARTIAL FINETUNING CLEAN BEST + FINAL")
    print("=" * 80)
    print("Fold:", fold)
    print("Unfreeze last N blocks:", unfreeze_last_n_blocks)
    print("Backbone LR:", backbone_lr)
    print("Classifier LR:", CLASSIFIER_LR)
    print("Weight decay:", WEIGHT_DECAY)
    print("Batch size:", BATCH_SIZE)
    print("Max epochs:", MAX_EPOCHS)
    print("Train CSV:", train_csv)
    print("Val CSV:", val_csv)
    print("Output dir:", output_dir)
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        raise RuntimeError("CUDA GPU was not detected. This partial fine-tuning run should use a GPU.")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    print("\nTrain label counts:")
    print(train_df["label"].value_counts().sort_index())

    print("\nVal label counts:")
    print(val_df["label"].value_counts().sort_index())

    transform = get_cavmal_transform(IMAGE_SIZE)

    train_dataset = CavMalDataset(train_csv, ROOT_DIR, transform=transform)
    val_dataset = CavMalDataset(val_csv, ROOT_DIR, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    model = build_model()
    model = apply_partial_finetuning(model, unfreeze_last_n_blocks)
    model = model.to(device)

    print_trainable_parameters(model)

    # Class-weighted cross entropy based on training fold only
    train_labels = torch.tensor(train_df["label"].values, dtype=torch.long)
    counts = torch.bincount(train_labels, minlength=NUM_CLASSES).float()
    class_weights = len(train_labels) / (NUM_CLASSES * counts)
    class_weights[torch.isinf(class_weights)] = 0.0
    class_weights = class_weights.to(device)

    print("\nClass counts:", counts.tolist())
    print("Class weights:", class_weights.detach().cpu().tolist())

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    backbone_params = []
    classifier_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("classifier") or ".classifier" in name:
            classifier_params.append(p)
        else:
            backbone_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": classifier_params, "lr": CLASSIFIER_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and device.type == "cuda"))

    history_rows = []
    best_bal_acc = -1.0
    best_epoch = -1
    best_path = os.path.join(output_dir, "best_checkpoint_model.pt")
    final_path = os.path.join(output_dir, "final_epoch_model.pt")

    print("\nStarting training...")

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for batch in train_loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(USE_AMP and device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item()
            train_batches += 1

        train_loss = train_loss_sum / max(train_batches, 1)

        train_eval = evaluate(model, train_eval_loader, device, criterion)
        val_eval = evaluate(model, val_loader, device, criterion)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_eval_loss": train_eval["loss"],
            "train_accuracy": train_eval["accuracy"],
            "train_balanced_accuracy": train_eval["balanced_accuracy"],
            "train_macro_f1": train_eval["macro_f1"],
            "train_weighted_f1": train_eval["weighted_f1"],
            "val_loss": val_eval["loss"],
            "val_accuracy": val_eval["accuracy"],
            "val_balanced_accuracy": val_eval["balanced_accuracy"],
            "val_macro_f1": val_eval["macro_f1"],
            "val_weighted_f1": val_eval["weighted_f1"],
        }

        history_rows.append(row)

        print(
            f"Epoch {epoch:03d}/{MAX_EPOCHS} | "
            f"train loss {train_loss:.4f} | "
            f"train BA {train_eval['balanced_accuracy']:.3f} | "
            f"val BA {val_eval['balanced_accuracy']:.3f} | "
            f"val macro-F1 {val_eval['macro_f1']:.3f} | "
            f"val acc {val_eval['accuracy']:.3f}"
        )

        if val_eval["balanced_accuracy"] > best_bal_acc:
            best_bal_acc = val_eval["balanced_accuracy"]
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_balanced_accuracy": best_bal_acc,
                    "config": {
                        "fold": fold,
                        "unfreeze_last_n_blocks": unfreeze_last_n_blocks,
                        "backbone_lr": backbone_lr,
                        "classifier_lr": CLASSIFIER_LR,
                        "weight_decay": WEIGHT_DECAY,
                        "batch_size": BATCH_SIZE,
                        "max_epochs": MAX_EPOCHS,
                    },
                },
                best_path,
            )

    # Save training history
    history_df = pd.DataFrame(history_rows)
    history_path = os.path.join(output_dir, "training_history.csv")
    history_df.to_csv(history_path, index=False)
    print("Saved training history:", history_path)

    # ============================================================
    # Save FINAL epoch model and FINAL epoch predictions
    # This happens BEFORE loading the best checkpoint.
    # ============================================================
    torch.save(
        {
            "epoch": MAX_EPOCHS,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {
                "fold": fold,
                "unfreeze_last_n_blocks": unfreeze_last_n_blocks,
                "backbone_lr": backbone_lr,
                "classifier_lr": CLASSIFIER_LR,
                "weight_decay": WEIGHT_DECAY,
                "batch_size": BATCH_SIZE,
                "max_epochs": MAX_EPOCHS,
            },
        },
        final_path,
    )
    print("Saved final epoch model:", final_path)

    final_eval = evaluate(model, val_loader, device, criterion)
    final_config = {
        "epoch_used": MAX_EPOCHS,
        "unfreeze_last_n_blocks": unfreeze_last_n_blocks,
        "backbone_lr": backbone_lr,
        "classifier_lr": CLASSIFIER_LR,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
    }
    final_metrics = save_eval_outputs(
        final_eval,
        output_dir,
        prefix="final_epoch",
        fold=fold,
        config=final_config,
    )

    # ============================================================
    # Load BEST checkpoint and save BEST checkpoint predictions
    # ============================================================
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    best_eval = evaluate(model, val_loader, device, criterion)
    best_config = {
        "epoch_used": int(checkpoint["epoch"]),
        "unfreeze_last_n_blocks": unfreeze_last_n_blocks,
        "backbone_lr": backbone_lr,
        "classifier_lr": CLASSIFIER_LR,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
    }
    best_metrics = save_eval_outputs(
        best_eval,
        output_dir,
        prefix="best_checkpoint",
        fold=fold,
        config=best_config,
    )

    # Save combined fold summary
    summary_path = os.path.join(output_dir, "fold_summary_best_and_final.csv")
    pd.DataFrame([best_metrics, final_metrics]).to_csv(summary_path, index=False)

    print("\nTraining complete.")
    print("Fold:", fold)
    print("Best epoch:", best_epoch)
    print("Best val balanced accuracy:", best_bal_acc)
    print("Saved best model:", best_path)
    print("Saved final model:", final_path)
    print("Saved fold summary:", summary_path)


if __name__ == "__main__":
    main()
