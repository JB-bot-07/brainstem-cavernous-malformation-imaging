import os
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# CONFIG
# ============================================================

FEATURE_PT = "/scratch/jbayasi/Cavmalproject1/features/brainiac_features_127.pt"
FOLD_CSV = "/scratch/jbayasi/Cavmalproject1/csvs/cavmal_master_127_folds.csv"

OUTPUT_DIR = "/scratch/jbayasi/Cavmalproject1/results/frozen_brainiac_mlp_5fold_150ep_from_pt_clean"

NUM_CLASSES = 7
NUM_FOLDS = 5

MAX_EPOCHS = 150
BATCH_SIZE = 4

CLASSIFIER_LR = 1e-3
WEIGHT_DECAY = 1e-3

HIDDEN_DIM = 128
DROPOUT = 0.3

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================
# DATASET
# ============================================================

class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ============================================================
# MODEL
# ============================================================

class FrozenFeatureMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# HELPERS
# ============================================================

def compute_class_weights(labels, num_classes):
    labels = np.asarray(labels).astype(int)
    counts = np.bincount(labels, minlength=num_classes)
    total = counts.sum()

    weights = np.zeros(num_classes, dtype=np.float32)

    for c in range(num_classes):
        if counts[c] > 0:
            weights[c] = total / (num_classes * counts[c])
        else:
            weights[c] = 0.0

    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model, loader, criterion):
    model.eval()

    all_labels = []
    all_preds = []

    total_loss = 0.0
    total_n = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            preds = torch.argmax(logits, dim=1)

            all_labels.extend(y.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

            total_loss += loss.item() * y.size(0)
            total_n += y.size(0)

    avg_loss = total_loss / max(total_n, 1)

    acc = accuracy_score(all_labels, all_preds)
    ba = balanced_accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "balanced_accuracy": ba,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "labels": all_labels,
        "preds": all_preds,
    }


def save_confusion_matrix(labels, preds, out_path):
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{i}" for i in range(NUM_CLASSES)],
        columns=[f"pred_{i}" for i in range(NUM_CLASSES)],
    )

    cm_df.to_csv(out_path)
    return cm_df


def save_classification_report(labels, preds, out_path):
    report = classification_report(
        labels,
        preds,
        labels=list(range(NUM_CLASSES)),
        output_dict=True,
        zero_division=0,
    )

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)


def load_features_from_pt(pt_path):
    """
    Robust loader for brainiac_features_127.pt.

    Supports:
    1. Raw tensor: shape [N, D]
    2. Dictionary containing one of:
       features, embeddings, x, X, feature_tensor
    """

    obj = torch.load(pt_path, map_location="cpu")

    print("Loaded PT object type:", type(obj))

    if isinstance(obj, torch.Tensor):
        features = obj
        extra = {}
    elif isinstance(obj, dict):
        print("PT dictionary keys:", obj.keys())

        possible_feature_keys = [
            "features",
            "embeddings",
            "feature_tensor",
            "x",
            "X",
            "brainiac_features",
        ]

        feature_key = None
        for k in possible_feature_keys:
            if k in obj:
                feature_key = k
                break

        if feature_key is None:
            tensor_keys = []
            for k, v in obj.items():
                if isinstance(v, torch.Tensor) and len(v.shape) == 2:
                    tensor_keys.append(k)

            if len(tensor_keys) == 1:
                feature_key = tensor_keys[0]
            else:
                raise ValueError(
                    "Could not identify feature tensor in PT dict. "
                    f"Keys: {list(obj.keys())}, 2D tensor keys: {tensor_keys}"
                )

        print(f"Using PT feature key: {feature_key}")
        features = obj[feature_key]
        extra = obj
    else:
        raise TypeError(f"Unsupported PT object type: {type(obj)}")

    if not isinstance(features, torch.Tensor):
        features = torch.tensor(features)

    features = features.float()

    if len(features.shape) != 2:
        raise ValueError(f"Expected features to be 2D [N, D], got shape {features.shape}")

    print("Feature tensor shape:", features.shape)

    return features.numpy(), extra


def summarize_metrics(metrics_df, prefix):
    return pd.DataFrame([{
        f"{prefix}_mean_accuracy": metrics_df["val_accuracy"].mean(),
        f"{prefix}_std_accuracy": metrics_df["val_accuracy"].std(ddof=1),

        f"{prefix}_mean_balanced_accuracy": metrics_df["val_balanced_accuracy"].mean(),
        f"{prefix}_std_balanced_accuracy": metrics_df["val_balanced_accuracy"].std(ddof=1),

        f"{prefix}_mean_macro_f1": metrics_df["val_macro_f1"].mean(),
        f"{prefix}_std_macro_f1": metrics_df["val_macro_f1"].std(ddof=1),

        f"{prefix}_mean_weighted_f1": metrics_df["val_weighted_f1"].mean(),
        f"{prefix}_std_weighted_f1": metrics_df["val_weighted_f1"].std(ddof=1),

        "num_folds": NUM_FOLDS,
        "max_epochs": MAX_EPOCHS,
        "batch_size": BATCH_SIZE,
        "classifier_lr": CLASSIFIER_LR,
        "weight_decay": WEIGHT_DECAY,
        "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT,
        "optimizer": "AdamW",
        "layer_norm": True,
        "encoder": "Frozen BrainIAC features from brainiac_features_127.pt",
    }])


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 80)
    print("FROZEN BRAINIAC FEATURE MLP 5-FOLD")
    print("=" * 80)

    print("FEATURE_PT:", FEATURE_PT)
    print("FOLD_CSV:", FOLD_CSV)
    print("OUTPUT_DIR:", OUTPUT_DIR)

    features, extra = load_features_from_pt(FEATURE_PT)

    folds_df = pd.read_csv(FOLD_CSV)

    print("\nFold CSV columns:")
    print(folds_df.columns.tolist())

    required_cols = ["pat_id", "label", "fold"]
    for c in required_cols:
        if c not in folds_df.columns:
            raise ValueError(f"Missing required column in fold CSV: {c}")

    if len(features) != len(folds_df):
        raise ValueError(
            f"Feature count and fold CSV row count do not match: "
            f"features={len(features)}, fold_csv={len(folds_df)}"
        )

    labels = folds_df["label"].astype(int).values
    folds = folds_df["fold"].astype(int).values
    pat_ids = folds_df["pat_id"].astype(str).values

    # Optional sanity check if labels are stored inside PT file.
    possible_label_keys = ["labels", "label", "y", "Y", "GroundTruthClassLabel"]
    for key in possible_label_keys:
        if isinstance(extra, dict) and key in extra:
            pt_labels = extra[key]
            if isinstance(pt_labels, torch.Tensor):
                pt_labels = pt_labels.cpu().numpy()
            pt_labels = np.asarray(pt_labels).astype(int).reshape(-1)

            if len(pt_labels) == len(labels):
                mismatches = int((pt_labels != labels).sum())
                print(f"Label mismatches between PT labels key '{key}' and fold CSV: {mismatches}")

                if mismatches > 0:
                    raise ValueError(
                        f"PT labels under key '{key}' do not match fold CSV labels."
                    )

    print("\nTotal samples:", len(labels))
    print("Feature dimension:", features.shape[1])
    print("Overall label counts:")
    print(pd.Series(labels).value_counts().sort_index())
    print("Fold counts:")
    print(pd.Series(folds).value_counts().sort_index())

    input_dim = features.shape[1]

    best_fold_metrics = []
    final_fold_metrics = []

    best_all_predictions = []
    final_all_predictions = []

    best_pooled_true = []
    best_pooled_pred = []

    final_pooled_true = []
    final_pooled_pred = []

    for fold in range(NUM_FOLDS):
        print("\n" + "=" * 80)
        print(f"FOLD {fold}")
        print("=" * 80)

        set_seed(SEED + fold)

        train_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]

        x_train = features[train_idx]
        y_train = labels[train_idx]

        x_val = features[val_idx]
        y_val = labels[val_idx]

        val_pat_ids = pat_ids[val_idx]

        print("Train size:", len(train_idx))
        print("Val size:", len(val_idx))

        print("Train label counts:")
        print(pd.Series(y_train).value_counts().sort_index())

        print("Val label counts:")
        print(pd.Series(y_val).value_counts().sort_index())

        train_dataset = FeatureDataset(x_train, y_train)
        val_dataset = FeatureDataset(x_val, y_val)

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )

        model = FrozenFeatureMLP(
            input_dim=input_dim,
            hidden_dim=HIDDEN_DIM,
            num_classes=NUM_CLASSES,
            dropout=DROPOUT,
        ).to(device)

        class_weights = compute_class_weights(y_train, NUM_CLASSES).to(device)
        print("Class weights:", class_weights.detach().cpu().numpy())

        criterion = nn.CrossEntropyLoss(weight=class_weights)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=CLASSIFIER_LR,
            weight_decay=WEIGHT_DECAY,
        )

        best_epoch = -1
        best_val_ba = -1.0
        best_state = None

        history = []

        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()

            train_labels = []
            train_preds = []

            train_loss_total = 0.0
            train_n = 0

            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()

                logits = model(x)
                loss = criterion(logits, y)

                loss.backward()
                optimizer.step()

                preds = torch.argmax(logits, dim=1)

                train_labels.extend(y.detach().cpu().numpy().tolist())
                train_preds.extend(preds.detach().cpu().numpy().tolist())

                train_loss_total += loss.item() * y.size(0)
                train_n += y.size(0)

            train_loss = train_loss_total / max(train_n, 1)
            train_acc = accuracy_score(train_labels, train_preds)
            train_ba = balanced_accuracy_score(train_labels, train_preds)
            train_macro_f1 = f1_score(train_labels, train_preds, average="macro", zero_division=0)
            train_weighted_f1 = f1_score(train_labels, train_preds, average="weighted", zero_division=0)

            val_result = evaluate(model, val_loader, criterion)

            row = {
                "fold": fold,
                "epoch": epoch,

                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "train_balanced_accuracy": train_ba,
                "train_macro_f1": train_macro_f1,
                "train_weighted_f1": train_weighted_f1,

                "val_loss": val_result["loss"],
                "val_accuracy": val_result["accuracy"],
                "val_balanced_accuracy": val_result["balanced_accuracy"],
                "val_macro_f1": val_result["macro_f1"],
                "val_weighted_f1": val_result["weighted_f1"],
            }

            history.append(row)

            print(
                f"Fold {fold} | Epoch {epoch:03d}/{MAX_EPOCHS} | "
                f"train_loss={train_loss:.4f} train_BA={train_ba:.4f} train_macroF1={train_macro_f1:.4f} | "
                f"val_loss={val_result['loss']:.4f} val_BA={val_result['balanced_accuracy']:.4f} "
                f"val_macroF1={val_result['macro_f1']:.4f}"
            )

            if val_result["balanced_accuracy"] > best_val_ba:
                best_val_ba = val_result["balanced_accuracy"]
                best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

        fold_dir = Path(OUTPUT_DIR) / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        history_df = pd.DataFrame(history)
        history_df.to_csv(fold_dir / "training_history.csv", index=False)

        # -------------------------
        # Final epoch evaluation
        # -------------------------
        final_result = evaluate(model, val_loader, criterion)

        final_pred_df = pd.DataFrame({
            "pat_id": val_pat_ids,
            "label": final_result["labels"],
            "pred": final_result["preds"],
            "fold": fold,
            "selection": "final_epoch",
            "epoch_used": MAX_EPOCHS,
        })

        final_pred_df.to_csv(fold_dir / "final_epoch_predictions.csv", index=False)

        save_confusion_matrix(
            final_result["labels"],
            final_result["preds"],
            fold_dir / "final_epoch_confusion_matrix.csv",
        )

        save_classification_report(
            final_result["labels"],
            final_result["preds"],
            fold_dir / "final_epoch_classification_report.json",
        )

        torch.save(
            model.state_dict(),
            fold_dir / "final_epoch_model.pt",
        )

        final_fold_metrics.append({
            "fold": fold,
            "selection": "final_epoch",
            "epoch_used": MAX_EPOCHS,
            "val_loss": final_result["loss"],
            "val_accuracy": final_result["accuracy"],
            "val_balanced_accuracy": final_result["balanced_accuracy"],
            "val_macro_f1": final_result["macro_f1"],
            "val_weighted_f1": final_result["weighted_f1"],
            "max_epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "classifier_lr": CLASSIFIER_LR,
            "weight_decay": WEIGHT_DECAY,
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
            "optimizer": "AdamW",
            "layer_norm": True,
        })

        final_pooled_true.extend(final_result["labels"])
        final_pooled_pred.extend(final_result["preds"])
        final_all_predictions.append(final_pred_df)

        # -------------------------
        # Best checkpoint evaluation
        # -------------------------
        model.load_state_dict(best_state)
        best_result = evaluate(model, val_loader, criterion)

        best_pred_df = pd.DataFrame({
            "pat_id": val_pat_ids,
            "label": best_result["labels"],
            "pred": best_result["preds"],
            "fold": fold,
            "selection": "best_validation_balanced_accuracy",
            "epoch_used": best_epoch,
        })

        best_pred_df.to_csv(fold_dir / "best_checkpoint_predictions.csv", index=False)

        save_confusion_matrix(
            best_result["labels"],
            best_result["preds"],
            fold_dir / "best_checkpoint_confusion_matrix.csv",
        )

        save_classification_report(
            best_result["labels"],
            best_result["preds"],
            fold_dir / "best_checkpoint_classification_report.json",
        )

        torch.save(
            best_state,
            fold_dir / "best_checkpoint_model.pt",
        )

        best_fold_metrics.append({
            "fold": fold,
            "selection": "best_validation_balanced_accuracy",
            "epoch_used": best_epoch,
            "val_loss": best_result["loss"],
            "val_accuracy": best_result["accuracy"],
            "val_balanced_accuracy": best_result["balanced_accuracy"],
            "val_macro_f1": best_result["macro_f1"],
            "val_weighted_f1": best_result["weighted_f1"],
            "max_epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "classifier_lr": CLASSIFIER_LR,
            "weight_decay": WEIGHT_DECAY,
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
            "optimizer": "AdamW",
            "layer_norm": True,
        })

        best_pooled_true.extend(best_result["labels"])
        best_pooled_pred.extend(best_result["preds"])
        best_all_predictions.append(best_pred_df)

        print(f"\nFold {fold} best epoch: {best_epoch}")
        print(f"Fold {fold} best val BA: {best_result['balanced_accuracy']:.4f}")
        print(f"Fold {fold} best val macro-F1: {best_result['macro_f1']:.4f}")

        print(f"Fold {fold} final epoch: {MAX_EPOCHS}")
        print(f"Fold {fold} final val BA: {final_result['balanced_accuracy']:.4f}")
        print(f"Fold {fold} final val macro-F1: {final_result['macro_f1']:.4f}")

    # ========================================================
    # Save overall summaries
    # ========================================================

    best_metrics_df = pd.DataFrame(best_fold_metrics)
    final_metrics_df = pd.DataFrame(final_fold_metrics)

    best_metrics_df.to_csv(Path(OUTPUT_DIR) / "best_checkpoint_fold_metrics.csv", index=False)
    final_metrics_df.to_csv(Path(OUTPUT_DIR) / "final_epoch_fold_metrics.csv", index=False)

    best_summary_df = summarize_metrics(best_metrics_df, "best")
    final_summary_df = summarize_metrics(final_metrics_df, "final")

    best_summary_df.to_csv(Path(OUTPUT_DIR) / "best_checkpoint_summary_metrics.csv", index=False)
    final_summary_df.to_csv(Path(OUTPUT_DIR) / "final_epoch_summary_metrics.csv", index=False)

    best_preds_df = pd.concat(best_all_predictions, ignore_index=True)
    final_preds_df = pd.concat(final_all_predictions, ignore_index=True)

    best_preds_df.to_csv(Path(OUTPUT_DIR) / "best_checkpoint_all_fold_predictions.csv", index=False)
    final_preds_df.to_csv(Path(OUTPUT_DIR) / "final_epoch_all_fold_predictions.csv", index=False)

    best_cm_df = save_confusion_matrix(
        best_pooled_true,
        best_pooled_pred,
        Path(OUTPUT_DIR) / "best_checkpoint_pooled_confusion_matrix.csv",
    )

    final_cm_df = save_confusion_matrix(
        final_pooled_true,
        final_pooled_pred,
        Path(OUTPUT_DIR) / "final_epoch_pooled_confusion_matrix.csv",
    )

    save_classification_report(
        best_pooled_true,
        best_pooled_pred,
        Path(OUTPUT_DIR) / "best_checkpoint_pooled_classification_report.json",
    )

    save_classification_report(
        final_pooled_true,
        final_pooled_pred,
        Path(OUTPUT_DIR) / "final_epoch_pooled_classification_report.json",
    )

    print("\n" + "=" * 80)
    print("DONE: FROZEN BRAINIAC MLP 5-FOLD FROM PT")
    print("=" * 80)

    print("\n===== BEST CHECKPOINT SUMMARY =====")
    print(best_summary_df.T)

    print("\n===== FINAL EPOCH SUMMARY =====")
    print(final_summary_df.T)

    print("\n===== BEST CHECKPOINT POOLED CONFUSION MATRIX =====")
    print(best_cm_df)

    print("\n===== FINAL EPOCH POOLED CONFUSION MATRIX =====")
    print(final_cm_df)

    print("\nSaved results to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
