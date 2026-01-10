import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
from peft import LoraConfig, get_peft_model
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# -------------------------
# CONFIG
# -------------------------
MODEL_NAME = os.getenv("MODEL_NAME", "roberta-large")
EXPERIMENT_NAME = "roberta_large_lora_regression_sequ4-sigmoid"

MAX_LENGTH = 512
BATCH_SIZE = 4
LR = 2e-4
EPOCHS = 3
# NUM_WORKERS = 4
NUM_WORKERS = 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# DATASET
# -------------------------
class RegressionDataset(Dataset):
    def __init__(self, csv_path=None, tokenizer=None):
        if csv_path is not None:
            df = pd.read_csv(csv_path)
            self.texts = df["text"].tolist()
            self.labels = df["label"].astype(float).tolist()
        else:
            self.texts = []
            self.labels = []
        self.tokenizer = tokenizer

    @classmethod
    def from_dataframe(cls, df, tokenizer):
        obj = cls(tokenizer=tokenizer)
        obj.texts = df["text"].tolist()
        obj.labels = df["label"].astype(float).tolist()
        return obj

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.float),
        }

# -------------------------
# MODEL
# -------------------------
class RobertaLargeLoRA(pl.LightningModule):
    def __init__(self):
        super().__init__()
        base_model = RobertaModel.from_pretrained(MODEL_NAME)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["query", "value"],
            lora_dropout=0.05,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        self.model = get_peft_model(base_model, lora_config)
        self.regressor = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, 1),
            nn.Sigmoid()
        )
        self.loss_fn = nn.MSELoss()

        self.train_losses = []
        self.val_losses = []

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        return self.regressor(pooled).squeeze(-1)

    def training_step(self, batch, batch_idx):
        preds = self(batch["input_ids"], batch["attention_mask"])
        loss = self.loss_fn(preds, batch["label"])
        self.log("train_loss_step", loss, on_step=True, prog_bar=True)
        self.train_losses.append(loss.detach().cpu().item())
        return loss

    def on_train_epoch_end(self):
        if self.train_losses:
            avg_loss = sum(self.train_losses[-len(self.train_losses):]) / len(self.train_losses)
            mlflow.log_metric("train_loss_epoch", avg_loss, step=self.current_epoch)
            print(f"Epoch {self.current_epoch}: avg_train_loss = {avg_loss:.4f}")

    def validation_step(self, batch, batch_idx):
        preds = self(batch["input_ids"], batch["attention_mask"])
        loss = self.loss_fn(preds, batch["label"])
        self.log("val_loss_step", loss, on_step=True, prog_bar=True)
        self.val_losses.append(loss.detach().cpu().item())
        return loss

    def on_validation_epoch_end(self):
        if self.val_losses:
            avg_loss = sum(self.val_losses[-len(self.val_losses):]) / len(self.val_losses)
            self.log("val_loss_epoch", avg_loss, on_epoch=True, prog_bar=True)
            mlflow.log_metric("val_loss_epoch", avg_loss, step=self.current_epoch)
            print(f"Epoch {self.current_epoch}: avg_val_loss = {avg_loss:.4f}")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=LR)

# -------------------------
# DATASET LINEAGE (Dynamic)
# -------------------------
def attach_dataset_lineage(csv_path: str, dataset_run_id: str, dataset_checksum: str, dataset_experiment: str = "data_registry"):
    from mlflow.tracking import MlflowClient

    mlflow.set_tag("dataset_run_id", dataset_run_id)
    mlflow.set_tag("dataset_checksum", dataset_checksum)
    mlflow.set_tag("dataset_experiment", dataset_experiment)
    mlflow.set_tag("dataset_path", os.path.abspath(csv_path))
    mlflow.set_tag("dataset_filename", os.path.basename(csv_path))
    mlflow.set_tag("lineage_type", "dataset->model")

    # fetch metrics dynamically from dataset run
    client = MlflowClient()
    try:
        dataset_run = client.get_run(dataset_run_id)
        metrics = dataset_run.data.metrics
        for key in ["num_rows", "label_mean", "label_std", "label_min", "label_max"]:
            if key in metrics:
                mlflow.log_param(f"dataset_{key}", metrics[key])
    except Exception as e:
        print(f"⚠️ Could not fetch dataset metrics dynamically: {e}")

    print("🔗 Dataset lineage attached and metrics logged dynamically")

# -------------------------
# Evaluation + Plots
# -------------------------
def evaluate_and_log(model, val_loader, tokenizer):
    model.eval()
    y_true, y_pred, text_lengths = [], [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].cpu().numpy()
            device = next(model.parameters()).device  # get model's device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            preds = model(input_ids, attention_mask).cpu().numpy()

            # preds = model(input_ids, attention_mask).cpu().numpy()
            y_true.extend(labels)
            y_pred.extend(preds)
            text_lengths.extend([len(tokenizer.decode(ids, skip_special_tokens=True)) for ids in batch["input_ids"]])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    text_lengths = np.array(text_lengths)

    # Residual distribution
    residuals = y_true - y_pred
    plt.figure(figsize=(6,4))
    sns.histplot(residuals, kde=True, bins=30, color="skyblue")
    plt.title("Residual Distribution")
    plt.xlabel("y_true - y_pred")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("residual_distribution.png")
    mlflow.log_artifact("residual_distribution.png")

    # Calibration curve
    plt.figure(figsize=(6,6))
    sns.scatterplot(x=y_pred, y=y_true)
    plt.plot([0,1],[0,1], linestyle='--', color='red')
    plt.title("Calibration Curve")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("calibration_curve.png")
    mlflow.log_artifact("calibration_curve.png")

    # Error vs Text Length
    errors = np.abs(y_true - y_pred)
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=text_lengths, y=errors)
    plt.title("Error vs Text Length")
    plt.xlabel("Text Length")
    plt.ylabel("|y_true - y_pred|")
    plt.tight_layout()
    plt.savefig("error_vs_text_length.png")
    mlflow.log_artifact("error_vs_text_length.png")

    # Confusion matrices at thresholds
    thresholds = [0.3, 0.5, 0.7]
    for t in thresholds:
        y_label = np.zeros_like(y_pred)
        y_label[y_pred >= t] = 1  # 1 = Human-written
        cm = confusion_matrix(y_true >= 0.5, y_label)  # true human labels >=0.5
        plt.figure(figsize=(4,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix @ threshold={t}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(f"confusion_matrix_{t}.png")
        mlflow.log_artifact(f"confusion_matrix_{t}.png")

    print("✅ Evaluation plots logged to MLflow")
    return y_true, y_pred

# -------------------------
# Thresholding & Risk Scoring
# -------------------------
def risk_scoring(y_pred):
    labels = []
    for s in y_pred:
        if s < 0.35:
            labels.append("AI-generated")
        elif s <= 0.65:
            labels.append("Human-edited AI")
        else:
            labels.append("Human-written")
    return labels

def evaluate_thresholds(y_true, y_pred):
    thresholds = [0.35, 0.65]
    results = []

    for low, high in [(0, thresholds[0]), (thresholds[0], thresholds[1]), (thresholds[1], 1)]:
        pred_labels = ((y_pred >= low) & (y_pred <= high)).astype(int)
        true_labels = (y_true >= 0.5).astype(int)
        precision = precision_score(true_labels, pred_labels, zero_division=0)
        recall = recall_score(true_labels, pred_labels, zero_division=0)
        fp = ((pred_labels==1) & (true_labels==0)).sum()
        fn = ((pred_labels==0) & (true_labels==1)).sum()
        results.append({
            "range": f"{low:.2f}-{high:.2f}",
            "precision": precision,
            "recall": recall,
            "false_positives": int(fp),
            "false_negatives": int(fn)
        })
    # log to MLflow
    with open("threshold_metrics.json", "w") as f:
        json.dump(results, f, indent=4)
    mlflow.log_artifact("threshold_metrics.json")
    print("✅ Threshold metrics logged to MLflow")
    return results

# -------------------------
# TRAINING
# -------------------------
def train(csv_path):
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    # 1️⃣ Load full CSV
    df = pd.read_csv(csv_path)

    # 2️⃣ Shuffle and split 80/20
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    split_idx = int(0.8 * len(df))
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df = df.iloc[split_idx:].reset_index(drop=True)

    # 3️⃣ Create datasets from split dataframes
    train_ds = RegressionDataset.from_dataframe(train_df, tokenizer)
    val_ds = RegressionDataset.from_dataframe(val_df, tokenizer)

    # train_ds = RegressionDataset(csv_path, tokenizer)
    # val_ds = RegressionDataset(csv_path, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///./mlflow.db"))
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        # Log training params
        mlflow.log_params({
            "model": MODEL_NAME,
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "epochs": EPOCHS,
        })

        # Dataset lineage
        attach_dataset_lineage(
            csv_path=csv_path,
            dataset_run_id="8efb6263ae364da2aafefcc4228bb62c",
            dataset_checksum="ec7cdf90cf2a10c6ace8b24cde5aacfc46b1d8e93c44898ff5349aab73c458a8",
        )

        # Model
        model = RobertaLargeLoRA()

        trainer = pl.Trainer(
            accelerator="cuda" if torch.cuda.is_available() else "cpu",
            devices=1,
            precision="16-mixed",
            max_epochs=EPOCHS,
            log_every_n_steps=10,
            callbacks=[
                pl.callbacks.EarlyStopping(monitor="val_loss_epoch", patience=2, mode="min")
            ],
        )

        trainer.fit(model, train_loader, val_loader)

        # Log model
        mlflow.pytorch.log_model(model, artifact_path="roberta_large_lora-sigmoid")
        print("✅ Training complete & model logged to MLflow")

        # -------------------------
        # Evaluation & risk scoring
        # -------------------------
        y_true, y_pred = evaluate_and_log(model, val_loader, tokenizer)
        risk_labels = risk_scoring(y_pred)
        evaluate_thresholds(np.array(y_true), np.array(y_pred))

# -------------------------
# ENTRY
# -------------------------
if __name__ == "__main__":
    csv_path = "./v4-training-without-test.csv"
    train(csv_path)
