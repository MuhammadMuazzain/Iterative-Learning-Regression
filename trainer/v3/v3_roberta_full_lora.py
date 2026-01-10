import os
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
from peft import LoraConfig, get_peft_model
import mlflow
import mlflow.pytorch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# -------------------------
# CONFIG
# -------------------------
MODEL_NAME = os.getenv("MODEL_NAME", "roberta-large")
EXPERIMENT_NAME = "roberta_large_lora_regression_versioned3"

MAX_LENGTH = 512
BATCH_SIZE = 4
LR = 2e-4
EPOCHS = 3
NUM_WORKERS = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# DATASET
# -------------------------
class RegressionDataset(Dataset):
    def __init__(self, csv_path, tokenizer):
        df = pd.read_csv(csv_path)
        self.texts = df["text"].tolist()
        self.labels = df["label"].astype(float).tolist()
        self.tokenizer = tokenizer

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
        self.regressor = nn.Linear(self.model.config.hidden_size, 1)
        self.loss_fn = nn.MSELoss()

        # store step losses for MLflow
        self.train_losses = []
        self.val_losses = []

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        return self.regressor(pooled).squeeze(-1)

    # ---------------- Training ----------------
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

    # ---------------- Validation ----------------
    def validation_step(self, batch, batch_idx):
        preds = self(batch["input_ids"], batch["attention_mask"])
        loss = self.loss_fn(preds, batch["label"])
        self.log("val_loss_step", loss, on_step=True, prog_bar=True)
        self.val_losses.append(loss.detach().cpu().item())
        return loss

    def on_validation_epoch_end(self):
        if self.val_losses:
            avg_loss = sum(self.val_losses[-len(self.val_losses):]) / len(self.val_losses)
            mlflow.log_metric("val_loss_epoch", avg_loss, step=self.current_epoch)
            print(f"Epoch {self.current_epoch}: avg_val_loss = {avg_loss:.4f}")

    # ---------------- Optimizer ----------------
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=LR)

    def on_validation_epoch_end(self):
        if self.val_losses:
            avg_loss = sum(self.val_losses[-len(self.val_losses):]) / len(self.val_losses)
        
        # Log to Lightning so EarlyStopping can see it
            self.log("val_loss_epoch", avg_loss, prog_bar=True, on_epoch=True)
        
        # Also log to MLflow (optional)
            mlflow.log_metric("val_loss_epoch", avg_loss, step=self.current_epoch)
        
            print(f"Epoch {self.current_epoch}: avg_val_loss = {avg_loss:.4f}")



# =========================
# DATASET ↔ MODEL LINKAGE
# =========================
# def attach_dataset_lineage(
#     csv_path: str,
#     dataset_run_id: str,
#     dataset_checksum: str,
#     dataset_experiment: str = "data_registry",
# ):
#     # Required lineage tags
#     mlflow.set_tag("dataset_run_id", dataset_run_id)
#     mlflow.set_tag("dataset_checksum", dataset_checksum)
#     mlflow.set_tag("dataset_experiment", dataset_experiment)

#     # Helpful metadata
#     mlflow.set_tag("dataset_path", os.path.abspath(csv_path))
#     mlflow.set_tag("dataset_filename", os.path.basename(csv_path))
#     mlflow.set_tag("lineage_type", "dataset->model")

#     mlflow.log_param("dataset_num_rows", 10000)
#     mlflow.log_param("dataset_label_mean", 0.54)
#     mlflow.log_param("dataset_label_std", 0.12)


#     print("🔗 Dataset lineage attached")




# =========================
# DATASET ↔ MODEL LINEAGE (DYNAMIC METRICS)
# =========================
def attach_dataset_lineage(
    csv_path: str,
    dataset_run_id: str,
    dataset_checksum: str,
    dataset_experiment: str = "data_registry",
):
    import mlflow
    from mlflow.tracking import MlflowClient
    import os

    # --- Required lineage tags ---
    mlflow.set_tag("dataset_run_id", dataset_run_id)
    mlflow.set_tag("dataset_checksum", dataset_checksum)
    mlflow.set_tag("dataset_experiment", dataset_experiment)

    # --- Helpful metadata ---
    mlflow.set_tag("dataset_path", os.path.abspath(csv_path))
    mlflow.set_tag("dataset_filename", os.path.basename(csv_path))
    mlflow.set_tag("lineage_type", "dataset->model")

    # --- Fetch metrics from dataset run dynamically ---
    client = MlflowClient()
    try:
        dataset_run = client.get_run(dataset_run_id)
        metrics = dataset_run.data.metrics

        # List of metrics you want to log as model params
        for key in ["num_rows", "label_mean", "label_std", "label_min", "label_max"]:
            if key in metrics:
                mlflow.log_param(f"dataset_{key}", metrics[key])
    except Exception as e:
        print(f"⚠️ Could not fetch dataset metrics dynamically: {e}")

    print("🔗 Dataset lineage attached and metrics logged dynamically")



# -------------------------
# TRAINING
# -------------------------
def train(csv_path):
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    train_ds = RegressionDataset(csv_path, tokenizer)
    val_ds = RegressionDataset(csv_path, tokenizer)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///./mlflow.db"))
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():

        # -------------------------
        # PARAMS (already correct)
        # -------------------------
        mlflow.log_params({
            "model": MODEL_NAME,
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "epochs": EPOCHS,
        })

        # -------------------------
        # 🔗 ATTACH DATASET LINEAGE
        # -------------------------
        attach_dataset_lineage(
            csv_path=csv_path,
            dataset_run_id="2553c4eb72924595984c737cca8c786e",     
            dataset_checksum="ec7cdf90cf2a10c6ace8b24cde5aacfc46b1d8e93c44898ff5349aab73c458a8",   
        )

        model = RobertaLargeLoRA()

        trainer = pl.Trainer(
            accelerator="cuda" if torch.cuda.is_available() else "cpu",
            devices=1,
            precision="16-mixed",
            max_epochs=EPOCHS,
            log_every_n_steps=10,
            callbacks=[
                pl.callbacks.EarlyStopping(
                    monitor="val_loss_epoch",
                    patience=2,
                    mode="min"
                )
            ],
        )

        trainer.fit(model, train_loader, val_loader)

        mlflow.pytorch.log_model(model, artifact_path="roberta_large_lora")
        print("✅ Training complete & model logged to MLflow")

# -------------------------
# ENTRY
# -------------------------
if __name__ == "__main__":
    csv_path = "./v3-training-without-test.csv"
    train(csv_path)
