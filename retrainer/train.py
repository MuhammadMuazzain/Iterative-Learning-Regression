import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
from peft import LoraConfig, get_peft_model

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.pytorch
from dotenv import load_dotenv

# -------------------------
# ENV
# -------------------------
load_dotenv()

# -------------------------
# CONFIG
# -------------------------
MODEL_NAME = os.getenv("MODEL_NAME", "roberta-large")
EXPERIMENT_NAME = "roberta_lora_regression_continuous_v3"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

DATA_PATH = "clean_feedback.csv" 

OUTPUT_MODEL_PATH = "models/candidate_model.pt"
OUTPUT_METRICS_PATH = "models/candidate_metrics.json"

MAX_LENGTH = 512
BATCH_SIZE = 4
LR = 2e-4
EPOCHS = 5
NUM_WORKERS = 0
SEED = 42

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Thresholds for production promotion
MIN_VAL_MAE = 0.15  # Only promote if MAE < 0.15
MAX_VAL_RMSE = 0.20

# -------------------------
# REPRODUCIBILITY
# -------------------------
pl.seed_everything(SEED, workers=True)

# -------------------------
# DATASET
# -------------------------
class RegressionDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts = df["text"].tolist()
        self.labels = df["target_score"].astype(float).tolist()
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

        self.encoder = get_peft_model(base_model, lora_config)

        self.regressor = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 1),
            nn.Sigmoid()
        )

        self.loss_fn = nn.MSELoss()
        
        # Store predictions for epoch-end metrics
        self.validation_step_outputs = []

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        return self.regressor(pooled).squeeze(-1)

    def training_step(self, batch, batch_idx):
        preds = self(batch["input_ids"], batch["attention_mask"])
        loss = self.loss_fn(preds, batch["label"])
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        preds = self(batch["input_ids"], batch["attention_mask"])
        loss = self.loss_fn(preds, batch["label"])
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        
        # Store for epoch-end aggregation
        self.validation_step_outputs.append({
            "preds": preds.detach(),
            "labels": batch["label"].detach()
        })
        return loss

    def on_validation_epoch_end(self):
        """Compute metrics at end of validation epoch"""
        if not self.validation_step_outputs:
            return
        
        all_preds = torch.cat([x["preds"] for x in self.validation_step_outputs])
        all_labels = torch.cat([x["labels"] for x in self.validation_step_outputs])
        
        # Move to CPU for sklearn
        y_pred = all_preds.cpu().numpy()
        y_true = all_labels.cpu().numpy()
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred)
        
        self.log("val_mae", mae, prog_bar=True)
        self.log("val_rmse", rmse, prog_bar=True)
        
        # Clear for next epoch
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=LR)

# -------------------------
# TRAINING
# -------------------------
def train():
    assert os.path.exists(DATA_PATH), f"❌ {DATA_PATH} not found"

    df = pd.read_csv(DATA_PATH)
    
    print(f"📊 Loaded {len(df)} samples from {DATA_PATH}")

    # Three-way split: train/val/test
    train_df, temp_df = train_test_split(
        df, train_size=TRAIN_RATIO, random_state=SEED, shuffle=True
    )
    val_df, test_df = train_test_split(
        temp_df, 
        train_size=VAL_RATIO/(VAL_RATIO + TEST_RATIO), 
        random_state=SEED
    )
    
    print(f"📈 Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    train_ds = RegressionDataset(train_df, tokenizer)
    val_ds = RegressionDataset(val_df, tokenizer)
    test_ds = RegressionDataset(test_df, tokenizer)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # -------------------------
    # MLFLOW SETUP - FIXED
    # -------------------------
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Get or create experiment
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    
    if experiment is None:
        # Create new experiment
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
        print(f"📌 Created new MLflow experiment: {EXPERIMENT_NAME} (ID: {experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        # Restore if deleted
        if experiment.lifecycle_stage != "active":
            mlflow.tracking.MlflowClient().restore_experiment(experiment_id)
            print(f"♻️ Restored deleted experiment: {EXPERIMENT_NAME} (ID: {experiment_id})")
        else:
            print(f"📌 Using existing experiment: {EXPERIMENT_NAME} (ID: {experiment_id})")

    # Set experiment explicitly by name
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Start run with explicit experiment_id
    with mlflow.start_run(run_name="candidate_training", experiment_id=experiment_id) as run:

        mlflow.log_params({
            "model": MODEL_NAME,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "seed": SEED,
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
            "experiment_name": EXPERIMENT_NAME,
            "experiment_id": experiment_id,
        })

        model = RobertaLargeLoRA()

        # Callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints",
            filename="best-{epoch:02d}-{val_mae:.3f}",
            monitor="val_mae",
            mode="min",
            save_top_k=1,
            save_last=True,
        )

        early_stop_callback = EarlyStopping(
            monitor="val_mae",
            patience=2,
            mode="min",
            verbose=True,
        )

        trainer = pl.Trainer(
            accelerator="cuda" if torch.cuda.is_available() else "cpu",
            devices=1,
            max_epochs=EPOCHS,
            precision="16-mixed" if torch.cuda.is_available() else 32,
            log_every_n_steps=10,
            callbacks=[checkpoint_callback, early_stop_callback],
            deterministic=True,
        )

        trainer.fit(model, train_loader, val_loader)

        # -------------------------
        # LOAD BEST CHECKPOINT
        # -------------------------
        best_model_path = checkpoint_callback.best_model_path
        print(f"🏆 Loading best model from: {best_model_path}")
        
        # Log checkpoint path to MLflow for traceability
        mlflow.log_param("best_checkpoint", best_model_path)
        
        best_model = RobertaLargeLoRA.load_from_checkpoint(best_model_path)
        best_model.to(DEVICE)
        best_model.eval()

        # -------------------------
        # FINAL EVALUATION ON TEST SET
        # -------------------------
        print("\n📊 Evaluating on test set...")
        y_true, y_pred = [], []

        with torch.no_grad():
            for batch in test_loader:
                preds = best_model(
                    batch["input_ids"].to(DEVICE),
                    batch["attention_mask"].to(DEVICE)
                ).cpu().numpy()

                y_pred.extend(preds)
                y_true.extend(batch["label"].numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        test_mae = mean_absolute_error(y_true, y_pred)
        test_rmse = mean_squared_error(y_true, y_pred)

        metrics = {
            "test_mae": float(test_mae),
            "test_rmse": float(test_rmse),
            "test_samples": len(y_true),
            "experiment_name": EXPERIMENT_NAME,
            "experiment_id": experiment_id,
        }

        # -------------------------
        # SAVE CANDIDATE MODEL
        # -------------------------
        os.makedirs("models", exist_ok=True)
        
        # Save state dict
        torch.save(best_model.state_dict(), OUTPUT_MODEL_PATH)
        
        # Save metrics (including experiment info)
        with open(OUTPUT_METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=2)

        # Log to MLflow
        mlflow.log_metrics({
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_samples": len(y_true),
        })
        mlflow.log_param("run_id", run.info.run_id)
        mlflow.log_artifact(OUTPUT_METRICS_PATH)
        
        # Log model to MLflow
        mlflow.pytorch.log_model(best_model, "model")

        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE")
        print("="*60)
        print(f"Experiment:   {EXPERIMENT_NAME}")
        print(f"Experiment ID: {experiment_id}")
        print(f"Run ID:       {run.info.run_id}")
        print(f"Test MAE:     {test_mae:.4f}")
        print(f"Test RMSE:    {test_rmse:.4f}")
        print(f"Model saved to: {OUTPUT_MODEL_PATH}")
        print("="*60)
        print("\n💡 Next step: Run post_training_promotion.py to decide deployment")
        
        return run.info.run_id, metrics

# -------------------------
# ENTRY
# -------------------------
if __name__ == "__main__":
    train()