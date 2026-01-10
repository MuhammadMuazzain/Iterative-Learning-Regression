# rob_base_lightning.py
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from transformers import RobertaTokenizer, RobertaModel
from torch.optim import AdamW

import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.pytorch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =========================
# 1️⃣ Dataset
# =========================
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.float)
        }

# =========================
# 2️⃣ Lightning Module
# =========================
class RobertaRegression(pl.LightningModule):
    def __init__(self, model_name='roberta-base', lr=2e-5):
        super().__init__()
        self.save_hyperparameters()
        self.model = RobertaModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.regressor = nn.Linear(self.model.config.hidden_size, 1)
        self.loss_fn = nn.MSELoss()

        # store all step losses for logging to MLflow
        self.train_losses = []
        self.val_losses = []

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        pred = self.regressor(cls_output)
        return pred.squeeze(-1)

    # ---------------- Training ----------------
    def training_step(self, batch, batch_idx):
        labels = batch['label']
        preds = self(batch['input_ids'], batch['attention_mask'])
        loss = self.loss_fn(preds, labels)
        self.log('train_loss_step', loss, prog_bar=True)
        self.train_losses.append(loss.detach().cpu().item())
        return {'loss': loss}

    def on_train_epoch_end(self):
        if self.train_losses:  # use stored step losses
            avg_train_loss = sum(self.train_losses[-len(self.train_losses):]) / len(self.train_losses)
            mlflow.log_metric('train_loss_epoch', avg_train_loss, step=self.current_epoch)
            print(f"Epoch {self.current_epoch}: avg_train_loss = {avg_train_loss:.4f}")
        else:
            print("No training losses to log.")

    # ---------------- Validation ----------------
    def validation_step(self, batch, batch_idx):
        labels = batch['label']
        preds = self(batch['input_ids'], batch['attention_mask'])
        loss = self.loss_fn(preds, labels)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)  # <--- key
        self.val_losses.append(loss.detach().cpu().item())
        return {'val_loss': loss, 'preds': preds.detach(), 'labels': labels.detach()}


    def on_validation_epoch_end(self):
        if self.val_losses: # use stored step losses
            avg_val_loss = sum(self.val_losses[-len(self.val_losses):]) / len(self.val_losses)
            mlflow.log_metric('val_loss_epoch', avg_val_loss, step=self.current_epoch)
            print(f"Epoch {self.current_epoch}: avg_val_loss = {avg_val_loss:.4f}")
        else:
            print("No validation losses to log.")

    # ---------------- Optimizer ----------------
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.hparams.lr)

    # ---------------- End of Training ----------------
    def on_train_end(self):
        # log full learning curves to MLflow
        for step, loss in enumerate(self.train_losses):
            mlflow.log_metric('train_loss_step', loss, step=step)
        for step, loss in enumerate(self.val_losses):
            mlflow.log_metric('val_loss_step', loss, step=step)
        print("Full training/validation curves logged to MLflow.")



# =========================
# 3️⃣ Data Preparation
# =========================
def load_data(csv_path, tokenizer, max_length=512, batch_size=8):
    df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42, stratify=None)
    
    train_dataset = TextDataset(
        train_df['text'].tolist(), 
        train_df['label'].tolist(),
        tokenizer,
        max_length=max_length
    )
    val_dataset = TextDataset(
        val_df['text'].tolist(),
        val_df['label'].tolist(),
        tokenizer,
        max_length=max_length
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# =========================
# 4️⃣ Training
# =========================
def train_model(csv_path, max_length=512, epochs=3, batch_size=8, lr=2e-5):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    train_loader, val_loader = load_data(csv_path, tokenizer, max_length, batch_size)
    
    model = RobertaRegression(lr=lr)

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "sqlite:///./mlflow.db")
    )
    
    # MLflow logging
    mlflow.set_experiment("roberta_base_regression_v1")
    mlflow.start_run()
    mlflow.log_params({'model': 'roberta-base', 'max_length': max_length, 'batch_size': batch_size, 'lr': lr, 'epochs': epochs})
    
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision='16-mixed',        # mixed precision for speed
        log_every_n_steps=10,
        callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode='min')]
    )
    
    trainer.fit(model, train_loader, val_loader)
    
    # Save model to MLflow
    mlflow.pytorch.log_model(model, "roberta_base_regression")
    mlflow.end_run()
    
    print("Training complete and model saved to MLflow!")

# =========================
# 5️⃣ Run
# =========================
if __name__ == "__main__":
    csv_path = "v1-training-without-test.csv"  # your CSV with columns: text,label
    train_model(csv_path)
