# RoBERTa Regression Text Risk Scoring

This project trains a **RoBERTa-based regression model** to score text for AI vs Human vs Human-edited AI style.  
It uses **PyTorch Lightning** for training and **MLflow** for logging experiments, parameters, and model artifacts.

---

## **Dataset**

- CSV file with columns: `text` and `label`
- Label mapping:
  - `0.0` → Human-written text
  - `0.5` → Human-edited AI text
  - `1.0` →  AI-generated text
- Text length: normalized ~1200 characters
- Train/Validation split: 85% / 15%

---

## **Model**

- Base architecture: `roberta-base` (Hugging Face Transformers)
- Regression head:
  - Linear layer on `[CLS]` token output
  - Dropout: 0.3
- Loss function: `MSELoss`
- Task: Risk scoring (continuous output 0 → 1)

---

## **Tokenizer & Input**

- `RobertaTokenizer.from_pretrained('roberta-base')`
- Max tokens: 512 (can be increased to 1024 in experiments)
- Padding: `max_length`
- Truncation: `True`
- Batch size: 8 (modifiable)
- Attention mask used

---

## **Training Settings**

| Parameter             | Default Value |
|-----------------------|---------------|
| Max epochs            | 3             |
| Learning rate (AdamW) | 2e-5          |
| Batch size            | 8             |
| Early stopping        | val_loss, patience=2 |
| Mixed precision       | 16-bit (fp16) |
| GPU support           | automatic if available |

---

## **Logging / Tracking**

- MLflow tracks:
  - Model parameters
  - Hyperparameters: lr, batch_size, max_length, model_name
  - Training / validation loss
  - Model artifact saved in MLflow

---

## **Usage**

```bash
# Install dependencies
pip install -r requirements.txt

# Run training
python rob_base_lightning.py
