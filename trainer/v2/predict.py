import os
import torch
import mlflow.pytorch
from transformers import RobertaTokenizer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "sqlite:///./mlflow.db")
)

# -------------------------
# CONFIG
# -------------------------
RUN_ID = "2487a8fc6c5548d0aa4b26cefcaab209" # 2487a8fc6c5548d0aa4b26cefcaab209
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 512

# -------------------------
# LOAD MODEL
# -------------------------
model = mlflow.pytorch.load_model(
    f"runs:/{RUN_ID}/roberta_large_lora",
    map_location=DEVICE
)

model.to(DEVICE)
model.eval()

# -------------------------
# TOKENIZER
# -------------------------
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# -------------------------
# PREDICTION
# -------------------------
def predict(text: str) -> float:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    return output.squeeze().item()

# -------------------------
# TEST
# -------------------------
if __name__ == "__main__":
    text = "RoBERTa-Large has 512M parameters and needed a fair amount of GPU memory during training. An RTX A2000 typically comes with 6 GB or 12 GB of VRAM (desktop/workstation variants), which is modest for large transformer models."

    score = predict(text)
    print("Predicted regression score:", score)
