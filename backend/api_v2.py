from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from datetime import datetime
import torch
from transformers import RobertaTokenizer
import mlflow
import mlflow.pytorch
from langdetect import detect, LangDetectException
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# -----------------------------
# CONFIG
# -----------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///./mlflow.db")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.getenv("MODEL_PATH", "runs:/bf6fa44badde4e9b80412dce11457a0c/roberta_large_lora-sigmoid")
MODEL_NAME = os.getenv("MODEL_NAME", "roberta-large")

ALLOWED_FEEDBACK = {"human", "ai", "human_edited_ai"}

model = None
tokenizer = None

# -----------------------------
# LIFESPAN
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    print("🚀 Loading model...")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    model = mlflow.pytorch.load_model(MODEL_PATH)
    model.eval()
    model.to(DEVICE)
    print("✅ Model loaded successfully")
    yield
    print("🛑 Shutting down FastAPI...")

# -----------------------------
# APP
# -----------------------------
app = FastAPI(title="Model Feedback API", lifespan=lifespan)

# -----------------------------
# CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all for dev; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# SERVE FRONTEND
# -----------------------------
FRONTEND_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

app.mount("/static", StaticFiles(directory=FRONTEND_PATH), name="static")

@app.get("/")
def root():
    return FileResponse(os.path.join(FRONTEND_PATH, "index.html"))

# -----------------------------
# SCHEMAS
# -----------------------------
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=5)

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def apply_threshold(score: float) -> str:
    if score < 0.35:
        return "Human-written"
    elif score <= 0.65:
        return "Human-edited AI"
    else:
        return "AI-generated"

# -----------------------------
# PREDICT ENDPOINT
# -----------------------------
@app.post("/predict", response_model=dict)
def predict(payload: PredictRequest):
    text = payload.text

    # Language check
    try:
        lang = detect(text)
    except LangDetectException:
        raise HTTPException(status_code=400, detail="Language detection failed")
    if lang != "en":
        raise HTTPException(status_code=400, detail=f"Only English text allowed (detected: {lang})")

    # Tokenize & inference
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    try:
        with torch.no_grad():
            score = model(input_ids=input_ids, attention_mask=attention_mask).cpu().item()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    label = apply_threshold(score)

    return {"text": text, "score": score, "label": label}

# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "device": DEVICE}

# -----------------------------
# FEEDBACK ENDPOINT
# -----------------------------
class FeedbackRequest(BaseModel):
    text: str
    feedback: str = Field(..., regex="^(human|ai|human_edited_ai)$")

feedback_store = []

@app.post("/feedback")
def save_feedback(payload: FeedbackRequest):
    feedback_store.append({
        "text": payload.text,
        "feedback": payload.feedback,
        "timestamp": datetime.utcnow().isoformat()
    })
    return {"status": "saved"}














































import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
import hashlib
import torch
from transformers import RobertaTokenizer
import mlflow
import mlflow.pytorch
from langdetect import detect, LangDetectException
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# -----------------------------
# CONFIG
# -----------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///./mlflow.db")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.getenv("MODEL_PATH", "runs:/bf6fa44badde4e9b80412dce11457a0c/roberta_large_lora-sigmoid")
MODEL_NAME = os.getenv("MODEL_NAME", "roberta-large")

ALLOWED_FEEDBACK = {"human", "ai", "human_edited_ai"}

model = None
tokenizer = None

# -----------------------------
# LIFESPAN (startup/shutdown)
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    print("🚀 Loading model...")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    model = mlflow.pytorch.load_model(MODEL_PATH)
    model.eval()
    model.to(DEVICE)
    print("✅ Model loaded successfully")
    yield
    print("🛑 Shutting down FastAPI...")

# -----------------------------
# APP
# -----------------------------
app = FastAPI(title="Model Feedback API", lifespan=lifespan)

# -----------------------------
# SCHEMAS
# -----------------------------
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=5)

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def apply_threshold(score: float) -> str:
    """
    Lesser score = human-written
    Thresholds:
    <0.35: Human
    0.35-0.65: Human-edited AI
    >0.65: AI-generated
    """
    if score < 0.35:
        return "Human-written"
    elif score <= 0.65:
        return "Human-edited AI"
    else:
        return "AI-generated"

# -----------------------------
# PREDICT ENDPOINT
# -----------------------------
@app.post("/predict", response_model=dict)
def predict(payload: PredictRequest):
    text = payload.text

    # 1️⃣ Language check
    try:
        lang = detect(text)
    except LangDetectException:
        raise HTTPException(status_code=400, detail="Language detection failed")
    if lang != "en":
        raise HTTPException(status_code=400, detail=f"Only English text allowed (detected: {lang})")

    # 2️⃣ Tokenize & inference
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    try:
        with torch.no_grad():
            score = model(input_ids=input_ids, attention_mask=attention_mask).cpu().item()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # 3️⃣ Apply threshold label
    label = apply_threshold(score)

    return {
        "text": text,
        "score": score,
        "label": label
    }

# -----------------------------
# OPTIONAL: Simple health check
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "device": DEVICE}

