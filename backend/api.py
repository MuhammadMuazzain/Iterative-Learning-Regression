import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime
import torch
from transformers import RobertaTokenizer
import mlflow
import mlflow.pytorch
from langdetect import detect, LangDetectException
import psycopg2
import hashlib
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

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "ai_text_feedback"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
}

ALLOWED_FEEDBACK = {"human", "ai", "human_edited_ai"}

model = None
tokenizer = None

# -----------------------------
# APP INIT
# -----------------------------
app = FastAPI(title="ML Feedback API")

# -----------------------------
# CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# FRONTEND
# -----------------------------
FRONTEND_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_PATH), name="static")

@app.get("/")
def root():
    return FileResponse(os.path.join(FRONTEND_PATH, "index.html"))

# -----------------------------
# STARTUP EVENT: LOAD MODEL
# -----------------------------
@app.on_event("startup")
def load_model():
    global model, tokenizer
    print("🚀 Loading model...")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    model = mlflow.pytorch.load_model(MODEL_PATH)
    model.eval()
    model.to(DEVICE)
    print("✅ Model loaded successfully")

# -----------------------------
# SCHEMAS
# -----------------------------
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=5)

class FeedbackRequest(BaseModel):
    text: str = Field(..., min_length=5)
    model_score: float = Field(..., ge=0.0, le=1.0)
    user_feedback: str = Field(..., pattern="^(human|ai|human_edited_ai)$")
    model_version: str

# -----------------------------
# DB UTILITIES
# -----------------------------
def get_conn():
    return psycopg2.connect(**DB_CONFIG)

def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

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
    global model, tokenizer
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

    label = apply_threshold(score)
    return {"text": text, "score": score, "label": label}

# -----------------------------
# FEEDBACK ENDPOINT
# -----------------------------
@app.post("/feedback")
def submit_feedback(payload: FeedbackRequest):
    # Validate user feedback
    if payload.user_feedback not in ALLOWED_FEEDBACK:
        raise HTTPException(
            status_code=400,
            detail=f"user_feedback must be one of {ALLOWED_FEEDBACK}"
        )

    # Language check
    try:
        lang = detect(payload.text)
    except LangDetectException:
        raise HTTPException(status_code=400, detail="Language detection failed")
    if lang != "en":
        raise HTTPException(
            status_code=400,
            detail=f"Only English text allowed (detected: {lang})"
        )

    # Deduplication hash
    text_hash = hash_text(payload.text)

    # Insert into DB
    conn = get_conn()
    cur = conn.cursor()

    # Check duplicate
    cur.execute(
        "SELECT 1 FROM feedback_events WHERE text_hash = %s LIMIT 1",
        (text_hash,)
    )
    if cur.fetchone():
        cur.close()
        conn.close()
        raise HTTPException(status_code=409, detail="Duplicate feedback")

    # Insert feedback
    cur.execute(
        """
        INSERT INTO feedback_events (
            text,
            text_hash,
            model_score,
            user_feedback,
            model_version,
            event_timestamp
        )
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (
            payload.text,
            text_hash,
            payload.model_score,
            payload.user_feedback,
            payload.model_version,
            datetime.utcnow()
        )
    )

    conn.commit()
    cur.close()
    conn.close()

    return {
        "status": "ok",
        "language": lang,
        "stored_at": datetime.utcnow().isoformat()
    }

# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "device": DEVICE}
