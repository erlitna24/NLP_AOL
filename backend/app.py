from tokenize import tokenize

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path

import re
import torch
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer


# =========================
# CONFIG LOKAL
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_ID = "littt24/indobert_model"

TOKENIZER_NAME = "indobenchmark/indobert-base-p1"

MAX_LENGTH = 96
LIME_NUM_SAMPLES = 150

# =========================
# TEMPERATURE SCALING
# =========================
# 1.0 = probability asli model
# >1.0 = confidence lebih lunak / tidak terlalu overconfident
# Contoh: 2, 3, 5, 7
# Untuk demo UI boleh pakai 7.
# Untuk laporan akademik/evaluasi asli, lebih aman pakai 1.0.
TEMPERATURE = 7.0

LABELS = {
    0: "Non-Clickbait",
    1: "Clickbait"
}

CLASS_NAMES = ["Non-Clickbait", "Clickbait"]


# =========================
# FASTAPI APP
# =========================

app = FastAPI(
    title="IndoBERT Clickbait Detection API",
    version="1.0.0"
)
tokenize
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# REQUEST BODY
# =========================

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)


# =========================
# TEXT CLEANING
# =========================

def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^\w\s.,!?-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


## =========================
# LOAD MODEL FROM HUGGING FACE
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("BASE_DIR:", BASE_DIR)
print("MODEL_ID:", MODEL_ID)
print("Device:", device)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

model.to(device)
model.eval()

print("Model loaded successfully from Hugging Face.")
print("Temperature:", TEMPERATURE)

# =========================
# PREDICTION HELPER
# =========================

def predict_proba(texts):
    """
    Function required by LIME.
    Input  : list of text
    Output : numpy array with shape [n_texts, n_classes]

    Temperature scaling:
    - logits / 1.0 = original probability
    - logits / 7.0 = softer confidence
    """

    cleaned_texts = [clean_text(text) for text in texts]

    inputs = tokenizer(
        cleaned_texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        # Temperature scaling ditambahkan dari file 1
        scaled_logits = logits / TEMPERATURE

        probabilities = torch.softmax(scaled_logits, dim=1)

    return probabilities.cpu().numpy()


def predict_single_text(text: str):
    cleaned_text = clean_text(text)
    probabilities = predict_proba([cleaned_text])[0]

    predicted_class = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_class])

    return cleaned_text, probabilities, predicted_class, confidence


def generate_lime_explanation(text: str, predicted_class: int):
    explainer = LimeTextExplainer(class_names=CLASS_NAMES)

    explanation = explainer.explain_instance(
        text_instance=text,
        classifier_fn=predict_proba,
        labels=[predicted_class],
        num_features=8,
        num_samples=LIME_NUM_SAMPLES
    )

    lime_list = explanation.as_list(label=predicted_class)

    formatted_explanation = []

    for word, weight in lime_list:
        formatted_explanation.append({
            "word": word,
            "weight": round(float(weight), 4),
            "abs_weight": round(abs(float(weight)), 4),
            "impact": "supports_prediction" if weight > 0 else "against_prediction"
        })

    return formatted_explanation


# =========================
# ROUTES
# =========================

@app.get("/")
def home():
    return {
        "message": "IndoBERT Clickbait Detection API is running",
        "model_path": str(MODEL_ID),
        "model_exists": MODEL_ID.exists(),
        "device": str(device),
        "temperature": TEMPERATURE,
        "lime_num_samples": LIME_NUM_SAMPLES
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "temperature": TEMPERATURE
    }


@app.post("/predict")
def predict(request: PredictRequest):
    text = request.text.strip()

    if text == "":
        return {
            "error": "Text cannot be empty"
        }

    cleaned_text, probabilities, predicted_class, confidence = predict_single_text(text)

    lime_explanation = generate_lime_explanation(
        text=cleaned_text,
        predicted_class=predicted_class
    )

    non_clickbait_prob = round(float(probabilities[0]), 4)
    clickbait_prob = round(float(probabilities[1]), 4)

    prediction_label = LABELS[predicted_class]
    is_clickbait = predicted_class == 1

    return {
        "input_text": text,
        "cleaned_text": cleaned_text,

        "prediction_id": predicted_class,
        "prediction_label": prediction_label,
        "prediction": prediction_label,
        "isClickbait": is_clickbait,

        "confidence": round(confidence, 4),

        "probabilities": {
            "Non-Clickbait": non_clickbait_prob,
            "Clickbait": clickbait_prob
        },

        "probability": {
            "non_clickbait": non_clickbait_prob,
            "clickbait": clickbait_prob
        },

        "lime_explanation": lime_explanation,
        "lime": lime_explanation,

        "temperature": TEMPERATURE
    }