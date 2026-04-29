from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


app = FastAPI(title="Spam Mail Classifier API")

MODEL_PATH = Path(__file__).resolve().parent / "model" / "spam_classifier_model.pkl"
model = joblib.load(MODEL_PATH)


class MessageInput(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        description="Message text to classify as spam or ham."
    )


class PredictionResponse(BaseModel):
    prediction: str


@app.get("/health")
def health_check():
    return {
        "message": "Spam Mail Classifier API is running"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_message(input_data: MessageInput):
    prediction = model.predict([input_data.message])

    return {
        "prediction": prediction[0]
    }