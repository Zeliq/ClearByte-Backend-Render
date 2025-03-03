from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model and tokenizer
MODEL_PATH = "./trained_model"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()

# Initialize FastAPI app
app = FastAPI()

class InputText(BaseModel):
    text: str

@app.post("/predict/")
def predict(input_data: InputText):
    """Run inference and return prediction"""
    inputs = tokenizer(input_data.text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    return {"predicted_class": predicted_class, "probabilities": probabilities.tolist()}

# Run with: uvicorn api:app --host 0.0.0.0 --port 8000
