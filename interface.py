import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model and tokenizer
MODEL_PATH = "./results/checkpoint-1380"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()

def predict(text):
    """Run inference on a given text"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    return predicted_class, probabilities.tolist()

# Example usage
text = "What is cucumber"
predicted_class, probabilities = predict(text)
print(f"Predicted class: {predicted_class}, Probabilities: {probabilities}")
