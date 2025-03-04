from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import httpx
import re
import pandas as pd

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OCR.space API configuration
OCR_API_URL = "https://api.ocr.space/parse/image"
OCR_API_KEY = "K84072634788957"  # Get free key from https://ocr.space/ocrapi

# Load ingredient patterns
def load_and_prepare_patterns(csv_path="Database1.csv"):
    df = pd.read_csv(csv_path)
    
    categories = {
        "halal": df["halal"].dropna().str.lower().tolist(),
        "haram": df["non_category"].dropna().str.lower().tolist(),
        "vegan": df["vegan"].dropna().str.lower().tolist(),
        "vegetarian": df["vegetarian"].dropna().str.lower().tolist(),
    }
    
    patterns = {}
    for name, words in categories.items():
        if words:
            patterns[name] = re.compile(r'\b(' + '|'.join(map(re.escape, words)) + r')\b')
        else:
            patterns[name] = re.compile(r'(?!x)x')
    
    return patterns

PATTERNS = load_and_prepare_patterns()

async def ocr_space_api(file_bytes: bytes) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            OCR_API_URL,
            files={"file": ("image.jpg", file_bytes)},
            data={
                "apikey": OCR_API_KEY,
                "language": "eng",
                "isOverlayRequired": False,
                "filetype": "JPG",
                "detectOrientation": True,
                "scale": True
            }
        )
    
    if response.status_code == 200:
        result = response.json()
        if result["IsErroredOnProcessing"]:
            return ""
        return "\n".join([res["ParsedText"] for res in result["ParsedResults"]])
    return ""

def classify_ingredients(text):
    text = text.lower()
    
    matches = {
        name: pattern.search(text) is not None
        for name, pattern in PATTERNS.items()
    }
    
    return {
        "halal": matches["halal"] or (not matches["haram"]),
        "haram": matches["haram"],
        "vegan": matches["vegan"] and not (matches["haram"] or matches["vegetarian"]),
        "vegetarian": matches["vegetarian"] and not matches["haram"]
    }

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    file_bytes = await file.read()
    
    # Perform OCR using external API
    text = await ocr_space_api(file_bytes)
    del file_bytes  # Free memory immediately
    
    # Classification
    classification = classify_ingredients(text)
    
    return {"text": text, "classification": classification}