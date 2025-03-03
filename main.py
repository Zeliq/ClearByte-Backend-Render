from fastapi import FastAPI, File, UploadFile
import io
import cv2
import numpy as np
import easyocr
import pytesseract
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change to ["http://localhost:3000"] for security)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Load OCR client
reader = easyocr.Reader(['en'])

# Sample ingredient categories
import pandas as pd

# Load ingredient categories from CSV
def load_ingredients_from_csv(csv_path="C:/Users/zeliqzayyan/OneDrive/Desktop/ingredient-checker/backend/Database1.csv"):
    df = pd.read_csv(csv_path)
    
    halal = df["halal"].dropna().str.lower().tolist()
    haram = df["non_category"].dropna().str.lower().tolist()
    vegan = df["vegan"].dropna().str.lower().tolist()
    vegetarian = df["vegetarian"].dropna().str.lower().tolist()
    
    return halal, haram, vegan, vegetarian

# Load categories
HALAL, HARAM, VEGAN, VEGETARIAN = load_ingredients_from_csv()


# Preprocess Image
def preprocess_image(image_bytes):
    image_np = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(image_np, cv2.IMREAD_GRAYSCALE)

    # Resize with aspect ratio
    height, width = img.shape
    new_width = 800
    new_height = int((height / width) * new_width)  # Maintain aspect ratio
    img = cv2.resize(img, (new_width, new_height))

    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return img


# EasyOCR (primary OCR)
def easyocr_ocr(image_bytes):
    image = preprocess_image(image_bytes)
    return " ".join([text[1] for text in reader.readtext(image)])

# Tesseract OCR (alternative)
def tesseract_ocr(image_bytes):
    image = preprocess_image(image_bytes)
    return pytesseract.image_to_string(image)

# Classify ingredients
def classify_ingredients(text):
    text = text.lower()

    contains_halal = any(word in text for word in HALAL)
    print(contains_halal)
    contains_haram = any(word in text for word in HARAM)
    contains_vegan = any(word in text for word in VEGAN)
    contains_vegetarian = any(word in text for word in VEGETARIAN)

    # Halal: Includes Vegan and Vegetarian items, but should not contain Haram
    is_halal = (contains_vegan or contains_vegetarian or not contains_haram) and not contains_haram or contains_halal

    # Vegan: Must contain at least one vegan ingredient and NO non-vegan (vegetarian/haram) ingredients
    is_vegan = contains_vegan and not any(word in text for word in VEGETARIAN + HARAM)

    # Vegetarian: Must contain at least one vegetarian ingredient and should not contain haram
    is_vegetarian = contains_vegetarian and not contains_haram

    classification = {
        "halal": is_halal,
        "haram": contains_haram,
        "vegan": is_vegan,
        "vegetarian": is_vegetarian
    }

    return classification

print("Hellooooo")
@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    image_bytes = await file.read()

    try:
        text = easyocr_ocr(image_bytes)  # Best free option
    except Exception:
        text = tesseract_ocr(image_bytes)  # Fallback OCR

    classification = classify_ingredients(text)

    print(classification)

    return {"text": text, "classification": classification}
