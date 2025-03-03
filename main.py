from fastapi import FastAPI, File, UploadFile
import io
import cv2
import numpy as np
import pytesseract
from fastapi.middleware.cors import CORSMiddleware
import os
import csv

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Replace pandas with direct CSV reading
def load_ingredients_from_csv(csv_filename="Database1.csv"):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, csv_filename)
    
    halal = []
    haram = []
    vegan = []
    vegetarian = []
    
    with open(csv_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get('halal') and row['halal'].strip():
                halal.append(row['halal'].lower())
            if row.get('non_category') and row['non_category'].strip():
                haram.append(row['non_category'].lower())
            if row.get('vegan') and row['vegan'].strip():
                vegan.append(row['vegan'].lower())
            if row.get('vegetarian') and row['vegetarian'].strip():
                vegetarian.append(row['vegetarian'].lower())
    
    return halal, haram, vegan, vegetarian

# Load categories only once at startup
HALAL, HARAM, VEGAN, VEGETARIAN = load_ingredients_from_csv()

# Preprocess Image
def preprocess_image(image_bytes):
    image_np = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(image_np, cv2.IMREAD_GRAYSCALE)
    
    # Resize with aspect ratio but limit max size
    height, width = img.shape
    max_dimension = 600  # Reduced from 800
    
    if width > height:
        new_width = min(width, max_dimension)
        new_height = int((height / width) * new_width)
    else:
        new_height = min(height, max_dimension)
        new_width = int((width / height) * new_height)
    
    img = cv2.resize(img, (new_width, new_height))
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return img

# Only use Tesseract OCR to reduce memory usage
def tesseract_ocr(image_bytes):
    image = preprocess_image(image_bytes)
    return pytesseract.image_to_string(image)

# Optimize classification by using sets for faster lookups
HALAL_SET = set(HALAL)
HARAM_SET = set(HARAM)
VEGAN_SET = set(VEGAN)
VEGETARIAN_SET = set(VEGETARIAN)

def text_contains_any(text, word_set):
    # Check if any words from the set appear in the text
    text_words = set(text.split())
    return bool(text_words.intersection(word_set))

def classify_ingredients(text):
    text = text.lower()
    
    contains_halal = text_contains_any(text, HALAL_SET)
    contains_haram = text_contains_any(text, HARAM_SET)
    contains_vegan = text_contains_any(text, VEGAN_SET)
    contains_vegetarian = text_contains_any(text, VEGETARIAN_SET)
    
    # Halal: Includes Vegan and Vegetarian items, but should not contain Haram
    is_halal = (contains_vegan or contains_vegetarian or contains_halal) and not contains_haram
    
    # Vegan: Must contain at least one vegan ingredient and NO non-vegan ingredients
    is_vegan = contains_vegan and not (contains_vegetarian or contains_haram)
    
    # Vegetarian: Must contain at least one vegetarian ingredient and should not contain haram
    is_vegetarian = contains_vegetarian and not contains_haram
    
    classification = {
        "halal": is_halal,
        "haram": contains_haram,
        "vegan": is_vegan,
        "vegetarian": is_vegetarian
    }
    return classification

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    text = tesseract_ocr(image_bytes)
    classification = classify_ingredients(text)
    return {"text": text, "classification": classification}

# Add healthcheck endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}