import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import easyocr
import pytesseract

# Set Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"


def preprocess_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Error: File '{image_path}' not found. Check the path.")
    
    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Error: Could not read '{image_path}'. File may be corrupted or unsupported.")
    
    # Resize for better OCR readability
    scale_percent = 150  # Increase size by 150%
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    
    # Noise Reduction - Bilateral Filtering
    img = cv2.bilateralFilter(img, 9, 75, 75)
    
    # Adaptive Thresholding for binarization
    adaptive_thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    _, img = cv2.threshold(adaptive_thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological Transformations (Closing to merge broken characters)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    
    # Show processed image (ensuring no rotation occurs)
    plt.imshow(img, cmap='gray')
    plt.title('Preprocessed Image for OCR')
    plt.axis('off')
    plt.show()
    
    return img


# File path
image_path = "C:/Users/zeliqzayyan/OneDrive/Desktop/ingredient-checker/backend/img.png"

try:
    processed_img = preprocess_image(image_path)
except Exception as e:
    print(e)
    exit()

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

def easyocr_ocr(processed_img):
    processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
    extracted_text_easyocr = reader.readtext(processed_img_rgb)
    easyocr_text = " ".join([text[1] for text in extracted_text_easyocr])
    return easyocr_text

def tesseract_ocr(processed_img):
    config = "--psm 6"
    tesseract_text = pytesseract.image_to_string(processed_img, config=config)
    return tesseract_text

# Extract text using both EasyOCR and Tesseract
easyocr_text = easyocr_ocr(processed_img)
tesseract_text = tesseract_ocr(processed_img)

# Combine results
final_text = easyocr_text if len(easyocr_text) > len(tesseract_text) else tesseract_text

print("\n🔹 Final Extracted Text:\n", final_text)
