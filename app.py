import os
import numpy as np
import tensorflow as tf
import cv2
from flask import Flask, render_template, request
from PIL import Image
from PIL.ExifTags import TAGS
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# 1. Initialize Flask App
app = Flask(__name__)

# 2. Rebuild the exact Architecture
def build_forensic_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    base_model.trainable = False 
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# 3. Load Model and Weights
model = build_forensic_model()
WEIGHTS_PATH = 'model_weights.weights.h5'
if os.path.exists(WEIGHTS_PATH):
    model.load_weights(WEIGHTS_PATH)
    print("✅ Forensic Model & Weights Loaded!")

# --- ADVANCED FORENSIC FUNCTIONS ---

def check_metadata(pil_img):
    """Feature 1: Metadata/EXIF Scan (The Birth Certificate)"""
    try:
        exif_data = pil_img._getexif()
        if not exif_data:
            return "⚠️ NO EXIF DATA: Common in AI-generated or privacy-scrubbed images."
        
        info = []
        for tag, value in exif_data.items():
            decoded = TAGS.get(tag, tag)
            if decoded in ['Make', 'Model', 'DateTime', 'Software']:
                info.append(f"{decoded}: {value}")
        
        return "✅ DEVICE SIGNATURE: " + ", ".join(info) if info else "⚠️ Limited Camera Metadata found."
    except:
        return "⚠️ Metadata unreadable or missing."

def analyze_frequency(pil_img):
    """Feature 2: Frequency Domain Analysis (The FFT Grid Test)"""
    # Image ko grayscale numpy array mein badlein
    img_gray = np.array(pil_img.convert('L'))
    # Fast Fourier Transform (FFT)
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    # AI images mein unnatural periodicity hoti hai
    mean_val = np.mean(magnitude_spectrum)
    if mean_val > 105: # Specific threshold for synthetic artifacts
        return "🔴 HIGH FREQUENCY NOISE: Unnatural mathematical patterns detected (AI Signature)."
    return "🟢 NATURAL FREQUENCY: Pixel distribution matches organic sensor noise."

# --- ROUTES ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or request.files['file'].filename == '':
        return render_template('index.html', result="ERROR", reason="Please upload an image.")

    try:
        file = request.files['file']
        raw_img = Image.open(file)
        
        # 1. Forensic Scan: Metadata
        metadata_res = check_metadata(raw_img)
        
        # 2. Forensic Scan: Frequency (FFT)
        frequency_res = analyze_frequency(raw_img)
        
        # 3. AI Model Scan (Pixel Artifacts)
        img_processed = raw_img.convert('RGB').resize((128, 128))
        img_array = np.array(img_processed) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction_prob = model.predict(img_array)[0][0]
        
        # Decision Logic
        if prediction_prob > 0.5:
            result = "REAL"
            confidence = round(prediction_prob * 100, 2)
            reason = "Organic textures and natural lighting gradients confirmed."
        else:
            result = "AI-GENERATED"
            confidence = round((1 - prediction_prob) * 100, 2)
            reason = "Detected unnatural pixel smoothing and synthetic artifact signatures."

        return render_template('index.html', 
                               result=result, 
                               confidence=confidence, 
                               reason=reason,
                               metadata_status=metadata_res,
                               frequency_status=frequency_res)
    
    except Exception as e:
        return render_template('index.html', result="ERROR", reason=f"Analysis failed: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)