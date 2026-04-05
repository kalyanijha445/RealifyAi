import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from PIL import Image
from PIL.ExifTags import TAGS
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# ---------------- INIT ---------------- #
app = Flask(__name__)

# ---------------- MODEL ---------------- #

def build_forensic_model():
    # ❗ IMPORTANT: No internet download
    base_model = MobileNetV2(weights=None, include_top=False, input_shape=(128, 128, 3))
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),  # ✅ SAME as training
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

print("🔄 Loading model...")
model = build_forensic_model()

WEIGHTS_PATH = 'model_weights.weights.h5'

if os.path.exists(WEIGHTS_PATH):
    model.load_weights(WEIGHTS_PATH)
    print("✅ Model Loaded Successfully!")
else:
    print("❌ Model file not found!")

# 🔥 Warmup (important for Render)
try:
    model.predict(np.zeros((1, 128, 128, 3)))
    print("🔥 Model Warmed Up!")
except:
    print("⚠️ Warmup failed")

# ---------------- FORENSIC FUNCTIONS ---------------- #

def check_metadata(pil_img):
    try:
        exif_data = pil_img._getexif()
        if not exif_data:
            return "⚠️ No EXIF data found."

        info = []
        for tag, value in exif_data.items():
            decoded = TAGS.get(tag, tag)
            if decoded in ['Make', 'Model', 'DateTime']:
                info.append(f"{decoded}: {value}")

        return "✅ " + ", ".join(info) if info else "⚠️ Limited metadata"
    except:
        return "⚠️ Metadata not readable"

def analyze_frequency(pil_img):
    img_gray = np.array(pil_img.convert('L'))
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1)

    if np.mean(magnitude) > 105:
        return "🔴 AI-like frequency patterns detected"
    return "🟢 Natural image frequency"

# ---------------- ROUTES ---------------- #

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or request.files['file'].filename == '':
        return render_template('index.html', result="ERROR", reason="Upload image")

    try:
        file = request.files['file']
        raw_img = Image.open(file)

        # Metadata
        metadata_res = check_metadata(raw_img)

        # Frequency
        frequency_res = analyze_frequency(raw_img)

        # Preprocess
        img = raw_img.convert('RGB').resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = model.predict(img_array)[0][0]

        if prediction > 0.5:
            result = "REAL"
            confidence = round(prediction * 100, 2)
            reason = "Natural textures detected"
        else:
            result = "AI-GENERATED"
            confidence = round((1 - prediction) * 100, 2)
            reason = "Synthetic patterns detected"

        return render_template(
            'index.html',
            result=result,
            confidence=confidence,
            reason=reason,
            metadata_status=metadata_res,
            frequency_status=frequency_res
        )

    except Exception as e:
        return render_template('index.html', result="ERROR", reason=str(e))

# ---------------- RUN ---------------- #

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
