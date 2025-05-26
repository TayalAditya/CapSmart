from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import tempfile
import os
import io
import json
import logging
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== CONFIG =====
MODEL_PATH = os.getenv("MODEL_PATH", "model/blur_detection_model_v2.h5")
IMAGE_SIZE = (224, 224)
SETTINGS_FILE = "settings.json"

# ===== Load Model =====
model = None
if os.path.exists(MODEL_PATH):
    try:
        custom_objects = {'mse': tf.keras.losses.MeanSquaredError()}
        model = load_model(MODEL_PATH, custom_objects=custom_objects)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Model load error: {e}")
else:
    logger.error(f"Model file not found at {MODEL_PATH}")

# ===== Blur Metrics =====
def detect_blur_laplacian(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def detect_blur_tenengrad(image, ksize=3):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    fm = np.sqrt(gx**2 + gy**2)
    return np.mean(fm)

def perceptual_blur_metric(image, threshold=0.1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    magnitude = np.sqrt(dx**2 + dy**2)
    edge_widths = [1.0 / magnitude[y, x] for y, x in np.column_stack(np.where(edges > 0)) if magnitude[y, x] > threshold]
    return float(np.mean(edge_widths)) if edge_widths else 0.0

def predict_blur_with_model(image):
    if model is None:
        return None
    try:
        image_resized = cv2.resize(image, IMAGE_SIZE)
        image_array = preprocess_input(img_to_array(image_resized))
        image_array = np.expand_dims(image_array, axis=0)
        predicted_blur = model.predict(image_array)[0][0]
        return float(predicted_blur)
    except Exception as e:
        logger.error(f"Model prediction error: {e}")
        return None

# ===== Unblurring (sharpening) =====
def perform_unblurring(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    unblurred_img = cv2.filter2D(img, -1, kernel)
    success, buffer = cv2.imencode(".jpg", unblurred_img)
    return io.BytesIO(buffer) if success else None

# ======= Routes =======

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    img_path = tmp.name
    try:
        request.files['image'].save(img_path)
        tmp.close()
        image = cv2.imread(img_path)
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400

        lap = detect_blur_laplacian(image)
        ten = detect_blur_tenengrad(image)
        pbm = perceptual_blur_metric(image)
        predicted_score = predict_blur_with_model(image)
        if predicted_score is None:
            return jsonify({'error': 'Model prediction failed'}), 500

        final_score = min(predicted_score * 5, 100)
        return jsonify({
            'laplacian_variance': lap,
            'tenengrad_score': ten,
            'perceptual_blur_metric': pbm,
            'predicted_blur_score': final_score
        })
    finally:
        if os.path.exists(img_path):
            os.unlink(img_path)

@app.route('/')
def health_check():
    return jsonify({"status": "healthy", "message": "Blur detection API is running"}), 200
    
@app.route('/unblur', methods=['POST'])
def unblur_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    img_path = tmp.name
    try:
        request.files['image'].save(img_path)
        tmp.close()
        result = perform_unblurring(img_path)
        if result is None:
            return jsonify({'error': 'Failed to unblur image'}), 500
        result.seek(0)
        return send_file(result, mimetype='image/jpeg')
    finally:
        if os.path.exists(img_path):
            os.unlink(img_path)

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if request.method == 'GET':
        if not os.path.exists(SETTINGS_FILE):
            return jsonify({'message': 'No settings saved yet'}), 404
        with open(SETTINGS_FILE, 'r') as f:
            return jsonify(json.load(f))
    elif request.method == 'POST':
        try:
            data = request.get_json()
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(data, f, indent=2)
            return jsonify({'message': 'Settings saved'}), 200
        except Exception as e:
            return jsonify({'error': f'Failed to save settings: {e}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use Render's PORT env var or default to 5000
    if model:
        app.run(host='0.0.0.0', port=port)
    else:
        logger.critical("Model could not be loaded. Exiting.")
        # Exit the app if model is mandatory
        import sys
        sys.exit(1)
