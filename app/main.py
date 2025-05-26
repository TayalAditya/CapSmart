from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import tempfile
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Only show errors
import io
import json
import logging
from tensorflow.keras.models import load_model
from concurrent.futures import ThreadPoolExecutor
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import joblib
from werkzeug.middleware.proxy_fix import ProxyFix

# Initialize Flask app first
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== Constants and Paths =====
MODEL_PATH = os.getenv("MODEL_PATH", "app/model/blur_detection_model_v2.h5")
ISO_MODEL_PATH = os.path.join("app/model", "iso_classifier_model.h5")
ISO_SCALER_PATH = os.path.join("app/model", "iso_feature_scaler.pkl")
ISO_MAP_PATH = os.path.join("app/model", "iso_class_to_label.pkl")
SS_MODEL_PATH = os.path.join("app/model", "stable_shutter_model.h5")
SS_SCALER_PATH = os.path.join("app/model", "stable_shutter_scaler.pkl")
IMAGE_SIZE = (224, 224)
SETTINGS_FILE = "settings.json"

# ===== Model Container =====
models = {
    'blur': None,
    'iso_model': None,
    'iso_scaler': None,
    'class_to_iso': None,
    'ss_model': None,
    'ss_scaler': None
}

# ===== Helper Functions =====
def safe_load(model_key, loader, path):
    """Safely load a model/scaler with error handling"""
    try:
        if os.path.exists(path):
            models[model_key] = loader(path)
            logger.info(f"Loaded {model_key} from {path}")
        else:
            logger.warning(f"Missing model file: {path}")
    except Exception as e:
        logger.error(f"Failed to load {model_key}: {str(e)}")

def validate_model_loading():
    """Validate essential models are loaded"""
    essential_models = [
        ('iso_model', ISO_MODEL_PATH),
        ('iso_scaler', ISO_SCALER_PATH),
        ('class_to_iso', ISO_MAP_PATH),
        ('ss_model', SS_MODEL_PATH),
        ('ss_scaler', SS_SCALER_PATH)
    ]
    
    missing = []
    for name, path in essential_models:
        if not models[name]:
            missing.append(f"{name} ({path})")
    
    if missing:
        logger.critical(f"CRITICAL: Missing essential models: {', '.join(missing)}")
        raise SystemExit(1)

# ===== Model Loading =====
try:
    logger.info("Starting model loading sequence")
    
    # Load models using safe_load
    safe_load('blur', lambda p: load_model(p, custom_objects={'mse': tf.keras.losses.MeanSquaredError()}), MODEL_PATH)
    safe_load('iso_model', tf.keras.models.load_model, ISO_MODEL_PATH)
    safe_load('iso_scaler', joblib.load, ISO_SCALER_PATH)
    safe_load('class_to_iso', joblib.load, ISO_MAP_PATH)
    safe_load('ss_model', lambda p: load_model(p, compile=False), SS_MODEL_PATH)
    safe_load('ss_scaler', joblib.load, SS_SCALER_PATH)

    # Post-process ISO mapping
    if models['class_to_iso'] is not None:
        models['class_to_iso'] = [models['class_to_iso'][i] for i in range(len(models['class_to_iso']))]

    validate_model_loading()
    logger.info("All essential models loaded successfully")

except Exception as e:
    logger.critical(f"Model loading failed: {e}")
    raise SystemExit(1)

# ===== Image Processing Functions =====
def detect_blur_laplacian(image):
    """Calculate Laplacian variance for blur detection"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def detect_blur_tenengrad(image, ksize=3):
    """Calculate Tenengrad score for blur detection"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    return float(np.mean(gx**2 + gy**2))

def perceptual_blur_metric(image, threshold=0.1):
    """Calculate perceptual blur metric using edge widths"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        magnitude = np.sqrt(dx**2 + dy**2)
        edge_coords = np.column_stack(np.where(edges > 0))
        edge_widths = [1.0 / magnitude[y, x] for y, x in edge_coords if magnitude[y, x] > threshold]
        return float(np.mean(edge_widths)) if edge_widths else 0.0
    except Exception as e:
        logger.error(f"Perceptual blur error: {e}")
        return 0.0

# ===== Feature Calculation Functions =====
def brightness(image):
    """Calculate image brightness"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))
    except Exception as e:
        logger.error(f"Brightness error: {e}")
        return 0.0

def histogram_stats(image):
    """Calculate histogram statistics"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        return float(np.mean(hist)), float(np.var(hist))
    except Exception as e:
        logger.error(f"Histogram error: {e}")
        return 0.0, 0.0

def edge_density(image):
    """Calculate edge density"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return float(np.sum(edges > 0) / edges.size) if edges.size > 0 else 0.0
    except Exception as e:
        logger.error(f"Edge density error: {e}")
        return 0.0

# ===== Prediction Functions =====
def predict_shutter_speed(image):
    """Predict shutter speed using loaded model"""
    if not models['ss_model'] or not models['ss_scaler']:
        logger.error("Shutter speed model/scaler not loaded!")
        return None
    
    try:
        # Feature extraction
        lap = detect_blur_laplacian(image)
        ten = detect_blur_tenengrad(image)
        pbm = perceptual_blur_metric(image)
        edge = edge_density(image)
        bright = brightness(image)
        hist_mean, hist_var = histogram_stats(image)
        
        features = [lap, ten, pbm, edge, bright, hist_mean, hist_var]
        
        if not all(np.isfinite(f) for f in features):
            logger.error("Invalid feature values detected")
            return None

        scaled = models['ss_scaler'].transform([features])
        prediction = models['ss_model'].predict(scaled, verbose=0)
        return float(prediction[0][0])
    
    except Exception as e:
        logger.error(f"Shutter speed prediction failed: {e}")
        return None

@app.route('/recommend_settings', methods=['POST'])
def recommend_settings():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        with tempfile.NamedTemporaryFile() as tmp:
            # 1. Save and resize image
            request.files['image'].save(tmp.name)
            img = cv2.imread(tmp.name)
            if img is None:
                return jsonify({'error': 'Invalid image'}), 400

            # Resize to reduce processing time
            img = cv2.resize(img, (640, 480))  # Adjust dimensions as needed

            # 2. Parallelize predictions
            with ThreadPoolExecutor() as executor:
                iso_future = executor.submit(predict_iso, img)
                shutter_future = executor.submit(predict_shutter_speed, img)
                iso = iso_future.result(timeout=30)
                shutter_speed = shutter_future.result(timeout=30)

            return jsonify({
                'recommended_iso': iso,
                'recommended_shutter_speed': float(shutter_speed)
            })
            
    except TimeoutError:
        logger.error("Recommendation timed out")
        return jsonify({'error': 'Processing timeout'}), 504
    except Exception as e:
        logger.error(f"Recommendation error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Server error'}), 500

# @app.route('/')
# def health_check():
#     return jsonify({
#         "status": "running",
#         "endpoints": ["/analyze", "/unblur", "/settings", "/recommend"]
#     })

@app.route('/')
def health_check():
    return jsonify({
        "status": "healthy",
        "loaded_models": {
            "blur": models['blur'] is not None,
            "iso": models['iso_model'] is not None,
            "shutter": models['ss_model'] is not None
        }
    }), 200


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
        
    if request.content_length > 5 * 1024 * 1024:  # 5MB limit
        return jsonify({'error': 'Image too large (max 5MB)'}), 413
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            request.files['image'].save(tmp.name)
            img = cv2.imread(tmp.name)
            if img is None:
                return jsonify({'error': 'Invalid image'}), 400

            results = {
                'laplacian': detect_blur_laplacian(img),
                'tenengrad': detect_blur_tenengrad(img),
                'perceptual_blur': perceptual_blur_metric(img),
                'cnn_blur_score': min(predict_blur_with_model(img) or 0, 100)
            }
            return jsonify(results)
    finally:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)

@app.route('/unblur', methods=['POST'])
def unblur():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    try:
        with tempfile.NamedTemporaryFile() as tmp:
            request.files['image'].save(tmp.name)
            img = cv2.imread(tmp.name)
            if img is None:
                return jsonify({'error': 'Invalid image'}), 400

            sharpened = cv2.filter2D(img, -1, np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]))
            _, buffer = cv2.imencode('.jpg', sharpened)
            return send_file(io.BytesIO(buffer), mimetype='image/jpeg')
    except Exception as e:
        logger.error(f"Unblur error: {e}")
        return jsonify({'error': 'Processing failed'}), 500

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if request.method == 'POST':
        try:
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(request.json, f)
            return jsonify({'status': 'settings saved'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        if not os.path.exists(SETTINGS_FILE):
            return jsonify({'error': 'No settings found'}), 404
        with open(SETTINGS_FILE) as f:
            return jsonify(json.load(f))


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    if models['blur'] or models['iso_model']:
        app.run(host='0.0.0.0', port=port)
    else:
        logger.critical("Critical models failed to load!")
        exit(1)
