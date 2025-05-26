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
import joblib
from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== Environment Variables & Paths =====
MODEL_PATH = os.getenv("MODEL_PATH", "app/model/blur_detection_model_v2.h5")
ISO_MODEL_PATH = os.path.join("app/model", "iso_classifier_model.h5")
ISO_SCALER_PATH = os.path.join("app/model", "iso_feature_scaler.pkl")
ISO_MAP_PATH = os.path.join("app/model", "iso_class_to_label.pkl")
SS_MODEL_PATH = os.path.join("app/model", "stable_shutter_model.h5")
SS_SCALER_PATH = os.path.join("app/model", "stable_shutter_scaler.pkl")
IMAGE_SIZE = (224, 224)
SETTINGS_FILE = "settings.json"

# ===== Load All Models =====
models = {
    'blur': None,
    'iso_model': None,
    'iso_scaler': None,
    'class_to_iso': None,
    'ss_model': None,
    'ss_scaler': None
}

# try:
#     # Load blur detection model
#     if os.path.exists(MODEL_PATH):
#         models['blur'] = load_model(
#             MODEL_PATH,
#             custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
#         )
#         logger.info(f"Blur model loaded from {MODEL_PATH}")
    
#     # Load ISO recommendation models
#     models['iso_model'] = tf.keras.models.load_model(ISO_MODEL_PATH)
#     models['iso_scaler'] = joblib.load(ISO_SCALER_PATH)
#     class_to_iso = joblib.load(ISO_MAP_PATH)
#     models['class_to_iso'] = [class_to_iso[i] for i in range(len(class_to_iso))]
    
#     # Load shutter speed model
#     models['ss_model'] = load_model(SS_MODEL_PATH, compile=False)
#     models['ss_scaler'] = joblib.load(SS_SCALER_PATH)

def safe_load(model_key, loader, path):
    try:
        if os.path.exists(path):
            models[model_key] = loader(path)
            logger.info(f"Loaded {model_key} from {path}")
        else:
            logger.warning(f"Missing model file: {path}")
    except Exception as e:
        logger.error(f"Failed to load {model_key}: {str(e)}")

# Load models
safe_load('blur', lambda p: load_model(p, custom_objects={'mse': tf.keras.losses.MeanSquaredError()}), MODEL_PATH)
safe_load('iso_model', tf.keras.models.load_model, ISO_MODEL_PATH)
safe_load('iso_scaler', joblib.load, ISO_SCALER_PATH)
safe_load('class_to_iso', joblib.load, ISO_MAP_PATH)
safe_load('ss_model', lambda p: load_model(p, compile=False), SS_MODEL_PATH)
safe_load('ss_scaler', joblib.load, SS_SCALER_PATH)

if models['class_to_iso']:
    models['class_to_iso'] = [models['class_to_iso'][i] for i in range(len(models['class_to_iso']))]
    
except Exception as e:
    logger.error(f"Model loading failed: {e}")
    raise SystemExit(1)

# ===== Blur Detection Functions =====
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

def predict_blur_with_model(image):
    """Predict blur score using the loaded CNN model"""
    if not models['blur'] or image is None:
        return None
    try:
        resized = cv2.resize(image, IMAGE_SIZE)
        processed = preprocess_input(img_to_array(resized))
        prediction = models['blur'].predict(np.expand_dims(processed, 0))
        return float(prediction[0][0]) * 5  # Scale to 0-100 range
    except Exception as e:
        logger.error(f"CNN blur prediction failed: {e}")
        return None
        
# ===== Heatmap Generation =====
def generate_heatmap_overlay(image):
    """Generates heatmap visualization for camera shake detection"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        shifted = np.roll(gray, -30, axis=0)  # Detect vertical shifts
        diff = cv2.absdiff(gray, shifted)
        inv = cv2.bitwise_not(diff)
        blur = cv2.GaussianBlur(inv, (11, 11), 0)
        norm = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(norm.astype('uint8'), cv2.COLORMAP_JET)
        return cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)
    except Exception as e:
        logger.error(f"Heatmap generation failed: {e}")
        return None

# ===== Recommendation System =====
def predict_iso(image):
    """Optimized ISO prediction"""
    try:
        # Downsample image for faster processing
        small_img = cv2.resize(image, (224, 224))
        gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
        
        # Simplified histogram calculation
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])  # Reduced bins
        
        features = [
            np.mean(gray),
            np.mean(hist),
            np.var(hist),
            detect_blur_laplacian(small_img),  # Use smaller image
            perceptual_blur_metric(small_img)
        ]
        
        scaled = models['iso_scaler'].transform([features])
        proba = models['iso_model'].predict(scaled, verbose=0)[0]  # Disable logging
        return int(models['class_to_iso'][np.argmax(proba)])
    except Exception as e:
        logger.error(f"ISO prediction failed: {str(e)}", exc_info=True)
        return None

# ===== All Endpoints =====
@app.route('/generate_heatmap', methods=['POST'])
def generate_heatmap():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        with tempfile.NamedTemporaryFile() as tmp:
            request.files['image'].save(tmp.name)
            img = cv2.imread(tmp.name)
            if img is None:
                return jsonify({'error': 'Invalid image'}), 400
                
            heatmap_img = generate_heatmap_overlay(img)
            _, buffer = cv2.imencode('.jpg', heatmap_img)
            return send_file(io.BytesIO(buffer), mimetype='image/jpeg')
    except Exception as e:
        logger.error(f"Heatmap error: {e}")
        return jsonify({'error': 'Heatmap generation failed'}), 500

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

@app.route('/')
def health_check():
    return jsonify({
        "status": "running",
        "endpoints": ["/analyze", "/unblur", "/settings", "/recommend"]
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

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
