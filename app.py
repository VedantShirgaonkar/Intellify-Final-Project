from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import tempfile
from datetime import datetime
import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from datetime import datetime
import io
import os
import logging
import time
import traceback

# ML / CV deps
import numpy as np
import cv2 as cv
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from revtrans import gloss_to_english_llm

# Project model/utils
from model import DETR
try:
    from utils.setup import get_classes, get_colors
    from utils.boxes import rescale_bboxes
except Exception:  # Fallbacks if utils not available
    def get_classes():
        return ["class_0", "class_1", "class_2"]
    def get_colors():
        return [(0,255,0), (255,0,0), (0,0,255)]
    def rescale_bboxes(bboxes, size_hw):
        # assume normalized cxcywh -> xyxy in pixel space, passthrough if already pixel
        H, W = size_hw
        b = bboxes.detach().cpu().numpy()
        if b.shape[-1] == 4:
            # If already looks like xyxy in pixels, just return tensor-like np
            return torch.tensor(b)
        return torch.tensor(b)

print("✅ ML dependencies loaded successfully")

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
ALLOWED_IMAGE_MIME = {"image/jpeg", "image/png"}
CONFIDENCE_THRESHOLD = 0.8

# Model configuration (.pt)
PT_MODEL_PATH = 'pretrained/4426_model.pt'
NUM_CLASSES = 3

# Global variables for model
torch_model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASSES = get_classes()

# Preprocessing pipeline (aligned with realtime.py)
transforms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])


def load_model():
    """Load the PyTorch DETR model for real-time inference."""
    global torch_model
    try:
        m = DETR(num_classes=NUM_CLASSES)
        # Use the model's helper to load weights
        m.load_pretrained(PT_MODEL_PATH)
        m.to(device)
        m.eval()
        torch_model = m
        logger.info("✅ PyTorch model loaded and ready on %s", device)
        return True
    except Exception as e:
        logger.error("Error loading PyTorch model: %s\n%s", str(e), traceback.format_exc())
        return False


def run_inference_on_frame(frame_bgr: np.ndarray):
    """Run model inference on a single BGR frame and return best detection.

    Returns dict: {detected_sign, confidence, detections: [...]} or None if no detection.
    """
    global torch_model
    if torch_model is None:
        raise RuntimeError("Model not loaded")

    # Albumentations expects HWC (BGR ok, normalization treats as numbers)
    transformed = transforms(image=frame_bgr)
    input_tensor = transformed['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        result = torch_model(input_tensor)

    # result: pred_logits [B,Q,C+1], pred_boxes [B,Q,4]
    probabilities = result['pred_logits'].softmax(-1)[:, :, :-1]  # drop no-object
    max_probs, max_classes = probabilities.max(-1)
    keep_mask = max_probs > CONFIDENCE_THRESHOLD

    batch_indices, query_indices = torch.where(keep_mask)
    if batch_indices.numel() == 0:
        return {
            "detected_sign": None,
            "confidence": 0.0,
            "detections": []
        }

    # Rescale boxes back to original frame size
    H, W = frame_bgr.shape[:2]
    # rescale_bboxes expects (H,W) or (W,H) depending on implementation; realtime.py passes (1920,1080)
    # We'll pass (H,W) consistent with our code here
    bboxes = rescale_bboxes(result['pred_boxes'][batch_indices, query_indices, :], (H, W))
    classes = max_classes[batch_indices, query_indices]
    probas = max_probs[batch_indices, query_indices]

    detections = []
    for bclass, bprob, bbox in zip(classes, probas, bboxes):
        cls_idx = int(bclass.detach().cpu().item())
        prob_val = float(bprob.detach().cpu().item())
        # Allow both tensor and numpy for bbox
        if hasattr(bbox, 'detach'):
            bb = bbox.detach().cpu().numpy()
        else:
            bb = np.array(bbox)
        x1, y1, x2, y2 = [float(v) for v in bb]
        detections.append({
            'class': CLASSES[cls_idx] if 0 <= cls_idx < len(CLASSES) else f'class_{cls_idx}',
            'confidence': prob_val,
            'bbox': [x1, y1, x2, y2]
        })

    best = max(detections, key=lambda d: d['confidence']) if detections else None
    return {
        "detected_sign": best['class'] if best else None,
        "confidence": best['confidence'] if best else 0.0,
        "detections": detections
    }


@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/about')
def about():
    return render_template('about.html')


@app.route('/learn')
def learn():
    # Prefer the newer page if present
    try:
        return render_template('learn_new.html')
    except Exception:
        return render_template('learn.html')


@app.route('/infer-frame', methods=['POST'])
def infer_frame():
    """Accept a single image frame (jpeg/png) and return detection JSON without saving files."""
    t0 = time.time()
    try:
        if torch_model is None:
            return jsonify({
                'error': 'Model not loaded',
                'model_loaded': False
            }), 503

        if 'frame' not in request.files:
            return jsonify({'error': 'No frame provided (expect form-data field "frame")'}), 400

        file = request.files['frame']
        if file.mimetype not in ALLOWED_IMAGE_MIME:
            return jsonify({'error': f'Unsupported content type: {file.mimetype}'}), 415

        # Read image bytes into numpy array
        file_bytes = np.frombuffer(file.read(), np.uint8)
        frame = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
        if frame is None:
            return jsonify({'error': 'Could not decode image'}), 415

        t1 = time.time()
        result = run_inference_on_frame(frame)
        t2 = time.time()

        result['timing'] = {
            'decode': t1 - t0,
            'inference': t2 - t1,
            'total': t2 - t0
        }
        return jsonify(result)
    except Exception as e:
        logger.error("/infer-frame error: %s\n%s", str(e), traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/model-status', methods=['GET'])
def model_status():
    return jsonify({
        'ml_libraries_available': True,
        'model_loaded': torch_model is not None,
        'device': str(device),
        'classes': CLASSES,
        'actions_count': len(CLASSES),
        'demo_mode': torch_model is None
    })


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'model_loaded': torch_model is not None})


@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'Payload too large'}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404



@app.route('/process-confirmed-words', methods=['POST'])
def process_confirmed_words():
    try:
        # Get the confirmedWords array from the request
        data = request.get_json()  # Use get_json() for better error handling
        if not data:
            return jsonify({'error': 'Invalid JSON payload'}), 400

        confirmed_words = data.get('confirmedWords', [])
        
        if not confirmed_words or not isinstance(confirmed_words, list):
            return jsonify({'error': 'Invalid or missing "confirmedWords". Expected a non-empty list.'}), 400

        # Log the received confirmed words for debugging
        logger.info("Received confirmed words: %s", confirmed_words)

        # Call the gloss_to_english_llm function
        gloss = gloss_to_english_llm(confirmed_words)

        # Return the refined output
        return jsonify({'gloss': gloss}), 200
    except KeyError as e:
        logger.error("KeyError: Missing key in request data: %s", str(e))
        return jsonify({'error': f'Missing key: {str(e)}'}), 400
    except Exception as e:
        logger.error("Error processing confirmed words: %s\n%s", str(e), traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("🚀 Starting Sign Language Translator...")
    print("📁 Loading PyTorch model...")

    success = load_model()
    if success:
        print("✅ Model ready!")
    else:
        print("⚠️ Model failed to load. Endpoints will return 503 for inference.")

    print("🌐 Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)

