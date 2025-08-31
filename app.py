from flask import Flask, request, jsonify, render_template, send_from_directory, Response
from flask_cors import CORS
from datetime import datetime
import shutil  # for moving generated videos
import os
import tempfile
import logging
import io
import time
import traceback
import json
# ML / CV deps
import numpy as np
import cv2 as cv
import torch
#import albumentations as A
#from albumentations.pytorch import ToTensorV2
import traceback
import joblib
import mediapipe as mp
import math


mp_drawing = mp.solutions.drawing_utils


# Avoid importing revtrans at module import time because it asserts OPENAI_API_KEY.
# We'll import within request handlers when needed.

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

# Output directory for generated artifacts (e.g., composed videos)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

#model loading
#mp_hands = mp.solutions.hands
#hands = mp_hands.Hands(max_num_hands=2,
#                       min_detection_confidence=0.7,
#                       min_tracking_confidence=0.7)


mp_hands = mp.solutions.hands

# Model Loading
MODEL_FILE = "./pretrained/gesture_model.pkl"
try:
    model = joblib.load(MODEL_FILE)
    print("✅ PKL Model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load PKL model: {e}")
    model = None



# Add these utility functions after the model loading:

def normalize_landmarks(landmarks):
    """
    Normalize landmarks like during training:
    - Translate so wrist is at origin.
    - Scale by wrist → middle finger MCP distance.
    """
    wrist = landmarks[0]
    translated = [(lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z) for lm in landmarks]

    scale = math.sqrt(translated[9][0]**2 + translated[9][1]**2 + translated[9][2]**2)
    if scale < 1e-6:
        scale = 1.0

    normalized = [(x/scale, y/scale, z/scale) for (x, y, z) in translated]
    return [coord for triple in normalized for coord in triple]

def empty_hand():
    return [0.0] * (21 * 3)


# Replace the entire run_inference_on_frame function:

def run_inference_on_frame(frame_bgr: np.ndarray):
    """Run model inference on a single BGR frame and return best detection with hand landmarks drawn.

    Returns dict: {detected_sign, confidence, detections: [...], annotated_frame} or None if no detection.
    """
    global model
    if model is None:
        raise RuntimeError("Model not loaded")

    # Create a new MediaPipe hands instance for each request to avoid timestamp conflicts
    hands = mp_hands.Hands(max_num_hands=2,
                          min_detection_confidence=0.7,
                          min_tracking_confidence=0.7)

    try:
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Create a copy of the frame for annotation
        annotated_frame = frame_bgr.copy()

        left_hand, right_hand = None, None

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                label = handedness.classification[0].label  # "Left" or "Right"
                score = handedness.classification[0].score

                if score < 0.7:
                    continue  # skip low-confidence

                # Draw hand landmarks on the annotated frame - THIS WAS MISSING!
                mp_drawing.draw_landmarks(
                    annotated_frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )

                norm = normalize_landmarks(hand_landmarks.landmark)

                if label == "Left":
                    left_hand = norm
                else:
                    right_hand = norm

        if left_hand is None:
            left_hand = empty_hand()
        if right_hand is None:
            right_hand = empty_hand()

        features = np.array([left_hand + right_hand])

        # Only predict if at least one hand is visible
        if any(v != 0.0 for v in features[0]):
            pred = model.predict(features)[0]

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features).max()
                confidence = float(proba)
                text = f"Gesture: {pred} ({confidence:.2f})"
            else:
                confidence = 1.0
                text = f"Gesture: {pred}"

            # Add text overlay on the annotated frame
            cv.putText(annotated_frame, text, (10, 40),
                      cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            return {
                "detected_sign": pred,
                "confidence": confidence,
                "detections": [{"class": pred, "confidence": confidence}],
                "annotated_frame": annotated_frame
            }
        else:
            return {
                "detected_sign": None,
                "confidence": 0.0,
                "detections": [],
                "annotated_frame": annotated_frame
            }
    finally:
        # Clean up the MediaPipe instance
        hands.close()

        
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


# Replace the infer_frame function:

@app.route('/infer-frame', methods=['POST'])
def infer_frame():
    """Accept a single image frame (jpeg/png) and return detection JSON with optional annotated frame."""
    t0 = time.time()
    try:
        if model is None:
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

        # Check if client wants annotated frame
        return_annotated = request.form.get('return_annotated', 'false').lower() == 'true'
        
        response_data = {
            'detected_sign': result['detected_sign'],
            'confidence': result['confidence'],
            'detections': result['detections'],
            'timing': {
                'decode': t1 - t0,
                'inference': t2 - t1,
                'total': t2 - t0
            }
        }

        if return_annotated and 'annotated_frame' in result:
            # Encode annotated frame as base64 JPEG
            import base64
            _, buffer = cv.imencode('.jpg', result['annotated_frame'])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            response_data['annotated_frame'] = f"data:image/jpeg;base64,{frame_base64}"

        return jsonify(response_data)
    except Exception as e:
        logger.error("/infer-frame error: %s\n%s", str(e), traceback.format_exc())
        return jsonify({'error': str(e)}), 500
# Replace the model_status function:

@app.route('/model-status', methods=['GET'])
def model_status():
    # Get model classes if available
    classes = []
    if model is not None and hasattr(model, 'classes_'):
        classes = model.classes_.tolist()
    elif model is not None:
        classes = ["gesture_class"]  # fallback if classes not available
    
    return jsonify({
        'ml_libraries_available': True,
        'model_loaded': model is not None,
        'model_type': '.pkl',
        'classes': classes,
        'actions_count': len(classes),
        'demo_mode': model is None
    })

# Replace the health_check function:

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})

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

        # Lazy import to avoid startup failures when no API key is present
        try:
            from revtrans import gloss_to_english_llm as _gloss_to_english_llm
        except Exception as ie:
            logger.error("revtrans import failed: %s", ie)
            return jsonify({'error': 'LLM module unavailable. Set OPENAI_API_KEY.'}), 503

        # Call the LLM to convert tokens to an English sentence
        gloss = _gloss_to_english_llm(confirmed_words)

        # Return the refined output
        return jsonify({'gloss': gloss}), 200
    except KeyError as e:
        logger.error("KeyError: Missing key in request data: %s", str(e))
        return jsonify({'error': f'Missing key: {str(e)}'}), 400
    except Exception as e:
        logger.error("Error processing confirmed words: %s\n%s", str(e), traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/outputs/<path:filename>', methods=['GET'])
def serve_output_file(filename: str):
    """Serve generated files from outputs/ with basic HTTP Range support for videos.

    Many browsers request video files with Range headers. If we detect a Range request,
    return a 206 Partial Content response with appropriate headers; otherwise fall back
    to a normal send_from_directory.
    """
    full_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.isfile(full_path):
        return jsonify({'error': 'File not found'}), 404

    range_header = request.headers.get('Range', None)
    if not range_header:
        resp = send_from_directory(OUTPUT_DIR, filename, as_attachment=False)
        # Advertise support for ranges to help the <video> element
        try:
            resp.headers.add('Accept-Ranges', 'bytes')
        except Exception:
            pass
        return resp

    # Parse Range header: e.g. "bytes=START-END"
    try:
        # Expected format 'bytes=start-end'
        units, rng = range_header.split('=', 1)
        if units.strip().lower() != 'bytes':
            raise ValueError('Only bytes unit is supported')
        start_str, end_str = (rng.split('-', 1) + [''])[:2]
        file_size = os.path.getsize(full_path)
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else file_size - 1
        start = max(0, start)
        end = min(end, file_size - 1)
        length = (end - start) + 1

        with open(full_path, 'rb') as f:
            f.seek(start)
            data = f.read(length)

        resp = Response(data, 206, mimetype='video/mp4', direct_passthrough=True)
        resp.headers.add('Content-Range', f'bytes {start}-{end}/{file_size}')
        resp.headers.add('Accept-Ranges', 'bytes')
        resp.headers.add('Content-Length', str(length))
        return resp
    except Exception as e:
        logger.warning('Range request failed, falling back to full file: %s', e)
        return send_from_directory(OUTPUT_DIR, filename, as_attachment=False)


def _list_available_video_tokens() -> list[str]:
    """Return available token basenames from the videos directory (without extension)."""
    base_vid_dir = os.path.join(os.path.dirname(__file__), 'videos')
    if not os.path.isdir(base_vid_dir):
        return []
    toks = []
    for name in os.listdir(base_vid_dir):
        if name.lower().endswith('.mp4'):
            toks.append(os.path.splitext(name)[0].lower())
    return sorted(set(toks))


def _text_to_gloss_tokens(text: str) -> list[str]:
    """Convert a free-form sentence into a list of gloss-like tokens, aligned to available clips.

    Strategy:
    - Lowercase and strip punctuation
    - Simple stemming for common endings (ing/ed/s)
    - Drop stopwords
    - Keep only tokens that have a matching videos/<token>.mp4 clip
    """
    if not text:
        return []
    import re

    # Normalize
    s = text.strip().lower()
    s = re.sub(r"[^a-z0-9'\s]", " ", s)
    words = [w for w in re.split(r"\s+", s) if w]

    # Simple contractions map and lemmatization hints
    contr = {
        "i'm": "i", "im": "i", "we're": "we", "you're": "you", "they're": "they",
        "can't": "cannot", "won't": "will", "don't": "do", "doesn't": "do", "didn't": "do",
    }
    words = [contr.get(w, w) for w in words]

    # Basic stemming try
    def stem(w: str) -> str:
        for suf in ("ing", "ed", "es", "s"):
            if len(w) > 3 and w.endswith(suf):
                return w[: -len(suf)]
        return w

    # Stopwords (keep short list)
    stop = {"the", "a", "an", "is", "am", "are", "to", "at", "in", "on", "for", "of", "and", "or", "with", "be", "was", "were", "will", "would", "should", "could", "have", "has", "had", "do", "did", "does", "that", "this", "these", "those", "it"}

    candidates = [stem(w) for w in words if w not in stop]

    avail = set(_list_available_video_tokens())
    filtered = [w for w in candidates if w in avail]
    # If nothing matched, return the raw candidates to allow caller to handle missing clips
    return filtered or candidates


def compose_video_from_gloss(gloss_tokens):
    """Concatenate per-token mp4 clips from videos/<token>.mp4 into a single mp4 in outputs.

    Returns (filename, meta) where filename is the saved file name under OUTPUT_DIR.
    """
    # Map tokens to available video files
    base_vid_dir = os.path.join(os.path.dirname(__file__), 'videos')
    if not os.path.exists(base_vid_dir):
        raise FileNotFoundError(f"Videos directory not found: {base_vid_dir}")
    
    files = []
    missing = []
    for t in gloss_tokens:
        name = str(t).strip().lower()
        if not name:
            continue
        cand = os.path.join(base_vid_dir, f"{name}.mp4")
        if os.path.exists(cand):
            files.append(cand)
            logger.info(f"Found video for token '{name}': {cand}")
        else:
            missing.append(name)
            logger.warning(f"Missing video for token '{name}': {cand}")

    if not files:
        available_tokens = _list_available_video_tokens()
        raise FileNotFoundError(f"No matching video clips found for tokens: {gloss_tokens}. Available tokens: {available_tokens[:10]}...")

    # Open first clip to get properties
    first = cv.VideoCapture(files[0])
    if not first.isOpened():
        raise RuntimeError(f"Could not open first video clip: {files[0]}")
    fps = first.get(cv.CAP_PROP_FPS) or 25.0
    width = int(first.get(cv.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(first.get(cv.CAP_PROP_FRAME_HEIGHT) or 480)
    first.release()
    
    logger.info(f"Video properties: {width}x{height}, {fps} FPS")

    # Prepare writer
    out_name = f"reverse_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}.mp4"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    
    # Use H.264 codec which is universally supported by browsers
    fourcc = cv.VideoWriter_fourcc(*'avc1')  # H.264 codec
    writer = cv.VideoWriter(out_path, fourcc, fps, (width, height))
    
    if not writer.isOpened():
        # Try alternative H.264 codec identifier
        fourcc = cv.VideoWriter_fourcc(*'H264')
        writer = cv.VideoWriter(out_path, fourcc, fps, (width, height))
        
    if not writer.isOpened():
        # Final fallback to MP4V
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        writer = cv.VideoWriter(out_path, fourcc, fps, (width, height))
        
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video writer at {out_path}")
    
    logger.info(f"Created video writer: {out_path} with codec: {fourcc}")

    total_frames = 0
    processed_files = 0
    try:
        for fp in files:
            logger.info(f"Processing video file: {fp}")
            cap = cv.VideoCapture(fp)
            if not cap.isOpened():
                logger.warning("Skip unreadable clip: %s", fp)
                continue
            
            file_frames = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                # Ensure frame is the right size
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv.resize(frame, (width, height))
                writer.write(frame)
                total_frames += 1
                file_frames += 1
            cap.release()
            processed_files += 1
            logger.info(f"Processed {file_frames} frames from {os.path.basename(fp)}")
            
        # If we have no frames, add a placeholder black frame to prevent empty video
        if total_frames == 0:
            logger.warning("No frames processed, adding placeholder frame")
            black_frame = np.zeros((height, width, 3), dtype=np.uint8)
            for _ in range(int(fps)):  # 1 second of black frames
                writer.write(black_frame)
                total_frames += 1
    finally:
        writer.release()
        logger.info(f"Video composition complete: {total_frames} total frames from {processed_files} files")

    meta = { 'fps': fps, 'width': width, 'height': height, 'frames': total_frames, 'missing': missing, 'codec': 'avc1' }
    if total_frames == 0:
        # Clean up empty file and surface a clear error
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
        except Exception:
            pass
        raise RuntimeError("Output video had 0 frames. Check source clips and codecs.")
    return out_name, meta


@app.route('/reverse-translate-video', methods=['POST'])
def reverse_translate_video():
    try:
        data = request.get_json(silent=True) or {}
        logger.info(f"🎬 reverse_translate_video called with data: {data}")
        gloss_tokens = data.get('glossTokens')
        text = data.get('text')
        logger.info(f"📝 gloss_tokens: {gloss_tokens}, text: {text}")

        # Prefer sentence input via revtrans if provided
        if isinstance(text, str) and text.strip():
            try:
                from revtrans import sentence_to_gloss_tokens as _sentence_to_gloss_tokens
            except Exception as ie:
                logger.warning('revtrans.sentence_to_gloss_tokens import failed, falling back: %s', ie)
                _sentence_to_gloss_tokens = None

            if _sentence_to_gloss_tokens is not None:
                available = _list_available_video_tokens()
                gloss_tokens = _sentence_to_gloss_tokens(text.strip(), available_tokens=available)
            else:
                gloss_tokens = _text_to_gloss_tokens(text)

        if not isinstance(gloss_tokens, list) or not gloss_tokens:
            available = _list_available_video_tokens()[:30]
            logger.warning(f"❌ Invalid payload. Available tokens: {available}")
            return jsonify({
                'error': 'Invalid payload. Provide a sentence via "text" or a token list "glossTokens".',
                'hint': 'Example text: "We go college" or glossTokens: ["we", "go", "college"]',
                'available_tokens_preview': available
            }), 400

        # ✅ Compose video from gloss tokens
        logger.info(f"🎥 Composing video from tokens: {gloss_tokens}")
        fname, meta = compose_video_from_gloss(gloss_tokens)
        logger.info(f"✅ Video composed: {fname}, meta: {meta}")

        # ✅ Use existing /outputs/<filename> route (no moving needed)
        url = f"/outputs/{fname}"
        logger.info(f"🌐 Video URL: {url}")

        # Warn clearly if codec is not H.264 which some browsers require
        if meta.get('codec') and meta['codec'].lower() not in ('avc1', 'h264', 'x264', 'mp4v'):
            meta['playback_warning'] = 'Browser may not play this codec. Prefer H.264 (avc1).'

        return jsonify({
            'video_url': url,
            'file': os.path.basename(fname),
            'meta': meta,
            'tokens': gloss_tokens
        }), 200

    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        tb = traceback.format_exc()
        logger.error('/reverse-translate-video error: %s\n%s', str(e), tb)
        return jsonify({'error': str(e), 'trace': tb}), 500
    
# Replace the main block:

if __name__ == '__main__':
    print("🚀 Starting Sign Language Translator...")
    print("📁 Loading PKL model...")

    if model is not None:
        print("✅ PKL Model ready!")
    else:
        print("⚠️ PKL Model failed to load. Endpoints will return 503 for inference.")

    print("🌐 Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)

