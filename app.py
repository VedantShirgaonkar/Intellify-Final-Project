from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import tempfile
from datetime import datetime
import logging
import numpy as np
import time

# Import ML dependencies
import tensorflow as tf
import cv2 as cv
import mediapipe as mp

print("✅ ML dependencies loaded successfully")

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'webm', 'mp4', 'avi', 'mov'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Model configuration
MODEL_PATH = 'models/main_wsl_model.h5'  # Update path as needed
LABEL_MAP_PATH = 'models/label_map.npy'  # Update path as needed
SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.7

# Global variables for model
model = None
label_map = None
actions = None
mp_holistic = mp.solutions.holistic

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def load_model():
    """Load the trained model and label map"""
    global model, label_map, actions
        
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(LABEL_MAP_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            label_map = np.load(LABEL_MAP_PATH, allow_pickle=True).item()
            actions = list(label_map.keys())
            logger.info(f"Model loaded successfully with {len(actions)} actions")
            return True
        else:
            logger.warning(f"Model files not found at {MODEL_PATH} or {LABEL_MAP_PATH}")
            return False
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def mediapipe_detection(image, model):
    """Perform mediapipe detection"""
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    """Extract keypoints from mediapipe results"""
    #pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([face, lh, rh])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_sign_language_video(video_path):
    """
    Process video for sign language detection using the trained model
    """
    global model, actions
    
    # Start overall timing
    start_time = time.time()
    timing_data = {}
    
    if model is None:
        logger.error("Model not loaded")
        return {
            "detected_sign": "Error",
            "confidence": 0.0,
            "error": "Model not loaded"
        }
    
    try:
        # Variables for prediction
        sequence = []
        predictions = []
        confidences = []
        frame_count = 0
        cleanup_paths = []
        
        # Time video opening
        video_open_start = time.time()
        logger.info(f"🎬 Starting video opening process...")
        
        # Open video with optimized WebM support
        cap = None
        backends_to_try = [
            cv.CAP_FFMPEG,    # FFmpeg backend (best for WebM)
            cv.CAP_GSTREAMER, # GStreamer backend
            cv.CAP_MSMF,      # Microsoft Media Foundation
            cv.CAP_ANY        # Default backend
        ]
        
        backend_attempt_count = 0
        for backend in backends_to_try:
            try:
                backend_attempt_count += 1
                backend_start = time.time()
                cap = cv.VideoCapture(video_path, backend)
                if cap.isOpened():
                    backend_time = time.time() - backend_start
                    video_open_time = time.time() - video_open_start
                    timing_data['video_opening'] = video_open_time
                    timing_data['backend_attempts'] = backend_attempt_count
                    timing_data['successful_backend'] = backend
                    logger.info(f"✅ Video opened with backend {backend} in {backend_time:.3f}s (total: {video_open_time:.3f}s)")
                    break
                else:
                    cap.release()
                    cap = None
                    backend_time = time.time() - backend_start
                    logger.debug(f"❌ Backend {backend} failed in {backend_time:.3f}s")
            except Exception as e:
                backend_time = time.time() - backend_start if 'backend_start' in locals() else 0
                logger.debug(f"💥 Backend {backend} exception in {backend_time:.3f}s: {e}")
                continue
        
        if cap is None:
            # Last resort: try imageio for direct WebM decoding
            imageio_start = time.time()
            logger.warning("🔄 All OpenCV backends failed. Trying imageio for direct WebM processing...")
            try:
                import imageio
                reader = imageio.get_reader(video_path)
                
                # Process frames directly from imageio reader for real-time performance
                mediapipe_start = time.time()
                timing_data['video_opening'] = time.time() - video_open_start
                timing_data['backend_attempts'] = len(backends_to_try)
                timing_data['successful_backend'] = 'imageio'
                logger.info(f"📹 ImageIO reader created in {time.time() - imageio_start:.3f}s")
                
                with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                    for frame in reader:
                        # Convert from RGB to BGR for OpenCV/MediaPipe
                        frame_bgr = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
                        frame_count += 1
                        
                        # Resize for consistency
                        frame_bgr = cv.resize(frame_bgr, (640, 480))
                        
                        # Detection
                        image, results = mediapipe_detection(frame_bgr, holistic)
                        
                        # Extract keypoints
                        keypoints = extract_keypoints(results)
                        sequence.append(keypoints)
                        
                        # Predict every SEQUENCE_LENGTH frames
                        if len(sequence) == SEQUENCE_LENGTH:
                            prediction_start = time.time()
                            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                            prediction_time = time.time() - prediction_start
                            
                            predictions.append(np.argmax(res))
                            confidences.append(np.max(res))
                            sequence = []
                            
                            logger.debug(f"🧠 Model prediction took {prediction_time:.3f}s")
                    
                reader.close()
                
                # Calculate timing
                mediapipe_time = time.time() - mediapipe_start
                timing_data['mediapipe_processing'] = mediapipe_time
                logger.info(f"🎯 MediaPipe processing completed in {mediapipe_time:.3f}s")
                
                # Final cleanup and prediction
                cleanup_paths = []  # No temp files with imageio
                
            except Exception as imageio_err:
                logger.error(f"ImageIO fallback failed: {imageio_err}")
                return {
                    "detected_sign": "Error",
                    "confidence": 0.0,
                    "error": "Could not decode WebM video with any method. Try recording in a different format.",
                    "details": str(imageio_err)
                }
        
        # If we successfully opened with OpenCV, process with the optimized pipeline
        if cap is not None:
            mediapipe_start = time.time()
            logger.info(f"🎯 Starting MediaPipe processing...")
            
            with mp_holistic.Holistic(
                min_detection_confidence=0.3,    # Lower = faster (was 0.5)
                min_tracking_confidence=0.3,     # Lower = faster (was 0.5)
                model_complexity=0,              # 0 = fastest, 2 = most accurate
                static_image_mode=False,         # Optimize for video
                smooth_landmarks=False           # Disable smoothing for speed
            ) as holistic:
                frame_skip = 5  # Process every 5th frame for better speed (was 3)
                processed_frames = 0
                max_frames_to_process = 15  # Limit total frames processed for speed
                max_processing_time = 3.0   # Maximum 3 seconds for MediaPipe processing
                
                while cap.isOpened() and processed_frames < max_frames_to_process:
                    # Check if we've exceeded max processing time
                    if time.time() - mediapipe_start > max_processing_time:
                        logger.info(f"⏰ Timeout: Stopping after {max_processing_time}s to maintain responsiveness")
                        break
                        
                    ret, frame = cap.read()
                    if not ret:
                        break
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    
                    # Skip frames for performance optimization
                    if frame_count % frame_skip != 0:
                        continue
                    
                    processed_frames += 1
                    
                    # Resize for consistency
                    frame = cv.resize(frame, (640, 480))
                    
                    # Detection
                    image, results = mediapipe_detection(frame, holistic)
                    
                    # Extract keypoints
                    keypoints = extract_keypoints(results)
                    sequence.append(keypoints)
                    
                    # Predict every SEQUENCE_LENGTH frames for real-time performance
                    if len(sequence) == SEQUENCE_LENGTH:
                        prediction_start = time.time()
                        res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                        prediction_time = time.time() - prediction_start
                        
                        predictions.append(np.argmax(res))
                        confidences.append(np.max(res))
                        sequence = []
                        
                        logger.debug(f"🧠 Model prediction took {prediction_time:.3f}s")
                        
                        # Early exit if confident - more aggressive
                        if len(predictions) >= 1 and max(confidences) > 0.75:  # Lower threshold, faster exit
                            logger.info(f"⚡ Early exit: High confidence ({max(confidences):.2f}) reached after {processed_frames} frames")
                            break
            
            cap.release()
            
            # Calculate timing
            mediapipe_time = time.time() - mediapipe_start
            timing_data['mediapipe_processing'] = mediapipe_time
            timing_data['frames_processed'] = processed_frames
            timing_data['total_frames'] = frame_count
            timing_data['frame_skip'] = frame_skip
            logger.info(f"🎯 MediaPipe processing completed in {mediapipe_time:.3f}s")
            logger.info(f"📊 Processed {processed_frames}/{frame_count} frames (every {frame_skip}th frame)")
            logger.info(f"⚡ Processing rate: {processed_frames/mediapipe_time:.1f} frames/sec")
        
        # Cleanup any converted files (none with direct WebM processing)
        for p in cleanup_paths:
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except Exception as e:
                logger.warning(f"Failed to remove temp file {p}: {e}")
        
        # Final timing calculation
        total_time = time.time() - start_time
        timing_data['total_processing'] = total_time
        
        # Log detailed timing breakdown
        logger.info(f"⏱️ TIMING BREAKDOWN:")
        logger.info(f"   📹 Video Opening: {timing_data.get('video_opening', 0):.3f}s")
        logger.info(f"   🎯 MediaPipe Processing: {timing_data.get('mediapipe_processing', 0):.3f}s") 
        logger.info(f"   🔄 Backend Attempts: {timing_data.get('backend_attempts', 0)}")
        logger.info(f"   🛠️  Successful Backend: {timing_data.get('successful_backend', 'unknown')}")
        logger.info(f"   ⚡ Total Processing: {total_time:.3f}s")
        
        # Return the most common prediction or highest confidence prediction
        if predictions and confidences:
            # Get the prediction with highest confidence
            best_idx = np.argmax(confidences)
            best_prediction_idx = predictions[best_idx]
            best_confidence = confidences[best_idx]
            
            # Convert prediction index to action name
            if best_prediction_idx < len(actions):
                best_prediction = actions[best_prediction_idx]
            else:
                best_prediction = "Unknown"
            
            logger.info(f"✅ Detected sign: {best_prediction} with confidence: {best_confidence:.2f}")
            
            return {
                "detected_sign": best_prediction,
                "confidence": float(best_confidence),
                "total_frames": frame_count,
                "valid_predictions": len(predictions),
                "timing": timing_data  # Include timing data in response
            }
        else:
            logger.warning("⚠️ No confident predictions found")
            return {
                "detected_sign": "No sign detected",
                "confidence": 0.0,
                "total_frames": frame_count,
                "valid_predictions": 0,
                "timing": timing_data  # Include timing data in response
            }
            
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return {
            "detected_sign": "Error",
            "confidence": 0.0,
            "error": "Processing failed on server",
            "details": str(e)
        }


@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/about')
def about():
    """Serve the about page"""
    return render_template('about.html')

@app.route('/learn')
def learn():
    """Serve the learn page"""
    return render_template('learn.html')

@app.route('/process', methods=['POST'])
def process_video():
    # 🚀 SERVER PROCESSING STARTS HERE - This is when video reaches the server
    server_start_time = time.time()
    endpoint_timing = {}
    
    logger.info(f"🔥 SERVER PROCESSING STARTED at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
    
    try:
        # Check if video file is present
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        timestamp = request.form.get('timestamp')
        duration = request.form.get('duration')
        
        # Validate file
        if video_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(video_file.filename):
            return jsonify({'error': 'Invalid file type. Allowed types: webm, mp4, avi, mov'}), 400
        
        # Check file size
        video_file.seek(0, os.SEEK_END)
        file_size = video_file.tell()
        video_file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            return jsonify({'error': 'File too large. Maximum size: 50MB'}), 400
        
        # Log received data with timing
        validation_time = time.time() - server_start_time
        endpoint_timing['validation'] = validation_time
        logger.info(f"📨 Received video: {video_file.filename}")
        logger.info(f"📅 Timestamp: {timestamp}")
        logger.info(f"⏱️ Duration: {duration} seconds")
        logger.info(f"📊 File size: {file_size / 1024:.2f} KB")
        logger.info(f"✅ Validation completed in {validation_time:.3f}s")
        
        # Save video file temporarily with unique filename
        save_start = time.time()
        import uuid
        file_extension = video_file.filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}.{file_extension}"
        temp_video_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        video_file.save(temp_video_path)
        
        save_time = time.time() - save_start
        endpoint_timing['file_save'] = save_time
        logger.info(f"💾 File saved in {save_time:.3f}s")

        try:
            # Process the video for sign language detection
            processing_start = time.time()
            result = process_sign_language_video(temp_video_path)
            processing_time = time.time() - processing_start
            endpoint_timing['video_processing'] = processing_time
            
            logger.info(f"🎬 Video processing completed in {processing_time:.3f}s")
            
            # Check if there was an error during processing
            if 'error' in result:
                status_code = 500
                # Differentiate common client-side issues
                if result['error'] in [
                    'No video file provided',
                    'No file selected',
                    'Invalid file type. Allowed types: webm, mp4, avi, mov',
                    'File too large. Maximum size: 50MB'
                ]:
                    status_code = 400
                elif 'Could not open video' in result['error']:
                    status_code = 415  # Unsupported Media Type / cannot read
                return jsonify({
                    'status': 'error',
                    'error': result['error'],
                    'details': result.get('details')
                }), status_code
            
            # Prepare response
            total_endpoint_time = time.time() - server_start_time
            endpoint_timing['total_endpoint'] = total_endpoint_time
            endpoint_timing['cleanup'] = total_endpoint_time - sum([t for k, t in endpoint_timing.items() if k != 'total_endpoint'])
            
            # Log endpoint timing breakdown
            logger.info(f"🚀 ENDPOINT TIMING BREAKDOWN:")
            logger.info(f"   ✅ Validation: {endpoint_timing.get('validation', 0):.3f}s")
            logger.info(f"   💾 File Save: {endpoint_timing.get('file_save', 0):.3f}s")
            logger.info(f"   🎬 Video Processing: {endpoint_timing.get('video_processing', 0):.3f}s")
            logger.info(f"   🔄 Cleanup: {endpoint_timing.get('cleanup', 0):.3f}s")
            logger.info(f"   ⚡ Total Endpoint: {total_endpoint_time:.3f}s")
            
            response_data = {
                'status': 'success',
                'message': 'Video processed successfully',
                'detected_sign': result['detected_sign'],
                'confidence': result['confidence'],
                'timestamp': timestamp,
                'duration': duration,
                'processing_time': datetime.now().isoformat(),
                'total_frames': result.get('total_frames', 0),
                'valid_predictions': result.get('valid_predictions', 0),
                'timing': {
                    'endpoint': endpoint_timing,
                    'processing': result.get('timing', {})
                }
            }
            
            logger.info(f"✅ Processing result: {result['detected_sign']} (confidence: {result['confidence']:.2f})")
            
            return jsonify(response_data), 200
            
        finally:
            # Clean up temporary file
            try:
                if os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)
                    logger.info(f"Cleaned up temporary file: {temp_video_path}")
            except OSError as e:
                logger.warning(f"Could not delete temporary file: {temp_video_path}, Error: {e}")
    
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': 'Failed to process video',
            'status': 'error'
        }), 500

@app.route('/model-status', methods=['GET'])
def model_status():
    """Check model loading status"""
    global model, actions
    is_loaded = model is not None
    return jsonify({
        'model_loaded': is_loaded,
        'ml_libraries_available': True,
        'demo_mode': not is_loaded,
        'actions_count': len(actions) if actions else 0,
        'actions': actions if actions else [],
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'Sign Language Video Processor'
    }), 200

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

if __name__ == '__main__':
    print("🚀 Starting Sign Language Translator...")
    print("📁 Checking model files...")
    
    # Load model before starting the app
    success = load_model()
    if success:
        print("✅ Model loaded successfully!")
        print(f"📊 Available actions: {len(actions) if actions else 0}")
        if actions:
            print(f"🎯 Actions: {', '.join(actions[:5])}{'...' if len(actions) > 5 else ''}")
    else:
        print("⚠️  Model failed to load - running in demo mode")
        print("💡 To enable full functionality, ensure model files are in the 'models/' directory")
    
    print("🌐 Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)