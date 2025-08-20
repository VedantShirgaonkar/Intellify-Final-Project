from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import tempfile
from datetime import datetime
import logging
import tensorflow as tf
import numpy as np
import cv2 as cv
import mediapipe as mp

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
    
    if model is None:
        logger.error("Model not loaded")
        return {"detected_sign": "Error", "confidence": 0.0, "error": "Model not loaded"}
    
    try:
        # Variables for prediction
        sequence = []
        predictions = []
        confidences = []
        
        # Open video
        cap = cv.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error("Could not open video file")
            return {"detected_sign": "Error", "confidence": 0.0, "error": "Could not open video"}
        
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Resize for consistency
                frame = cv.resize(frame, (640, 480))
                
                # Detection
                image, results = mediapipe_detection(frame, holistic)
                
                # Extract keypoints
                keypoints = extract_keypoints(results)
                
                # Append to sequence
                sequence.append(keypoints)
                sequence = sequence[-SEQUENCE_LENGTH:]
                
                if len(sequence) == SEQUENCE_LENGTH:
                    input_seq = np.expand_dims(sequence, axis=0)  # shape (1, 30, 1530)
                    
                    # Predict
                    res = model.predict(input_seq, verbose=0)[0]
                    predicted_action = actions[np.argmax(res)]
                    confidence = np.max(res)
                    
                    # Store prediction if above threshold
                    if confidence > CONFIDENCE_THRESHOLD:
                        predictions.append(predicted_action)
                        confidences.append(confidence)
        
        cap.release()
        
        # Return the most common prediction or highest confidence prediction
        if predictions:
            # Get the prediction with highest confidence
            best_idx = np.argmax(confidences)
            best_prediction = predictions[best_idx]
            best_confidence = confidences[best_idx]
            
            logger.info(f"Detected sign: {best_prediction} with confidence: {best_confidence:.2f}")
            
            return {
                "detected_sign": best_prediction,
                "confidence": float(best_confidence),
                "total_frames": frame_count,
                "valid_predictions": len(predictions)
            }
        else:
            logger.warning("No confident predictions found")
            return {
                "detected_sign": "No sign detected",
                "confidence": 0.0,
                "total_frames": frame_count,
                "valid_predictions": 0
            }
            
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return {"detected_sign": "Error", "confidence": 0.0, "error": str(e)}


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
        
        # Log received data
        logger.info(f"Received video: {video_file.filename}")
        logger.info(f"Timestamp: {timestamp}")
        logger.info(f"Duration: {duration} seconds")
        logger.info(f"File size: {file_size / 1024:.2f} KB")
        
        # Save video file temporarily
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm', dir=UPLOAD_FOLDER) as temp_file:
            video_file.save(temp_file.name)
            temp_video_path = temp_file.name
        """

        
        
        unique_filename = f"D:\\Intellify_hackathon\\uploads\\z.webm"
        video_file.save(unique_filename)
        temp_video_path = unique_filename

        try:
            # Process the video for sign language detection
            result = process_sign_language_video(temp_video_path)
            
            # Check if there was an error during processing
            if 'error' in result:
                return jsonify({
                    'error': 'Processing failed',
                    'message': result['error'],
                    'status': 'error'
                }), 500
            
            # Prepare response
            response_data = {
                'status': 'success',
                'message': 'Video processed successfully',
                'detected_sign': result['detected_sign'],
                'confidence': result['confidence'],
                'timestamp': timestamp,
                'duration': duration,
                'processing_time': datetime.now().isoformat(),
                'total_frames': result.get('total_frames', 0),
                'valid_predictions': result.get('valid_predictions', 0)
            }
            
            logger.info(f"Processing result: {result['detected_sign']} (confidence: {result['confidence']:.2f})")
            
            return jsonify(response_data), 200
            
        finally:
            # Clean up temporary file
            try:
                #os.unlink(temp_video_path)
                a=1
            except OSError:
                logger.warning(f"Could not delete temporary file: {temp_video_path}")
    
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
    # Load model before starting the app
    success = load_model()
    print("Model loaded successfully")
    if not success:
        logger.warning("Model failed to load - using mock responses")

    app.run(debug=True, host='0.0.0.0', port=5000)