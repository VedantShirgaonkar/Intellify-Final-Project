from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import tempfile
from datetime import datetime
import logging

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)  # Enable CORS for all routes
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'webm', 'mp4', 'avi', 'mov'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_sign_language_video(video_path):
    """
    Placeholder function for sign language processing.
    Replace this with your actual ML model/processing logic.
    """
    # Simulate processing time
    import time
    time.sleep(1)
    
    # Mock results - replace with actual sign language detection
    mock_results = [
        {"detected_sign": "Hello", "confidence": 0.95},
        {"detected_sign": "Thank you", "confidence": 0.87},
        {"detected_sign": "Please", "confidence": 0.92},
        {"detected_sign": "Yes", "confidence": 0.89},
        {"detected_sign": "No", "confidence": 0.94}
    ]
    
    # Return a random result for demo purposes
    import random
    return random.choice(mock_results)

@app.route('/')
def index():
    """Serve the main HTML page"""
    # Option 1: Serve from templates folder (default)
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
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm', dir=UPLOAD_FOLDER) as temp_file:
            video_file.save(temp_file.name)
            temp_video_path = temp_file.name
        
        try:
            # Process the video for sign language detection
            result = process_sign_language_video(temp_video_path)
            
            # Prepare response
            response_data = {
                'status': 'success',
                'message': 'Video processed successfully',
                'detected_sign': result['detected_sign'],
                'confidence': result['confidence'],
                'timestamp': timestamp,
                'duration': duration,
                'processing_time': datetime.now().isoformat()
            }
            
            logger.info(f"Processing result: {result['detected_sign']} (confidence: {result['confidence']:.2f})")
            
            return jsonify(response_data), 200
            
        finally:
            # Clean up temporary file
            try:
                a = 1
               ## os.unlink(temp_video_path)
            except OSError:
                logger.warning(f"Could not delete temporary file: {temp_video_path}")
    
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': 'Failed to process video',
            'status': 'error'
        }), 500

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
    app.run(debug=True, host='0.0.0.0', port=5000)