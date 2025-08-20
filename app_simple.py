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

@app.route('/')
def index():
    """Serve the main page"""
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
    """Mock processing endpoint"""
    return jsonify({
        'success': True,
        'message': 'Video processed successfully (mock)',
        'prediction': 'Hello',
        'confidence': 0.85,
        'audio_path': None
    })

@app.route('/model-status', methods=['GET'])
def model_status():
    """Check model loading status"""
    return jsonify({
        'status': 'ready',
        'model_loaded': True,
        'message': 'Mock model ready'
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'message': 'Server is running (mock mode)'
    })

if __name__ == '__main__':
    print("Starting Flask app in mock mode...")
    app.run(host='0.0.0.0', port=5000, debug=True)
