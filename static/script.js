let videoStream = null;
let selectedVoice = null;
let availableVoices = [];
let mediaRecorder = null;
let recordedChunks = [];
let isRecording = false;
// let hands = null;
// let faceMesh = null;
// let camera = null;
let frameDetections = []
let confirmedWords = []

// Add letter inference interval variable at the top of your script
let letterInferInterval = null;


let inferInterval = null;

async function startRealtimeInfer(videoEl) {
    // Reduce inference rate significantly for better performance
    const FPS = 2;  // Reduced from 3 to 2 - only 2 inferences per second
    const WINDOW_SIZE = 1;  // Reduced from 6 to 4 - faster consensus
    const PERIOD = 1000 / FPS;  // Now 500ms between inferences

    // Smaller canvas for faster processing
    const sendCanvas = document.createElement('canvas');
    const sendCtx = sendCanvas.getContext('2d');
    sendCanvas.width = 160;  // Reduced from 224 to 160
    sendCanvas.height = 160; // Reduced from 224 to 160

    if (inferInterval) clearInterval(inferInterval);
    
    // Track if inference is in progress to prevent overlapping requests
    let inferenceInProgress = false;
    
    inferInterval = setInterval(async () => {
        // Skip if previous inference still running
        if (inferenceInProgress || !videoEl || videoEl.readyState < 2) return;
        
        inferenceInProgress = true;
        
        try {
            sendCtx.drawImage(videoEl, 0, 0, sendCanvas.width, sendCanvas.height);
            const blob = await new Promise(res => sendCanvas.toBlob(res, 'image/jpeg', 0.6)); // Reduced quality from 0.8 to 0.6
            if (!blob) return;

            const form = new FormData();
            form.append('frame', blob, 'frame.jpg');
            // Remove annotated frame request to reduce processing time
            // form.append('return_annotated', 'true'); 

            const resp = await fetch('/infer-frame', { 
                method: 'POST', 
                body: form,
                signal: AbortSignal.timeout(2000) // 2 second timeout
            });
            if (!resp.ok) return;
            
            const data = await resp.json();

            if (data) {
                // Only record valid detections
                const label = data.detected_sign;
                const rawConf = data.confidence || 0;
                
                if (label && rawConf >= 0.7) { // Reduced threshold from 0.8 to 0.7
                    frameDetections.push(label);
                }
                
                // When we have enough frames, pick the majority word
                if (frameDetections.length >= WINDOW_SIZE) {
                    const freq = {};
                    frameDetections.forEach(w => { freq[w] = (freq[w] || 0) + 1; });
                    const majority = Object.keys(freq).reduce((a, b) => freq[a] > freq[b] ? a : b);
                    confirmedWords.push(majority);
                    console.log('✅ Confirmed words so far:', confirmedWords);
                    
                    // Update displayed text to the majority vote
                    document.getElementById('detectedText').textContent = majority;
                    frameDetections = [];
                } else {
                    // For intermediate frames, still show live detection
                    if (data.detected_sign) {
                        document.getElementById('detectedText').textContent = data.detected_sign;
                    }
                }
                
                // Update confidence bar
                const conf = Math.round((data.confidence || 0) * 100);
                const confEl = document.getElementById('confidence-value');
                const barEl = document.getElementById('confidence-fill');
                if (confEl) confEl.textContent = `${conf}%`;
                if (barEl) barEl.style.width = `${conf}%`;

                // Remove annotated frame processing to reduce lag
                // if (data.annotated_frame) { ... }
            }
        } catch (e) {
            // ignore transient errors
        } finally {
            inferenceInProgress = false;
        }
    }, PERIOD);
}

function stopRealtimeInfer() {
    if (inferInterval) {
        clearInterval(inferInterval);
        inferInterval = null;
    }
}

// Wait for the page to load before initializing MediaPipe
window.addEventListener('load', async function () {
 //   await initializeMediaPipe();
    await checkModelStatus();

    // Load voices for testing
    loadVoicesForTesting();
    speechSynthesis.addEventListener('voiceschanged', loadVoicesForTesting);

    // Set up test phrase dropdown
    const testPhrases = document.getElementById('testPhrases');
    if (testPhrases) {
        testPhrases.addEventListener('change', function () {
            if (this.value) {
                document.getElementById('detectedText').textContent = this.value;
                document.getElementById('confidence-value').textContent = '95%';
                document.getElementById('confidence-fill').style.width = '95%';
            }
        });
    }

    // Set up play audio button
    const playAudioButton = document.getElementById('play-audio');
    if (playAudioButton) {
        playAudioButton.addEventListener('click', processAndSpeak);
    }
});


async function sendConfirmedWords(words = confirmedWords) {
    console.log('[LLM Flow] ▶️ Sending confirmedWords:', words);
    const response = await fetch('/process-confirmed-words', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ confirmedWords: words }),
    });
    console.log('[LLM Flow] 📡 HTTP status:', response.status);
    if (!response.ok) {
        const err = await response.text().catch(() => '(no body)');
        console.error('[LLM Flow] ❌ Error body:', err);
        throw new Error(`Server returned ${response.status}`);
    }
    const payload = await response.json();
    console.log('[LLM Flow] ✅ Payload:', payload);
    return payload;
}
// Camera and recording functionality
async function toggleCamera() {
    const video = document.getElementById('videoElement');
    const placeholder = document.getElementById('cameraPlaceholder');
    const button = document.getElementById('cameraButton');

    if (!videoStream) {
        try {
            // Request camera access with lower resolution for better performance
            videoStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },  // Reduced from 1280
                    height: { ideal: 480 }, // Reduced from 720
                    facingMode: 'user',
                    frameRate: { ideal: 15, max: 20 } // Limit frame rate
                },
                audio: false
            });

            video.srcObject = videoStream;
            video.style.display = 'block';
            placeholder.style.display = 'none';
            button.textContent = 'Stop Camera';
            button.style.background = 'linear-gradient(135deg, #ff4757, #ff3838)';
            
            // Set analyzing placeholder until a confident detection is made
            const detectedEl = document.getElementById('detectedText');
            if (detectedEl) detectedEl.textContent = 'Analyzing…';

            // Show overlay frame when camera is on
            const overlayDiv = document.querySelector('.camera-overlay');
            if (overlayDiv) overlayDiv.style.display = 'block';
            
            // Start realtime inference without MediaPipe
            startRealtimeInfer(video);

        } catch (error) {
            console.error('Error accessing camera:', error);
            alert('Unable to access camera. Please ensure you have granted camera permissions.');
        }
    } else {
        // If camera is already active, stop it
        stopCamera();
    }
}

// Stop and clean up the camera stream
function stopCamera() {
    const video = document.getElementById('videoElement');
    const placeholder = document.getElementById('cameraPlaceholder');
    const button = document.getElementById('cameraButton');
    const overlay = document.querySelector('.camera-overlay');

    if (videoStream) {
        // Stop all media tracks
        videoStream.getTracks().forEach(track => track.stop());
        // Reset video element
        video.srcObject = null;
        video.style.display = 'none';
        // Show placeholder again
        if (placeholder) placeholder.style.display = 'block';
        // Hide overlay
        if (overlay) overlay.style.display = 'none';
        // Reset button text and style
        button.textContent = 'Start Camera';
        button.style.background = ''; // revert to original styling
        // Stop ongoing inference
        stopRealtimeInfer();
        // Clear stream reference
        videoStream = null;
    }
}

// Deprecated: recording flow not used in realtime mode but kept for fallback/demo
function startRecording() {
    try {
        recordedChunks = [];

        // Prefer a broadly compatible MIME type; fall back progressively
        let mimeType = 'video/webm;codecs=vp9';
        if (typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported) {
            if (!MediaRecorder.isTypeSupported(mimeType)) {
                mimeType = 'video/webm;codecs=vp8';
            }
            if (!MediaRecorder.isTypeSupported(mimeType)) {
                mimeType = 'video/webm';
            }
        }

        mediaRecorder = new MediaRecorder(videoStream, { mimeType });

        mediaRecorder.ondataavailable = function (event) {
            if (event.data.size > 0) {
                recordedChunks.push(event.data);
            }
        };

        // Record in chunks every second to maintain 30-second buffer
        mediaRecorder.start(1000);
        isRecording = true;

        console.log('Recording started');
    } catch (error) {
        console.error('Error starting recording:', error);
        // Fallback to basic recording
        try {
            mediaRecorder = new MediaRecorder(videoStream);
            mediaRecorder.ondataavailable = function (event) {
                if (event.data.size > 0) {
                    recordedChunks.push(event.data);
                }
            };
            mediaRecorder.start(1000);
            isRecording = true;
        } catch (fallbackError) {
            console.error('Fallback recording also failed:', fallbackError);
            alert('Recording not supported in this browser');
        }
    }
}

async function stopRecordingAndProcess() {
    if (!mediaRecorder || !isRecording) {
        console.log('No recording to stop');
        return;
    }

    // 🎬 START TIMING: From the moment camera stops
    const cameraStopTime = performance.now();
    console.log(`🔴 Camera stopped at ${new Date().toLocaleTimeString()}.${Date.now() % 1000}`);

    return new Promise((resolve) => {
        mediaRecorder.onstop = async function () {
            console.log('📹 Recording stopped, processing video...');

            // Show loading overlay
            showLoadingOverlay();

            // Create video blob from last 30 seconds of chunks
            const last30SecondsChunks = recordedChunks.slice(-30); // Approximate last 30 chunks
            const blob = new Blob(last30SecondsChunks, { type: 'video/webm' });

            const blobCreationTime = performance.now();
            const blobTime = (blobCreationTime - cameraStopTime) / 1000;

            console.log(`📦 Video blob created: ${blob.size} bytes in ${blobTime.toFixed(3)}s`);

            // Send to server with camera stop time
            await sendVideoToServer(blob, cameraStopTime);

            // Hide loading overlay
            hideLoadingOverlay();

            resolve();
        };

        mediaRecorder.stop();
        isRecording = false;
    });
}

async function sendVideoToServer(videoBlob, cameraStopTime) {
    // Use camera stop time as the baseline for all measurements
    let uploadStartTime, uploadEndTime;

    try {
        const formData = new FormData();
        formData.append('video', videoBlob, 'sign_language_video.webm');
        formData.append('timestamp', new Date().toISOString());
        formData.append('duration', '30'); // 30 seconds

        console.log('📤 Sending video to server...');
        console.log(`📊 Video blob size: ${(videoBlob.size / 1024).toFixed(2)} KB`);

        // Record upload start time
        uploadStartTime = performance.now();
        const preUploadTime = (uploadStartTime - cameraStopTime) / 1000;
        console.log(`⚡ Pre-upload preparation: ${preUploadTime.toFixed(3)}s`);

        const response = await fetch('/process', {
            method: 'POST',
            body: formData,
            headers: {
                // Don't set Content-Type, let browser set it with boundary for FormData
            }
        });

        // Record when server starts responding
        uploadEndTime = performance.now();
        const serverResponseTime = (uploadEndTime - uploadStartTime) / 1000;
        const totalTimeToResponse = (uploadEndTime - cameraStopTime) / 1000;
        console.log(`🌐 Server response received in ${serverResponseTime.toFixed(3)}s`);

        if (response.ok) {
            const result = await response.json();

            // Calculate complete timing from camera stop
            const completionTime = performance.now();
            const totalTimeFromCameraStop = (completionTime - cameraStopTime) / 1000;
            const jsonParsingTime = (completionTime - uploadEndTime) / 1000;

            console.log('🎯 COMPLETE TIMING FROM CAMERA STOP:');
            console.log(`   ⚡ Pre-upload prep: ${((uploadStartTime - cameraStopTime) / 1000).toFixed(3)}s`);
            console.log(`   🌐 Network + Server: ${serverResponseTime.toFixed(3)}s`);
            console.log(`   📄 JSON parsing: ${jsonParsingTime.toFixed(3)}s`);
            console.log(`   🏁 TOTAL FROM CAMERA STOP: ${totalTimeFromCameraStop.toFixed(3)}s`);

            // Log server-side timing breakdown
            if (result.timing) {
                console.log('🔍 Server Processing Breakdown:');
                if (result.timing.endpoint) {
                    console.log(`   📝 Validation: ${result.timing.endpoint.validation?.toFixed(3) || 0}s`);
                    console.log(`   💾 File Save: ${result.timing.endpoint.file_save?.toFixed(3) || 0}s`);
                    console.log(`   🎬 Video Processing: ${result.timing.endpoint.video_processing?.toFixed(3) || 0}s`);
                    console.log(`   ⚡ Total Server Time: ${result.timing.endpoint.total_endpoint?.toFixed(3) || 0}s`);
                }
                if (result.timing.processing) {
                    console.log(`   📹 Video Opening: ${result.timing.processing.video_opening?.toFixed(3) || 0}s`);
                    console.log(`   🎯 MediaPipe: ${result.timing.processing.mediapipe_processing?.toFixed(3) || 0}s`);
                    console.log(`   🛠️  Backend: ${result.timing.processing.successful_backend || 'unknown'}`);
                    console.log(`   📊 Frames: ${result.timing.processing.frames_processed || 0}/${result.timing.processing.total_frames || 0}`);
                }
            }

            console.log('✅ Server response:', result);

            // Update UI with server response if available
            if (result.detected_sign) {
                document.getElementById('detectedText').textContent = result.detected_sign;
            }
            if (result.confidence) {
                const confidencePercentage = Math.round(result.confidence * 100);
                document.querySelector('.confidence-label span:last-child').textContent = `${confidencePercentage}%`;
                document.querySelector('.confidence-fill').style.width = `${confidencePercentage}%`;
            }

            // Log additional model information if available
            if (result.total_frames) {
                console.log(`📊 Processed ${result.total_frames} frames`);
            }
            if (result.valid_predictions) {
                console.log(`🎯 Found ${result.valid_predictions} confident predictions`);
            }

            // Show success message with total time from camera stop
            showTemporaryMessage(`Detected: ${result.detected_sign || 'Unknown'} (${Math.round((result.confidence || 0) * 100)}%) - ${totalTimeFromCameraStop.toFixed(2)}s total`, 'success');

        } else {
            // Try to surface server-side error details
            let serverError = `Server responded with status: ${response.status}`;
            try {
                const errJson = await response.json();
                if (errJson && (errJson.error || errJson.message || errJson.details)) {
                    serverError += ` - ${errJson.error || errJson.message}${errJson.details ? ' (' + errJson.details + ')' : ''}`;
                }
                // Specialized hint when decoding fails
                if (response.status === 415 || (errJson && /Could not open video/i.test(errJson.error || ''))) {
                    serverError += ' | Note: Server now processes WebM natively for optimal real-time performance. If error persists, try recording again or check browser compatibility.';
                }
            } catch (_) { /* ignore JSON parse errors */ }
            throw new Error(serverError);
        }

    } catch (error) {
        console.error('Error sending video to server:', error);

        // Show error message but don't break the flow
        showTemporaryMessage(`Processing error: ${error.message}`, 'warning');

        // For demo purposes, simulate a response
        simulateServerResponse();
    }
}

function simulateServerResponse() {
    // Simulate processing with random results for demo
    const signs = ['Hello', 'Thank You', 'Please', 'Good Morning', 'How Are You?', 'Yes', 'No'];
    const randomSign = signs[Math.floor(Math.random() * signs.length)];
    const randomConfidence = 75 + Math.random() * 20; // 75-95%

    setTimeout(() => {
        document.getElementById('detectedText').textContent = randomSign;
        document.querySelector('.confidence-label span:last-child').textContent = `${Math.round(randomConfidence)}%`;
        document.querySelector('.confidence-fill').style.width = `${randomConfidence}%`;
    }, 1000);
}

function showLoadingOverlay() {
    const overlay = document.getElementById('loadingOverlay');
    overlay.classList.add('active');
}

function hideLoadingOverlay() {
    const overlay = document.getElementById('loadingOverlay');
    overlay.classList.remove('active');
}

function showTemporaryMessage(message, type = 'info') {
    // Create temporary message element
    const messageDiv = document.createElement('div');
    messageDiv.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 1rem 1.5rem;
                border-radius: 10px;
                color: white;
                font-weight: 600;
                z-index: 1000;
                opacity: 0;
                transform: translateX(100%);
                transition: all 0.3s ease;
                ${type === 'success' ? 'background: linear-gradient(135deg, #00ff88, #00cc6a);' : ''}
                ${type === 'warning' ? 'background: linear-gradient(135deg, #ffa500, #ff8c00);' : ''}
                ${type === 'info' ? 'background: linear-gradient(135deg, #00ffff, #8b5cf6);' : ''}
            `;
    messageDiv.textContent = message;
    document.body.appendChild(messageDiv);

    // Animate in
    setTimeout(() => {
        messageDiv.style.opacity = '1';
        messageDiv.style.transform = 'translateX(0)';
    }, 100);

    // Remove after 3 seconds
    setTimeout(() => {
        messageDiv.style.opacity = '0';
        messageDiv.style.transform = 'translateX(100%)';
        setTimeout(() => document.body.removeChild(messageDiv), 300);
    }, 3000);
}

// Load available voices
function loadVoices() {
    availableVoices = speechSynthesis.getVoices();
    const voiceMenu = document.getElementById('voiceMenu');
    const voiceButton = document.getElementById('voiceButton');

    if (!voiceMenu) return;

    voiceMenu.innerHTML = '';

    if (availableVoices.length === 0) {
        voiceMenu.innerHTML = '<div class="dropdown-item">No voices available</div>';
        return;
    }

    console.log(`Found ${availableVoices.length} voices`);

    // Add default option
    const defaultOption = document.createElement('div');
    defaultOption.className = 'dropdown-item';
    defaultOption.textContent = 'Default';
    defaultOption.addEventListener('click', function () {
        selectedVoice = null;
        voiceButton.querySelector('span').textContent = 'Default';
        // Close dropdown
        const dropdown = voiceButton.parentElement;
        dropdown.classList.remove('active');
        voiceButton.nextElementSibling.style.display = 'none';
    });
    voiceMenu.appendChild(defaultOption);

    // Group voices by language
    const voiceGroups = {};
    availableVoices.forEach(voice => {
        const langCode = voice.lang.split('-')[0];
        if (!voiceGroups[langCode]) {
            voiceGroups[langCode] = [];
        }
        voiceGroups[langCode].push(voice);
    });

    // Priority languages
    const priorityLanguages = ['en', 'hi', 'es', 'fr', 'de', 'ja', 'zh', 'ko'];

    priorityLanguages.forEach(langCode => {
        if (voiceGroups[langCode]) {
            // Add language header
            const langHeader = document.createElement('div');
            langHeader.className = 'dropdown-header';
            langHeader.textContent = getLanguageName(langCode);
            voiceMenu.appendChild(langHeader);

            // Add voices for this language
            voiceGroups[langCode].forEach(voice => {
                const voiceOption = document.createElement('div');
                voiceOption.className = 'dropdown-item';

                // Create a cleaner display name
                let displayName = voice.name;

                // Clean up common voice name patterns
                if (voice.name.includes('Google')) {
                    displayName = voice.name.replace('Google ', '').replace(' HD', '') + ' (Google)';
                } else if (voice.name.includes('Microsoft')) {
                    displayName = voice.name.replace('Microsoft ', '') + ' (Microsoft)';
                }

                // Add gender/type indicators
                if (voice.name.toLowerCase().includes('female') || voice.name.toLowerCase().includes('woman')) {
                    displayName += ' ♀';
                } else if (voice.name.toLowerCase().includes('male') || voice.name.toLowerCase().includes('man')) {
                    displayName += ' ♂';
                }

                voiceOption.textContent = displayName;
                voiceOption.addEventListener('click', function () {
                    selectedVoice = voice;
                    voiceButton.querySelector('span').textContent = displayName;
                    // Close dropdown
                    const dropdown = voiceButton.parentElement;
                    dropdown.classList.remove('active');
                    voiceButton.nextElementSibling.style.display = 'none';
                    console.log('Selected voice:', voice.name, voice.lang);
                });
                voiceMenu.appendChild(voiceOption);
            });

            delete voiceGroups[langCode];
        }
    });

    // Add remaining languages
    Object.keys(voiceGroups).forEach(langCode => {
        if (voiceGroups[langCode].length > 0) {
            const langHeader = document.createElement('div');
            langHeader.className = 'dropdown-header';
            langHeader.textContent = getLanguageName(langCode);
            voiceMenu.appendChild(langHeader);

            voiceGroups[langCode].forEach(voice => {
                const voiceOption = document.createElement('div');
                voiceOption.className = 'dropdown-item';
                voiceOption.textContent = voice.name;
                voiceOption.addEventListener('click', function () {
                    selectedVoice = voice;
                    voiceButton.querySelector('span').textContent = voice.name;
                    // Close dropdown
                    const dropdown = voiceButton.parentElement;
                    dropdown.classList.remove('active');
                    voiceButton.nextElementSibling.style.display = 'none';
                });
                voiceMenu.appendChild(voiceOption);
            });
        }
    });

    // Set up dropdown toggle
    voiceButton.addEventListener('click', function (e) {
        e.stopPropagation();
        const dropdown = this.parentElement; // Get the .custom-dropdown parent
        const dropdownMenu = this.nextElementSibling;
        const isOpen = dropdown.classList.contains('active');

        // Close all other dropdowns first
        document.querySelectorAll('.custom-dropdown').forEach(dd => {
            dd.classList.remove('active');
        });

        // Toggle this dropdown
        if (!isOpen) {
            dropdown.classList.add('active');
            dropdownMenu.style.display = 'block';
        } else {
            dropdown.classList.remove('active');
            dropdownMenu.style.display = 'none';
        }
    });

    // Close dropdown when clicking outside
    document.addEventListener('click', function () {
        document.querySelectorAll('.custom-dropdown').forEach(dropdown => {
            dropdown.classList.remove('active');
            const menu = dropdown.querySelector('.dropdown-menu');
            if (menu) menu.style.display = 'none';
        });
    });
}

function getLanguageName(langCode) {
    const languageNames = {
        'en': 'English',
        'hi': 'Hindi',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'ja': 'Japanese',
        'zh': 'Chinese',
        'ko': 'Korean',
        'it': 'Italian',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'ar': 'Arabic',
        'th': 'Thai',
        'vi': 'Vietnamese',
        'nl': 'Dutch',
        'sv': 'Swedish',
        'da': 'Danish',
        'no': 'Norwegian',
        'fi': 'Finnish',
        'pl': 'Polish',
        'tr': 'Turkish'
    };
    return languageNames[langCode] || langCode.toUpperCase();
}

function toggleDropdown(dropdown, show) {
    if (show) {
        dropdown.classList.add('active');
        dropdown.style.display = 'block';
    } else {
        dropdown.classList.remove('active');
        setTimeout(() => {
            if (!dropdown.classList.contains('active')) {
                dropdown.style.display = 'none';
            }
        }, 200);
    }
}

async function processAndSpeak() {
    // Get the detected text
    const detectedText = document.getElementById('detectedText').textContent;

    // Try to get refined gloss first and use it for speaking
     let speakText = '';

    try {
        showLoadingOverlay();
        const result = await sendConfirmedWords();
        // first try "sentence", fallback to old "gloss"
        const refined = result.sentence || result.gloss;
        if (refined) {
            speakText = refined.trim();
            const outEl = document.getElementById('sentenceOutput');
            if (outEl) outEl.textContent = speakText;
            console.log('LLM sentence:', speakText);
        }
    } catch (err) {
        console.error('LLM call failed:', err);
        // fallback to raw joined words
        speakText = confirmedWords.join(' ');
    } finally {
        hideLoadingOverlay();
    }

    if (!speakText) {
        showTemporaryMessage('No sentence to speak', 'warning');
        return;
    }

    // Check if Speech Synthesis is supported
    if ('speechSynthesis' in window) {
        // Cancel any ongoing speech
        speechSynthesis.cancel();

        // Create a new SpeechSynthesisUtterance with refined/fallback text
        const utterance = new SpeechSynthesisUtterance(speakText);

        // Configure voice settings - simple defaults
        utterance.rate = 0.9;
        utterance.pitch = 1.0;
        utterance.volume = 1;

        // Set voice if one is selected
        if (selectedVoice) {
            utterance.voice = selectedVoice;
            utterance.lang = selectedVoice.lang;
            console.log(`Using voice: ${selectedVoice.name} (${selectedVoice.lang})`);
        } else {
            // Set language based on language selection
            const languageSelect = document.getElementById('language-select');
            if (languageSelect) {
                const selectedLang = languageSelect.value;
                switch (selectedLang) {
                    case 'hi':
                        utterance.lang = 'hi-IN';
                        break;
                    case 'es':
                        utterance.lang = 'es-ES';
                        break;
                    case 'fr':
                        utterance.lang = 'fr-FR';
                        break;
                    default:
                        utterance.lang = 'en-US';
                }
            } else {
                utterance.lang = 'en-US';
            }
        }

        // Visual feedback
        const button = document.getElementById('processButton');
        const playButton = document.getElementById('play-audio');
        const originalButtonText = button ? button.innerHTML : '';

        if (button) {
            button.innerHTML = '<i class="fas fa-volume-up"></i> Speaking...';
            button.disabled = true;
        }

        if (playButton) {
            playButton.innerHTML = '<i class="fas fa-stop"></i>';
            playButton.style.background = 'linear-gradient(135deg, #ff4757, #ff3838)';
        }

        // Animate audio visualizer
        animateAudioVisualizer(true);

        // Event handlers
        utterance.onstart = function () {
            console.log('Speech started');
            showTemporaryMessage('🔊 Playing audio...', 'info');
        };

        utterance.onend = function () {
            console.log('Speech ended');
            if (button) {
                button.innerHTML = originalButtonText;
                button.disabled = false;
            }

            if (playButton) {
                playButton.innerHTML = '<i class="fas fa-volume-up"></i>';
                playButton.style.background = '';
            }

            animateAudioVisualizer(false);
            showTemporaryMessage('✅ Audio playback completed', 'success');
        };

        utterance.onerror = function (event) {
            console.error('Speech synthesis error:', event.error);
            if (button) {
                button.innerHTML = originalButtonText;
                button.disabled = false;
            }

            if (playButton) {
                playButton.innerHTML = '<i class="fas fa-volume-up"></i>';
                playButton.style.background = '';
            }

            animateAudioVisualizer(false);
            showTemporaryMessage('❌ Speech error: ' + event.error, 'warning');
        };

        // Speak the text
        speechSynthesis.speak(utterance);
        console.log(`Speaking: "${speakText}" with rate=${utterance.rate}, pitch=${utterance.pitch}`);

    } else {
        showTemporaryMessage('❌ Speech synthesis not supported in this browser', 'warning');
    }
}

function clearDetection() {
    // Clear the detected text
    document.getElementById('detectedText').textContent = 'Click "Start Camera" to begin detection';

    // Reset confidence
    document.getElementById('confidence-value').textContent = '--';
    document.getElementById('confidence-fill').style.width = '0%';

    // Reset test phrases dropdown
    const testPhrases = document.getElementById('testPhrases');
    if (testPhrases) {
        testPhrases.value = '';
    }

    // Stop any ongoing speech
    if ('speechSynthesis' in window) {
        speechSynthesis.cancel();
    }

    // Stop audio visualizer
    animateAudioVisualizer(false);

    showTemporaryMessage('Detection cleared', 'info');
}

// Load voices for testing
function loadVoicesForTesting() {
    availableVoices = speechSynthesis.getVoices();
    const voiceMenu = document.getElementById('voiceMenu');
    const voiceButton = document.getElementById('voiceButton');

    if (!voiceMenu || !voiceButton) return;

    voiceMenu.innerHTML = '';

    if (availableVoices.length === 0) {
        voiceMenu.innerHTML = '<div class="dropdown-item">No voices available</div>';
        return;
    }

    console.log(`Found ${availableVoices.length} voices`);

    // Add default option
    const defaultOption = document.createElement('div');
    defaultOption.className = 'dropdown-item';
    defaultOption.textContent = 'Default Voice';
    defaultOption.addEventListener('click', function () {
        selectedVoice = null;
        voiceButton.querySelector('span').textContent = 'Default Voice';
        // Close dropdown
        const dropdown = voiceButton.parentElement;
        dropdown.classList.remove('active');
        voiceButton.nextElementSibling.style.display = 'none';
    });
    voiceMenu.appendChild(defaultOption);

    // Filter voices for English and other common languages
    const priorityLanguages = ['en', 'es', 'fr', 'de'];

    priorityLanguages.forEach(langCode => {
        const langVoices = availableVoices.filter(voice =>
            voice.lang.toLowerCase().startsWith(langCode)
        );

        if (langVoices.length > 0) {
            // Add language header
            const langHeader = document.createElement('div');
            langHeader.className = 'dropdown-header';
            langHeader.textContent = getLanguageName(langCode);
            voiceMenu.appendChild(langHeader);

            // Add first few voices for each language
            langVoices.slice(0, 3).forEach(voice => {
                const voiceOption = document.createElement('div');
                voiceOption.className = 'dropdown-item';

                // Create cleaner display name
                let displayName = voice.name;
                if (voice.name.includes('Google')) {
                    displayName = voice.name.replace('Google ', '').replace(' HD', '') + ' (Google)';
                } else if (voice.name.includes('Microsoft')) {
                    displayName = voice.name.replace('Microsoft ', '') + ' (Microsoft)';
                }

                voiceOption.textContent = displayName;
                voiceOption.addEventListener('click', function () {
                    selectedVoice = voice;
                    voiceButton.querySelector('span').textContent = displayName;
                    // Close dropdown
                    const dropdown = voiceButton.parentElement;
                    dropdown.classList.remove('active');
                    voiceButton.nextElementSibling.style.display = 'none';
                    console.log('Selected voice:', voice.name, voice.lang);
                });
                voiceMenu.appendChild(voiceOption);
            });
        }
    });

    // Set up dropdown toggle
    voiceButton.addEventListener('click', function (e) {
        e.stopPropagation();
        const dropdown = this.parentElement; // Get the .custom-dropdown parent
        const dropdownMenu = this.nextElementSibling;
        const isOpen = dropdown.classList.contains('active');

        // Close all other dropdowns first
        document.querySelectorAll('.custom-dropdown').forEach(dd => {
            dd.classList.remove('active');
            const menu = dd.querySelector('.dropdown-menu');
            if (menu) menu.style.display = 'none';
        });

        // Toggle this dropdown
        if (!isOpen) {
            dropdown.classList.add('active');
            dropdownMenu.style.display = 'block';
        }
    });

    // Close dropdown when clicking outside
    document.addEventListener('click', function () {
        document.querySelectorAll('.custom-dropdown').forEach(dropdown => {
            dropdown.classList.remove('active');
            const menu = dropdown.querySelector('.dropdown-menu');
            if (menu) menu.style.display = 'none';
        });
    });
}

function getLanguageName(langCode) {
    const languageNames = {
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German'
    };
    return languageNames[langCode] || langCode.toUpperCase();
}

function toggleDropdown(dropdown, show) {
    if (show) {
        dropdown.classList.add('active');
        dropdown.style.display = 'block';
        const button = dropdown.previousElementSibling;
        if (button) {
            button.classList.add('active');
        }
    } else {
        dropdown.classList.remove('active');
        const button = dropdown.previousElementSibling;
        if (button) {
            button.classList.remove('active');
        }
        setTimeout(() => {
            if (!dropdown.classList.contains('active')) {
                dropdown.style.display = 'none';
            }
        }, 200);
    }
}

function animateAudioVisualizer(isPlaying) {
    const bars = document.querySelectorAll('.waveform-bar');

    if (isPlaying) {
        bars.forEach((bar, index) => {
            bar.style.animation = `audioWave 0.5s ease-in-out infinite alternate`;
            bar.style.animationDelay = `${index * 0.1}s`;
        });
    } else {
        bars.forEach(bar => {
            bar.style.animation = '';
        });
    }
}

// Check model status on startup
async function checkModelStatus() {
    try {
        const response = await fetch('/model-status');
        if (response.ok) {
            const status = await response.json();
            console.log('Model Status:', status);

            if (status.demo_mode) {
                if (!status.ml_libraries_available) {
                    console.log('⚠️ ML libraries not available - running in demo mode');
                    showTemporaryMessage('Running in Demo Mode (ML libraries unavailable)', 'warning');
                } else if (!status.model_loaded) {
                    console.log('⚠️ Model not loaded - running in demo mode');
                    showTemporaryMessage('Running in Demo Mode (Model not loaded)', 'warning');
                }
            } else {
                console.log(`✅ Model loaded successfully with ${status.actions_count} actions`);
                console.log('Available actions:', status.actions);
                showTemporaryMessage('AI Model Ready! 🤖', 'success');
            }
        } else {
            console.log('⚠️ Could not check model status');
            showTemporaryMessage('Running in Demo Mode', 'warning');
        }
    } catch (error) {
        console.error('Error checking model status:', error);
        showTemporaryMessage('Running in Demo Mode', 'warning');
    }
}

// Build a video from English sentence and display it
async function composeReverseVideo() {
    console.log('🎬 composeReverseVideo called');
    const input = document.getElementById('glossTokensInput');
    const videoEl = document.getElementById('reverseVideo');
    const metaEl = document.getElementById('reverseMeta');
    if (!input || !videoEl) {
        console.error('❌ Missing input or video element', { input, videoEl });
        return;
    }
    const sentence = (input.value || '').trim();
    console.log('📝 Input sentence:', sentence);
    if (!sentence) {
        showTemporaryMessage('Enter an English sentence', 'warning');
        return;
    }
    try {
        showLoadingOverlay();
        console.log('🌐 Sending request to /reverse-translate-video');
        const resp = await fetch('/reverse-translate-video', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: sentence })
        });
        console.log('📡 Response status:', resp.status);
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            console.error('❌ Server error:', err);
            const more = err.hint ? `\nHint: ${err.hint}` : '';
            const prev = err.available_tokens_preview ? `\nAvailable (sample): ${err.available_tokens_preview.join(', ')}` : '';
            throw new Error((err.error || `Server error: ${resp.status}`) + more + prev);
        }
        const data = await resp.json();
        console.log('✅ Response data:', data);
        const url = data.video_url; // Use the direct URL returned by the backend
        console.log('🎥 Video URL:', url);
        
        // Hide placeholder and show video
        const placeholder = document.getElementById('videoPlaceholder');
        if (placeholder) placeholder.style.display = 'none';
        
        videoEl.src = url;
        videoEl.style.display = 'block';
        console.log('🔄 Loading video...');
        videoEl.load();
        videoEl.play().catch((playErr) => {
            console.error('▶️ Play error:', playErr);
        });
        showTemporaryMessage('Reverse video ready', 'success');
    } catch (e) {
        console.error('composeReverseVideo error:', e);
        showTemporaryMessage(e.message || 'Failed to compose video', 'warning');
    } finally {
        hideLoadingOverlay();
    }
}
function fillSampleGloss() {
    const input = document.getElementById('glossTokensInput');
    if (input) input.value = 'We are going to college for exam';
}



// Add this after your existing startRealtimeInfer function
async function startRealtimeLetterInfer(videoEl) {
    // Similar to gesture inference but for letters
    const FPS = 3;  // Slightly higher for letter detection
    const WINDOW_SIZE = 2;  // Faster consensus for letters
    const PERIOD = 1000 / FPS;

    const sendCanvas = document.createElement('canvas');
    const sendCtx = sendCanvas.getContext('2d');
    sendCanvas.width = 160;
    sendCanvas.height = 160;

    if (letterInferInterval) clearInterval(letterInferInterval);
    
    let inferenceInProgress = false;
    
    letterInferInterval = setInterval(async () => {
        if (inferenceInProgress || !videoEl || videoEl.readyState < 2) return;
        
        inferenceInProgress = true;
        
        try {
            sendCtx.drawImage(videoEl, 0, 0, sendCanvas.width, sendCanvas.height);
            const blob = await new Promise(res => sendCanvas.toBlob(res, 'image/jpeg', 0.6));
            if (!blob) return;

            const form = new FormData();
            form.append('frame', blob, 'frame.jpg');

            const resp = await fetch('/infer-letter', { 
                method: 'POST', 
                body: form,
                signal: AbortSignal.timeout(2000)
            });
            if (!resp.ok) return;
            
            const data = await resp.json();

            if (data && data.detected_letter) {
                // Update letter display elements
                const letterEl = document.getElementById('detectedLetter');
                if (letterEl) letterEl.textContent = data.detected_letter;
                
                const conf = Math.round((data.confidence || 0) * 100);
                const confEl = document.getElementById('letter-confidence-value');
                const barEl = document.getElementById('letter-confidence-fill');
                if (confEl) confEl.textContent = `${conf}%`;
                if (barEl) barEl.style.width = `${conf}%`;
            }
        } catch (e) {
            // ignore transient errors
        } finally {
            inferenceInProgress = false;
        }
    }, PERIOD);
}


// Add stop function
function stopRealtimeLetterInfer() {
    if (letterInferInterval) {
        clearInterval(letterInferInterval);
        letterInferInterval = null;
    }
}

// Add toggle function for letter mode
async function toggleLetterMode() {
    const video = document.getElementById('videoElement');
    
    if (currentMode === 'gesture') {
        // Switch to letter mode
        stopRealtimeInfer();
        startRealtimeLetterInfer(video);
        currentMode = 'letter';
        document.getElementById('modeButton').textContent = 'Switch to Gesture Mode';
    } else {
        // Switch to gesture mode
        stopRealtimeLetterInfer();
        startRealtimeInfer(video);
        currentMode = 'gesture';
        document.getElementById('modeButton').textContent = 'Switch to Letter Mode';
    }
}

// Mode tracking and letter spelling
let currentMode = 'gesture';
let spelledLetters = [];

// Toggle between gesture and letter detection modes
function toggleDetectionMode() {
    const video = document.getElementById('videoElement');
    const modeButton = document.getElementById('modeButton');
    const gestureCard = document.getElementById('gestureCard');
    const letterCard = document.getElementById('letterCard');
    const audioCard = document.getElementById('audioCard');
    const spellingCard = document.getElementById('spellingCard');
    const refineOutput = document.getElementById('refineOutput');
    
    if (currentMode === 'gesture') {
        // Switch to letter mode
        stopRealtimeInfer();
        if (video && video.srcObject) {
            startRealtimeLetterInfer(video);
        }
        currentMode = 'letter';
        
        // Update UI
        modeButton.innerHTML = '<i class="fas fa-hand-paper"></i> Switch to Gesture Mode';
        gestureCard.style.display = 'none';
        letterCard.style.display = 'block';
        audioCard.style.display = 'none';
        spellingCard.style.display = 'block';
        refineOutput.style.display = 'none';
        
    } else {
        // Switch to gesture mode
        stopRealtimeLetterInfer();
        if (video && video.srcObject) {
            startRealtimeInfer(video);
        }
        currentMode = 'gesture';
        
        // Update UI
        modeButton.innerHTML = '<i class="fas fa-font"></i> Switch to Letter Mode';
        gestureCard.style.display = 'block';
        letterCard.style.display = 'none';
        audioCard.style.display = 'block';
        spellingCard.style.display = 'none';
        refineOutput.style.display = 'block';
    }
}

// Clear spelled word
function clearSpelling() {
    spelledLetters = [];
    document.getElementById('spelledWord').textContent = 'Start spelling...';
}

// Add letter to spelled word (call this when a letter is detected)
function addToSpelling(letter) {
    if (letter && letter !== spelledLetters[spelledLetters.length - 1]) {
        spelledLetters.push(letter);
        document.getElementById('spelledWord').textContent = spelledLetters.join('');
    }
}

// =============================================================================
// ALPHABET DETECTION SECTION - SEPARATE FROM MAIN TRANSLATOR
// =============================================================================

// Alphabet detection variables
let alphabetVideoStream = null;
let alphabetInferInterval = null;
let alphabetSpelledLetters = [];
let alphabetLetterHistory = [];
let isAlphabetDetectionActive = false;

// Start alphabet detection camera
async function toggleAlphabetCamera() {
    const video = document.getElementById('alphabetVideoElement');
    const placeholder = document.getElementById('alphabetCameraPlaceholder');
    const button = document.getElementById('alphabetCameraButton');
    const statusDot = document.getElementById('alphabetStatusDot');
    const status = document.getElementById('alphabetStatus');

    if (!alphabetVideoStream) {
        try {
            // Request camera access
            alphabetVideoStream = await navigator.mediaDevices.getUserMedia({
                video: { 
                    width: { ideal: 640 }, 
                    height: { ideal: 480 },
                    facingMode: 'user'
                }
            });
            
            video.srcObject = alphabetVideoStream;
            video.style.display = 'block';
            placeholder.style.display = 'none';
            
            button.innerHTML = '<i class="fas fa-pause"></i> Pause Detection';
            button.classList.remove('primary');
            button.classList.add('secondary');
            
            statusDot.style.backgroundColor = '#10b981';
            status.textContent = 'Detecting';
            
            // Start letter detection
            await startAlphabetInference(video);
            
        } catch (error) {
            console.error('Error accessing camera for alphabet detection:', error);
            showTemporaryMessage('Camera access denied for alphabet detection', 'error');
        }
    } else {
        // Stop detection
        stopAlphabetCamera();
    }
}

// Stop alphabet detection camera
function stopAlphabetCamera() {
    const video = document.getElementById('alphabetVideoElement');
    const placeholder = document.getElementById('alphabetCameraPlaceholder');
    const button = document.getElementById('alphabetCameraButton');
    const statusDot = document.getElementById('alphabetStatusDot');
    const status = document.getElementById('alphabetStatus');

    if (alphabetVideoStream) {
        alphabetVideoStream.getTracks().forEach(track => track.stop());
        alphabetVideoStream = null;
    }
    
    stopAlphabetInference();
    
    video.style.display = 'none';
    placeholder.style.display = 'flex';
    
    button.innerHTML = '<i class="fas fa-play"></i> Start Letter Detection';
    button.classList.remove('secondary');
    button.classList.add('primary');
    
    statusDot.style.backgroundColor = '#6b7280';
    status.textContent = 'Ready';
    
    // Reset detection display
    document.getElementById('alphabetDetectedLetter').textContent = '?';
    document.getElementById('alphabet-confidence-value').textContent = '--';
    document.getElementById('alphabet-confidence-fill').style.width = '0%';
}

// Start alphabet inference
async function startAlphabetInference(videoEl) {
    if (!videoEl) return;
    
    const FPS = 3; // 3 inferences per second for alphabet detection
    const PERIOD = 1000 / FPS;
    
    // Create canvas for sending frames
    const sendCanvas = document.createElement('canvas');
    const sendCtx = sendCanvas.getContext('2d');
    sendCanvas.width = 224; // Slightly larger for better letter recognition
    sendCanvas.height = 224;
    
    if (alphabetInferInterval) clearInterval(alphabetInferInterval);
    
    let inferenceInProgress = false;
    isAlphabetDetectionActive = true;
    
    alphabetInferInterval = setInterval(async () => {
        if (inferenceInProgress || !videoEl || videoEl.readyState < 2 || !isAlphabetDetectionActive) return;
        
        inferenceInProgress = true;
        
        try {
            // Draw video frame to canvas
            sendCtx.drawImage(videoEl, 0, 0, sendCanvas.width, sendCanvas.height);
            const blob = await new Promise(res => sendCanvas.toBlob(res, 'image/jpeg', 0.8));
            if (!blob) return;

            // Send to letter detection endpoint
            const form = new FormData();
            form.append('frame', blob, 'frame.jpg');

            const response = await fetch('/infer-letter', {
                method: 'POST',
                body: form,
                signal: AbortSignal.timeout(3000)
            });

            if (!response.ok) {
                console.warn('Alphabet detection request failed:', response.status);
                return;
            }

            const data = await response.json();
            
            if (data && data.detected_letter) {
                updateAlphabetDisplay(data.detected_letter, data.confidence || 0);
                
                // Add to history if confidence is high enough
                if (data.confidence > 0.8) {
                    addLetterToHistory(data.detected_letter, data.confidence);
                    
                    // Auto-add to spelled word if confidence is very high
                    if (data.confidence > 0.9) {
                        addToAlphabetSpelling(data.detected_letter);
                    }
                }
            } else {
                // No letter detected
                updateAlphabetDisplay('?', 0);
            }
            
        } catch (error) {
            if (error.name !== 'AbortError') {
                console.warn('Alphabet inference error:', error);
            }
        } finally {
            inferenceInProgress = false;
        }
    }, PERIOD);
}

// Stop alphabet inference
function stopAlphabetInference() {
    if (alphabetInferInterval) {
        clearInterval(alphabetInferInterval);
        alphabetInferInterval = null;
    }
    isAlphabetDetectionActive = false;
}

// Update alphabet detection display
function updateAlphabetDisplay(letter, confidence) {
    const letterEl = document.getElementById('alphabetDetectedLetter');
    const confValueEl = document.getElementById('alphabet-confidence-value');
    const confFillEl = document.getElementById('alphabet-confidence-fill');
    
    if (letterEl) letterEl.textContent = letter || '?';
    
    const confPercent = Math.round((confidence || 0) * 100);
    if (confValueEl) confValueEl.textContent = `${confPercent}%`;
    if (confFillEl) confFillEl.style.width = `${confPercent}%`;
    
    // Color-code the confidence
    if (confFillEl) {
        if (confPercent >= 80) {
            confFillEl.style.backgroundColor = '#10b981'; // Green
        } else if (confPercent >= 60) {
            confFillEl.style.backgroundColor = '#f59e0b'; // Yellow
        } else {
            confFillEl.style.backgroundColor = '#ef4444'; // Red
        }
    }
}

// Add letter to alphabet spelling
function addToAlphabetSpelling(letter) {
    if (!letter || letter === '?') return;
    
    // Prevent duplicate consecutive letters
    if (letter === alphabetSpelledLetters[alphabetSpelledLetters.length - 1]) return;
    
    alphabetSpelledLetters.push(letter.toUpperCase());
    updateAlphabetSpelledWord();
}

// Update spelled word display
function updateAlphabetSpelledWord() {
    const spelledWordEl = document.getElementById('alphabetSpelledWord');
    const wordLengthEl = document.getElementById('wordLength');
    
    if (alphabetSpelledLetters.length === 0) {
        spelledWordEl.textContent = 'START SPELLING...';
        spelledWordEl.style.color = '#9ca3af';
    } else {
        spelledWordEl.textContent = alphabetSpelledLetters.join('');
        spelledWordEl.style.color = '#10b981';
    }
    
    if (wordLengthEl) {
        wordLengthEl.textContent = alphabetSpelledLetters.length;
    }
}

// Clear alphabet spelling
function clearAlphabetSpelling() {
    alphabetSpelledLetters = [];
    updateAlphabetSpelledWord();
}

// Add letter to history
function addLetterToHistory(letter, confidence) {
    const historyEl = document.getElementById('alphabetLetterHistory');
    if (!historyEl || !letter || letter === '?') return;
    
    // Add to history array
    alphabetLetterHistory.unshift({
        letter: letter.toUpperCase(),
        confidence: confidence,
        timestamp: new Date()
    });
    
    // Keep only last 10 letters
    if (alphabetLetterHistory.length > 10) {
        alphabetLetterHistory = alphabetLetterHistory.slice(0, 10);
    }
    
    // Update display
    updateLetterHistoryDisplay();
}

// Update letter history display
function updateLetterHistoryDisplay() {
    const historyEl = document.getElementById('alphabetLetterHistory');
    if (!historyEl) return;
    
    if (alphabetLetterHistory.length === 0) {
        historyEl.innerHTML = '<span style="color: #9ca3af; font-style: italic;">No letters detected yet</span>';
        return;
    }
    
    historyEl.innerHTML = alphabetLetterHistory.map(item => {
        const confPercent = Math.round(item.confidence * 100);
        const confColor = confPercent >= 80 ? '#10b981' : confPercent >= 60 ? '#f59e0b' : '#ef4444';
        
        return `
            <span style="
                display: inline-block;
                background: ${confColor};
                color: white;
                padding: 8px 12px;
                border-radius: 8px;
                font-weight: 600;
                font-size: 14px;
                margin: 2px;
            ">
                ${item.letter}
                <small style="opacity: 0.8;">${confPercent}%</small>
            </span>
        `;
    }).join('');
}

// Clear letter history
function clearLetterHistory() {
    alphabetLetterHistory = [];
    updateLetterHistoryDisplay();
}

// Scroll to alphabet section
function scrollToAlphabet() {
    document.getElementById('alphabet').scrollIntoView({ behavior: 'smooth' });
}