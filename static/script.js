let videoStream = null;
        let selectedVoice = null;
        let availableVoices = [];
        let mediaRecorder = null;
        let recordedChunks = [];
        let isRecording = false;
        let hands = null;
        let faceMesh = null;
        let camera = null;

        // Wait for the page to load before initializing MediaPipe
        window.addEventListener('load', async function() {
            await initializeMediaPipe();
            
            // Load voices initially
            loadVoices();

            // Some browsers load voices asynchronously
            speechSynthesis.addEventListener('voiceschanged', loadVoices);

            // Set up language dropdown functionality
            document.querySelectorAll('.dropdown-item').forEach(item => {
                if (item.parentElement.parentElement.querySelector('#languageButton')) {
                    item.addEventListener('click', function () {
                        document.getElementById('languageButton').textContent = this.textContent;
                        // Reload voices when language changes
                        setTimeout(loadVoices, 100);
                    });
                }
            });
        });

        async function initializeMediaPipe() {
            try {
                // Initialize MediaPipe Hands
                hands = new Hands({
                    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
                });
                hands.setOptions({
                    maxNumHands: 2,
                    modelComplexity: 1,
                    minDetectionConfidence: 0.5,
                    minTrackingConfidence: 0.5,
                });

                // Initialize MediaPipe Face Mesh
                faceMesh = new FaceMesh({
                    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
                });
                faceMesh.setOptions({
                    maxNumFaces: 1,
                    refineLandmarks: true,
                    minDetectionConfidence: 0.5,
                    minTrackingConfidence: 0.5,
                });

                // Set up canvas and video elements
                const videoElement = document.getElementById('videoElement');
                const canvasElement = document.getElementById('canvasOverlay');
                const canvasCtx = canvasElement.getContext('2d');

                // Set canvas size to match container
                function resizeCanvas() {
                    const cameraFeed = document.querySelector('.camera-feed');
                    canvasElement.width = cameraFeed.clientWidth;
                    canvasElement.height = cameraFeed.clientHeight;
                }

                resizeCanvas();
                window.addEventListener('resize', resizeCanvas);

                // Set up MediaPipe results handlers
                hands.onResults((results) => {
                    // Clear canvas
                    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
                    
                    // Draw hand landmarks if detected
                    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
                        for (const landmarks of results.multiHandLandmarks) {
                            // Draw connections
                            drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, 
                                {color: '#00FF41', lineWidth: 2});
                            // Draw landmarks
                            drawLandmarks(canvasCtx, landmarks, 
                                {color: '#FF0041', lineWidth: 1, radius: 3});
                        }
                        console.log(`Detected ${results.multiHandLandmarks.length} hand(s)`);
                    }
                });

                faceMesh.onResults((results) => {
                    // Draw face mesh if detected (overlay on existing canvas content)
                    if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
                        for (const landmarks of results.multiFaceLandmarks) {
                            // Draw face tesselation (subtle)
                            drawConnectors(canvasCtx, landmarks, FACEMESH_TESSELATION, 
                                {color: '#C0C0C070', lineWidth: 1});
                            // Draw eyes
                            drawConnectors(canvasCtx, landmarks, FACEMESH_RIGHT_EYE, 
                                {color: '#FF3030', lineWidth: 1});
                            drawConnectors(canvasCtx, landmarks, FACEMESH_LEFT_EYE, 
                                {color: '#30FF30', lineWidth: 1});
                            // Draw lips
                            drawConnectors(canvasCtx, landmarks, FACEMESH_LIPS, 
                                {color: '#E0E0E0', lineWidth: 1});
                        }
                        console.log(`Detected ${results.multiFaceLandmarks.length} face(s)`);
                    }
                });

                console.log('MediaPipe initialized successfully');

            } catch (error) {
                console.error('Error initializing MediaPipe:', error);
            }
        }

        // Camera and recording functionality
        async function toggleCamera() {
            const video = document.getElementById('videoElement');
            const placeholder = document.getElementById('cameraPlaceholder');
            const button = document.getElementById('cameraButton');

            if (!videoStream) {
                try {
                    // Request camera access
                    videoStream = await navigator.mediaDevices.getUserMedia({
                        video: {
                            width: { ideal: 1280 },
                            height: { ideal: 720 },
                            facingMode: 'user'
                        },
                        audio: false
                    });

                    video.srcObject = videoStream;
                    video.style.display = 'block';
                    placeholder.style.display = 'none';
                    button.textContent = 'Stop Camera';
                    button.style.background = 'linear-gradient(135deg, #ff4757, #ff3838)';

                    // Initialize camera for MediaPipe
                    if (hands && faceMesh) {
                        camera = new Camera(video, {
                            onFrame: async () => {
                                try {
                                    await hands.send({ image: video });
                                    await faceMesh.send({ image: video });
                                } catch (error) {
                                    console.error('Error processing frame:', error);
                                }
                            },
                            width: 1280,
                            height: 720,
                        });
                        camera.start();
                        console.log('Camera started for MediaPipe processing');
                    }

                    // Start recording immediately
                    startRecording();

                } catch (error) {
                    console.error('Error accessing camera:', error);
                    alert('Unable to access camera. Please ensure you have granted camera permissions.');
                }
            } else {
                // Stop recording and process video
                await stopRecordingAndProcess();

                // Stop camera
                if (camera) {
                    camera.stop();
                    camera = null;
                }
                videoStream.getTracks().forEach(track => track.stop());
                videoStream = null;
                video.style.display = 'none';
                placeholder.style.display = 'block';
                button.textContent = 'Start Camera';
                button.style.background = 'var(--gradient-primary)';
            }
        }

        function startRecording() {
            try {
                recordedChunks = [];
                mediaRecorder = new MediaRecorder(videoStream, {
                    mimeType: 'video/webm;codecs=vp9' // High quality codec
                });

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

            return new Promise((resolve) => {
                mediaRecorder.onstop = async function () {
                    console.log('Recording stopped, processing video...');

                    // Show loading overlay
                    showLoadingOverlay();

                    // Create video blob from last 30 seconds of chunks
                    const last30SecondsChunks = recordedChunks.slice(-30); // Approximate last 30 chunks
                    const blob = new Blob(last30SecondsChunks, { type: 'video/webm' });

                    console.log(`Video blob created: ${blob.size} bytes`);

                    // Send to server
                    await sendVideoToServer(blob);

                    // Hide loading overlay
                    hideLoadingOverlay();

                    resolve();
                };

                mediaRecorder.stop();
                isRecording = false;
            });
        }

        async function sendVideoToServer(videoBlob) {
            try {
                const formData = new FormData();
                formData.append('video', videoBlob, 'sign_language_video.webm');
                formData.append('timestamp', new Date().toISOString());
                formData.append('duration', '30'); // 30 seconds

                console.log('Sending video to server...');

                const response = await fetch('/process', {
                    method: 'POST',
                    body: formData,
                    headers: {
                        // Don't set Content-Type, let browser set it with boundary for FormData
                    }
                });

                if (response.ok) {
                    const result = await response.json();
                    console.log('Server response:', result);

                    // Update UI with server response if available
                    if (result.detected_sign) {
                        document.getElementById('detectedText').textContent = result.detected_sign;
                    }
                    if (result.confidence) {
                        const confidencePercentage = Math.round(result.confidence * 100);
                        document.querySelector('.confidence-label span:last-child').textContent = `${confidencePercentage}%`;
                        document.querySelector('.confidence-fill').style.width = `${confidencePercentage}%`;
                    }

                    // Show success message
                    showTemporaryMessage('Video processed successfully!', 'success');

                } else {
                    throw new Error(`Server responded with status: ${response.status}`);
                }

            } catch (error) {
                console.error('Error sending video to server:', error);

                // Show error message but don't break the flow
                showTemporaryMessage('Processing complete (server unavailable)', 'warning');

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
            voiceMenu.innerHTML = '';

            if (availableVoices.length === 0) {
                voiceMenu.innerHTML = '<div class="dropdown-item">No voices available</div>';
                return;
            }

            // Add default option
            const defaultOption = document.createElement('div');
            defaultOption.className = 'dropdown-item';
            defaultOption.textContent = 'Default';
            defaultOption.addEventListener('click', function () {
                selectedVoice = null;
                document.getElementById('voiceButton').textContent = 'Default';
            });
            voiceMenu.appendChild(defaultOption);

            // Group voices by language and show accents
            const currentLang = document.getElementById('languageButton').textContent;
            let langCode = 'en';

            switch (currentLang) {
                case 'Hindi': langCode = 'hi'; break;
                case 'Spanish': langCode = 'es'; break;
                case 'French': langCode = 'fr'; break;
                case 'German': langCode = 'de'; break;
                case 'Japanese': langCode = 'ja'; break;
                case 'Chinese': langCode = 'zh'; break;
                case 'Korean': langCode = 'ko'; break;
                default: langCode = 'en';
            }

            // Filter and add voices
            const filteredVoices = availableVoices.filter(voice =>
                voice.lang.toLowerCase().startsWith(langCode.toLowerCase())
            );

            // If no voices for selected language, show English voices
            const voicesToShow = filteredVoices.length > 0 ? filteredVoices :
                availableVoices.filter(voice => voice.lang.toLowerCase().startsWith('en'));

            voicesToShow.forEach(voice => {
                const voiceOption = document.createElement('div');
                voiceOption.className = 'dropdown-item';

                // Create a more descriptive name
                let displayName = voice.name;
                if (voice.name.includes('Google')) {
                    displayName = voice.name.replace('Google ', '') + ' (Google)';
                }

                // Add locale info for better identification
                displayName += ` [${voice.lang}]`;

                voiceOption.textContent = displayName;
                voiceOption.addEventListener('click', function () {
                    selectedVoice = voice;
                    document.getElementById('voiceButton').textContent = displayName;
                });
                voiceMenu.appendChild(voiceOption);
            });
        }

        function processAndSpeak() {
            // Get the detected text
            const detectedText = document.getElementById('detectedText').textContent;

            // Check if Speech Synthesis is supported
            if ('speechSynthesis' in window) {
                // Cancel any ongoing speech
                speechSynthesis.cancel();

                // Create a new SpeechSynthesisUtterance
                const utterance = new SpeechSynthesisUtterance(detectedText);

                // Configure voice settings
                utterance.rate = 0.9; // Slightly slower for clarity
                utterance.pitch = 1;
                utterance.volume = 1;

                // Set voice if one is selected
                if (selectedVoice) {
                    utterance.voice = selectedVoice;
                } else {
                    // Set language based on selection
                    const selectedLanguage = document.getElementById('languageButton').textContent;
                    switch (selectedLanguage) {
                        case 'Hindi':
                            utterance.lang = 'hi-IN';
                            break;
                        case 'Spanish':
                            utterance.lang = 'es-ES';
                            break;
                        case 'French':
                            utterance.lang = 'fr-FR';
                            break;
                        case 'German':
                            utterance.lang = 'de-DE';
                            break;
                        case 'Japanese':
                            utterance.lang = 'ja-JP';
                            break;
                        case 'Chinese':
                            utterance.lang = 'zh-CN';
                            break;
                        case 'Korean':
                            utterance.lang = 'ko-KR';
                            break;
                        default:
                            utterance.lang = 'en-US';
                    }
                }

                // Speak the text
                speechSynthesis.speak(utterance);

                // Visual feedback
                const button = document.getElementById('processButton');
                button.textContent = 'Speaking...';

                utterance.onend = function () {
                    button.textContent = 'Process';
                };

                utterance.onerror = function (event) {
                    console.error('Speech synthesis error:', event.error);
                    button.textContent = 'Process';
                    alert('Error occurred during speech synthesis: ' + event.error);
                };
            } else {
                alert('Speech synthesis is not supported in your browser.');
            }
        }