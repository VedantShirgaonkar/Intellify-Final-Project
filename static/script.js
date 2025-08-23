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
            await checkModelStatus();

            // Load voices for testing
            loadVoicesForTesting();
            speechSynthesis.addEventListener('voiceschanged', loadVoicesForTesting);

            // Set up test phrase dropdown
            const testPhrases = document.getElementById('testPhrases');
            if (testPhrases) {
                testPhrases.addEventListener('change', function() {
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

                    // Log additional model information if available
                    if (result.total_frames) {
                        console.log(`Processed ${result.total_frames} frames`);
                    }
                    if (result.valid_predictions) {
                        console.log(`Found ${result.valid_predictions} confident predictions`);
                    }

                    // Show success message
                    showTemporaryMessage(`Detected: ${result.detected_sign || 'Unknown'} (${Math.round((result.confidence || 0) * 100)}%)`, 'success');

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
                            serverError += ' | Hint: Try recording with a different browser or upload MP4. Admins can install moviepy and imageio-ffmpeg on server to enable WebM conversion.';
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
            voiceButton.addEventListener('click', function(e) {
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
            document.addEventListener('click', function() {
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

        function processAndSpeak() {
            // Get the detected text
            const detectedText = document.getElementById('detectedText').textContent;

            if (!detectedText || detectedText === 'Click "Start Camera" to begin detection' || detectedText.trim() === '' || detectedText === '--') {
                showTemporaryMessage('No text to speak', 'warning');
                return;
            }

            // Check if Speech Synthesis is supported
            if ('speechSynthesis' in window) {
                // Cancel any ongoing speech
                speechSynthesis.cancel();

                // Create a new SpeechSynthesisUtterance
                const utterance = new SpeechSynthesisUtterance(detectedText);

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
                utterance.onstart = function() {
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
                console.log(`Speaking: "${detectedText}" with rate=${utterance.rate}, pitch=${utterance.pitch}`);

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
            voiceButton.addEventListener('click', function(e) {
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
            document.addEventListener('click', function() {
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