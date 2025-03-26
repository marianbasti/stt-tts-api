document.addEventListener('DOMContentLoaded', () => {
    const talkButton = document.getElementById('talk-button');
    const speakerInput = document.getElementById('speaker-sample');
    const messagesContainer = document.getElementById('messages');
    const statusDiv = document.getElementById('status');
    const apiKeyInput = document.getElementById('api-key');
    const chatToggle = document.getElementById('chat-toggle');
    
    let mediaRecorder;
    let audioChunks = [];
    let speakerSample = null;
    let totalStartTime;

    // Create and insert configuration section
    const configSection = document.createElement('div');
    configSection.className = 'config-section model-section';
    document.querySelector('.api-config').appendChild(configSection);

    // Create model selector container
    const modelContainer = document.createElement('div');
    modelContainer.className = 'config-row';
    modelContainer.innerHTML = `
        <label for="model-select">Model:</label>
        <select id="model-select" class="model-select">
            <option value="">Auto-select model</option>
        </select>
    `;
    configSection.appendChild(modelContainer);

    // Get the elements after they're created
    const modelSelect = document.getElementById('model-select');
    const apiUrlInput = document.getElementById('api-url');
    const saveConfigButton = document.getElementById('save-config');
    const apiStatusIndicator = document.getElementById('api-status');

    // Load saved API configuration
    const savedApiKey = localStorage.getItem('openai_api_key');
    const savedApiUrl = localStorage.getItem('openai_api_url');
    if (savedApiKey) apiKeyInput.value = savedApiKey;
    if (savedApiUrl) apiUrlInput.value = savedApiUrl;

    // Load saved chat toggle state
    const savedChatEnabled = localStorage.getItem('chat_enabled');
    if (savedChatEnabled !== null) {
        chatToggle.checked = savedChatEnabled === 'true';
    }

    // Save chat toggle state
    chatToggle.addEventListener('change', () => {
        localStorage.setItem('chat_enabled', chatToggle.checked);
    });

    // Save API configuration
    saveConfigButton.addEventListener('click', async () => {
        const apiUrl = apiUrlInput.value.trim();
        
        try {
            // Update backend configuration
            const formData = new FormData();
            formData.append('api_url', apiUrl);
            const response = await fetch('/api/config', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('Failed to update configuration');
            }
            
            // Save to localStorage only after successful backend update
            if (apiUrl) {
                localStorage.setItem('openai_api_url', apiUrl);
            } else {
                localStorage.removeItem('openai_api_url');
            }
            
            checkApiHealth();
            await updateModelsList();
        } catch (error) {
            console.error('Failed to save configuration:', error);
            apiStatusIndicator.classList.add('error');
        }
    });

    // Check API health status
    async function checkApiHealth() {
        try {
            apiStatusIndicator.className = 'status-indicator';
            const response = await fetch('/health');
            const data = await response.json();
            console.log('Health check response:', data);  // Debug log
            
            if (data.status === 'healthy' && data.models_available && data.models_count > 0) {
                apiStatusIndicator.classList.add('active');
                // Also update models list when health check passes
                await updateModelsList();
            } else {
                apiStatusIndicator.classList.add('error');
            }
        } catch (error) {
            console.error('Health check failed:', error);
            apiStatusIndicator.classList.add('error');
        }
    }

    // Add function to fetch and populate models
    async function updateModelsList() {
        try {
            const response = await fetch('/api/models');
            const data = await response.json();
            
            modelSelect.innerHTML = '<option value="">Auto-select model</option>';
            data.models.forEach(modelId => {
                const option = document.createElement('option');
                option.value = modelId;
                option.textContent = modelId;
                modelSelect.appendChild(option);
            });
        } catch (error) {
            console.error('Failed to fetch models:', error);
        }
    }

    // Initial health check
    checkApiHealth();
    // Regular health check interval
    setInterval(checkApiHealth, 30000); // Check every 30 seconds

    // Clear any existing messages
    messagesContainer.innerHTML = '';

    // Initialize media recorder
    async function setupRecorder() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            
            mediaRecorder.ondataavailable = (event) => {
                audioChunks.push(event.data);
            };

            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                audioChunks = [];
                totalStartTime = performance.now();
                await processConversation(audioBlob);
            };

            statusDiv.textContent = 'Ready';
            talkButton.disabled = false;
        } catch (err) {
            console.error('Error accessing microphone:', err);
            statusDiv.textContent = 'Error: Microphone access denied';
        }
    }

    // Handle speaker sample selection
    speakerInput.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (file) {
            speakerSample = file;
            statusDiv.textContent = 'Voice sample selected';
        }
    });

    // Add message to chat with timing info
    function addMessage(text, role, timing = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}-message`;
        
        const textDiv = document.createElement('div');
        textDiv.className = 'message-text';
        textDiv.textContent = text;
        messageDiv.appendChild(textDiv);
        
        if (timing) {
            const timingDiv = document.createElement('div');
            timingDiv.className = 'message-timing';
            timingDiv.textContent = `⏱️ ${timing.toFixed(2)}s`;
            messageDiv.appendChild(timingDiv);
        }
        
        messagesContainer.appendChild(messageDiv);
        messageDiv.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }

    // Process the conversation flow
    async function processConversation(audioBlob) {
        try {
            statusDiv.textContent = 'Transcribing...';
            
            // Step 1: Transcribe audio
            const formData = new FormData();
            formData.append('file', audioBlob);
            
            const transcribeResponse = await fetch('/api/transcribe', {
                method: 'POST',
                body: formData
            });
            
            if (!transcribeResponse.ok) throw new Error('Transcription failed');
            const transcribeData = await transcribeResponse.json();
            
            // Add user message to chat
            addMessage(transcribeData.text, 'user', transcribeData.elapsed_time);

            let responseText;
            let elapsedTime;
            
            // Step 2: Get chat response or use transcription directly
            if (chatToggle.checked) {
                statusDiv.textContent = 'Getting response...';
                const chatFormData = new FormData();
                chatFormData.append('message', transcribeData.text);
                if (modelSelect.value) {
                    chatFormData.append('model', modelSelect.value);
                }
                
                const chatResponse = await fetch('/api/chat', {
                    method: 'POST',
                    body: chatFormData
                });
                
                if (!chatResponse.ok) throw new Error('Chat failed');
                const chatData = await chatResponse.json();
                responseText = chatData.response;
                elapsedTime = chatData.elapsed_time;
            } else {
                responseText = transcribeData.text;
                elapsedTime = 0;
            }
            
            // Add response message to chat
            addMessage(responseText, 'assistant', elapsedTime);
            
            // Step 3: Convert response to speech
            if (!speakerSample) {
                statusDiv.textContent = 'Please select a voice sample first';
                return;
            }
            
            statusDiv.textContent = 'Generating speech...';
            const ttsFormData = new FormData();
            ttsFormData.append('text', responseText);
            ttsFormData.append('speaker_wav', speakerSample);
            
            const ttsResponse = await fetch('/api/tts', {
                method: 'POST',
                body: ttsFormData
            });
            
            if (!ttsResponse.ok) throw new Error('Speech synthesis failed');
            
            // Play the audio response
            const ttsAudioBlob = await ttsResponse.blob();
            const ttsTime = parseFloat(ttsResponse.headers.get('X-Processing-Time') || '0');
            const audioUrl = URL.createObjectURL(ttsAudioBlob);
            const audio = new Audio(audioUrl);
            await audio.play();
            
            const totalTime = (performance.now() - totalStartTime) / 1000;
            statusDiv.textContent = `Ready (Total processing time: ${totalTime.toFixed(2)}s)`;
        } catch (error) {
            console.error('Error:', error);
            statusDiv.textContent = `Error: ${error.message}`;
        }
    }

    // Setup recording controls
    talkButton.addEventListener('mousedown', () => {
        if (mediaRecorder && mediaRecorder.state === 'inactive') {
            audioChunks = [];
            mediaRecorder.start();
            talkButton.classList.add('recording');
            statusDiv.textContent = 'Recording...';
        }
    });

    talkButton.addEventListener('mouseup', () => {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            mediaRecorder.stop();
            talkButton.classList.remove('recording');
        }
    });

    // Handle touch events for mobile devices
    talkButton.addEventListener('touchstart', (e) => {
        e.preventDefault();
        talkButton.dispatchEvent(new MouseEvent('mousedown'));
    });

    talkButton.addEventListener('touchend', (e) => {
        e.preventDefault();
        talkButton.dispatchEvent(new MouseEvent('mouseup'));
    });

    // Initial setup
    talkButton.disabled = true;
    setupRecorder();
});