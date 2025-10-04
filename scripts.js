/**
 * scripts.js
 * Client-side JavaScript for the Blink Morse Communicator application.
 * Handles webcam streaming, navigation, Morse code input, and Socket.IO communication.
 */

let socket;
let stream;
let video; // Reference to the <video> element with id="webcam"
let canvas;
let ctx;
let currentSelectedUser = null; // Stores username of currently selected user
let communicationStartTime = null;
let timerInterval = null;
let currentMode = 'idle'; // Tracks current mode: 'idle', 'navigation', 'morse_input'
let currentNavIndex = -1; // Index of the currently highlighted navigable element
let navigableElements = []; // Array of DOM elements that can be highlighted
let accumulatedMessage = ''; // Tracks accumulated Morse-decoded message for message display

// Element references (declared here for global access, assigned in DOMContentLoaded)
let userSelect, startButton, stopButton, clearButton, messageDisplay, morseSequenceDisplay;
let overallStatusText, cooldownProgressBar, letterTimerInfo, spaceTimerInfo;

/**
 * Initializes the webcam stream for video capture.
 * This is now ONLY called after user selection is confirmed by backend.
 */
function initWebcam() {
    video = document.getElementById('webcam');
    canvas = document.getElementById('canvas');
    ctx = canvas.getContext('2d');

    // Check if stream is already active to avoid re-initialization
    if (stream && stream.active) {
        console.log(`${new Date().toLocaleTimeString()} Webcam stream already active, skipping re-initialization.`);
        // Ensure frame sending is active
        if (!this._frameSendingInterval) {
            startSendingFrames();
        }
        return;
    }

    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(s => {
                stream = s;
                video.srcObject = stream;
                // Set canvas dimensions to match video to avoid stretching
                video.addEventListener('loadedmetadata', () => {
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    console.log(`${new Date().toLocaleTimeString()} Webcam stream dimensions: ${canvas.width}x${canvas.height}`);
                    video.play();
                    console.log(`${new Date().toLocaleTimeString()} Webcam initialized successfully.`);
                    // Start sending frames immediately AFTER successful webcam init
                    startSendingFrames(); 
                });
            })
            .catch(err => {
                console.error(`${new Date().toLocaleTimeString()} Error accessing webcam: `, err);
                alert("Could not access webcam. Please ensure it is connected and permissions are granted. Stopping communication.");
                stopCommunication(); // Ensure clean state if camera fails
            });
    } else {
        console.error(`${new Date().toLocaleTimeString()} Webcam not supported by this browser.`);
        alert("Your browser does not support webcam access. Stopping communication.");
        stopCommunication(); // Ensure clean state if not supported
    }
}



/**
 * Sends video frames to the server for blink detection.
 */
function startSendingFrames() {
    if (!video || !canvas || !ctx || !stream || !stream.active) {
        console.error(`${new Date().toLocaleTimeString()} Cannot start sending frames: Webcam stream not active or elements not ready.`);
        return;
    }

    // Ensure only one frame sending loop is active
    if (this._frameSendingInterval) {
        clearInterval(this._frameSendingInterval);
        this._frameSendingInterval = null;
        console.log(`${new Date().toLocaleTimeString()} Cleared existing frame sending interval.`);
    }

    const sendFrame = () => {
        if (!stream || !stream.active) { // Check stream active status inside the loop
            console.log(`${new Date().toLocaleTimeString()} Stream stopped, ceasing frame sending loop.`);
            clearInterval(this._frameSendingInterval); // Stop the loop
            this._frameSendingInterval = null;
            return;
        }

        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const frameData = canvas.toDataURL('image/jpeg', 0.5); 
        socket.emit('frame', { image: frameData });
    };

    // Start the loop with a fixed interval
    this._frameSendingInterval = setInterval(sendFrame, 100); // Send frames at ~10 FPS
    console.log(`${new Date().toLocaleTimeString()} Started sending frames to backend.`);
}


/**
 * Stops webcam stream and communication.
 */
function stopCommunication() {
    console.log(`${new Date().toLocaleTimeString()} Attempting to stop communication...`);
    // Clear any active frame sending interval
    if (this._frameSendingInterval) {
        clearInterval(this._frameSendingInterval);
        this._frameSendingInterval = null;
        console.log(`${new Date().toLocaleTimeString()} Stopped frame sending loop.`);
    }

    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
        console.log(`${new Date().toLocaleTimeString()} Webcam stream tracks stopped.`);
    }
    if (video) video.srcObject = null;
    if (canvas && ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    clearInterval(timerInterval);
    communicationStartTime = null;
    timerInterval = null;
    
    // Update UI button states
    if (startButton) startButton.disabled = false;
    if (stopButton) stopButton.disabled = true;
    if (clearButton) clearButton.disabled = true;
    
    // Reset frontend mode and inform backend
    currentMode = 'idle'; // Reset frontend mode to idle
    if (socket && socket.connected) {
        socket.emit('stop_stream'); // Inform backend to stop processing
        socket.emit('set_mode', { mode: 'idle' }); // Explicitly set backend mode to idle
    }
    
    // Reset display messages
    if (overallStatusText) overallStatusText.textContent = 'Status: Stopped';
    if (messageDisplay) messageDisplay.textContent = 'Message:';
    if (morseSequenceDisplay) morseSequenceDisplay.textContent = 'Current:';
    accumulatedMessage = ''; // Clear accumulated message
    
    // Remove all highlights
    navigableElements.forEach(el => el.classList.remove('highlighted'));
    currentNavIndex = -1; // Reset nav index
    
    // Store currentSelectedUser for persistence across page reloads
    if (userSelect && userSelect.value) { // Only store if a value is selected
        sessionStorage.setItem('currentSelectedUser', userSelect.value);
    } else {
        sessionStorage.removeItem('currentSelectedUser'); // Clear if no user selected
    }

    console.log(`${new Date().toLocaleTimeString()} Communication stopped completely.`);
}

/**
 * Updates the timer display for communication duration.
 */
function updateTimerDisplay() {
    if (!communicationStartTime) return;
    const elapsedSeconds = Math.floor((Date.now() - communicationStartTime) / 1000);
    const hours = Math.floor(elapsedSeconds / 3600);
    const minutes = Math.floor((elapsedSeconds % 3600) / 60);
    const seconds = elapsedSeconds % 60;
    
    document.getElementById('hoursDisplay').textContent = hours.toString().padStart(2, '0');
    document.getElementById('minutesDisplay').textContent = minutes.toString().padStart(2, '0');
    document.getElementById('secondsDisplay').textContent = seconds.toString().padStart(2, '0');
}

/**
 * Initializes Socket.IO event listeners.
 */
function setupSocketEvents() {
    // Only initialize if socket is not already connected or doesn't exist
    if (socket && socket.connected) {
        console.log(`${new Date().toLocaleTimeString()} Socket already connected, skipping re-initialization.`);
        return;
    }
    
    socket = io(); // Connect to Socket.IO server

    socket.on('connect', () => {
        console.log(`${new Date().toLocaleTimeString()} Connected to Socket.IO server.`);
        
        // Populate user dropdown on connect (if userSelect element exists on the page)
        if (userSelect) { 
            populateUserDropdown(); // This will also handle loading stored user.
        }
        // After connecting, if the page context suggests an active mode (e.g., quick_messages was loaded directly)
        // ensure backend is aware.
        if (currentMode !== 'idle') {
            socket.emit('set_mode', { mode: currentMode });
        }
    });

    socket.on('disconnect', () => {
        console.log(`${new Date().toLocaleTimeString()} Disconnected from Socket.IO server. Running stopCommunication.`);
        stopCommunication(); // Ensure frontend cleanup on disconnect
    });

socket.on('blink_detected', (data) => {
    console.log(`${new Date().toLocaleTimeString()} Frontend blink_detected event received: ${data.type} in mode: ${currentMode}`);

    // Debounce logic to prevent confusion between dot and dash
    if (!socket._lastBlinkTime) {
        socket._lastBlinkTime = 0;
        socket._blinkTimeout = null;
    }

    const now = Date.now();
    const timeSinceLastBlink = now - socket._lastBlinkTime;

    // Minimum interval between blinks to avoid confusion (e.g., 300ms)
    const minBlinkInterval = 300;

    if (timeSinceLastBlink < minBlinkInterval) {
        console.log(`${new Date().toLocaleTimeString()} Ignoring blink due to debounce interval.`);
        return;
    }

    socket._lastBlinkTime = now;

    // This event is primarily for navigation in 'navigation' mode.
    if (currentMode === 'navigation') {
        // Allow navigation on room-control-page and other pages
        if (data.type === 'dot') {
            moveHighlight(1); // Short blink moves to next option
        } else if (data.type === 'dash') {
            selectHighlightedElement(); // Long blink selects current option
        }
    }
});

socket.on('update_ui', (data) => {
        // Only update message display with new decoded characters if in morse_input mode
        if (currentMode === 'morse_input') {
            if (data.message && data.message.trim() !== '') {
                accumulatedMessage = data.message; // Replace with full accumulated message
                messageDisplay.textContent = `Message: ${accumulatedMessage}`;
            }
        } else if (messageDisplay) {
             // In navigation or idle mode, messageDisplay should reflect fixed text or selected QM.
             if (document.body.classList.contains('quick-messages-page')) {
                 if (!messageDisplay.textContent.startsWith('Selected Message: ')) {
                     messageDisplay.textContent = 'Selected Message: '; // Ensure prefix is there
                 }
             } else {
                 messageDisplay.textContent = `Message: ${accumulatedMessage}`; // Keep showing what was accumulated (empty if not morse_input)
             }
        }
        
        // Ensure morseSequenceDisplay is updated only in morse_input mode
        if (morseSequenceDisplay) {
            morseSequenceDisplay.textContent = `Current: ${currentMode === 'morse_input' ? (data.morse_sequence || '') : ''}`;
        }

        if (overallStatusText) overallStatusText.textContent = `${data.status || 'Processing...'}`;

        // Update progress bar and timers based on data (always show for feedback)
        if (cooldownProgressBar && data.cooldown_percent !== undefined) {
            cooldownProgressBar.style.width = `${data.cooldown_percent * 100}%`;
            cooldownProgressBar.style.backgroundColor = data.cooldown_percent < 1 ? 'orange' : 'var(--success-green)';
        } else if (cooldownProgressBar) { // Reset progress bar if cooldown_percent is not provided
            cooldownProgressBar.style.width = '0%';
            cooldownProgressBar.style.backgroundColor = 'var(--success-green)';
        }

        if (letterTimerInfo) letterTimerInfo.textContent = data.letter_timer > 0 ? `Letter in: ${data.letter_timer.toFixed(1)}s` : '';
        if (spaceTimerInfo) spaceTimerInfo.textContent = data.space_timer > 0 ? `Space in: ${data.space_timer.toFixed(1)}s` : '';
    });

    socket.on('stream_started', (data) => {
        console.log(`${new Date().toLocaleTimeString()} ${data.message}`);
        if (overallStatusText) overallStatusText.textContent = `${data.message}`;
    });

    socket.on('status', (data) => {
        console.log(`${new Date().toLocaleTimeString()} Status: ${data.message}`);
        if (overallStatusText) overallStatusText.textContent = `${data.message}`;
        // If the backend indicates an issue (e.g., user not selected/trained), stop communication.
        if (data.message.includes('No user selected') || data.message.includes('Model for') || data.message.includes('not trained')) {
            stopCommunication();
        }
    });
}

/**
 * Initializes navigable elements based on the current page.
 */
function initializeNavigation() {
    navigableElements = []; // Clear array first
    currentNavIndex = -1; // Reset index on re-initialization

    // Dynamically select elements based on page context
    if (document.body.classList.contains('main-page')) {
        navigableElements = [
            document.getElementById('navSpeakBtn'),
            document.getElementById('navMessageBtn'),
            document.getElementById('navCallBtn'),
            document.getElementById('navRoomControlBtn'), // Room Control button
            document.getElementById('startButton'),
            document.getElementById('stopButton'),
            document.getElementById('clearButton'),
            document.getElementById('controlCallBtn'),
            document.getElementById('controlSettingsBtn'),
            document.getElementById('quickMessageNavBtn'), // This is an <a> tag
            document.getElementById('navHealthBtn'),
            document.getElementById('navMediaBtn')
        ].filter(el => el !== null); // Filter out any nulls if elements don't exist
    } else if (document.body.classList.contains('room-control-page')) {
        navigableElements = [
            document.getElementById('light1Btn'),
            document.getElementById('light2Btn'),
            document.getElementById('fanBtn'),
            document.getElementById('acBtn'),
            document.getElementById('backBtn')
        ].filter(el => el !== null);
    } else if (document.body.classList.contains('device-control-page')) {
        navigableElements = [
            document.getElementById('onBtn'),
            document.getElementById('offBtn'),
            document.getElementById('backBtn')
        ].filter(el => el !== null);
    } else if (document.body.classList.contains('quick-messages-page')) {
        navigableElements = [
            document.getElementById('qmHungry'),
            document.getElementById('qmWater'),
            document.getElementById('qmCold'),
            document.getElementById('qmPain'),
            document.getElementById('qmTired'),
            document.getElementById('qmBathroom'),
            document.getElementById('qmCallNurse'),
            document.getElementById('qmTvOn1'),
            document.getElementById('backButton')
        ].filter(el => el !== null);
    }

    if (navigableElements.length > 0) {
        setHighlight(0); // Highlight the first element when navigation starts/re-initializes
    } else {
        console.warn(`${new Date().toLocaleTimeString()} No navigable elements found on this page for navigation.`);
    }
}

/**
 * Applies the 'highlighted' CSS class to a specific element at the given index.
 * Removes 'highlighted' class from all other navigable elements.
 * @param {number} index The index of the element in `navigableElements` to highlight.
 */
function setHighlight(index) {
    // Remove 'highlighted' class from ALL currently highlighted navigable elements first
    navigableElements.forEach(el => {
        if (el && el.classList.contains('highlighted')) {
            el.classList.remove('highlighted');
        }
    });
    
    // Update the current highlighted index
    currentNavIndex = index;
    
    // Apply highlight to the new element and scroll it into view
    if (navigableElements[currentNavIndex]) {
        navigableElements[currentNavIndex].classList.add('highlighted');
        navigableElements[currentNavIndex].scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
}

/**
 * Moves the highlight to the next or previous navigable element.
 * @param {number} direction 1 for next, -1 for previous.
 */
function moveHighlight(direction) {
    if (navigableElements.length === 0) {
        console.warn(`${new Date().toLocaleTimeString()} No navigable elements to move highlight.`);
        return;
    }

    let newIndex = (currentNavIndex + direction);
    // Handle wrap-around for both positive and negative directions
    newIndex = (newIndex % navigableElements.length + navigableElements.length) % navigableElements.length;
    
    setHighlight(newIndex);
}

/**
 * Selects the currently highlighted element and performs its action.
 */
function selectHighlightedElement() {
    if (currentNavIndex < 0 || currentNavIndex >= navigableElements.length) {
        console.warn(`${new Date().toLocaleTimeString()} No element highlighted to select or invalid index.`);
        return;
    }

    const selectedElement = navigableElements[currentNavIndex];
    if (!selectedElement) {
        console.error(`${new Date().toLocaleTimeString()} Selected element is null, cannot perform action.`);
        return;
    }

    // Remove highlight immediately upon selection to prevent visual glitches
    selectedElement.classList.remove('highlighted');

    console.log(`${new Date().toLocaleTimeString()} Selecting element: ${selectedElement.id || selectedElement.textContent.trim()}`);

    // Handle navigation links first, as they cause full page reloads
    if (selectedElement.tagName === 'A') { // If it's an anchor tag (link)
        console.log(`${new Date().toLocaleTimeString()} Navigating to: ${selectedElement.href}`);
        window.location.assign(selectedElement.href); // Directly navigate to href
        return; // Exit function as page will reload
    }

    // Handle specific button actions based on ID
    if (selectedElement.id === 'navMessageBtn') {
        // This button is for switching to Morse Input mode on the main page
        if (document.body.classList.contains('main-page')) {
            currentMode = 'morse_input'; // Set frontend mode
            accumulatedMessage = ''; // Clear previous message
            messageDisplay.textContent = 'Message:'; // Update UI
            morseSequenceDisplay.textContent = 'Current:'; // Update UI
            
            if (socket && socket.connected) {
                socket.emit('set_mode', { mode: 'morse_input' }); // Inform backend
            }
            if (overallStatusText) overallStatusText.textContent = 'Status: Morse Input Mode';
            console.log(`${new Date().toLocaleTimeString()} Switched to Morse Input Mode.`);
            currentNavIndex = -1; // Reset nav index as we're no longer navigating with blinks
        }
    } else if (selectedElement.id === 'startButton') {
        // Ensure user is selected before starting (check in startCommunication)
        startCommunication(); 
    } else if (selectedElement.id === 'stopButton') {
        stopCommunication();
    } else if (selectedElement.id === 'clearButton') {
        accumulatedMessage = '';
        if (messageDisplay) messageDisplay.textContent = 'Message:';
        if (morseSequenceDisplay) morseSequenceDisplay.textContent = 'Current:';
        console.log(`${new Date().toLocaleTimeString()} Messages cleared.`);
    } else if (selectedElement.id === 'backButton') {
        // Navigate back to index page on long blink (selection)
        stopCommunication(); // Stop stream before navigating away
        console.log(`${new Date().toLocaleTimeString()} Navigating back to index page from backButton.`);
        window.location.assign('/'); // Navigate to index page
        return; // Exit function as page will reload
    } else if (selectedElement.classList.contains('quick-message-button')) {
        // Handle selection of a quick message button
        const messageText = selectedElement.querySelector('.text').textContent;
        if (messageDisplay) messageDisplay.textContent = `Selected Message: ${messageText}`;
        console.log(`${new Date().toLocaleTimeString()} Quick Message Selected: "${messageText}"`);
        speakMessage(messageText); // TTS on blink selection
        // Stay in 'navigation' mode on quick messages page, no Morse decoding here.
    } else if (selectedElement.id === 'sendQuickMessageBtn') {
        const selectedMsg = messageDisplay.textContent.replace('Selected Message: ', '').trim();
        if (selectedMsg && selectedMsg !== 'Message:' && selectedMsg !== '') {
            if (socket && socket.connected) {
                socket.emit('send_quick_message', { message: selectedMsg });
                alert(`Quick Message sent to server: "${selectedMsg}"`);
                if (messageDisplay) messageDisplay.textContent = 'Selected Message: '; // Clear after sending
            } else {
                console.warn(`${new Date().toLocaleTimeString()} Socket not connected, attempting to re-establish for quick message send.`);
                setupSocketEvents(); // Try to reconnect
                setTimeout(() => { // Give it time to connect
                            if (socket && socket.connected) {
                                socket.emit('send_quick_message', { message: selectedMsg });
                                alert(`Quick Message sent to server: "${selectedMsg}"`);
                                if (messageDisplay) messageDisplay.textContent = 'Selected Message: ';
                            } else {
                                alert('Could not connect to server to send message. Please ensure the server is running.');
                            }
                        }, 500);
            }
        } else {
            alert('No quick message selected to send.');
        }
    } else if (selectedElement.id === 'light1Btn' || selectedElement.id === 'light2Btn' || selectedElement.id === 'fanBtn' || selectedElement.id === 'acBtn') {
        // Handle device selection in roomcontrol.html
        const device = selectedElement.dataset.device;
        console.log(`${new Date().toLocaleTimeString()} Navigating to devicecontrol.html for device: ${device}`);
        window.location.assign(`/devicecontrol.html?device=${device}`); // Navigate to device control page
        return; // Exit function as page will reload
    } else if (selectedElement.id === 'onBtn' || selectedElement.id === 'offBtn') {
        // Handle ON/OFF button selection in devicecontrol.html
        const device = new URLSearchParams(window.location.search).get('device');
        const action = selectedElement.dataset.action;
        if (device && action && socket && socket.connected) {
            socket.emit('room_command', { device: device, action: action });
            console.log(`${new Date().toLocaleTimeString()} Sent room_command: device=${device}, action=${action}`);
        } else {
            console.warn(`${new Date().toLocaleTimeString()} Could not send room_command: device=${device}, action=${action}, socket connected=${socket && socket.connected}`);
        }
    } else if (selectedElement.id === 'backBtn') {
        // Handle back button in roomcontrol.html or devicecontrol.html
        if (document.body.classList.contains('device-control-page')) {
            console.log(`${new Date().toLocaleTimeString()} Back button selected - navigating to roomcontrol.html`);
            window.location.assign('/roomcontrol.html'); // Navigate back to room control
        } else if (document.body.classList.contains('room-control-page')) {
            console.log(`${new Date().toLocaleTimeString()} Back button selected - navigating to index.html`);
            window.location.assign('/'); // Navigate back to main page
        }
        return; // Exit function as page will reload
    } else {
        // For other generic buttons (Speak, Call, Settings, Health, Media)
        console.log(`${new Date().toLocaleTimeString()} Selected generic element: ${selectedElement.id}. No specific action defined.`);
        if (selectedElement.tagName === 'BUTTON') {
            console.log(`${new Date().toLocaleTimeString()} ${selectedElement.id} clicked (no specific JS action).`);
        }
    }

    // After selection, if not transitioning page or changing to morse_input mode,
    // move highlight to the next element for continuous navigation.
    if (
        currentMode !== 'morse_input' &&
        selectedElement.tagName !== 'A' &&
        !selectedElement.classList.contains('quick-message-button') // Prevent auto-move for quick-message buttons
    ) {
        moveHighlight(1); // Move highlight to the next element for continued navigation
    }
}

/**
 * Populates the user dropdown by fetching users from the backend.
 */
async function populateUserDropdown() {
    try {
        const response = await fetch('/users');
        const users = await response.json();
        userSelect.innerHTML = '<option value="">-- Select User --</option>'; // Clear existing options
        let foundStoredUser = false;
        let foundDefaultUser = false;

        for (const username in users) {
            const option = document.createElement('option');
            option.value = username;
            option.textContent = `${username} (${users[username].trained ? 'Trained' : 'Not Trained'})`;
            userSelect.appendChild(option);
        }

        // 1. Try to restore previously selected user from sessionStorage
        const storedUser = sessionStorage.getItem('currentSelectedUser');
        if (storedUser && userSelect.querySelector(`option[value="${storedUser}"]`)) {
            userSelect.value = storedUser;
            currentSelectedUser = storedUser; // Update JS variable
            foundStoredUser = true;
            console.log(`${new Date().toLocaleTimeString()} Restored user from sessionStorage: ${storedUser}`);
        } 
        
        // 2. If no stored user or invalid, try to default to 'test'
        if (!foundStoredUser) {
            const testUserOption = userSelect.querySelector('option[value="test"]');
            if (testUserOption) {
                userSelect.value = 'test';
                currentSelectedUser = 'test';
                foundDefaultUser = true;
                console.log(`${new Date().toLocaleTimeString()} Defaulting user to 'test'.`);
            }
        }

        // 3. If a user is now selected (either stored or default), inform backend
        if (currentSelectedUser && socket && socket.connected) {
            socket.emit('select_user', { username: currentSelectedUser }, (response) => {
                console.log(`${new Date().toLocaleTimeString()} Backend select_user response for loaded/default user: ${response.message}`);
                if (response.status === 'error' || response.status === 'warning') {
                    // Alert only if it's an error for a user we tried to select automatically
                    alert(`Issue with user '${currentSelectedUser}': ${response.message}. Please select another user or train.`);
                    // Clear problematic selection if it failed on backend
                    sessionStorage.removeItem('currentSelectedUser');
                    userSelect.value = '';
                    currentSelectedUser = null;
                }
            });
        } else {
            console.log(`${new Date().toLocaleTimeString()} No user to select or socket not connected.`);
        }

    } catch (error) {
        console.error(`${new Date().toLocaleTimeString()} Error fetching users:`, error);
    }
}

/**
 * Event listener for creating a new user.
 */
async function handleCreateUser() {
    const newUsernameInput = document.getElementById('newUsernameInput');
    const username = newUsernameInput.value.trim();
    if (!username) {
        alert("Please enter a username.");
        return;
    }
    try {
        const response = await fetch(`/create_user/${username}`);
        const data = await response.json();
        alert(data.message);
        if (data.status === 'success') {
            newUsernameInput.value = ''; // Clear input
            populateUserDropdown(); // Refresh dropdown
        }
    } catch (error) {
        console.error(`${new Date().toLocaleTimeString()} Error creating user:`, error);
        alert("Failed to create user. See console for details.");
    }
}

/**
 * Event listener for training a user's model.
 */
async function handleTrainUser() {
    const username = userSelect.value;
    if (!username) {
        alert("Please select a user to train.");
        return;
    }
    alert(`Please run the 'Train.py' script separately in your terminal to train user '${username}'. 
           Ensure you select this user in the 'Train.py' menu, then re-select the user here.`);
    populateUserDropdown(); // To update (Trained) status
}

// === TTS (Text-to-Speech) Utility ===
function speakMessage(text) {
    if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel(); // Stop any current speech
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = 'en-US';
        window.speechSynthesis.speak(utterance);
    } else {
        console.warn('TTS not supported in this browser.');
    }
}

/**
 * Initializes the page on load.
 */
document.addEventListener('DOMContentLoaded', () => {
    // Assign common element references
    userSelect = document.getElementById('userSelect');
    messageDisplay = document.getElementById('messageDisplay');
    morseSequenceDisplay = document.getElementById('morseSequenceDisplay');
    overallStatusText = document.getElementById('overallStatusText');
    cooldownProgressBar = document.getElementById('cooldownProgressBar');
    letterTimerInfo = document.getElementById('letterTimerInfo');
    spaceTimerInfo = document.getElementById('spaceTimerInfo');
    
    // User Management buttons
    const createUserBtn = document.getElementById('createUserBtn');
    const trainUserBtn = document.getElementById('trainUserBtn');

    if (createUserBtn) createUserBtn.addEventListener('click', handleCreateUser);
    if (trainUserBtn) trainUserBtn.addEventListener('click', handleTrainUser);
    
    // setupSocketEvents() must be called early to establish connection
    setupSocketEvents(); 
    
    // Page-specific initialization
    if (document.body.classList.contains('main-page')) {
        console.log(`${new Date().toLocaleTimeString()} Initializing Main Page (index.html)...`);

        startButton = document.getElementById('startButton');
        stopButton = document.getElementById('stopButton');
        clearButton = document.getElementById('clearButton');

        if (startButton) startButton.addEventListener('click', startCommunication);
        if (stopButton) stopButton.addEventListener('click', stopCommunication);
        if (clearButton) clearButton.addEventListener('click', () => {
            accumulatedMessage = '';
            if (messageDisplay) messageDisplay.textContent = 'Message:';
            if (morseSequenceDisplay) morseSequenceDisplay.textContent = 'Current:';
            console.log(`${new Date().toLocaleTimeString()} Messages cleared.`);
        });

        if (messageDisplay) messageDisplay.textContent = 'Message:';
        if (morseSequenceDisplay) morseSequenceDisplay.textContent = 'Current:';
        if (stopButton) stopButton.disabled = true;
        if (clearButton) clearButton.disabled = true;

        // Auto-start communication on index.html if a user is already selected from sessionStorage.
        populateUserDropdown().then(() => {
            if (userSelect && userSelect.value) {
                startCommunication();
            }
        });

        setTimeout(() => {
            initializeNavigation(); // Setup highlight navigation
        }, 500);

    } else if (document.body.classList.contains('quick-messages-page')) {
        console.log(`${new Date().toLocaleTimeString()} Initializing Quick Messages Page (quick_messages.html)...`);

        // Auto-start communication on quick_messages.html if a user is already selected from sessionStorage.
        populateUserDropdown().then(() => {
            if (userSelect && userSelect.value) {
                startCommunication();
            }
        });
document.querySelectorAll('.quick-message-button').forEach(button => {
    if (button.id === 'backButton') return; // Skip backButton as it's not a quick message
    button.addEventListener('click', () => {
        const messageText = button.querySelector('.text').textContent;
        if (messageDisplay) messageDisplay.textContent = `Selected Message: ${messageText}`;
        console.log(`${new Date().toLocaleTimeString()} Quick Message Selected: "${messageText}"`);
        speakMessage(messageText); // TTS on click
    });
});

// Add click event listener for backButton to navigate to index page
const backButton = document.getElementById('backButton');
if (backButton) {
    backButton.addEventListener('click', () => {
        stopCommunication(); // Stop stream before navigating away
        console.log(`${new Date().toLocaleTimeString()} Back button clicked - navigating to index page.`);
        window.location.assign('/'); // Navigate to index page
    });
}
    } else if (document.body.classList.contains('room-control-page')) {
        console.log(`${new Date().toLocaleTimeString()} Initializing Room Control Page (roomcontrol.html)...`);

        // Auto-start communication on roomcontrol.html if a user is already selected from sessionStorage.
        populateUserDropdown().then(() => {
            if (userSelect && userSelect.value) {
                startCommunication();
            }
        });

        // Add click event listeners for device buttons
        document.getElementById('light1Btn').addEventListener('click', () => {
            stopCommunication();
            window.location.assign('/devicecontrol.html?device=light1');
        });
        document.getElementById('light2Btn').addEventListener('click', () => {
            stopCommunication();
            window.location.assign('/devicecontrol.html?device=light2');
        });
        document.getElementById('fanBtn').addEventListener('click', () => {
            stopCommunication();
            window.location.assign('/devicecontrol.html?device=fan');
        });
        document.getElementById('acBtn').addEventListener('click', () => {
            stopCommunication();
            window.location.assign('/devicecontrol.html?device=ac');
        });
        document.getElementById('backBtn').addEventListener('click', () => {
            window.location.assign('/');
        });



        // Add touch event listener for Light1 button to navigate immediately
        const light1Btn = document.getElementById('light1Btn');
        if (light1Btn) {
            light1Btn.addEventListener('touchstart', () => {
                stopCommunication();
                window.location.assign('/devicecontrol.html?device=light1');
            });
        }

        setTimeout(() => {
            initializeNavigation(); // Setup highlight navigation
        }, 500);
    } else if (document.body.classList.contains('device-control-page')) {
        console.log(`${new Date().toLocaleTimeString()} Initializing Device Control Page (devicecontrol.html)...`);

        // Get device name from URL query parameter
        const urlParams = new URLSearchParams(window.location.search);
        const device = urlParams.get('device');
        if (device) {
            console.log(`${new Date().toLocaleTimeString()} Device control page loaded for device: ${device}`);
            // Update page title or display device name if needed
            const deviceTitle = document.getElementById('deviceTitle');
            if (deviceTitle) deviceTitle.textContent = `Control ${device.charAt(0).toUpperCase() + device.slice(1)}`;
        }

        // Auto-start communication on devicecontrol.html if a user is already selected from sessionStorage.
        populateUserDropdown().then(() => {
            if (userSelect && userSelect.value) {
                startCommunication();
            }
        });

        setTimeout(() => {
            initializeNavigation(); // Setup highlight navigation
        }, 500);
    }
    else if (document.body.classList.contains('flappy-bird-page')) {
        console.log(`${new Date().toLocaleTimeString()} Initializing Flappy Bird Game Page (flappy_bird.html)...`);
        setTimeout(() => {
            // Try to auto-select a user if none is selected
            userSelect = document.getElementById('userSelect');
            if (userSelect) {
                if (!userSelect.value) {
                    // Try to select 'test' user if available
                    const testOption = userSelect.querySelector('option[value="test"]');
                    if (testOption) {
                        userSelect.value = 'test';
                        currentSelectedUser = 'test';
                    }
                } else {
                    currentSelectedUser = userSelect.value;
                }
            }
            // If a user is now selected, start backend stream for blink detection
            if (currentSelectedUser && currentSelectedUser !== "") {
                startCommunication();
            } else {
                // Fallback: just start webcam (no blink detection)
                initWebcam();
            }
        }, 100);

        // Game variables
        let canvas = document.getElementById('flappyCanvas');
        let ctx = canvas.getContext('2d');
        let bird, pipes, score, gameOver, gravity, flapStrength, pipeGap, pipeWidth, pipeSpeed, frameCount;
        let menuOptions = ['retryBtn', 'quitBtn'];
        let menuIndex = 0;
        let inGameOverMenu = false;
        let gameStarted = false;

        function resetGame() {
            bird = {
                x: 80,
                y: canvas.height / 2,
                radius: 20,
                velocity: 0
            };
            pipes = [];
            score = 0;
            gameOver = false;
            gravity = 0.5;
            flapStrength = -8;
            pipeGap = 150;
            pipeWidth = 60;
            pipeSpeed = 2.5;
            frameCount = 0;
            inGameOverMenu = false;
            gameStarted = false;
            document.getElementById('gameOverMenu').style.display = 'none';
            highlightMenuOption(menuIndex);
            drawStartScreen();
        }

        function drawStartScreen() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.save();
            ctx.fillStyle = '#70c5ce';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.font = 'bold 36px Arial';
            ctx.fillStyle = '#fff';
            ctx.textAlign = 'center';
            ctx.fillText('Flappy Bird', canvas.width / 2, canvas.height / 2 - 60);
            ctx.font = '24px Arial';
            ctx.fillText('Long blink to start', canvas.width / 2, canvas.height / 2);
            ctx.restore();
        }

        function drawBird() {
            ctx.save();
            ctx.beginPath();
            ctx.arc(bird.x, bird.y, bird.radius, 0, Math.PI * 2);
            ctx.fillStyle = '#FFD700';
            ctx.fill();
            ctx.strokeStyle = '#FFA500';
            ctx.lineWidth = 3;
            ctx.stroke();
            ctx.restore();
        }

        function drawPipes() {
            ctx.fillStyle = '#228B22';
            pipes.forEach(pipe => {
                // Top pipe
                ctx.fillRect(pipe.x, 0, pipeWidth, pipe.top);
                // Bottom pipe
                ctx.fillRect(pipe.x, pipe.bottom, pipeWidth, canvas.height - pipe.bottom);
            });
        }

        function drawScore() {
            ctx.save();
            ctx.font = '32px Arial';
            ctx.fillStyle = '#fff';
            ctx.strokeStyle = '#000';
            ctx.lineWidth = 3;
            ctx.strokeText(score, canvas.width / 2 - 10, 60);
            ctx.fillText(score, canvas.width / 2 - 10, 60);
            ctx.restore();
        }

        function updateBird() {
            bird.velocity += gravity;
            bird.y += bird.velocity;
        }

        function addPipe() {
            let top = Math.random() * (canvas.height - pipeGap - 100) + 50;
            let bottom = top + pipeGap;
            pipes.push({
                x: canvas.width,
                top: top,
                bottom: bottom,
                passed: false
            });
        }

        function updatePipes() {
            pipes.forEach(pipe => {
                pipe.x -= pipeSpeed;
            });
            // Remove pipes that are off screen
            if (pipes.length && pipes[0].x + pipeWidth < 0) {
                pipes.shift();
            }
        }

        function checkCollision() {
            // Ground or ceiling
            if (bird.y + bird.radius > canvas.height || bird.y - bird.radius < 0) {
                return true;
            }
            // Pipes
            for (let pipe of pipes) {
                if (
                    bird.x + bird.radius > pipe.x &&
                    bird.x - bird.radius < pipe.x + pipeWidth &&
                    (bird.y - bird.radius < pipe.top || bird.y + bird.radius > pipe.bottom)
                ) {
                    return true;
                }
            }
            return false;
        }

        function checkScore() {
            pipes.forEach(pipe => {
                if (!pipe.passed && bird.x > pipe.x + pipeWidth) {
                    score++;
                    pipe.passed = true;
                }
            });
        }

        function gameLoop() {
            if (!gameStarted) {
                drawStartScreen();
                return;
            }
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            if (!gameOver) {
                updateBird();
                if (frameCount % 90 === 0) {
                    addPipe();
                }
                updatePipes();
                checkScore();
                drawPipes();
                drawBird();
                drawScore();
                if (checkCollision()) {
                    gameOver = true;
                    setTimeout(showGameOverMenu, 500);
                }
                frameCount++;
                requestAnimationFrame(gameLoop);
            } else {
                drawPipes();
                drawBird();
                drawScore();
            }
        }

        function flap() {
            if (!gameOver && !inGameOverMenu && gameStarted) {
                bird.velocity = flapStrength;
            }
        }

        function showGameOverMenu() {
            inGameOverMenu = true;
            document.getElementById('gameOverMenu').style.display = 'block';
            menuIndex = 0;
            highlightMenuOption(menuIndex);
        }

        function highlightMenuOption(idx) {
            menuOptions.forEach((id, i) => {
                let btn = document.getElementById(id);
                if (btn) {
                    if (i === idx) {
                        btn.classList.add('highlighted');
                    } else {
                        btn.classList.remove('highlighted');
                    }
                }
            });
        }

        function selectMenuOption() {
            let id = menuOptions[menuIndex];
            if (id === 'retryBtn') {
                resetGame();
                // Wait for long blink to start again
            } else if (id === 'quitBtn') {
                window.location.href = '/'; // Go back to main page
            }
        }

        // Blink event integration
        function handleBlinkFlappy(type) {
            if (!gameStarted && type === 'dash') {
                gameStarted = true;
                requestAnimationFrame(gameLoop);
                return;
            }
            if (!inGameOverMenu) {
                if (type === 'dot') {
                    flap();
                }
            } else {
                if (type === 'dot') {
                    // Move highlight
                    menuIndex = (menuIndex + 1) % menuOptions.length;
                    highlightMenuOption(menuIndex);
                } else if (type === 'dash') {
                    // Select
                    selectMenuOption();
                }
            }
        }

        // Listen for blink_detected events from backend
        if (typeof socket !== 'undefined') {
            socket.on('blink_detected', (data) => {
                if (document.body.classList.contains('flappy-bird-page')) {
                    handleBlinkFlappy(data.type);
                }
            });
        }

        // Also allow keyboard for testing
        document.addEventListener('keydown', (e) => {
            if (!gameStarted && e.key === 'Enter') {
                gameStarted = true;
                requestAnimationFrame(gameLoop);
            } else if (!inGameOverMenu && (e.code === 'Space' || e.key === 'w')) {
                flap();
            } else if (inGameOverMenu) {
                if (e.key === 'ArrowRight' || e.key === 'd') {
                    menuIndex = (menuIndex + 1) % menuOptions.length;
                    highlightMenuOption(menuIndex);
                } else if (e.key === 'Enter') {
                    selectMenuOption();
                }
            }
        });

        // Button click handlers for accessibility
        document.getElementById('retryBtn').addEventListener('click', () => {
            resetGame();
            // Wait for long blink to start again
        });
        document.getElementById('quitBtn').addEventListener('click', () => {
            window.location.href = '/';
        });

        // Start with start screen
        resetGame();
        // Do not start game loop until long blink
    }
});

/**
 * Core function to start communication flow.
 * Called by start button click or auto-start logic on page load.
 * This function orchestrates user selection, webcam init, and stream start.
 */
function startCommunication() {
    console.log(`${new Date().toLocaleTimeString()} startCommunication called from button.`);
    
    // Check currentSelectedUser directly now. This is a local variable.
    // It should have been set by populateUserDropdown or a manual change.
    if (!currentSelectedUser || currentSelectedUser === "") { 
        alert('No user selected. Please select a user from the dropdown before starting communication.');
        return;
    }

    // Disable start button, enable stop/clear immediately
    if (startButton) startButton.disabled = true;
    if (stopButton) stopButton.disabled = false;
    if (clearButton) clearButton.disabled = false;

    // Ensure socket is connected before proceeding
    if (!socket || !socket.connected) {
        console.log(`${new Date().toLocaleTimeString()} Socket not connected, attempting to connect and then proceed.`);
        setupSocketEvents(); 
        setTimeout(() => { // Add a delay to ensure connection
            if (socket && socket.connected) {
                _startCommunicationFlow();
            } else {
                alert("Could not establish Socket.IO connection. Please check server status and try again.");
                stopCommunication(); // Fallback to stopped state
            }
        }, 500); 
    } else {
        _startCommunicationFlow();
    }
}

/**
 * Private helper function to consolidate the full communication start sequence.
 * This is the ONLY place where webcam/stream initiation should begin.
 */
function _startCommunicationFlow() {
    console.log(`${new Date().toLocaleTimeString()} _startCommunicationFlow initiated.`);

    // 1. Ensure a user is selected locally before telling backend.
    if (!currentSelectedUser || currentSelectedUser === "") {
        console.error(`${new Date().toLocaleTimeString()} currentSelectedUser is not set in _startCommunicationFlow. Aborting.`);
        alert("Internal error: No user selected. Please refresh and try again.");
        stopCommunication();
        return;
    }

    // 2. Tell backend to select user and load model, then wait for response.
    // The callback ensures synchronization: initWebcam/start_stream only after backend confirms.
    socket.emit('select_user', { username: currentSelectedUser }, (response) => {
        console.log(`${new Date().toLocaleTimeString()} Backend select_user response in _startCommunicationFlow: ${response.message}`);
        if (response && response.status === 'success') {
            // 3. User selection successful, now init webcam and start stream.
            initWebcam(); // Initialize webcam (this will also call startSendingFrames upon success)
            
            currentMode = 'navigation'; // Set frontend mode
            socket.emit('start_stream'); // Tell backend to start processing frames
            socket.emit('set_mode', { mode: 'navigation' }); // Backend mode

            communicationStartTime = Date.now();
            if (!timerInterval) { // Prevent multiple intervals
                timerInterval = setInterval(updateTimerDisplay, 1000);
            }

            initializeNavigation(); // Re-initialize navigation elements (ensure highlights start correctly)
            if (overallStatusText) overallStatusText.textContent = 'Status: Navigation Mode';
            console.log(`${new Date().toLocaleTimeString()} Communication started. Mode: Navigation.`);
        } else {
            // Backend reported an error or warning (e.g., user not trained or model load failed)
            // Use logical OR (||) to provide a fallback message if response.message is undefined.
            alert(response && response.message ? response.message : `${new Date().toLocaleTimeString()} Failed to select user on backend (Unknown reason). See console.`);
            stopCommunication(); // Revert to stopped state
        }
    });
}



