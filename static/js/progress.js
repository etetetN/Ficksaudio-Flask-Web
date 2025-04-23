document.addEventListener('DOMContentLoaded', function() {
    const progressUrl = new URL('/progress', window.location.origin).href;
    const progressMessage = document.getElementById('progress-message');
    const progressCount = document.getElementById('progress-count');
    const progressFill = document.getElementById('progress-fill');
    
    let retryCount = 0;
    const MAX_RETRIES = 5;
    const RETRY_DELAY = 2000; // 2 seconds
    
    function updateProgress(progress) {
        if (!progress) return;
        
        // Update message
        if (progress.message) {
            progressMessage.textContent = progress.message;
        }
        
        // Update progress bar
        if (progress.total > 0) {
            const percent = Math.min(100, Math.round((progress.current / progress.total) * 100));
            progressFill.style.width = percent + '%';
            
            // Update count text
            if (progress.current !== undefined && progress.total !== undefined) {
                progressCount.textContent = `Processing segment ${progress.current} of ${progress.total}`;
            }
        } else if (progress.percent !== undefined) {
            const percent = Math.min(100, Math.round(progress.percent * 100));
            progressFill.style.width = percent + '%';
        }
        
        // Update UI for step tracking
        updateProgressUI(progress);
        
        // Check if processing is complete
        if (progress.done) {
            progressMessage.textContent = 'Processing complete!';
            progressFill.style.width = '100%';
            
            setTimeout(() => {
                window.location.href = '/result';
            }, 1500);
        }
    }
    
    function updateProgressUI(progress) {
        if (!progress) return;
        
        // Update steps based on progress message
        let currentStep = 'upload';
        if (progress.current > 0 && progress.current < progress.total) {
            currentStep = 'enhance';
            document.getElementById('step-upload').classList.add('completed');
            document.getElementById('step-convert').classList.add('completed');
            document.getElementById('step-enhance').classList.add('active');
        } else if (progress.message && progress.message.includes('Converting')) {
            currentStep = 'convert';
            document.getElementById('step-upload').classList.add('completed');
            document.getElementById('step-convert').classList.add('active');
        } else if (progress.done) {
            document.getElementById('step-upload').classList.add('completed');
            document.getElementById('step-convert').classList.add('completed');
            document.getElementById('step-enhance').classList.add('completed');
            document.getElementById('step-finalize').classList.add('active');
        }
        
        // Show random fun facts every 5 seconds
        if (!window.factInterval) {
            window.factInterval = setInterval(() => {
                const randomFact = funFacts[Math.floor(Math.random() * funFacts.length)];
                const factElement = document.getElementById('fun-fact');
                factElement.style.opacity = 0;
                
                setTimeout(() => {
                    document.getElementById('fun-fact').innerHTML = 
                        `<i class="fas fa-lightbulb"></i><p>${randomFact}</p>`;
                    factElement.style.opacity = 1;
                }, 500);
            }, 8000);
        }
    }
    
    function connectEventSource() {
        console.log('Connecting to event source:', progressUrl);
        progressMessage.textContent = 'Connecting to server...';
        
        // Create a new EventSource
        const eventSource = new EventSource(progressUrl);
        
        // Handle connection open
        eventSource.onopen = function() {
            console.log('SSE connection established');
            progressMessage.textContent = 'Connection established, waiting for processing to begin...';
            retryCount = 0; // Reset retry count on successful connection
        };
        
        // Handle incoming messages
        eventSource.onmessage = function(e) {
            console.log('SSE message received:', e.data);
            try {
                const progress = JSON.parse(e.data);
                updateProgress(progress);
                
                // When complete, close the connection and clear intervals
                if (progress.done) {
                    eventSource.close();
                    clearInterval(window.factInterval);
                }
            } catch (error) {
                console.error('Error parsing progress data:', error, e.data);
                progressMessage.textContent = 'Error updating progress. Processing continues in background.';
            }
        };
        
        // Handle errors
        eventSource.onerror = function(e) {
            console.error('SSE connection error:', e);
            
            // Close the connection
            eventSource.close();
            
            // Attempt to reconnect with exponential backoff
            if (retryCount < MAX_RETRIES) {
                retryCount++;
                const delay = RETRY_DELAY * Math.pow(2, retryCount - 1);
                progressMessage.textContent = `Connection lost. Reconnecting in ${delay/1000} seconds...`;
                
                setTimeout(() => {
                    progressMessage.textContent = 'Reconnecting...';
                    connectEventSource();
                }, delay);
            } else {
                // After max retries, show message but don't attempt reconnection
                progressMessage.textContent = 'Cannot connect to progress feed. Processing continues in the background.';
                
                // Setup polling as fallback
                setupPolling();
            }
        };
        
        return eventSource;
    }
    
    function setupPolling() {
        console.log('Setting up polling fallback');
        const pollInterval = 3000; // 3 seconds
        
        const checkProgress = async () => {
            try {
                const response = await fetch('/progress_status');
                if (response.ok) {
                    const progress = await response.json();
                    updateProgress(progress);
                    
                    if (progress.done) {
                        clearInterval(pollingId);
                    }
                } else {
                    console.error('Error fetching progress status:', response.statusText);
                }
            } catch (error) {
                console.error('Error in polling:', error);
            }
        };
        
        // Start polling
        const pollingId = setInterval(checkProgress, pollInterval);
        checkProgress(); // Call immediately for first update
    }
    
    // Fun facts about audio processing
    const funFacts = [
        "Our AI model processes audio in frequency-time space (spectrograms) instead of raw waveforms.",
        "The model was trained on thousands of hours of audio data to learn how to distinguish noise from signal.",
        "Audio enhancement uses a special algorithm known as a 'diffusion model', which gradually removes noise.",
        "Each audio segment is processed through dozens of neural network steps to achieve the best quality.",
        "The spectrogram visualization shows you the frequency content of your audio over time.",
        "Lower frequencies (bass sounds) appear at the bottom of spectrograms, while higher frequencies are at the top.",
        "Our model can help reduce background noise, hiss, and improve overall clarity.",
        "Bright spots in the enhanced spectrogram indicate stronger audio signals.",
        "The neural network has over 30 million parameters that work together to enhance your audio.",
        "Audio processing utilizes your computer's CPU and memory resources to run complex calculations."
    ];
    
    // Connect to EventSource
    window.eventSource = connectEventSource();
    
    // Handle page unload to close connection properly
    window.addEventListener('beforeunload', function() {
        if (window.eventSource) {
            window.eventSource.close();
        }
        if (window.factInterval) {
            clearInterval(window.factInterval);
        }
    });
}); 