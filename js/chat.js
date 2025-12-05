class ChatBot {
    constructor() {
        this.chatToggle = document.getElementById('chatToggle');
        this.chatWidget = document.getElementById('aiChatWidget');
        this.chatInput = document.getElementById('chatInput');
        this.chatFileInput = document.getElementById('chatFileInput');
        this.attachButton = document.getElementById('attachButton');
        this.chatMessages = document.getElementById('chatMessages');
        this.sendButton = document.getElementById('sendMessage');
        this.typingIndicator = document.getElementById('typingIndicator');
        this.chatSuggestions = document.getElementById('chatSuggestions');
        this.suggestionButtons = document.querySelectorAll('.suggestion-btn');

        this.conversationHistory = [];
        this.isTyping = false;

        this.initializeEventListeners();
        this.hasShownWelcome = false;

        // Check if we are in the pop-out window (chat.html)
        const isPopout = window.location.pathname.endsWith('chat.html');

        if (isPopout) {
            // Load history from session storage if in pop-out
            this.loadHistory();
        } else {
            // Clear history if on main page to start fresh
            sessionStorage.removeItem('chatHistory');
        }
    }

    initializeEventListeners() {
        // Toggle chat
        if (this.chatToggle) {
            this.chatToggle.addEventListener('click', (e) => {
                e.stopPropagation();
                this.toggleChat();
            });
        }

        // Close button
        const closeButton = document.querySelector('.chat-close');
        if (closeButton) {
            closeButton.addEventListener('click', (e) => {
                e.stopPropagation();
                this.chatWidget.classList.remove('active');
            });
        }

        // Minimize button
        const minimizeButton = document.querySelector('.chat-minimize');
        if (minimizeButton) {
            minimizeButton.addEventListener('click', (e) => {
                e.stopPropagation();
                this.chatWidget.classList.toggle('minimized');
            });
        }

        // Pop-out button
        const popoutButton = document.getElementById('popoutChat');
        if (popoutButton) {
            popoutButton.addEventListener('click', (e) => {
                e.stopPropagation();
                window.open('chat.html', '_blank');
            });
        }

        // Check for full page mode
        if (document.body.classList.contains('full-page-chat')) {
            this.chatWidget.classList.add('active');
            // Ensure it stays active
            this.chatWidget.classList.remove('minimized');
        }

        // Send message on button click
        if (this.sendButton) {
            this.sendButton.addEventListener('click', (e) => {
                e.preventDefault();
                this.handleUserMessage();
            });
        }

        // Send message on Enter key
        if (this.chatInput) {
            this.chatInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.handleUserMessage();
                }
            });
        }

        // Attach button click
        if (this.attachButton && this.chatFileInput) {
            this.attachButton.addEventListener('click', (e) => {
                e.preventDefault();
                this.chatFileInput.click();
            });

            this.chatFileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    this.handleFileUpload(e.target.files[0]);
                }
            });
        }

        // Handle suggestion clicks
        if (this.suggestionButtons && this.suggestionButtons.length > 0) {
            this.suggestionButtons.forEach(button => {
                button.addEventListener('click', (e) => {
                    e.preventDefault();
                    this.chatInput.value = e.target.textContent;
                    this.chatInput.focus();
                    this.handleUserMessage();
                });
            });
        }

        // Close chat when clicking outside
        document.addEventListener('click', (e) => {
            const isClickInsideChat = this.chatWidget.contains(e.target) || e.target === this.chatToggle;
            if (!isClickInsideChat && !this.chatWidget.classList.contains('minimized')) {
                this.chatWidget.classList.remove('active');
            } else if (e.target === this.chatToggle) {
                this.chatWidget.classList.toggle('active');
                if (this.chatWidget.classList.contains('active')) {
                    this.chatWidget.classList.remove('minimized');
                    this.chatInput.focus();
                }
            }
        });

        // Prevent clicks inside chat from closing it
        this.chatWidget.addEventListener('click', (e) => {
            e.stopPropagation();
        });
    }

    toggleChat(forceState) {
        // If forceState is provided, set the state directly
        if (typeof forceState === 'boolean') {
            this.chatWidget.classList.toggle('active', forceState);
        } else {
            this.chatWidget.classList.toggle('active');
        }

        // Focus the input when showing
        if (this.chatWidget.classList.contains('active')) {
            // Show welcome message only the first time the widget is opened
            if (!this.hasShownWelcome) {
                this.showWelcomeMessage();
                this.hasShownWelcome = true;
            }
            this.chatInput.focus();
            const notificationBadge = document.querySelector('.notification-badge');
            if (notificationBadge) {
                notificationBadge.classList.remove('active');
            }
        }
    }

    async handleFileUpload(file) {
        if (!file) return;

        // Validate file type
        if (!file.type.startsWith('image/')) {
            this.addMessage('Please upload an image file.', 'ai');
            return;
        }

        // Show image preview immediately
        const reader = new FileReader();
        reader.onload = (e) => {
            this.addMessage(`<img src="${e.target.result}" class="chat-upload-preview" alt="Uploaded Image">`, 'user');
        };
        reader.readAsDataURL(file);

        // Show typing indicator while analyzing
        this.showTypingIndicator(true);

        try {
            // Prepare form data for API
            const formData = new FormData();
            formData.append('file', file);
            formData.append('model', 'primary'); // Default to primary model

            // Determine API URL
            let apiUrl = '/api/analyze';
            if (window.location.protocol === 'file:') {
                apiUrl = 'http://127.0.0.1:8000/api/analyze';
            }

            // Call Analysis API
            const response = await fetch(apiUrl, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Analysis failed');
            }

            const data = await response.json();

            // Construct analysis context
            let analysisContext = `[System: User uploaded an image.`;

            if (data.classification && data.classification.class_name) {
                analysisContext += ` Analysis Result: ${data.classification.class_name}`;
                if (data.classification.confidence) {
                    analysisContext += ` (Confidence: ${(data.classification.confidence * 100).toFixed(1)}%)`;
                }
                analysisContext += `.`;
            } else if (data.error) {
                analysisContext += ` Analysis Error: ${data.error}.`;
            } else {
                analysisContext += ` Analysis Result: Indeterminate.`;
            }

            if (data.notes) {
                analysisContext += ` Notes: ${data.notes}.`;
            }

            if (data.segmentation && data.segmentation.performed) {
                analysisContext += ` Segmentation performed.`;
            }
            analysisContext += `]`;

            // Add to history (hidden from UI)
            this.conversationHistory.push({
                role: 'user',
                content: analysisContext
            });
            this.saveHistory();

            // Trigger AI response based on this context
            const aiResponse = await this.getAIResponse("I have uploaded a medical image. Please explain the analysis results provided in the system context.");
            this.addMessage(aiResponse, 'ai');

        } catch (error) {
            console.error('Error uploading/analyzing file:', error);
            this.addMessage(`I'm sorry, I couldn't analyze that image. Error: ${error.message}. Please try again.`, 'ai');
        } finally {
            this.showTypingIndicator(false);
            // Reset file input
            this.chatFileInput.value = '';
        }
    }

    async handleUserMessage() {
        const message = this.chatInput.value.trim();
        if (!message) return;

        await this.sendMessage(message);
        this.chatInput.value = '';
    }

    async sendMessage(message) {
        // Add user message to chat
        this.addMessage(message, 'user');

        // Show typing indicator
        this.showTypingIndicator(true);

        try {
            // Get AI response
            const response = await this.getAIResponse(message);

            // Simulate typing delay
            await new Promise(resolve => setTimeout(resolve, 1000));

            // Add AI response to chat
            this.addMessage(response, 'ai');

            // Hide suggestions after first message
            this.chatSuggestions.classList.add('hidden');

        } catch (error) {
            console.error('Error getting AI response:', error);
            this.addMessage("I'm sorry, I encountered an error. Please try again.", 'ai');
            this.addMessage("the error is: " + error, 'ai');
        } finally {
            this.showTypingIndicator(false);
        }
    }

    async getAIResponse(message) {
        // Note: User message is already added to history by addMessage()

        try {
            // Call the Gemini API
            const apiKey = 'AIzaSyAKZjVbYVoXIeOWotJhNOyzahb_JS7JWAI';
            const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${apiKey}`;

            const systemPrompt = "You are a helpful medical AI assistant. called (AI-Doc). Provide accurate, concise, and helpful information about medical conditions, symptoms, and general health advice. remind users to consult with healthcare professionals for medical advice when you gave them a treatment advice. Do not provide any details other medical related topics. I want your output to be in simple text format no in markdown style.";

            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    system_instruction: {
                        parts: [{ text: systemPrompt }]
                    },
                    contents: this.conversationHistory.map(msg => ({
                        role: msg.role === 'assistant' ? 'model' : 'user',
                        parts: [{ text: msg.content }]
                    }))
                })
            });


            if (!response.ok) {
                const errorData = await response.json();
                console.error('Gemini API Error:', errorData);
                throw new Error(errorData.error?.message || 'Failed to get response from AI');
            }

            const data = await response.json();
            const aiResponse = data.candidates[0].content.parts[0].text;

            // Note: AI response will be added to history by addMessage()

            return aiResponse;

        } catch (error) {
            console.error('Error calling Gemini API:', error);
            this.addMessage(`Error: ${error.message}`, 'ai');
            // Fallback to local responses if API call fails
            return this.getFallbackResponse(message);
        }
    }

    showWelcomeMessage() {
        this.addMessage("Hello! I'm your AI medical assistant. I can help you with health information, explain medical terms, and provide general wellness advice. Please remember that I'm not a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for any medical concerns.", 'ai');
    }

    addMessage(text, sender) {
        this.appendMessageToUI(text, sender);

        // Add to conversation history if it's a user message or AI response
        if (sender === 'user' || sender === 'ai') {
            this.conversationHistory.push({
                role: sender === 'user' ? 'user' : 'assistant',
                content: text
            });
            this.saveHistory();
        }
    }

    appendMessageToUI(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;

        const timeString = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        messageDiv.innerHTML = `
            <div class="message-avatar">
                <i class="fas fa-${sender === 'ai' ? 'robot' : 'user'}"></i>
            </div>
            <div class="message-content">
                <div class="message-text">
                    <p>${this.formatMessage(text)}</p>
                </div>
                <div class="message-time">${timeString}</div>
            </div>
        `;

        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }

    saveHistory() {
        try {
            sessionStorage.setItem('chatHistory', JSON.stringify(this.conversationHistory));
        } catch (e) {
            console.error('Failed to save chat history:', e);
        }
    }

    loadHistory() {
        try {
            const savedHistory = sessionStorage.getItem('chatHistory');
            if (savedHistory) {
                this.conversationHistory = JSON.parse(savedHistory);

                // If we have history, don't show welcome message again
                if (this.conversationHistory.length > 0) {
                    this.hasShownWelcome = true;

                    // Replay messages to UI
                    this.conversationHistory.forEach(msg => {
                        if (msg.role === 'user') {
                            this.appendMessageToUI(msg.content, 'user');
                        } else if (msg.role === 'assistant') {
                            this.appendMessageToUI(msg.content, 'ai');
                        }
                    });
                }
            }
        } catch (e) {
            console.error('Failed to load chat history:', e);
        }
    }

    formatMessage(text) {
        // Simple formatting for URLs and basic markdown
        return text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // Bold
            .replace(/\*(.*?)\*/g, '<em>$1</em>') // Italic
            .replace(/\n/g, '<br>') // Line breaks
            .replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>'); // URLs
    }

    showTypingIndicator(show) {
        this.isTyping = show;
        this.typingIndicator.classList.toggle('active', show);
        this.scrollToBottom();
    }

    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }
}

// Initialize the chat when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    window.chatBot = new ChatBot();
    // Note: do not auto-open the chat on page load. The chat will open when the user clicks the toggle.
    // Add a subtle pulse to the toggle button once (non-intrusive) to indicate availability
    const toggleButton = document.getElementById('chatToggle');
    if (toggleButton) {
        toggleButton.classList.add('pulse');
        setTimeout(() => toggleButton.classList.remove('pulse'), 3000);
    }
});

// Close chat when pressing Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        const chatWidget = document.getElementById('aiChatWidget');
        if (chatWidget && chatWidget.classList.contains('active')) {
            chatWidget.classList.remove('active');
            chatWidget.style.opacity = '0';
            chatWidget.style.transform = 'translateY(100px) scale(0.9)';
            setTimeout(() => {
                chatWidget.style.display = 'none';
            }, 300);
        }
    }
});
