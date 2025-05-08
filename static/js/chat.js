class ChatApp {
    constructor() {
        this.userInput = document.getElementById("user-input");
        this.sendButton = document.getElementById("send-button");
        this.chatBox = document.getElementById("chat-box");
        this.charCounter = document.querySelector(".char-counter");
        this.isProcessing = false;
        
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        // Input event listeners
        this.userInput.addEventListener("keypress", (event) => {
            if (event.key === "Enter" && !this.isProcessing) {
                event.preventDefault();
                this.sendMessage();
            }
        });

        this.userInput.addEventListener("input", () => this.updateCharCounter());

        // Suggested questions
        document.querySelectorAll('.suggested-question').forEach(button => {
            button.addEventListener('click', () => {
                this.userInput.value = button.textContent;
                this.updateCharCounter();
                this.sendMessage();
            });
        });

        // Send button click event
        this.sendButton.addEventListener('click', () => {
            if (!this.isProcessing) {
                this.sendMessage();
            }
        });

        // Global click handler for input focus
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.input-container')) {
                this.focusInput();
            }
        });
    }

    formatTimestamp(date) {
        return date.toLocaleTimeString('vi-VN', { 
            hour: '2-digit', 
            minute: '2-digit'
        });
    }

    updateCharCounter() {
        const length = this.userInput.value.length;
        this.charCounter.textContent = `${length}/500`;
        this.charCounter.style.color = length >= 450 ? 'var(--error-color)' : 'var(--counter-color)';
    }

    createMessageElement(message, isUser = false) {
        const container = document.createElement('div');
        container.className = 'message-container';

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
        
        if (!isUser) {
            messageDiv.innerHTML = marked.parse(message);
        } else {
            messageDiv.textContent = message;
        }

        const timestamp = document.createElement('div');
        timestamp.className = `timestamp ${isUser ? 'user-timestamp' : 'bot-timestamp'}`;
        timestamp.textContent = this.formatTimestamp(new Date());

        container.appendChild(messageDiv);
        container.appendChild(timestamp);
        return container;
    }

    showLoading() {
        const loading = document.createElement('div');
        loading.className = 'message bot-message';
        loading.innerHTML = 'Đang xử lý<div class="loading"><span></span><span></span><span></span></div>';
        this.chatBox.appendChild(loading);
        this.chatBox.scrollTop = this.chatBox.scrollHeight;
        return loading;
    }

    showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = message;
        this.chatBox.appendChild(errorDiv);
        this.chatBox.scrollTop = this.chatBox.scrollHeight;
        
        setTimeout(() => {
            errorDiv.style.opacity = '0';
            setTimeout(() => errorDiv.remove(), 300);
        }, 5000);
    }

    focusInput() {
        this.userInput.focus();
    }

    async sendMessage() {
        const message = this.userInput.value.trim();
        if (!message || this.isProcessing) return;

        this.isProcessing = true;
        this.sendButton.disabled = true;
        this.userInput.disabled = true;

        this.chatBox.appendChild(this.createMessageElement(message, true));
        this.userInput.value = '';
        this.updateCharCounter();

        const loadingDiv = this.showLoading();

        try {
            const response = await fetch("http://127.0.0.1:5001/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ 
                    session_id: "test", 
                    message: message 
                })
            });

            const data = await response.json();
            loadingDiv.remove();

            if (data.error) {
                this.showError(data.error);
            } else {
                this.chatBox.appendChild(this.createMessageElement(data.response));
            }
        } catch (error) {
            loadingDiv.remove();
            this.showError("Không thể kết nối với máy chủ. Vui lòng thử lại sau.");
        } finally {
            this.isProcessing = false;
            this.sendButton.disabled = false;
            this.userInput.disabled = false;
            this.focusInput();
            this.chatBox.scrollTop = this.chatBox.scrollHeight;
        }
    }
}

// Initialize the chat application when the DOM is loaded
window.addEventListener('DOMContentLoaded', () => {
    const chatApp = new ChatApp();
}); 