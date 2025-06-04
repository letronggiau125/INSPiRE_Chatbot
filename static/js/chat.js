// chat.js

// Configuration
const API_ENDPOINT = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
  ? 'http://127.0.0.1:5000/chat'
  : '/chat';  // For production, use relative path

// Theme configuration
const THEME_KEY = 'tdtu_chat_theme';
const THEME_TOGGLE = document.getElementById('theme-toggle');
const THEME_ICON = THEME_TOGGLE.querySelector('i');

// Debounce utility
function debounce(fn, delay) {
  let timeout;
  return (...args) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => fn(...args), delay);
  };
}

class ChatApp {
  constructor() {
    this.userInput = document.getElementById("user-input");
    this.sendButton = document.getElementById("send-button");
    this.chatBox = document.getElementById("chat-box");
    this.charCounter = document.querySelector(".char-counter");
    this.typingIndicator = document.getElementById("typing-indicator");
    this.errorNotice = document.getElementById("error-notice");
    this.feedbackBox = document.getElementById("feedback-box");
    this.feedbackBtns = document.querySelectorAll(".feedback-btn");

    this.isProcessing = false;
    this.currentTheme = localStorage.getItem(THEME_KEY) || 'light';

    // Clear chat history on initialization
    this.clearChatHistory();
    
    this.initializeEventListeners();
    this.initializeTheme();
  }

  initializeTheme() {
    document.documentElement.setAttribute('data-theme', this.currentTheme);
    THEME_ICON.className = this.currentTheme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
  }

  toggleTheme() {
    this.currentTheme = this.currentTheme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', this.currentTheme);
    localStorage.setItem(THEME_KEY, this.currentTheme);
    THEME_ICON.className = this.currentTheme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
  }

  initializeEventListeners() {
    // Theme toggle
    THEME_TOGGLE.addEventListener('click', () => this.toggleTheme());

    // Gửi khi Enter
    this.userInput.addEventListener("keypress", (event) => {
      if (event.key === "Enter" && !this.isProcessing) {
        event.preventDefault();
        this.sendMessage();
      }
    });

    // Send button click
    this.sendButton.addEventListener("click", () => {
      if (!this.isProcessing) {
        this.sendMessage();
      }
    });

    // Cập nhật char-counter
    this.userInput.addEventListener(
      "input",
      debounce(() => this.updateCharCounter(), 100)
    );

    // Bật/tắt nút Gửi
    this.userInput.addEventListener("input", () => {
      this.sendButton.disabled = this.userInput.value.trim() === "";
    });

    // Suggested questions
    document.querySelectorAll(".suggested-question").forEach((button) => {
      button.addEventListener("click", () => {
        this.userInput.value = button.textContent;
        this.updateCharCounter();
        this.sendMessage();
      });
    });

    // Global click để focus input
    document.addEventListener("click", (e) => {
      if (!e.target.closest(".input-container")) {
        this.focusInput();
      }
    });

    // Feedback buttons
    this.feedbackBtns.forEach((btn) => {
      btn.addEventListener("click", (e) => {
        const type = e.currentTarget.dataset.feedback;
        this.handleFeedback(type);
      });
    });

    // Khi focus input, ẩn error notice
    this.userInput.addEventListener("focus", () => {
      this.errorNotice.classList.add("hidden");
    });
  }

  handleFeedback(type) {
    const feedbackMessage = type === "thumbs-up"
      ? "Cảm ơn bạn đã đánh giá hữu ích!"
      : "Cảm ơn phản hồi. Chúng tôi sẽ cải thiện.";
    
    // Create a temporary notification
    const notification = document.createElement('div');
    notification.className = 'feedback-notification';
    notification.textContent = feedbackMessage;
    document.body.appendChild(notification);

    // Animate and remove the notification
    setTimeout(() => {
      notification.classList.add('show');
      setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => notification.remove(), 300);
      }, 2000);
    }, 100);

    this.hideFeedback();
  }

  formatTimestamp(date) {
    return date.toLocaleTimeString("vi-VN", {
      hour: "2-digit",
      minute: "2-digit",
    });
  }

  updateCharCounter() {
    if (!this.charCounter) return;  // Skip if element doesn't exist
    const length = this.userInput.value.length;
    this.charCounter.textContent = `${length}/500`;
    this.charCounter.style.color =
      length >= 450 ? "var(--error-color)" : "var(--muted-color)";
  }

  // createMessageElement(text, isUser = false, htmlContent = "") {
  //   const container = document.createElement("div");
  //   container.className = "message-container";
  //   const messageDiv = document.createElement("div");
  //   messageDiv.className = isUser ? "message user-message" : "message bot-message";

  //   if (isUser) {
  //     messageDiv.textContent = text;
  //   } else {
  //     if (htmlContent) {
  //       messageDiv.innerHTML = htmlContent;
  //     } else {
  //       // Convert line breaks to <br> tags and preserve formatting
  //       const formattedText = text
  //         .replace(/\n/g, '<br>')  // Convert newlines to <br>
  //         .replace(/(\d+\.)/g, '<br>$1')  // Add line break before numbered items
  //         .replace(/^<br>/, '');  // Remove leading <br>
        
  //       messageDiv.innerHTML = marked.parse(formattedText);
  //     }
  //   }

  //   const timestampEl = document.createElement("div");
  //   timestampEl.className = isUser ? "timestamp user-timestamp" : "timestamp bot-timestamp";
  //   timestampEl.textContent = this.formatTimestamp(new Date());

  //   container.appendChild(messageDiv);
  //   return { container, timestampEl };
  // }

  createMessageElement(text, isUser = false, htmlContent = "") {
    // Tạo container cha
    const container = document.createElement("div");
    container.className = "message-container";
  
    // Tạo bubble message
    const messageDiv = document.createElement("div");
    messageDiv.className = isUser ? "message user-message" : "message bot-message";
  
    if (isUser) {
      // Với tin nhắn của user: hiển thị nguyên văn text, giữ newline
      messageDiv.textContent = text;
    } else {
      if (htmlContent) {
        // Nếu backend đã trả về HTML-safe (ví dụ đã parse Markdown ở server),
        // dùng thẳng innerHTML
        messageDiv.innerHTML = htmlContent;
      } else {
        // Coi `text` là Markdown chuẩn, parse trực tiếp
        messageDiv.innerHTML = marked.parse(text);
      }
    }
  
    // Tạo timestamp
    const timestampEl = document.createElement("div");
    timestampEl.className = isUser ? "timestamp user-timestamp" : "timestamp bot-timestamp";
    timestampEl.textContent = this.formatTimestamp(new Date());
  
    container.appendChild(messageDiv);
    return { container, timestampEl };
  }
   
  showTypingIndicator() {
    this.typingIndicator.classList.remove("hidden");
  }

  hideTypingIndicator() {
    this.typingIndicator.classList.add("hidden");
  }

  showFeedback() {
    this.feedbackBox.classList.remove("hidden");
  }

  hideFeedback() {
    this.feedbackBox.classList.add("hidden");
  }

  showError(message) {
    this.errorNotice.textContent = message;
    this.errorNotice.classList.remove("hidden");
  }

  scrollToBottom() {
    this.chatBox.scrollTop = this.chatBox.scrollHeight;
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

    const userMsg = this.createMessageElement(message, true);
    this.chatBox.appendChild(userMsg.container);
    this.chatBox.appendChild(userMsg.timestampEl);
    this.scrollToBottom();

    this.userInput.value = "";
    this.updateCharCounter();

    this.showTypingIndicator();

    try {
      const response = await fetch(API_ENDPOINT, {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          "Accept": "application/json"
        },
        body: JSON.stringify({
          session_id: "test",
          message: message,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log("Server response:", data); // Debug log
      this.hideTypingIndicator();

      if (data.error) {
        this.showError(data.error);
      } else if (data.message || data.answer || data.response) {  // Added data.response check
        const responseText = data.message || data.answer || data.response;  // Use whichever is available
        console.log("Using response text:", responseText); // Debug log
        const botMsg = this.createMessageElement(responseText, false);
        this.chatBox.appendChild(botMsg.container);
        this.chatBox.appendChild(botMsg.timestampEl);
        this.scrollToBottom();
        setTimeout(() => this.showFeedback(), 200);
      } else {
        console.error("Invalid response data:", data); // Debug log
        throw new Error("Invalid response format from server");
      }
    } catch (error) {
      console.error("Error:", error);
      this.hideTypingIndicator();
      this.showError("Không thể kết nối với máy chủ. Vui lòng thử lại.");
    } finally {
      this.isProcessing = false;
      this.sendButton.disabled = false;
      this.userInput.disabled = false;
      this.focusInput();
    }
  }

  // Add method to clear chat history
  clearChatHistory() {
    // Keep only the welcome message
    const welcomeMessage = this.chatBox.querySelector('.welcome-message');
    this.chatBox.innerHTML = '';
    if (welcomeMessage) {
      this.chatBox.appendChild(welcomeMessage);
    }
  }
}

window.addEventListener("DOMContentLoaded", () => {
  new ChatApp();
});
  