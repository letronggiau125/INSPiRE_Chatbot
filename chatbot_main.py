import requests

# 🔹 URL API của Flask (chạy `serve.py` trước khi gọi API này)
# API_URL = "http://127.0.0.1:5001/chat"
API_URL = "http://172.16.69.118:5001/chat"

def chat_with_bot():
    """
    Chương trình CLI đơn giản để tương tác với chatbot qua API Flask.
    """
    print("🤖 Chatbot Thư viện Đại học Tôn Đức Thắng - TDTU 🤖")
    print("Nhập 'exit' để kết thúc chat.")

    session_id = "user_123"  # Mã định danh phiên chat

    while True:
        user_query = input("👨‍🎓 Bạn: ")
        if user_query.lower() == "exit":
            print("👋 Tạm biệt! Hẹn gặp lại bạn sau.")
            break
        
        # 🔹 Gửi yêu cầu đến API Flask
        response = requests.post(API_URL, json={
            "session_id": session_id,
            "query": user_query
        })

        # 🔹 Xử lý phản hồi
        if response.status_code == 200:
            bot_reply = response.json().get("content", "❌ Lỗi: Không nhận được phản hồi.")
            print(f"🤖 Bot: {bot_reply}")
        else:
            print(f"❌ Lỗi: Không thể kết nối API ({response.status_code})")

if __name__ == "__main__":
    chat_with_bot()
