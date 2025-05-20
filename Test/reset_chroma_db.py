from reflection import Reflection
import os
import shutil

db_path = "./chroma_db"

# 🛠 Xóa dữ liệu cũ nếu có (Hỗ trợ Windows)
if os.path.exists(db_path):
    shutil.rmtree(db_path)

# 🔥 Tạo lại Reflection
chatbot = Reflection(db_path=db_path)

# ✅ Kiểm tra phương thức thêm dữ liệu
if hasattr(chatbot, "add_document"):  # Kiểm tra nếu có phương thức `add_document`
    faq_data = [
        {"question": "Làm sao để gia hạn phòng?", "answer": "Bạn có thể gia hạn phòng nếu không có người đặt tiếp theo."},
        {"question": "Tôi có thể đăng ký phòng ở đâu?", "answer": "Bạn có thể đăng ký phòng trên website thư viện."},
    ]

    # 🔄 Thêm dữ liệu vào ChromaDB
    for item in faq_data:
        chatbot.add_document(item["question"], item["answer"])  # Nếu có phương thức này
    print("✅ Dữ liệu đã cập nhật lại!")

else:
    print("❌ Lỗi: `Reflection` không có phương thức `add_document`. Hãy kiểm tra `reflection.py`!")
