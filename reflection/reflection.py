import chromadb
import pandas as pd
import json
import uuid
from typing import Optional

class Reflection:
    def __init__(self, db_path="./chroma_db", history_collection="chat_history", faq_collection="faq_data"):
        """
        Khởi tạo Reflection với cơ sở dữ liệu vector (ChromaDB)
        """
        # Khởi tạo ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Tạo collection cho lịch sử chat
        self.chat_history_collection = self.client.get_or_create_collection(name=history_collection)
        
        # Tạo collection cho dữ liệu FAQ
        self.faq_collection = self.client.get_or_create_collection(name=faq_collection)

        # Nạp dữ liệu FAQ vào ChromaDB nếu chưa có
        self._load_faq_data("faq_all_pages.xlsx")

    def _load_faq_data(self, faq_file):
        """
        Nạp dữ liệu FAQ từ file Excel vào ChromaDB nếu chưa có
        """
        try:
            faq_df = pd.read_excel(faq_file)
            if self.faq_collection.count() == 0:  # Tránh nhập lại dữ liệu nếu đã tồn tại
                for _, row in faq_df.iterrows():
                    self.faq_collection.add(
                        ids=[str(uuid.uuid4())],
                        documents=[json.dumps({
                            "question": row["question"],
                            "answer": row["answer"],
                            "source": row.get("source", "N/A")  # Thêm giá trị mặc định nếu thiếu
                        })]
                    )
                print("✅ Dữ liệu FAQ đã được nạp vào ChromaDB!")
            else:
                print("⚡ Dữ liệu FAQ đã tồn tại, không cần nhập lại.")
        except Exception as e:
            print(f"⚠️ Lỗi khi nạp dữ liệu FAQ: {e}")

    def load_faq_data(self):
        """
        Phương thức công khai để nạp lại dữ liệu FAQ (nếu cần).
        """
        print("🔄 Đang nạp lại dữ liệu FAQ...")
        self._load_faq_data("faq_all_pages.xlsx")
        print("✅ Dữ liệu FAQ đã được cập nhật!")

    def __record_chat_history__(self, session_id: str, human_message: str, ai_response: str):
        """
        Ghi lại lịch sử hội thoại của người dùng vào ChromaDB
        """
        self.chat_history_collection.add(
            ids=[str(uuid.uuid4())],
            documents=[json.dumps({
                "session_id": session_id,
                "human_message": human_message,
                "ai_response": ai_response
            })]
        )

    def query_faq(self, query: str) -> Optional[str]:
        """
        Truy vấn dữ liệu FAQ để tìm câu trả lời phù hợp nhất.
        """
        try:
            search_results = self.faq_collection.query(
                query_texts=[query],
                n_results=3  # 📌 Trả về nhiều kết quả để debug
            )

            if not search_results['documents']:
                print("❌ Không tìm thấy dữ liệu phù hợp trong FAQ.")
                return None
            
            for idx, doc in enumerate(search_results['documents'][0]):
                faq_data = json.loads(doc)
                print(f"🔍 Kết quả {idx + 1}: {faq_data['question']} → {faq_data['answer']}")

            best_match = json.loads(search_results['documents'][0][0])
            return best_match.get("answer", "Xin lỗi, tôi không tìm thấy thông tin phù hợp.")
        
        except Exception as e:
            print(f"⚠️ Lỗi khi truy vấn FAQ: {e}")
            return None
    
    def chat(self, session_id: str, user_message: str):
        """
        Chat với người dùng: Trả lời từ FAQ trước, nếu không có thì dùng LLM.
        """
        # 🔍 Kiểm tra FAQ trước
        faq_answer = self.query_faq(user_message)
        if faq_answer:
            return faq_answer

        # ⚡ Nếu không tìm thấy trong FAQ, gọi AI model (LLM) để trả lời
        ai_response = "Xin lỗi, tôi chưa có câu trả lời chính xác. Bạn có thể liên hệ thư viện để biết thêm chi tiết."
        
        # 📝 Ghi lại hội thoại vào ChromaDB
        self.__record_chat_history__(session_id, user_message, ai_response)
        
        return ai_response
