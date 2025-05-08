# import openai
# import os
# from dotenv import load_dotenv

# # 🌟 Load API key từ biến môi trường `.env`
# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# class OpenAIClient:
#     def __init__(self, api_key=OPENAI_API_KEY):
#         if not api_key:
#             raise ValueError("⚠️ API Key không hợp lệ. Vui lòng kiểm tra file .env!")
#         self.api_key = api_key

#     def chat(self, messages):
#         """
#         Gửi đoạn hội thoại đến OpenAI và nhận phản hồi.
#         """
#         try:
#             response = openai.ChatCompletion.create(
#                 model="gpt-3.5-turbo",
#                 messages=messages,
#                 api_key=self.api_key
#             )
#             return response["choices"][0]["message"]["content"].strip()
#         except Exception as e:
#             return f"⚠️ Lỗi khi kết nối OpenAI: {e}"
        
import openai
import os
from dotenv import load_dotenv

# 🌟 Load API key từ file .env hoặc biến môi trường
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class OpenAIClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("⚠️ API Key không hợp lệ. Vui lòng kiểm tra biến môi trường hoặc file .env!")

        # Thiết lập API key cho openai module
        openai.api_key = self.api_key

    def chat(self, messages):
        """
        Gửi đoạn hội thoại đến OpenAI và nhận phản hồi.
        :param messages: List[Dict[str, str]] ví dụ [{"role": "user", "content": "Xin chào"}]
        :return: Nội dung phản hồi từ chatbot
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"⚠️ Lỗi khi kết nối OpenAI: {e}"
