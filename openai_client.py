import openai
import os
from dotenv import load_dotenv

# 🌟 Load API key từ biến môi trường `.env`
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class OpenAIClient:
    def __init__(self, api_key=OPENAI_API_KEY):
        if not api_key:
            raise ValueError("⚠️ API Key không hợp lệ. Vui lòng kiểm tra file .env!")
        self.api_key = api_key

    def chat(self, messages):
        """
        Gửi đoạn hội thoại đến OpenAI và nhận phản hồi.
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                api_key=self.api_key
            )
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"⚠️ Lỗi khi kết nối OpenAI: {e}"
        
