import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("❌ Không tìm thấy API Key!")
else:
    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Ping"}]
        )
        print("✅ API hoạt động! Trả lời từ OpenAI:")
        print(response.choices[0].message.content)
    except Exception as e:
        print("❌ Lỗi khi gọi OpenAI:", e)