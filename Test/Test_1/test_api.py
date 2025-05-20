import requests

# 🔗 Cấu hình URL API
url = "http://172.16.69.118:5001/chat"
headers = {"Content-Type": "application/json"}

# 1️⃣ Kiểm tra với phương thức GET
print("🟡 Gửi yêu cầu GET...")
get_response = requests.get(url)
print("📡 Status Code (GET):", get_response.status_code)

try:
    print("📩 Response JSON (GET):", get_response.json())
except Exception as e:
    print("❌ Không thể parse JSON:", e)
    print("🔍 Raw Response (GET):", get_response.text)

# 2️⃣ Kiểm tra với phương thức POST với nhiều tin nhắn
test_messages = [
    "CD/DVD kèm sách và DVD phim có cho mượn về nhà không?",
    "Quy định về số lượng tài liệu và thời gian mượn tài liệu đọc tại chỗ?",
    "Quy định về số ngày tăng thêm khi thực hiện lệnh gia hạn tài liệu cho mượn về?",
    "Cách tạo mật khẩu cá nhân?"
]

for msg in test_messages:
    print(f"\n🟢 Gửi: {msg}")
    post_response = requests.post(url, json={"message": msg}, headers=headers)
    print("📡 Status Code (POST):", post_response.status_code)

    try:
        print("📩 Response JSON (POST):", post_response.json())
    except Exception as e:
        print("❌ Không thể parse JSON:", e)
        print("🔍 Raw Response (POST):", post_response.text)
