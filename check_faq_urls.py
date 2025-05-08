import requests

# 📌 Danh sách URL FAQ (Tiếng Việt & Tiếng Anh)
faq_urls = [
    # 🌍 Tiếng Việt
    "https://lib.tdtu.edu.vn/dich-vu/tham-khao/faq-vi-tri",
    "https://lib.tdtu.edu.vn/dich-vu/tham-khao/faq-luu-hanh-tai-lieu",
    "https://lib.tdtu.edu.vn/dich-vu/tham-khao/faq-su-dung-may-tinh",
    "https://lib.tdtu.edu.vn/dich-vu/tham-khao/faq-dat-muon-su-dung-phong",
    "https://lib.tdtu.edu.vn/dich-vu/tham-khao/faq-in-photocopy-scan",
    "https://lib.tdtu.edu.vn/dich-vu/tham-khao/faq-lop-ky-nang",
    "https://lib.tdtu.edu.vn/dich-vu/tham-khao/faq-the-tai-khoan",
    "https://lib.tdtu.edu.vn/dich-vu/tham-khao/faq-kiem-tra-trung-lap-ithenticate",
    "https://lib.tdtu.edu.vn/dich-vu/tham-khao/faq-kiem-tra-dao-van-turnitin",
    "https://lib.tdtu.edu.vn/dich-vu/tham-khao/faq-cau-hoi-khac",

    # 🌍 Tiếng Anh
    "https://lib.tdtu.edu.vn/en/services/reference/faq-location",
    "https://lib.tdtu.edu.vn/index.php/en/services/reference/faq-circulation",
    "https://lib.tdtu.edu.vn/en/services/reference/faq-computer",
    "https://lib.tdtu.edu.vn/index.php/en/services/reference/faq-room-reservation",
    "https://lib.tdtu.edu.vn/index.php/en/services/reference/faq-printing-photocopy-scan",
    "https://lib.tdtu.edu.vn/index.php/en/services/reference/faq-orientation",
    "https://lib.tdtu.edu.vn/en/services/reference/faq-card-account",
    "https://lib.tdtu.edu.vn/en/services/reference/faq-ithenticate",
    "https://lib.tdtu.edu.vn/en/services/reference/faq-turnitin",
    "https://lib.tdtu.edu.vn/en/services/reference/faq-others"
]

print("🔍 Bắt đầu kiểm tra tất cả URL...")

# 📌 Kiểm tra từng URL
for url in faq_urls:
    try:
        print(f"📡 Đang kiểm tra: {url}")
        response = requests.get(url, timeout=5)  # Gửi request với timeout 5 giây
        if response.status_code == 200:
            print(f"✅ {url} - Status Code: 200 (Hoạt động)")
        elif response.status_code == 404:
            print(f"❌ {url} - Status Code: 404 (Không tìm thấy)")
        else:
            print(f"⚠️ {url} - Status Code: {response.status_code} (Lỗi khác)")
    except requests.exceptions.RequestException as e:
        print(f"🚨 {url} - Lỗi: {e}")

print("✅ Hoàn thành kiểm tra tất cả URL!")
