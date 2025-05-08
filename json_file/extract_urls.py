import json
import os

# Danh sách URL FAQ thủ công
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

# Tạo thư mục json_file nếu chưa tồn tại
output_dir = "json_file"
os.makedirs(output_dir, exist_ok=True)

# Lưu danh sách URL vào file JSON
output_path = os.path.join(output_dir, "faq_urls.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(faq_urls, f, indent=4, ensure_ascii=False)

print(f"✅ Đã lưu {len(faq_urls)} URL vào {output_path}")
