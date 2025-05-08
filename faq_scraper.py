from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time

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

# 📌 Cấu hình Chrome
options = Options()
options.add_argument("--headless")  # Chạy ẩn không hiển thị trình duyệt
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

# Khởi động trình duyệt
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

faq_data = []

# 📌 Lặp qua từng trang FAQ để lấy dữ liệu
for url in faq_urls:
    driver.get(url)
    
    # Chờ tối đa 10 giây để đảm bảo trang tải xong
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".views-row")))

    # 📌 Lấy danh sách câu hỏi
    questions = driver.find_elements(By.CSS_SELECTOR, ".views-accordion-header p")

    # 📌 Lấy danh sách câu trả lời (lấy tất cả nội dung trong div chứa câu trả lời)
    answers = []
    answer_elements = driver.find_elements(By.CSS_SELECTOR, ".views-row > div:nth-of-type(2)")

    for element in answer_elements:
        full_text = element.get_attribute("innerHTML").strip().replace("\n", " ")
        # Xóa các thẻ HTML không cần thiết
        clean_text = ' '.join(full_text.split()).replace("<br>", " ")
        answers.append(clean_text)

    # 📌 Xử lý nếu số lượng câu hỏi không khớp số lượng câu trả lời
    min_len = min(len(questions), len(answers))
    if min_len == 0:
        print(f"🚨 Không tìm thấy dữ liệu tại {url}")
        continue
    
    for i in range(min_len):
        faq_data.append({
            "question": questions[i].text.strip(),
            "answer": answers[i],
            "source": url
        })

    print(f"✅ Lấy dữ liệu thành công từ {url} ({min_len} mục).")

# 📌 Đóng trình duyệt
driver.quit()

# 📌 Kiểm tra dữ liệu trước khi lưu
if len(faq_data) > 0:
    # 📌 Lưu dữ liệu vào file Excel
    df = pd.DataFrame(faq_data)
    df.to_excel("faq_all_pages.xlsx", index=False, engine="openpyxl")
    print(f"✅ Đã lưu {len(faq_data)} FAQ vào 'faq_all_pages.xlsx'")
else:
    print("🚨 Không có dữ liệu nào được lưu. Kiểm tra lại CSS Selector hoặc mạng!")
