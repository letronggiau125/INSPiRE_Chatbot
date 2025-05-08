import pandas as pd

# 📌 Đọc dữ liệu từ file Excel
file_path = "faq_all_pages.xlsx"  # Đảm bảo đường dẫn đúng
df = pd.read_excel(file_path, engine="openpyxl")

# 📌 Kiểm tra số lượng dòng dữ liệu
print(f"📊 Tổng số FAQ thu thập được: {len(df)}\n")

# 📌 Hiển thị 5 dòng đầu tiên
print("🔍 5 dòng dữ liệu đầu tiên:")
print(df.head())

# 📌 Kiểm tra xem có cột nào bị thiếu dữ liệu không
missing_data = df.isnull().sum()
print("\n🛠 Kiểm tra dữ liệu bị thiếu:")
print(missing_data)

# 📌 Kiểm tra số lượng FAQ theo nguồn (URL)
print("\n📌 Số lượng FAQ theo từng nguồn:")
print(df["source"].value_counts())

# 📌 Kiểm tra xem có câu hỏi hoặc câu trả lời nào bị trùng không
duplicate_questions = df.duplicated(subset=["question"]).sum()
duplicate_answers = df.duplicated(subset=["answer"]).sum()

print(f"\n🔄 Số câu hỏi bị trùng: {duplicate_questions}")
print(f"🔄 Số câu trả lời bị trùng: {duplicate_answers}")

# 📌 Hiển thị một số câu hỏi bị trùng (nếu có)
if duplicate_questions > 0:
    print("\n🔍 Các câu hỏi bị trùng:")
    print(df[df.duplicated(subset=["question"], keep=False)][["question", "source"]].head(10))

# 📌 Hiển thị một số câu trả lời bị trùng (nếu có)
if duplicate_answers > 0:
    print("\n🔍 Các câu trả lời bị trùng:")
    print(df[df.duplicated(subset=["answer"], keep=False)][["answer", "source"]].head(10))
