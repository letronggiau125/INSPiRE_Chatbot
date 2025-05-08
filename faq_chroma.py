import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd

# 1️⃣ Khởi tạo ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_faq_db")

# 2️⃣ Tạo Collection để lưu dữ liệu FAQ
collection = chroma_client.get_or_create_collection(name="faq_collection")

# 3️⃣ Load dữ liệu từ file Excel
df = pd.read_excel("faq_all_pages.xlsx")

# 4️⃣ Dùng mô hình tạo vector cho câu hỏi
model = SentenceTransformer("all-MiniLM-L6-v2")

# 5️⃣ Thêm dữ liệu vào ChromaDB
for index, row in df.iterrows():
    question = row["question"]
    answer = row["answer"]
    vector = model.encode(question).tolist()  # Chuyển câu hỏi thành vector

    collection.add(
        ids=[str(index)],  # ID duy nhất cho mỗi câu hỏi
        embeddings=[vector],  # Vector embedding của câu hỏi
        metadatas=[{"question": question, "answer": answer}]
    )


print("✅ Dữ liệu đã được lưu vào ChromaDB!")

def search_faq(query, top_k=3):
    query_vector = model.encode(query).tolist()  # Chuyển câu hỏi thành vector
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k  # Số kết quả gần nhất cần tìm
    )

    print("\n📌 Kết quả tìm kiếm:")
    for i in range(len(results["ids"][0])):
        print(f"🔹 Câu hỏi: {results['metadatas'][0][i]['question']}")
        print(f"✅ Trả lời: {results['metadatas'][0][i]['answer']}\n")

# 🛠 Kiểm tra tìm kiếm với một câu hỏi
search_faq("Làm thế nào để mượn tài liệu?")
