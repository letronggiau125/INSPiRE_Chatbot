import os
import json
import chromadb
from dotenv import load_dotenv

load_dotenv()

DB_PATH         = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = "tdtu_faq"

# Thay đổi đây:
# FAQ_FILE = os.path.join(os.path.dirname(__file__), "..", "faq.json")

# Thành:
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
FAQ_FILE = os.path.join(BASE_DIR, "faq_data.json")

def ingest_all_faqs():
    from chromadb.utils import embedding_functions
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-ada-002"
    )
    client = chromadb.PersistentClient(path=DB_PATH)
    coll   = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)

    if not os.path.isfile(FAQ_FILE):
        raise FileNotFoundError(f"Không tìm thấy faq.json tại {FAQ_FILE}")

    with open(FAQ_FILE, "r", encoding="utf-8") as f:
        faqs = json.load(f)

    for idx, item in enumerate(faqs):
        coll.upsert(
            documents=[item["answer"]],
            metadatas=[{
                "question": item["question"],
                "category": item["category"],
                **{f"alias_{i}": a for i, a in enumerate(item.get("aliases", []))}
            }],
            ids=[str(idx)]
        )

    print("✅ Ingest completed.")

if __name__ == "__main__":
    ingest_all_faqs()
