import os
import json
import shutil
import chromadb
from dotenv import load_dotenv
from embedding_model.embedding import embedding_model

# Load environment variables
load_dotenv()

DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = "tdtu_faq"

# Path to FAQ data file
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
FAQ_FILE = os.path.join(BASE_DIR, "faq_data.json")


def ingest_all_faqs():
    """Ingest all FAQ entries into ChromaDB, recreating DB if schema mismatch."""
    # Remove old database to avoid schema mismatches
    if os.path.exists(DB_PATH):
        print(f"⚠️ Removing existing ChromaDB at {DB_PATH} to rebuild schema...")
        try:
            shutil.rmtree(DB_PATH)
        except Exception as e:
            print(f"Error removing old DB: {e}")

    # Initialize Chromadb client and collection (no embedding function passed)
    client = chromadb.PersistentClient(path=DB_PATH)
    coll = client.get_or_create_collection(name=COLLECTION_NAME)

    # Verify FAQ file exists
    if not os.path.isfile(FAQ_FILE):
        raise FileNotFoundError(f"FAQ file not found at {FAQ_FILE}")

    # Load FAQ data
    with open(FAQ_FILE, "r", encoding="utf-8") as f:
        faqs = json.load(f)

    # Prepare data for upsert
    ids, docs, metas = [], [], []
    for idx, item in enumerate(faqs):
        ids.append(str(idx))
        docs.append(item.get("answer", ""))
        metas.append({
            "question": item.get("question", ""),
            "category": item.get("category", ""),
            **{f"alias_{i}": a for i, a in enumerate(item.get("aliases", []))}
        })

    # Compute embeddings using the new EmbeddingModel
    embeddings = embedding_model.get_batch_embeddings(docs).tolist()

    # Upsert documents with explicit embeddings
    coll.upsert(
        ids=ids,
        embeddings=embeddings,
        metadatas=metas,
        documents=docs
    )

    print("✅ Ingest completed.")


if __name__ == "__main__":
    ingest_all_faqs()