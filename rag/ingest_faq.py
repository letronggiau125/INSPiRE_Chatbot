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

    # Prepare data for upsert - filter out entries without answers
    ids, docs, metas = [], [], []
    for idx, item in enumerate(faqs):
        # Skip entries without answers
        if not item.get("answer"):
            continue
            
        # Ensure answer is a string
        answer = str(item.get("answer", ""))
        if not answer.strip():
            continue
            
        ids.append(str(idx))
        docs.append(answer)
        metas.append({
            "question": str(item.get("question", "")),
            "category": str(item.get("category", "")),
            **{f"alias_{i}": str(a) for i, a in enumerate(item.get("aliases", []))}
        })

    if not docs:
        print("No valid FAQ entries found to ingest")
        return

    try:
        # Compute embeddings using the new EmbeddingModel
        embeddings = embedding_model.get_batch_embeddings(docs).tolist()

        # Verify lengths match before upserting
        if len(ids) != len(embeddings):
            print(f"Warning: Number of IDs ({len(ids)}) doesn't match number of embeddings ({len(embeddings)})")
            # Use the minimum length to ensure consistency
            min_len = min(len(ids), len(embeddings))
            ids = ids[:min_len]
            docs = docs[:min_len]
            metas = metas[:min_len]
            embeddings = embeddings[:min_len]

        # Upsert documents with explicit embeddings
        coll.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metas,
            documents=docs
        )

        print(f"✅ Ingest completed. Processed {len(ids)} FAQ entries.")
    except Exception as e:
        print(f"Error during ingestion: {e}")
        raise


if __name__ == "__main__":
    ingest_all_faqs()