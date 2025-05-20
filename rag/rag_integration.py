# rag_integration.py
import os
from langdetect import detect
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from openai import OpenAI

from rag.ingest_faq import DB_PATH, COLLECTION_NAME

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize OpenAI embedding function
ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-ada-002"
)

# Create client + collection
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)

class RAG:
    def __init__(self, collection):
        self.collection = collection
        self.client = client

    def hybrid_search(self, query: str, limit=5):
        # 1) Perform vector search
        vec_res = self.collection.query(
            query_texts=[query],
            n_results=limit
        )
        
        # 2) Perform keyword search
        kw_res = self.collection.query(
            query_texts=[query],
            n_results=limit
        )

        # 3) Build unified list
        docs = []
        for source in (vec_res, kw_res):
            ids = source["ids"][0]
            meta = source["metadatas"][0]
            dists = source["distances"][0]
            for i, doc_id in enumerate(ids):
                docs.append({
                    "id": doc_id,
                    "question": meta[i].get("question"),
                    "answer": meta[i].get("answer", None),
                    "distance": dists[i]
                })

        # 4) Deduplicate and sort by distance
        seen, fused = set(), []
        for d in sorted(docs, key=lambda x: x["distance"]):
            if d["id"] not in seen:
                fused.append(d)
                seen.add(d["id"])
        return fused[:limit]

    def enhance_prompt(self, query):
        hits = self.hybrid_search(query)
        prompt = ""
        for h in hits:
            prompt += f"QUESTION: {h['question']}\nANSWER: {h['answer']}\n---\n"
        return prompt
