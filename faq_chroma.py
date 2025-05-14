import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI embedding function
ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-ada-002"
)

# Initialize ChromaDB with OpenAI embeddings
chroma_client = chromadb.PersistentClient(path="./chroma_faq_db")

# Create Collection with OpenAI embeddings
collection = chroma_client.get_or_create_collection(
    name="faq_collection",
    embedding_function=ef
)

# Load data from Excel file
df = pd.read_excel("faq_all_pages.xlsx")

# Add data to ChromaDB
for index, row in df.iterrows():
    question = row["question"]
    answer = row["answer"]

    collection.add(
        ids=[str(index)],  # Unique ID for each question
        documents=[question],  # The question text
        metadatas=[{"question": question, "answer": answer}]
    )

print("✅ Data has been saved to ChromaDB!")

def search_faq(query, top_k=3):
    results = collection.query(
        query_texts=[query],
        n_results=top_k  # Number of closest results to find
    )

    print("\n📌 Search Results:")
    for i in range(len(results["ids"][0])):
        print(f"🔹 Question: {results['metadatas'][0][i]['question']}")
        print(f"✅ Answer: {results['metadatas'][0][i]['answer']}\n")

# Test search with a question
search_faq("Làm thế nào để mượn tài liệu?")
