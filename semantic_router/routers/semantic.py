# class SemanticRouter:
#     def __init__(self, routes):
#         self.routes = routes

#     def guide(self, query):
#         for route in self.routes:
#             if query in route.utterances:
#                 return (query, route.name)
#         return (query, "unknown")

# import numpy as np
# from sentence_transformers import SentenceTransformer

# class SemanticRouter:
#     def __init__(self, routes):
#         self.routes = routes
#         self.embedding_model = SentenceTransformer("keepitreal/vietnamese-sbert")
#         self.routesEmbedding = {}
#         self.routesEmbeddingCal = {}

#         # Tạo vector embedding cho từng route
#         for route in self.routes:
#             print(f"🔹 Generating embeddings for route: {route.name}")
#             embeddings = self.embedding_model.encode(route.utterances)
#             self.routesEmbedding[route.name] = embeddings
#             self.routesEmbeddingCal[route.name] = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

#     def guide(self, query):
#         print(f"🔍 Processing query: {query}")
#         queryEmbedding = self.embedding_model.encode([query])
#         queryEmbedding = queryEmbedding / np.linalg.norm(queryEmbedding)
        
#         scores = []
#         for route in self.routes:
#             routeEmbeddingCal = self.routesEmbeddingCal[route.name]
#             score = np.mean(np.dot(routeEmbeddingCal, queryEmbedding.T).flatten())
#             print(f"📌 Score for {route.name}: {score}")
#             scores.append((score, route.name))
        
#         scores.sort(reverse=True)
#         best_match = scores[0]
#         print(f"✅ Best match: {best_match[1]} (Score: {best_match[0]})")
        
#         return best_match if best_match[0] > 0.5 else (query, "unknown")

# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.preprocessing import normalize  # Thêm thư viện normalize

# class SemanticRouter:
#     def __init__(self, routes, threshold=0.5):  # Thêm tham số threshold
#         self.routes = routes
#         self.threshold = threshold  # Ngưỡng điểm tối thiểu để xác định danh mục
#         self.embedding_model = SentenceTransformer("keepitreal/vietnamese-sbert")
#         self.routesEmbedding = {}
#         self.routesEmbeddingCal = {}

#         # Tạo vector embedding cho từng route
#         for route in self.routes:
#             print(f"🔹 Generating embeddings for route: {route.name}")
#             embeddings = self.embedding_model.encode(route.utterances)
#             self.routesEmbedding[route.name] = embeddings
#             self.routesEmbeddingCal[route.name] = normalize(embeddings, axis=1)  # Chuẩn hóa đúng cách

#     def guide(self, query):
#         print(f"🔍 Processing query: {query}")
#         queryEmbedding = self.embedding_model.encode([query])
#         queryEmbedding = normalize(queryEmbedding)  # Dùng sklearn để chuẩn hóa
        
#         scores = []
#         for route in self.routes:
#             routeEmbeddingCal = self.routesEmbeddingCal[route.name]
#             score = np.mean(np.dot(routeEmbeddingCal, queryEmbedding.T).flatten())
#             print(f"📌 Score for {route.name}: {score}")
#             scores.append((score, route.name))
        
#         # Sắp xếp điểm số giảm dần
#         scores.sort(reverse=True)
#         best_match = scores[0]

#         # Kiểm tra ngưỡng điểm
#         if best_match[0] < self.threshold:
#             print(f"⚠️ Best score {best_match[0]} is below threshold ({self.threshold}). Returning 'unknown'.")
#             return (query, "unknown")

#         print(f"✅ Best match: {best_match[1]} (Score: {best_match[0]})")
#         return best_match

# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.preprocessing import normalize

# class SemanticRouter:
#     def __init__(self, routes, threshold=0.3):  # Hạ ngưỡng xuống 0.3
#         self.routes = routes
#         self.threshold = threshold
#         self.embedding_model = SentenceTransformer("keepitreal/vietnamese-sbert")
#         self.routesEmbedding = {}
#         self.routesEmbeddingCal = {}

#         # Tạo vector embedding cho từng route
#         for route in self.routes:
#             print(f"🔹 Generating embeddings for route: {route.name}")
#             embeddings = self.embedding_model.encode(route.utterances)
#             self.routesEmbedding[route.name] = embeddings
#             self.routesEmbeddingCal[route.name] = normalize(embeddings, axis=1)  # Dùng normalize() chính xác hơn

#     def guide(self, query):
#         print(f"🔍 Processing query: {query}")
#         queryEmbedding = self.embedding_model.encode([query])
#         queryEmbedding = normalize(queryEmbedding)  # Dùng sklearn normalize

#         scores = []
#         for route in self.routes:
#             routeEmbeddingCal = self.routesEmbeddingCal[route.name]
#             score = np.mean(np.dot(routeEmbeddingCal, queryEmbedding.T).flatten())
#             print(f"📌 Score for {route.name}: {score}")
#             scores.append((score, route.name))

#         scores.sort(reverse=True)
#         best_match = scores[0]

#         # Kiểm tra ngưỡng
#         if best_match[0] < self.threshold:
#             print(f"⚠️ Best score {best_match[0]} is below threshold ({self.threshold}). Returning 'unknown'.")
#             return (query, "unknown")

#         print(f"✅ Best match: {best_match[1]} (Score: {best_match[0]})")
#         return best_match


import numpy as np
from typing import List
import os
from chromadb.utils import embedding_functions
from sklearn.preprocessing import normalize

class SemanticRouter:
    def __init__(self, routes: List, threshold: float = 0.3):
        self.routes = routes
        self.threshold = threshold
        self.ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-ada-002"
        )
        self.routesEmbedding = {}
        self.routesEmbeddingCal = {}

        # Generate embeddings for each route
        for route in self.routes:
            print(f"🔹 Generating embeddings for route: {route.name}")
            embeddings = self.ef(route.samples)

            # Check if embeddings are empty
            if embeddings is None or len(embeddings) == 0:
                print(f"⚠️ Error: Empty embedding for route {route.name}. Using zeros vector.")
                embeddings = np.zeros((1, 1536))  # OpenAI embedding dimension

            # Ensure embeddings are 2D array before normalizing
            embeddings = np.array(embeddings)
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)

            self.routesEmbedding[route.name] = embeddings
            self.routesEmbeddingCal[route.name] = normalize(embeddings, axis=1)

    def guide(self, query: str):
        print(f"🔍 Processing query: {query}")
        queryEmbedding = self.ef([query])

        # Check if query embedding is empty
        if queryEmbedding is None or len(queryEmbedding) == 0:
            print("⚠️ Error: Empty query embedding! Using zeros vector.")
            queryEmbedding = np.zeros((1, 1536))  # OpenAI embedding dimension
        
        # Ensure query embedding is 2D before normalizing
        queryEmbedding = np.array(queryEmbedding)
        if queryEmbedding.ndim == 1:
            queryEmbedding = queryEmbedding.reshape(1, -1)

        queryEmbedding = normalize(queryEmbedding)

        scores = []
        for route in self.routes:
            routeEmbeddingCal = self.routesEmbeddingCal[route.name]
            score = np.mean(np.dot(routeEmbeddingCal, queryEmbedding.T).flatten())
            print(f"📌 Score for {route.name}: {score}")
            scores.append((score, route.name))

        scores.sort(reverse=True)
        best_match = scores[0]

        # Check threshold
        if best_match[0] < self.threshold:
            print(f"⚠️ Best score {best_match[0]} is below threshold ({self.threshold}). Returning 'unknown'.")
            return (query, "unknown")

        print(f"✅ Best match: {best_match[1]} (Score: {best_match[0]})")
        return best_match
