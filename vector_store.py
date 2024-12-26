import faiss
import numpy as np
import pickle

class VectorStore:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []

    def add(self, embedding, metadata):
        self.index.add(np.array([embedding], dtype=np.float32))
        self.metadata.append(metadata)

    def search(self, query_vector, k=5):
        query = np.array([query_vector], dtype=np.float32)
        distances, indices = self.index.search(query, k)
        results = [(self.metadata[i], distances[0][j]) for j, i in enumerate(indices[0])]
        return results
