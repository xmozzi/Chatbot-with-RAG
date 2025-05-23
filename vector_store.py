from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

# In-memory store
vector_db = []
texts = []

def create_vector_store(chunks):
    global vector_db, texts
    embeddings = model.encode(chunks)
    vector_db = embeddings
    texts = chunks
    return vector_db, texts

def search_similar_chunks(vector_db, query, top_k=3):
    query_vec = model.encode([query])
    scores = cosine_similarity(query_vec, vector_db)[0]
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return "\n".join([texts[i] for i in top_indices])