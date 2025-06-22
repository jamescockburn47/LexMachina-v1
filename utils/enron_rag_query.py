import os
import pickle
import traceback
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Paths for WSL environment
FAISS_PATH = '/home/jcockburn/LexMachina/vectorstore/db.faiss'
META_PATH = '/home/jcockburn/LexMachina/vectorstore/meta.pkl'
MODEL_NAME = 'all-MiniLM-L6-v2'


def main():
    try:
        # Ask for user question
        user_query = input("Enter your question: ")
        if not user_query.strip():
            print("No question entered. Exiting.")
            return

        # Load embedding model
        print("Loading embedding model...")
        model = SentenceTransformer(MODEL_NAME)
        print("Model loaded.")

        # Load FAISS index
        print("Loading FAISS index...")
        index = faiss.read_index(FAISS_PATH)
        print("FAISS index loaded.")

        # Load metadata
        print("Loading metadata...")
        with open(META_PATH, 'rb') as f:
            metadata = pickle.load(f)
        print(f"Loaded metadata for {len(metadata)} chunks.")

        # Embed the user query
        query_emb = model.encode(user_query).astype('float32')
        query_emb = np.expand_dims(query_emb, axis=0)

        # Search FAISS for top 5 most similar chunks
        D, I = index.search(query_emb, 5)  # D: distances, I: indices
        print("\nTop 5 most similar chunks:")
        for rank, (idx, dist) in enumerate(zip(I[0], D[0]), 1):
            if idx < 0 or idx >= len(metadata):
                print(f"Result {rank}: Invalid index {idx}")
                continue
            meta = metadata[idx]
            chunk_text = meta.get('chunk_text', '[Chunk text not stored in metadata]')
            print(f"Result {rank}:")
            print(f"Chunk text: {chunk_text}")
            print(f"ID: {meta.get('id')}")
            print(f"Sender: {meta.get('sender')}")
            print(f"Date: {meta.get('date')}")
            print(f"Subject: {meta.get('subject')}")
            print(f"Similarity score: {dist}")
            print("---")
    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()

# Entry point
if __name__ == "__main__":
    main() 