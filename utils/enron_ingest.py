# DEPRECATED: This script is no longer maintained.
# Please use the updated enron_ingest.py in the project root directory.
# For consistency, keep all future ingest scripts in a single, logical location (e.g., project root or a dedicated 'scripts' directory').
#
# --- Original script below ---
import os
import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import faiss
import traceback
import re

# File paths for WSL
CSV_PATH = '/home/jcockburn/LexMachina/enron/emails_clean.csv'
FAISS_PATH = '/home/jcockburn/LexMachina/vectorstore/db.faiss'
META_PATH = '/home/jcockburn/LexMachina/vectorstore/meta.pkl'
MODEL_NAME = 'all-MiniLM-L6-v2'
CHUNK_WORDS = 500
MIN_BODY_LEN = 20
# Set to None to process all, or an integer for a subset (e.g., 1000 for testing)
MAX_EMAILS = None  # Change to 1000 for quick tests


def chunk_text(text, max_words=500):
    """Split text into chunks of up to max_words words."""
    words = text.split()
    if len(words) <= max_words:
        return [text]
    return [
        ' '.join(words[i:i+max_words])
        for i in range(0, len(words), max_words)
    ]


def extract_sender(row, body):
    # Try several columns for sender
    for col in ['From', 'Sender', 'sender_email', 'from', 'Sender-Type']:
        sender = row.get(col)
        if sender and isinstance(sender, str) and sender.strip() and sender.strip().lower() != 'nan':
            return sender.strip()
    # Fallback: extract from body using regex
    if isinstance(body, str):
        match = re.search(r'from\s*:\s*"?([^"<]*)"?\s*<([^>]+)>', body, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            email = match.group(2).strip()
            return f'{name} <{email}>' if name else email
    return 'Unknown'


def main():
    print("Starting Enron email ingestion...")
    try:
        # Load CSV
        df = pd.read_csv(CSV_PATH)
        print(f"Loaded {len(df)} rows from CSV.")

        # Use the original body text, not lemmatized/stemmed
        body_col = 'Body' if 'Body' in df.columns else 'body'
        if body_col not in df.columns:
            raise ValueError("Original body column not found in CSV.")

        # Filter out empty bodies
        df = df[df[body_col].notnull() & (df[body_col].str.strip().str.len() >= MIN_BODY_LEN)]
        print(f"After filtering empty/short bodies: {len(df)} rows.")

        # Optionally process only a subset for testing
        if MAX_EMAILS is not None:
            df = df.iloc[:MAX_EMAILS]
            print(f"Processing only the first {MAX_EMAILS} emails for testing.")

        # Load embedding model
        print("Loading embedding model...")
        model = SentenceTransformer(MODEL_NAME)
        print("Model loaded.")

        embeddings = []
        metadata = []
        total_chunks = 0

        for idx, row in df.iterrows():
            body = row[body_col]
            chunks = chunk_text(body, CHUNK_WORDS)
            sender = extract_sender(row, body)
            # Compose date string if possible
            date = row.get('Date') or row.get('date')
            if not date:
                # Try to build from components
                year = str(row.get('Year', '')).zfill(4)
                month = str(row.get('Month', '')).zfill(2)
                day = str(row.get('Day', '')).zfill(2)
                hour = str(row.get('Hour', '')).zfill(2)
                date = f"{year}-{month}-{day} {hour}:00"
            subject = row.get('Subject') or row.get('subject')
            if not subject or str(subject).strip().lower() == 'nan':
                subject = '[No subject]'
            for chunk in chunks:
                # Ensure chunk is a string
                if not isinstance(chunk, str):
                    chunk = str(chunk)
                emb = model.encode(chunk)
                embeddings.append(emb)
                meta = {
                    'chunk_text': chunk,
                    'id': idx,
                    'sender': sender,
                    'date': date,
                    'subject': subject,
                }
                # Add any other useful fields present in the row
                for field in ['to', 'recipient', 'thread_id', 'Label', 'Unique-Mails-From-Sender', 'Contains-Reply-Forwards']:
                    if field in row:
                        meta[field] = row.get(field)
                metadata.append(meta)
                total_chunks += 1

        print(f"Finished processing emails. Total chunks: {total_chunks}.")

        # Convert embeddings to numpy array
        embeddings_np = np.vstack(embeddings).astype('float32')
        # Create FAISS index
        dim = embeddings_np.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings_np)
        # Save FAISS index
        os.makedirs(os.path.dirname(FAISS_PATH), exist_ok=True)
        faiss.write_index(index, FAISS_PATH)
        print(f"FAISS index saved to {FAISS_PATH}")
        # Save metadata
        with open(META_PATH, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"Metadata saved to {META_PATH}")

        # Output validation
        print(f"Total chunks saved: {len(metadata)}")
        if metadata:
            print("First metadata entry:")
            print(metadata[0])
        print("Ingestion complete.")
    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()


if __name__ == "__main__":
    main() 