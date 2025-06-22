import os
import pandas as pd
import numpy as np
import pickle
import faiss
import hashlib
from datetime import datetime
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
CSV_PATH = '/home/jcockburn/LexMachina/enron/emails_clean.csv'
FAISS_PATH = '/home/jcockburn/LexMachina/vectorstore/db.faiss'
META_PATH = '/home/jcockburn/LexMachina/vectorstore/meta.pkl'
MODEL_NAME = 'all-MiniLM-L6-v2'
CHUNK_WORDS = 500
BATCH_SIZE = 128
MIN_BODY_LEN = 20

# --- UTILITY FUNCTIONS ---
def clean_text(text):
    """Basic cleaning: strip whitespace and collapse multiple spaces."""
    if not isinstance(text, str):
        return ""
    return ' '.join(text.split())

def chunk_by_paragraph(text, min_len=MIN_BODY_LEN):
    """Chunk by paragraph, skipping short or non-content chunks."""
    paras = [p.strip() for p in text.split('\n\n') if len(p.strip().split()) >= min_len]
    return paras

def safe_str(row, col, default='unknown'):
    val = row.get(col)
    if isinstance(val, str) and not pd.isna(val):
        return val
    return default

def safe_date(row, col):
    val = row.get(col)
    if isinstance(val, str) and not pd.isna(val):
        try:
            date_parsed = pd.to_datetime(val, errors='coerce')
            if pd.notnull(date_parsed):
                if isinstance(date_parsed, pd.Timestamp):
                    return date_parsed.isoformat()
        except Exception:
            pass
        return val
    return 'unknown'

# --- MAIN INGESTION ---
def main():
    print(f"--- Ingestion started at {datetime.now().isoformat()} ---")
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows from CSV.")
    print(f"CSV columns: {df.columns.tolist()}")

    # --- COLUMN SELECTION ---
    # Only use original, fluent text for chunking/embedding
    if 'body' in df.columns:
        content_col = 'body'
    elif 'message' in df.columns:
        content_col = 'message'
    else:
        raise ValueError("No suitable original text column found (body or message).")
    # Warn if lemmatized columns are present
    for col in ['Body_lemma', 'body_lemma']:
        if col in df.columns:
            print(f"WARNING: Lemmatized column '{col}' found but will NOT be used for chunking/embedding.")

    id_col = 'id' if 'id' in df.columns else ('message_id' if 'message_id' in df.columns else None)

    chunk_texts = []
    chunk_metadata = []
    seen_hashes = set()
    total_emails = 0
    total_chunks = 0

    model = SentenceTransformer(MODEL_NAME)

    for idx, row in df.iterrows():
        content = safe_str(row, content_col, '')
        content = clean_text(content)
        if not content or len(content.split()) < MIN_BODY_LEN:
            continue
        # Use id/message_id or fallback to row index
        if id_col is not None:
            id_val = row.get(id_col)
            email_id = str(id_val) if isinstance(id_val, str) and not pd.isna(id_val) else str(idx)
        else:
            email_id = str(idx)
        from_ = safe_str(row, 'from')
        to = safe_str(row, 'to')
        subject = safe_str(row, 'subject', '[no subject]')
        date = safe_date(row, 'date')
        # Chunking by paragraph
        chunks = chunk_by_paragraph(content)
        total_emails += 1
        for chunk_index, chunk in enumerate(chunks):
            if len(chunk.split()) < MIN_BODY_LEN:
                continue
            chunk_hash = hashlib.sha256(chunk.encode('utf-8')).hexdigest()
            if chunk_hash in seen_hashes:
                continue
            seen_hashes.add(chunk_hash)
            meta = {
                'id': email_id,
                'from': str(from_),
                'to': str(to),
                'subject': str(subject),
                'date': str(date),
                'chunk_index': chunk_index,
                'total_chunks': len(chunks),
                'chunk_text': chunk
            }
            chunk_texts.append(chunk)
            chunk_metadata.append(meta)
            total_chunks += 1

    print(f"Total emails processed: {total_emails}")
    print(f"Total deduplicated chunks: {total_chunks}")
    if chunk_metadata:
        print(f"Example metadata: {chunk_metadata[0]}")

    # --- BATCH EMBEDDING ---
    embeddings = []
    for i in range(0, len(chunk_texts), BATCH_SIZE):
        batch = chunk_texts[i:i+BATCH_SIZE]
        batch_embs = model.encode(batch, show_progress_bar=True)
        embeddings.extend(batch_embs)

    # --- FAISS INDEXING ---
    embeddings_np = np.vstack(embeddings).astype('float32')
    dim = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings_np)
    os.makedirs(os.path.dirname(FAISS_PATH), exist_ok=True)
    faiss.write_index(index, FAISS_PATH)
    with open(META_PATH, 'wb') as f:
        pickle.dump(chunk_metadata, f)
    print(f"FAISS index saved to {FAISS_PATH}")
    print(f"Metadata saved to {META_PATH}")
    print("--- Ingestion complete ---")

if __name__ == "__main__":
    main() 