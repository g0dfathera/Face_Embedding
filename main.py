import sqlite3
import numpy as np
import cv2
import os
import pickle
import json
import faiss
import multiprocessing as mp
from tqdm import tqdm
from insightface.app import FaceAnalysis

# === Load Configuration ===
with open("config.json", "r") as f:
    config = json.load(f)

DB_PATH = config["DB_PATH"]
BATCH_SIZE = config.get("BATCH_SIZE", 5000)
EMBEDDINGS_PATH = config.get("EMBEDDINGS_PATH", "face_embeddings.npy")
LABELS_PATH = config.get("LABELS_PATH", "face_labels.pkl")
FAISS_INDEX_PATH = config.get("FAISS_INDEX_PATH", "faiss_index.bin")
NUM_WORKERS = config.get("NUM_WORKERS", mp.cpu_count() - 1 or 1)

app = FaceAnalysis(providers=['DmlExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def extract_embedding(rgb_img):
    faces = app.get(rgb_img)
    if len(faces) != 1:
        return None
    return faces[0].embedding

def worker(row):
    num, name, blob = row
    img_array = np.frombuffer(blob, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return None
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    embedding = extract_embedding(rgb_img)
    if embedding is None:
        return None
    return (embedding, name)

def load_db_batch(offset, limit):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT p.id, p.full_name, i.image_blob
        FROM people p
        JOIN images i ON p.id = i.person_id
        WHERE i.image_blob IS NOT NULL
        LIMIT ? OFFSET ?;
    """, (limit, offset))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

def save_progress(embeddings, labels):
    if os.path.exists(EMBEDDINGS_PATH):
        existing_embeddings = np.load(EMBEDDINGS_PATH)
        embeddings = np.vstack([existing_embeddings, embeddings])
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "rb") as f:
            existing_labels = pickle.load(f)
        labels = existing_labels + labels

    np.save(EMBEDDINGS_PATH, embeddings)
    with open(LABELS_PATH, "wb") as f:
        pickle.dump(labels, f)

def build_faiss_index():
    embeddings = np.load(EMBEDDINGS_PATH).astype('float32')
    labels = pickle.load(open(LABELS_PATH, "rb"))

    faiss.normalize_L2(embeddings)
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"[+] FAISS index saved to {FAISS_INDEX_PATH}")

def main():
    offset = 0

    while True:
        rows = load_db_batch(offset, BATCH_SIZE)
        if not rows:
            break

        print(f"\nProcessing batch {offset} to {offset + len(rows)}")

        with mp.Pool(NUM_WORKERS) as pool:
            results = list(tqdm(pool.imap(worker, rows), total=len(rows)))

        filtered = [r for r in results if r is not None]
        if not filtered:
            print("[!] No embeddings extracted.")
            offset += BATCH_SIZE
            continue

        embeddings, labels = zip(*filtered)
        embeddings = np.array(embeddings, dtype='float32')

        save_progress(embeddings, list(labels))
        offset += BATCH_SIZE

    print("\n[+] All batches processed. Building FAISS index...")
    build_faiss_index()
    print("[+] Done.")

if __name__ == "__main__":
    main()
