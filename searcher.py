import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sqlite3
import numpy as np
import cv2
import os
import pickle
import json
import faiss
from insightface.app import FaceAnalysis
from PIL import Image, ImageTk

with open("config.json", "r") as f:
    config = json.load(f)

DB_PATH = config["DB_PATH"]
FAISS_INDEX_PATH = config["FAISS_INDEX_PATH"]
LABELS_PATH = config["LABELS_PATH"]

app = FaceAnalysis(providers=['DmlExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def extract_embedding(rgb_img):
    faces = app.get(rgb_img)
    if len(faces) != 1:
        return None
    return faces[0].embedding

def get_person_info(fullname):
    try:
        first_name, last_name = fullname.split(maxsplit=1)
    except:
        first_name, last_name = fullname, ""

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT p.first_name, p.last_name, p.date_of_birth, d.district_name, r.region_name
        FROM people p
        LEFT JOIN districts d ON p.district_id = d.id
        LEFT JOIN regions r ON d.region_id = r.id
        WHERE p.first_name = ? AND p.last_name = ?
    """, (first_name, last_name))
    result = cursor.fetchone()
    cursor.close()
    conn.close()

    return result if result else ("Unknown", "", "", "", "")

def get_person_image(fullname):
    try:
        first_name, last_name = fullname.split(maxsplit=1)
    except:
        first_name, last_name = fullname, ""

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM people WHERE first_name = ? AND last_name = ?", (first_name, last_name))
    row = cursor.fetchone()
    if not row:
        cursor.close()
        conn.close()
        return None
    person_id = row[0]

    cursor.execute("SELECT image_blob FROM images WHERE person_id = ? LIMIT 1", (person_id,))
    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if not row or row[0] is None:
        return None
    img_array = np.frombuffer(row[0], dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

class FaceSearchApp:
    def __init__(self, root):
        self.root = root
        root.title("Face Search")
        root.geometry("720x650")

        if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(LABELS_PATH):
            messagebox.showerror("Error", "FAISS index or labels file not found.")
            root.destroy()
            return

        self.index = faiss.read_index(FAISS_INDEX_PATH)
        self.labels = pickle.load(open(LABELS_PATH, "rb"))

        self.photo_refs = []

        # Top Frame
        frame_top = ttk.LabelFrame(root, text="Query Image")
        frame_top.pack(fill=tk.X, padx=10, pady=5)

        self.query_img_label = ttk.Label(frame_top)
        self.query_img_label.pack(pady=5)

        self.btn_browse = ttk.Button(frame_top, text="Select Image to Search", command=self.browse_image)
        self.btn_browse.pack(pady=5)

        frame_results = ttk.LabelFrame(root, text="Top Matches")
        frame_results.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        canvas = tk.Canvas(frame_results)
        scrollbar = ttk.Scrollbar(frame_results, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def browse_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if not file_path:
            return
        self.search_face(file_path)

    def search_face(self, image_path, top_k=10):
        img = cv2.imread(image_path)
        if img is None:
            messagebox.showerror("Error", "Could not read image.")
            return

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        embedding = extract_embedding(rgb_img)
        if embedding is None:
            messagebox.showwarning("Warning", "No face or multiple faces detected.")
            return

        self.show_image_in_label(rgb_img, self.query_img_label, maxsize=(200, 200))

        embedding = embedding.astype('float32').reshape(1, -1)
        faiss.normalize_L2(embedding)
        distances, indices = self.index.search(embedding, top_k)

        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.photo_refs.clear()

        ttk.Label(self.scrollable_frame, text=f"Top {top_k} Matches", font=("Arial", 14, "bold")).pack(pady=10)

        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
            fullname = self.labels[idx]
            first, last, dob, district, region = get_person_info(fullname)
            person_img = get_person_image(fullname)

            frame = ttk.Frame(self.scrollable_frame, relief=tk.RIDGE, borderwidth=1)
            frame.pack(fill=tk.X, padx=10, pady=5)

            photo_label = ttk.Label(frame)
            photo_label.pack(side=tk.LEFT, padx=5, pady=5)
            if person_img is not None:
                self.show_image_in_label(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB), photo_label, maxsize=(100, 100))
                self.photo_refs.append(photo_label.image)
            else:
                photo_label.config(text="[No Image]", width=12, background="gray", foreground="white", anchor="center", font=("Arial", 9, "italic"))

            info_text = (
                f"Rank: {rank}\n"
                f"Name: {first} {last}\n"
                f"Date of Birth: {dob}\n"
                f"District: {district}\n"
                f"Region: {region}\n"
                f"Similarity: {dist:.4f}"
            )
            info_label = ttk.Label(frame, text=info_text, justify=tk.LEFT, anchor="w", font=("Arial", 10))
            info_label.pack(side=tk.LEFT, padx=10, pady=5)

    def show_image_in_label(self, rgb_img, label_widget, maxsize=(200, 200)):
        pil_img = Image.fromarray(rgb_img)
        pil_img.thumbnail(maxsize, Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(pil_img)
        label_widget.configure(image=tk_img)
        label_widget.image = tk_img

def main():
    root = tk.Tk()
    app = FaceSearchApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
