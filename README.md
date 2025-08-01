# Face Recognition Indexer & Search (FAISS + SQLite)

This project provides a fast and scalable face recognition system using **InsightFace**, **FAISS**, and **SQLite**. It supports:

- Batch extraction of face embeddings from a database
- Building a searchable FAISS index
- A GUI tool to perform image-based identity search with metadata display

## Features

-  High-speed embedding extraction with multiprocessing
-  Uses InsightFace for accurate face recognition
-  Fast approximate nearest neighbor search with FAISS
-  Tkinter GUI for visual face search and results
-  Compatible with SQLite databases storing image blobs and metadata

## Requirements

- Python 3+
- OpenCV, FAISS, InsightFace, NumPy, Pillow, tqdm

Install dependencies:
```bash
pip install -r requirements.txt
```
## Usage
- Configure paths and settings in config.json.

- Run the embedding/indexing script (MUST):

```bash
python3 main.py
```
- Launch the GUI search tool:

```bash
python3 search.py
```
## Notes
- THIS VERSION OF PROGRAM IS WORKING BEST ON AMD GPU, but you can still use it with Nvidia / intel GPUs.
- All configuration (DB path, batch size, etc.) is set in config.json.
