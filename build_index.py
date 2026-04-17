# =========================
# build_index.py
# =========================

import os
import faiss
import pickle

BASE_DIRS = [
    "data/part1/food_faiss_v7",
    "data/part3/food_faiss_v7"
]

OUTPUT_DIR = "data/final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FINAL_INDEX = os.path.join(OUTPUT_DIR, "final_index.faiss")
FINAL_META = os.path.join(OUTPUT_DIR, "final_meta.pkl")

main_index = None
all_meta = []

for folder in BASE_DIRS:

    idx = faiss.read_index(os.path.join(folder, "index.faiss"))

    if main_index is None:
        main_index = idx
    else:
        main_index.merge_from(idx)

    meta_dir = os.path.join(folder, "meta_chunks")

    for f in os.listdir(meta_dir):
        if f.endswith(".pkl"):
            all_meta.extend(
                pickle.load(open(os.path.join(meta_dir, f), "rb"))
            )

faiss.write_index(main_index, FINAL_INDEX)

with open(FINAL_META, "wb") as f:
    pickle.dump(all_meta, f)

print("DONE:", main_index.ntotal, len(all_meta))