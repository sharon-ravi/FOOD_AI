# ============================================
# 🔥 FOOD AI BACKEND (STREAMLIT CLOUD READY)
# ============================================

import os
import pickle
import numpy as np
import faiss
import gdown

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI

# ============================================
# 🔹 CONFIG
# ============================================

DATA_DIR = "data"
INDEX_PATH = f"{DATA_DIR}/final_index.faiss"
META_PATH = f"{DATA_DIR}/final_meta.pkl"

# 👉 GOOGLE DRIVE FILE IDS (REPLACE THESE)
INDEX_FILE_ID = "https://drive.google.com/file/d/1pFpCYY0KIjmmqNVZ5qtMsQqZmzIcbV1U/view?usp=sharing"
META_FILE_ID = "https://drive.google.com/file/d/1dCC8r1CoN7AJwifh7ZGHOWPu8mWSA6Df/view?usp=sharing"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

# ============================================
# 🔥 DOWNLOAD DATA FROM GOOGLE DRIVE
# ============================================

def download_if_missing():

    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(INDEX_PATH):
        print("⬇️ Downloading FAISS index...")
        gdown.download(
            f"https://drive.google.com/uc?id={INDEX_FILE_ID}",
            INDEX_PATH,
            quiet=False
        )

    if not os.path.exists(META_PATH):
        print("⬇️ Downloading metadata...")
        gdown.download(
            f"https://drive.google.com/uc?id={META_FILE_ID}",
            META_PATH,
            quiet=False
        )

# ============================================
# 🔹 LOAD DATA
# ============================================

def load_data():
    download_if_missing()

    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        print("❌ Data not found after download.")
        return None, None

    index = faiss.read_index(INDEX_PATH)

    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)

    return index, meta


index, meta = load_data()

# ============================================
# 🔹 MODELS
# ============================================

embed_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ============================================
# 🔹 BM25 INIT
# ============================================

bm25 = None

if meta:
    texts = [m["name"] + " " + str(m.get("brand", "")) for m in meta]
    corpus = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(corpus)

# ============================================
# 🔹 GROQ CLIENT
# ============================================

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# ============================================
# 🔹 CLEAN FUNCTION
# ============================================

def clean_name(x):
    if isinstance(x, list):
        for item in x:
            if item.get("lang") == "main":
                return item.get("text", "")
        return x[0].get("text", "")
    return str(x)

# ============================================
# 🔹 STRUCTURE TEXT
# ============================================

def build_structured_text(row):
    return (
        f"{row.get('name','')} | "
        f"protein: {row.get('proteins_100g','?')}g | "
        f"calories: {row.get('energy-kcal_100g','?')} kcal | "
        f"sugar: {row.get('sugars_100g','?')}g"
    )

# ============================================
# 🚨 FALLBACK MODE (NO DATA)
# ============================================

def fallback(query):
    return f"""
⚠️ Dataset not loaded.

Query: {query}

👉 Check Google Drive file IDs or internet connection.
"""

# ============================================
# 🔥 MAIN FUNCTION (USED BY STREAMLIT)
# ============================================

def generate_response(query, mode="chatbot"):

    if index is None or meta is None:
        return fallback(query)

    # ==============================
    # FAISS SEARCH
    # ==============================

    q_vec = embed_model.encode([query], normalize_embeddings=True)
    D, I = index.search(np.array(q_vec).astype("float32"), 10)

    results = [meta[i] for i in I[0] if i < len(meta)]

    if not results:
        return "No results found."

    # ==============================
    # OUTPUT FORMAT
    # ==============================

    output = "🔍 Top Food Results:\n\n"

    for i, r in enumerate(results[:5], 1):
        name = clean_name(r.get("name"))
        brand = r.get("brand", "")
        cal = r.get("energy-kcal_100g", "?")

        output += f"{i}. {name} ({brand}) - {cal} kcal\n"

    return output