# ============================================
# 🔥 FOOD AI BACKEND (STREAMLIT SAFE VERSION)
# ============================================

import os
import pickle
import numpy as np

import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI

# ============================================
# 🔹 CONFIG
# ============================================

INDEX_PATH = "data/final/final_index.faiss"
META_PATH = "data/final/final_meta.pkl"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

# ============================================
# 🔹 SAFE DATA LOADER (NO CRASH)
# ============================================

def load_data():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        print("⚠️ FAISS data not found. Running in SAFE MODE.")
        return None, None

    index = faiss.read_index(INDEX_PATH)

    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)

    return index, meta


index, meta = load_data()

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
# 🔹 EMBEDDING MODEL
# ============================================

embed_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ============================================
# 🔹 BM25 (SAFE INIT)
# ============================================

bm25 = None
corpus = None

if meta:
    texts = [m["name"] + " " + str(m.get("brand", "")) for m in meta]
    corpus = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(corpus)

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
# 🔥 SIMPLE FALLBACK (IF DATA NOT LOADED)
# ============================================

def fallback_response(query):
    return f"""
⚠️ Dataset not loaded on server.

Your app UI is working, but FAISS dataset is missing.

Query received: {query}

👉 Fix: Upload FAISS index OR connect cloud storage.
"""

# ============================================
# 🔥 MAIN FUNCTION (USED BY STREAMLIT)
# ============================================

def generate_response(query, mode="chatbot"):

    # SAFE MODE CHECK
    if index is None or meta is None:
        return fallback_response(query)

    # ==============================
    # FAISS SEARCH
    # ==============================

    q_vec = embed_model.encode([query], normalize_embeddings=True)
    D, I = index.search(np.array(q_vec).astype("float32"), 10)

    results = [meta[i] for i in I[0] if i < len(meta)]

    if not results:
        return "No results found."

    # ==============================
    # FORMAT OUTPUT
    # ==============================

    output = "🔍 Top Results:\n\n"

    for i, r in enumerate(results[:5], 1):
        name = clean_name(r.get("name"))
        brand = r.get("brand", "")
        cal = r.get("energy-kcal_100g", "?")

        output += f"{i}. {name} ({brand}) - {cal} kcal\n"

    return output