# =========================
# app.py (CLEAN UI)
# =========================

import streamlit as st
from backend import generate_response

st.set_page_config(
    page_title="Food AI",
    layout="wide"
)

st.title("🍎 Food Intelligence AI")

mode = st.selectbox(
    "Select Mode",
    ["fitness", "health", "diet", "chatbot"]
)

query = st.text_input("Enter your food query")

# caching responses (FAST)
@st.cache_data(show_spinner=False)
def get_response(q, m):
    return generate_response(q, m)

if st.button("Search"):
    if query:
        with st.spinner("Thinking..."):
            result = get_response(query, mode)
        st.success("Done")
        st.write(result)