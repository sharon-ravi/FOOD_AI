Got it—you want a **simple, clean README** (not too heavy, easy to understand, good for submission). Here you go:

---

# 🥗 NourishAI – Food Recommendation System

## 📌 Overview

NourishAI is an AI-based system that suggests food products based on user queries like:

* “high protein snacks”
* “low calorie foods”
* “healthy options”

It uses a combination of search techniques and AI models to give better results.

---

## ⚙️ How It Works

1. **User enters a query**
2. Query is improved using AI (query rewrite)
3. System searches using:

   * FAISS → finds similar meaning
   * BM25 → finds exact words
4. Results are combined and cleaned
5. CrossEncoder ranks the best results
6. AI selects the most relevant items
7. Final answer is shown in clean format

---

## 🧠 Modes

* **fitness** → high protein foods
* **diet** → low calorie / low sugar
* **health** → balanced food
* **chatbot** → general queries
* **barcode** → alternatives

---

## 🛠️ Technologies Used

* FAISS (vector search)
* BM25 (keyword search)
* SentenceTransformer (embeddings)
* CrossEncoder (reranking)
* Groq LLM (response generation)
* Streamlit (UI)

---

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API key

Windows:

```bash
setx GROQ_API_KEY "your_key"
```

Mac/Linux:

```bash
export GROQ_API_KEY="your_key"
```

---

### 3. Run app

CLI:

```bash
python app.py
```

Streamlit:

```bash
streamlit run app.py
```

---

## 💡 Example Queries

* high protein snacks
* low calorie foods
* healthy packaged food

---

## ⚠️ Note

* This system suggests food products only
* Not meant for medical advice

---

## 👨‍💻 Author

CHRISTABEL SHARON 

---

