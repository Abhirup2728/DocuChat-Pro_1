# 📖 DocuChat Pro: Enterprise RAG Architecture

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://docuchat-pro.streamlit.app)

DocuChat Pro is an end-to-end **Retrieval-Augmented Generation (RAG)** web application designed to allow users to interact dynamically with multi-page PDF documents. By leveraging semantic search and high-speed LLM inference, it eliminates the need for manual keyword searching and provides instant, context-grounded answers to complex queries.

## 🚀 Live Demo
**Try the application here:** [docuchat-pro.streamlit.app](https://docuchat-pro.streamlit.app)

---

## 🧠 System Architecture

The application follows a strict, linear RAG pipeline to ensure zero hallucinations and high-speed retrieval:

1. **Document Ingestion:** PyMuPDF (`fitz`) extracts raw text from user-uploaded PDFs.
2. **Semantic Chunking:** LangChain's `RecursiveCharacterTextSplitter` segments the text into 1000-character overlapping chunks to preserve context.
3. **Vector Embedding:** HuggingFace's open-source `all-MiniLM-L6-v2` model converts text chunks into dense numerical vectors.
4. **Vector Database:** An ephemeral, in-memory **ChromaDB** instance stores and indexes the embeddings for lightning-fast retrieval.
5. **Retrieval & Generation:** User queries are vectorized and matched against the database using Cosine Similarity. The top-K contexts are injected into a strict prompt template and passed to the **Groq API** (Llama-3) for natural language synthesis.

---

## 🛠️ Tech Stack

* **Frontend UI:** Streamlit
* **Orchestration:** LangChain
* **LLM Inference Engine:** Groq API (`llama-3.3-70b-versatile`)
* **Vector Store:** ChromaDB
* **Embeddings:** HuggingFace Sentence Transformers
* **Document Parsing:** PyMuPDF

---

## 💻 Run Locally

To run this application on your own machine, follow these steps:

### 1. Clone the repository
```bash
git clone [https://github.com/yourusername/DocuChat-Pro.git](https://github.com/yourusername/DocuChat-Pro.git)
cd DocuChat-Pro
