import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
# -------------------------------------------------

import os
import fitz  # PyMuPDF
# ... (the rest of your original imports)
import os
import fitz  # PyMuPDF
import streamlit as str_layout # We use this alias internally for layout management
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage


# 2. Initialize core AI components once to save memory (Caching)
@st.cache_resource
def load_llm_and_embeddings():
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return llm, embeddings

llm, embeddings = load_llm_and_embeddings()

# 3. Streamlit Web User Interface Styling
#st.set_page_config(page_title="DocuChat Pro", page_icon="📖", layout="wide")
# 3. Streamlit Web User Interface Styling
st.set_page_config(page_title="DocuChat Pro", page_icon="🌌", layout="wide")

# --- UI DECORATION: CUSTOM CSS ---
# 3. Streamlit Web User Interface Styling
st.set_page_config(page_title="DocuChat Pro", page_icon="✨", layout="wide")
# 3. Streamlit Web User Interface Styling
st.set_page_config(page_title="DocuChat Pro", page_icon="📘", layout="centered") # Changed layout to 'centered' for a cleaner document feel

# --- UI DECORATION: PREMIUM MINIMALIST SaaS CSS ---
# --- Production Deployment (GitHub & Streamlit Cloud) ---
# Cell 12: Complete, Secure, and Aesthetic RAG Application (DocuChat Pro)

# 1. Imports and Security: Set up core AI and Web logic once. We are deploying for real now.
import os
import streamlit as st
import base64 # to handle file data
import fitz # for PDF text extraction (PyMuPDF)
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage

# --- PRODUCTION ENVIRONMENT CONFIGURATION ---
# LangChain, ChromaDB, and Streamlit Community Cloud have specific dependency/OS issues. We fix them once here.
# Chromium (used by ChromaDB) requires specific background libraries on Streamlit Cloud.
# Protobuf version must be strict (v3), otherwise ChromaDB crashes.
os.environ["LC_ALL"] = "en_US.UTF-8"
os.environ["LANG"] = "en_US.UTF-8"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
# ---------------------------------------------

# --- PRODUCTION API KEY SECURITY ---
# DO NOT hardcode your API key anymore. It touched a local file and was nearly compromised.
# After rotating your key at console.groq.com, we will paste it securely into the Streamlit deployment settings.
# For now, your app will look for the secret 'GROQ_API_KEY' in its secure environment variable storage.
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
# ------------------------------------

# --- AESTHETIC DECORATION: PREMIUM SaaS MINIMALISM CSS ---
# We are creating a sophisticated enterprise aesthetic: Clean background, premium fonts, refined cards.
st.set_page_config(page_title="DocuChat Pro", page_icon="📘", layout="centered")

st.markdown("""
<style>
    /* Import 'Inter' font: The standard for premium web products */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Define text color, size, and globally apply the 'Inter' font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
        color: #1e293b; /* Sophisticated slate gray */
    }

    /* Calm, static off-white background */
    .stApp {
        background-color: #f8fafc;
    }

    /* Minimalist File Uploader container */
    [data-testid="stFileUploadDropzone"] {
        background-color: #ffffff;
        border: 1px dashed #cbd5e1;
        border-radius: 12px;
        padding: 30px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    
    /* Soft blue accent on hover */
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #3b82f6;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    /* Polished, minimalist title styling */
    h1 {
        font-weight: 700 !important;
        color: #0f172a !important;
        letter-spacing: -0.025em;
        margin-bottom: 0.5rem;
    }

    /* Clean Subtitle styling */
    p {
        color: #475569;
        font-size: 1.05rem;
    }

    /* Production Response Card: Minimalist and highly readable */
    .response-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 24px 32px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-top: 24px;
        color: #1e293b;
        font-size: 1rem;
        line-height: 1.7;
        border-top: 4px solid #3b82f6; /* Subtle blue top accent bar */
    }
</style>
""", unsafe_allow_html=True)
# --- UI DECORATION: PREMIUM GLASSMORPHISM CSS ---

st.title("📖 DocuChat Pro: Multi-PDF RAG Chatbot")
st.markdown("Upload any document and have a context-grounded conversation with it instantly.")

# 4. File Upload Section
uploaded_file = st.file_uploader("Upload your PDF document", type=["pdf"])

if uploaded_file is not None:
    st.success("📄 PDF uploaded successfully!")
    
    # Process file inside a spinner so the user knows the system is working
    with st.spinner("Processing document... Extracting text & generating vector index..."):
        # Read the uploaded PDF file bytes
        file_bytes = uploaded_file.read()
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        
        # Extract raw text
        full_text = ""
        for page in doc:
            full_text += page.get_text()
            
        # Segment text into overlapping chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(full_text)
        
        # Build local ephemeral ChromaDB instance
        vector_store = Chroma.from_texts(texts=chunks, embedding=embeddings)
    
    st.info(f"System Status: Document tokenized into {len(chunks)} searchable chunks.")
    
    # 5. Conversational Interface Section
    st.subheader("💬 Ask Your Document Anything")
    user_question = st.text_input("Enter your question:")
    
    if user_question:
        with st.spinner("Searching document data and synthesisng answer..."):
            # Search database for top 4 relevant matches to avoid missing context
            retrieved_docs = vector_store.similarity_search(user_question, k=4)
            context_text = "\n\n---\n\n".join([d.page_content for d in retrieved_docs])
            
            # Construct rigorous grounded system template
            system_prompt = f"""You are DocuChat Pro, an elite AI assistant. 
Your task is to answer the user's question using ONLY the provided context below.
If the context does not contain the answer, say exactly: 'I cannot find the answer in the provided document.' 
Do not make up facts or use outside knowledge.

CONTEXT FROM UPLOADED PDF:
{context_text}
"""
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_question)
            ]
            
            # Execute inference
            # Execute inference
            # Generate response from the secure Groq LLM (initialized from st.secrets)
            response = llm.invoke(messages)
            
            # Formatted Output Section using HTML and the 'response-card' CSS class
            st.markdown("### 🤖 DocuChat Pro Response:")
            
            html_response = f"""
            <div class="response-card">
                {response.content}
            </div>
            """
            st.markdown(html_response, unsafe_allow_html=True)
