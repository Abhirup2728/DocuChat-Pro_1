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

# --- UI DECORATION: PREMIUM GLASSMORPHISM CSS ---
st.markdown("""
<style>
    /* Import Google Font 'Poppins' for a sleek, modern look */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap');

    /* Apply the font globally */
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif !important;
    }

    /* Create a vibrant, animated gradient background */
    .stApp {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }

    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Apply a 'Frosted Glass' effect to the main content area */
    .block-container {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(15px);
        border-radius: 25px;
        padding: 3rem !important;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
        margin-top: 3rem;
        margin-bottom: 3rem;
    }

    /* Style the main title with a sleek text gradient */
    h1 {
        font-weight: 800 !important;
        text-align: center;
        background: -webkit-linear-gradient(#e73c7e, #23a6d5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
    }

    /* Soften and modernize the file uploader */
    [data-testid="stFileUploadDropzone"] {
        background-color: rgba(255, 255, 255, 0.5);
        border: 2px dashed #e73c7e;
        border-radius: 20px;
        padding: 30px;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploadDropzone"]:hover {
        background-color: rgba(255, 255, 255, 0.9);
        border-color: #23a6d5;
        transform: scale(1.01);
    }

    /* Premium AI Response Card */
    .response-card {
        background: #ffffff;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 15px 25px rgba(0,0,0,0.05);
        border-left: 8px solid #23a6d5;
        margin-top: 25px;
        color: #2b2b2b;
        font-size: 1.1rem;
        line-height: 1.7;
    }
</style>
""", unsafe_allow_html=True)
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
            response = llm.invoke(messages)
            
            # Render output using the custom CSS 'response-card'
            st.markdown("### 🤖 DocuChat Pro Response:")
            
            # We wrap the response in a styled HTML div
            html_response = f"""
            <div class="response-card">
                {response.content}
            </div>
            """
            st.markdown(html_response, unsafe_allow_html=True)
