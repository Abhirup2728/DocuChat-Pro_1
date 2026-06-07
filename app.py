# Cell 7: Complete Production Code for DocuChat Pro
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
st.set_page_config(page_title="DocuChat Pro", page_icon="📖", layout="wide")
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
            response = llm.invoke(messages)
            
            # Render output
            st.markdown("### 🤖 DocuChat Pro Response:")
            st.write(response.content)
