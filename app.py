import os
import streamlit as st
import shutil
from rag_utility import process_document_to_chroma_db, answer_question

# Get the directory where this script is running
working_dir = os.path.dirname(os.path.abspath(__file__))
VECTOR_STORE_PATH = os.path.join(working_dir, "doc_vectorstore")

st.title("🦙 Llama-3.3-70B - Document RAG")

# --- 1. SIDEBAR RESET BUTTON (UPDATED) ---
if st.sidebar.button("Clear System"):
    # Step A: Delete the Vector Database (AI Memory)
    if os.path.exists(VECTOR_STORE_PATH):
        shutil.rmtree(VECTOR_STORE_PATH, ignore_errors=True)
    
    # Step B: Delete all PDF files in the project folder (Local Storage)
    # We loop through every file in the directory
    for filename in os.listdir(working_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(working_dir, filename)
            try:
                os.remove(file_path)
            except Exception as e:
                st.sidebar.error(f"Error deleting {filename}: {e}")
                
    st.sidebar.success("✅ System Cleared: Memory wiped & PDFs deleted!")

# --- 2. FILE UPLOADER ---
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# --- 3. PROCESS BUTTON ---
if uploaded_files and st.button("Process Documents"):
    
    progress_text = "Operation in progress. Please wait..."
    my_bar = st.progress(0, text=progress_text)
    
    for i, file in enumerate(uploaded_files):
        # Save file locally
        save_path = os.path.join(working_dir, file.name)
        with open(save_path, "wb") as f:
            f.write(file.getbuffer())

        # Process file into Vector DB
        process_document_to_chroma_db(file.name)
        
        my_bar.progress((i + 1) / len(uploaded_files), text=f"Processed {file.name}")
        
    st.success("✅ All documents processed and added to vector DB.")

# --- 4. Q&A SECTION ---
user_question = st.text_area("Ask your question about the documents")

if st.button("Answer"):
    with st.spinner("Thinking..."):
        answer_text, sources = answer_question(user_question)
        
        st.markdown("### Llama-3.3-70B Response")
        st.markdown(answer_text)
        
        # Only show sources if the list is not empty
        # (Note: Current backend returns empty list, but this is safe to keep)
        if sources:
            st.markdown("---")
            st.subheader("Sources:")
            for source in sources:
                st.caption(f"📄 {source}")