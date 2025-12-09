import os
import shutil
from dotenv import load_dotenv
from pdf2image import convert_from_path
import pytesseract
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

working_dir = os.path.dirname(os.path.abspath(__file__))
VECTOR_STORE_PATH = os.path.join(working_dir, "doc_vectorstore")

# Initialize Embedding Model and LLM
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

def process_document_to_chroma_db(file_name: str):
    """
    Ingest PDF. Appends to DB.
    """
    file_path = os.path.join(working_dir, file_name)

    try:
        pages = convert_from_path(file_path)
    except Exception as e:
        raise ValueError(f"Error converting PDF: {e}")

    documents = []
    for i, page in enumerate(pages):
        text = pytesseract.image_to_string(page)
        if text.strip():
            # Clean text to fix simple OCR line-break issues
            clean_text = text.replace("\n", " ")
            documents.append(Document(page_content=clean_text, metadata={"page": i+1, "source": file_name}))

    if not documents:
        raise ValueError(f"No text extracted from {file_name}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=VECTOR_STORE_PATH
    )
    return vectordb

def answer_question(user_question: str):
    vectordb = Chroma(persist_directory=VECTOR_STORE_PATH, embedding_function=embedding)
    
    # 1. AGGRESSIVE RETRIEVAL
    # We ask for 15 chunks to make sure we get the WHOLE document content.
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 15})

    document_prompt = PromptTemplate(
        input_variables=["page_content", "source"],
        template="Content:\n{page_content}\nSource: {source}\n----------------\n"
    )

    # 2. "EXTRACT ALL" PROMPT
    system_template = """You are a meticulous Data Extraction AI.
    The context contains scanned exam questions or notes from PDF files.
    
    Instructions:
    1. Answer the user's question using ONLY the provided context.
    2. IF THE USER ASKS FOR A LIST (e.g., "important questions"):
       - You MUST extract EVERY single question found in the text.
       - Do NOT summarize. Do NOT skip items.
       - If the text has numbers (1, 2, 3...), keep the numbering.
       - Fix obvious OCR typos if possible (e.g., "Eolucation" -> "Education"), but otherwise keep the text as is.
    3. CITATION: At the end, list the Source Filename(s).

    Context:
    {context}

    Question: {question}
    """

    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=system_template
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": QA_CHAIN_PROMPT,
            "document_prompt": document_prompt,
        }
    )

    response = qa_chain.invoke({"query": user_question})
    result_text = response.get("result", "").strip()

    return result_text, []

if __name__ == "__main__":
    try:
        # Local Test
        answer, _ = answer_question("List all the important questions in the notes")
        print("Answer:", answer)
    except ValueError as e:
        print("❌ Error:", e)