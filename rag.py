import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_docs():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    notes_path = os.path.join(BASE_DIR, "data", "therapy_notes.txt")
    print(f"Loading therapy notes from: {notes_path}")

    loader = TextLoader(notes_path, encoding="utf-8")
    raw_docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(raw_docs)

def create_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    docs = load_docs()
    return FAISS.from_documents(docs, embeddings)

retriever = create_vector_store().as_retriever()


