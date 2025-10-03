# create_db.py

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

PDF_FOLDER = "RAGData"
PERSIST_DIR = "chroma_db"

def create_vector_database():
    documents = []
    for file in os.listdir(PDF_FOLDER):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(PDF_FOLDER, file))
            documents.extend(loader.load())
            print(f"Loaded {len(documents)} pages from {file}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    docs = text_splitter.split_documents(documents)

    embedding = OllamaEmbeddings(model="nomic-embed-text")

    Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory=PERSIST_DIR
    )
    print("✅ Vector database created and saved to disk.")

if __name__ == "__main__":
    create_vector_database()