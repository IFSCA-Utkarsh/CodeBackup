import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from config import SETTINGS

PDF_FOLDER = SETTINGS.PDF_FOLDER
PERSIST_DIR = SETTINGS.PERSIST_DIR


def create_vector_database():
    documents = []
    for file in os.listdir(PDF_FOLDER):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(PDF_FOLDER, file))
            pages = loader.load()
            documents.extend(pages)
            print(f"Loaded {len(pages)} pages from {file}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    docs = text_splitter.split_documents(documents)

    embedding = OllamaEmbeddings(model=SETTINGS.EMBEDDING_MODEL)

    Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory=PERSIST_DIR
    )
    print("✅ Vector database created and saved to disk.")

if __name__ == "__main__":
    create_vector_database()