import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from langchain_ollama import OllamaEmbeddings

PDF_FOLDER = "RAGData"
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "rag_docs"


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

    Milvus.from_documents(
        documents=docs,
        embedding=embedding,
        collection_name=COLLECTION_NAME,
        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
    )
    print("✅ Vector database created in Milvus")


if __name__ == "__main__":
    create_vector_database()
