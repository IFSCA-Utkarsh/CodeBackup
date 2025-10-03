# rag_pipeline.py

import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

class RAGPipeline:
    def __init__(self, pdf_folder="RAGData", persist_dir="./chroma_db"):
        self.pdf_folder = pdf_folder
        self.persist_dir = persist_dir
        self.qa_chain = self.setup_pipeline()

    def setup_pipeline(self):
        documents = []
        for file in os.listdir(self.pdf_folder):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(self.pdf_folder, file))
                documents.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        docs = text_splitter.split_documents(documents)

        embedding = OllamaEmbeddings(model="nomic-embed-text")
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embedding,
            persist_directory=self.persist_dir
        )

        llm = Ollama(model="llama3.1", num_ctx=2048)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        return qa_chain

    def ask(self, query: str):
        response = self.qa_chain.invoke(query)
        return {
            "query": query,
            "answer": response["result"],
            "sources": [
                {
                    "source": doc.metadata["source"],
                    "content": doc.page_content[:200]
                }
                for doc in response["source_documents"]
            ]
        }
