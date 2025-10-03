import os
from typing import List, Dict, Any

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

BASE_DIR = os.path.abspath(os.path.dirname(__file__))  # folder of this file
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")  # always in backend/chroma_db
#PERSIST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chroma_db"))


class RAGPipeline:
    def __init__(self, persist_dir: str = PERSIST_DIR):
        self.persist_dir = os.path.abspath(persist_dir)  # ensure absolute path
        self.vectorstore = None
        self.qa_chain = None

        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
            print(f"🔍 Loading vector store from: {self.persist_dir}")
            self.vectorstore = self.load_vectorstore()
            self.qa_chain = self.setup_chain()
        else:
            print(f"⚠️ No vector store found at {self.persist_dir}")

    def build_vectorstore(self, files: List[str]):
        """Takes a list of file paths and builds a Chroma vector store."""
        print("📚 Building vector store...")
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        docs = []

        for path in files:
            if not os.path.isfile(path):
                print(f"⚠️ Skipping missing file: {path}")
                continue
            loader = TextLoader(path)
            raw_docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs.extend(splitter.split_documents(raw_docs))

        self.vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=self.persist_dir
        )
        self.vectorstore.persist()
        print(f"✅ Vector store saved at {self.persist_dir}")

        self.qa_chain = self.setup_chain()

    def load_vectorstore(self):
        """Loads an existing Chroma vector store from disk."""
        print("🔍 Loading existing vector store...")
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        return Chroma(persist_directory=self.persist_dir, embedding_function=embeddings)

    def setup_chain(self):
        """Sets up the RetrievalQA chain."""
        llm = OllamaLLM(model="llama3:8b", num_ctx=2048)

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are a concise assistant that ONLY uses the provided context.\n"
                'If the answer is not in the context, reply: "Unable to tell from the provided documents."\n\n'
                "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:\n"
            )
        )

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})

        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

    def ask(self, question: str) -> Dict[str, Any]:
        """Asks a question to the QA chain."""
        if not self.qa_chain:
            return {"question": question, "answer": "❌ No QA chain. Build the vector store first.", "sources": []}

        response = self.qa_chain({"query": question})
        answer = response.get("result", "")
        sources = [
            {
                "source": doc.metadata.get("source", "N/A"),
                #"content": doc.page_content[:400]
            }
            for doc in response.get("source_documents", [])
        ]
        return {"question": question, "answer": answer, "sources": sources}