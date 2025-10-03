import os
from pathlib import Path
from typing import List, Dict, Any

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")

# Consistent chunking across the app
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3:8b"


class RAGPipeline:
    def __init__(self, persist_dir: str = PERSIST_DIR):
        self.persist_dir = os.path.abspath(persist_dir)
        self.vectorstore = None
        self.qa_chain = None

        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
            print(f"🔍 Loading vector store from: {self.persist_dir}")
            self.vectorstore = self.load_vectorstore()
            self.qa_chain = self.setup_chain()
        else:
            print(f"⚠️ No vector store found at {self.persist_dir}")

    # ---- Document loading (PDF + text + directories) ----
    def _load_docs(self, paths: List[str]):
        docs = []
        for p in map(Path, paths):
            items = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
            for f in items:
                suffix = f.suffix.lower()
                try:
                    if suffix == ".pdf":
                        loader = PyPDFLoader(str(f))
                    else:
                        loader = TextLoader(
                            str(f),
                            encoding="utf-8",
                            autodetect_encoding=True
                        )
                    docs.extend(loader.load())
                    print(f"Loaded: {f}")
                except Exception as e:
                    print(f"⚠️ Skipping {f} due to error: {e}")
        return docs

    def build_vectorstore(self, files: List[str]):
        """Build a Chroma vector store from mixed inputs (files or directories)."""
        print("📚 Building vector store (PDF + text supported)...")
        embeddings = OllamaEmbeddings(model=EMBED_MODEL)

        raw_docs = self._load_docs(files)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks = splitter.split_documents(raw_docs)

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=self.persist_dir
        )
        self.vectorstore.persist()
        print(f"✅ Vector store saved at {self.persist_dir}")
        self.qa_chain = self.setup_chain()

    def load_vectorstore(self):
        embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        return Chroma(persist_directory=self.persist_dir, embedding_function=embeddings)

    def setup_chain(self):
        # Increase context window to reduce truncation
        llm = OllamaLLM(model=LLM_MODEL, num_ctx=8192)

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are a concise assistant that ONLY uses the provided context.\n"
                'If the answer is not in the context, reply: "Unable to tell from the provided documents."\n\n'
                "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:\n"
            )
        )

        # MMR retrieval for better coverage, slightly higher k
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5}
        )

        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

    def ask(self, question: str) -> Dict[str, Any]:
        if not self.qa_chain:
            return {
                "question": question,
                "answer": "❌ No QA chain. Build the vector store first.",
                "sources": []
            }

        response = self.qa_chain({"query": question})
        answer = response.get("result", "")
        sources = [
            {"source": doc.metadata.get("source", "N/A")}
            for doc in response.get("source_documents", [])
        ]
        return {"question": question, "answer": answer, "sources": sources}
