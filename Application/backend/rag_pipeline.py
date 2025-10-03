import os
from pathlib import Path
from typing import List, Dict, Any

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")
PDF_FOLDER = os.path.abspath("RAGData")

# Config
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3:8b"


class RAGPipeline:
    def __init__(self, persist_dir: str = PERSIST_DIR):
        self.persist_dir = os.path.abspath(persist_dir)
        self.vectorstore = None
        self.user_chains: Dict[str, ConversationalRetrievalChain] = {}

        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
            print(f"🔍 Loading vector store from: {self.persist_dir}")
            self.vectorstore = self.load_vectorstore()
        else:
            print(f"⚠️ No vector store found at {self.persist_dir}")

    def _load_docs(self, paths: List[str]):
        """Load PDF and text files from given paths (files or dirs)."""
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
                            str(f), encoding="utf-8", autodetect_encoding=True
                        )
                    docs.extend(loader.load())
                    print(f"Loaded: {f}")
                except Exception as e:
                    print(f"⚠️ Skipping {f} due to error: {e}")
        return docs

    def build_vectorstore(self, files: List[str]):
        print("📚 Building vector store (PDF + text supported)...")
        embeddings = OllamaEmbeddings(model=EMBED_MODEL)

        raw_docs = self._load_docs(files)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        chunks = splitter.split_documents(raw_docs)

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=self.persist_dir,
        )
        self.vectorstore.persist()
        print(f"✅ Vector store saved at {self.persist_dir}")

        # Reset user chains since retriever changed
        self.user_chains.clear()

    def load_vectorstore(self):
        embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        return Chroma(persist_directory=self.persist_dir, embedding_function=embeddings)

    def _get_user_chain(self, user_id: str) -> ConversationalRetrievalChain:
        """Return/create a chain with memory for a specific user."""
        if user_id in self.user_chains:
            return self.user_chains[user_id]

        if not self.vectorstore:
            raise ValueError("❌ No vector store. Build it first.")

        llm = OllamaLLM(model=LLM_MODEL, num_ctx=8192)

        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5},
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",   # ✅ store only 'answer' in memory
        )

        # Prompt enforces grounded answers
        prompt = PromptTemplate(
            input_variables=["context", "question", "chat_history"],
            template=(
                "You are a helpful assistant that ONLY uses the provided context and chat history.\n"
                "If the answer is not in the context, say: 'Unable to tell from the provided documents.'\n\n"
                "Chat history:\n{chat_history}\n\n"
                "Context:\n{context}\n\n"
                "Question:\n{question}\n\nAnswer:"
            ),
        )

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True,
            output_key="answer",   # ✅ explicitly set output key
            verbose=False,
        )

        self.user_chains[user_id] = chain
        return chain


    def ask(self, user_id: str, question: str) -> Dict[str, Any]:
        if not self.vectorstore:
            return {
                "question": question,
                "answer": "❌ No vector store. Build the vector store first.",
                "sources": [],
            }

        chain = self._get_user_chain(user_id)
        response = chain({"question": question})

        answer = response.get("answer", "")
        sources = []
        for doc in response.get("source_documents", []):
            src = doc.metadata.get("source", "N/A")
            if src.endswith(".pdf") and os.path.exists(src):
                filename = os.path.basename(src)
                src_url = f"/files/{filename}"
            else:
                src_url = src
            sources.append({"source": src_url})

        return {"question": question, "answer": answer, "sources": sources}