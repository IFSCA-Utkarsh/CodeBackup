import os
from pathlib import Path
from typing import List, Dict, Any

from langchain_milvus import Milvus
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory

# Milvus Config
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "rag_docs"

# RAG Config
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3:8b"


class RAGPipeline:
    def __init__(self):
        self.vectorstore = None
        self.user_chains: Dict[str, ConversationalRetrievalChain] = {}

        try:
            embeddings = OllamaEmbeddings(model=EMBED_MODEL)
            self.vectorstore = Milvus(
                embedding_function=embeddings,
                collection_name=COLLECTION_NAME,
                connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
            )
            print("✅ Connected to Milvus collection:", COLLECTION_NAME)
        except Exception as e:
            print("❌ Failed to connect to Milvus:", e)

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
                        loader = TextLoader(str(f), encoding="utf-8", autodetect_encoding=True)
                    docs.extend(loader.load())
                    print(f"Loaded: {f}")
                except Exception as e:
                    print(f"⚠️ Skipping {f} due to error: {e}")
        return docs

    def build_vectorstore(self, files: List[str]):
        print("📚 Building Milvus vector store...")
        embeddings = OllamaEmbeddings(model=EMBED_MODEL)

        raw_docs = self._load_docs(files)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        chunks = splitter.split_documents(raw_docs)

        self.vectorstore = Milvus.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
        )
        print(f"✅ Documents indexed into Milvus collection: {COLLECTION_NAME}")

        # Reset user chains since retriever changed
        self.user_chains.clear()

    def _get_user_chain(self, user_id: str) -> ConversationalRetrievalChain:
        """Return/create a chain with memory for a specific user."""
        if user_id in self.user_chains:
            return self.user_chains[user_id]

        if not self.vectorstore:
            raise ValueError("❌ No Milvus vector store. Build it first.")

        llm = OllamaLLM(model=LLM_MODEL, num_ctx=8192)

        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5},
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )

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
            output_key="answer",
        )

        self.user_chains[user_id] = chain
        return chain

    def ask(self, user_id: str, question: str) -> Dict[str, Any]:
        if not self.vectorstore:
            return {
                "question": question,
                "answer": "❌ No vector store. Build the Milvus vector store first.",
                "sources": [],
            }

        chain = self._get_user_chain(user_id)
        response = chain({"question": question})

        answer = response.get("answer", "")
        sources = []
        for doc in response.get("source_documents", []):
            src = doc.metadata.get("source", "N/A")
            filename = os.path.basename(src) if os.path.exists(src) else src
            src_url = f"/files/{filename}" if filename.endswith(".pdf") else src
            sources.append({"source": src_url})

        return {"question": question, "answer": answer, "sources": sources}

