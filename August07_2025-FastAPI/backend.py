# rag_pipeline.py

from langchain_chroma import Chroma  # for vector store
from langchain_ollama import OllamaLLM, OllamaEmbeddings # for LLM
#from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

PERSIST_DIR = "chroma_db"

class RAGPipeline:
    def __init__(self):
        self.vectorstore = self.load_vectorstore()
        self.qa_chain = self.setup_chain()

    def load_vectorstore(self):
        embedding = OllamaEmbeddings(model="nomic-embed-text")
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding)

    def setup_chain(self):
        llm = OllamaLLM(model="llama3-chatqa", num_ctx=2048)

        # ✅ Custom Prompt Template
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a helpful, honest assistant with access to official documents.

Use the context below to answer the user's question. Be concise and summarize clearly.

⚠️ If the answer cannot be found in the context, do NOT make up anything. Just say: "Unable to tell from the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""
        )

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template}
        )
        return qa_chain

    def ask(self, question: str):
        response = self.qa_chain.invoke(question)
        return {
            "question": question,
            "answer": response["result"],
            "sources": [
                {
                    "source": doc.metadata.get("source", "N/A"),
                    "content": doc.page_content[:200]
                }
                for doc in response["source_documents"]
            ]
        }