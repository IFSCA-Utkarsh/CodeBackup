# RAG pipeline wrapper
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from config import SETTINGS

class RAGPipeline:
    def __init__(self):
        self.vectorstore = self.load_vectorstore()
        self.qa_chain = self.setup_chain()

    def load_vectorstore(self):
        embedding = OllamaEmbeddings(model=SETTINGS.EMBEDDING_MODEL)
        return Chroma(persist_directory=SETTINGS.PERSIST_DIR, embedding_function=embedding)

    def setup_chain(self):
        llm = OllamaLLM(model=SETTINGS.LLM_MODEL, num_ctx=SETTINGS.LLM_NUM_CTX)

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
        # Call the chain and return structured response
        try:
            res = self.qa_chain({"query": question})
        except Exception:
            # fallback to run if the chain interface differs
            res_text = self.qa_chain.run(question)
            return {"question": question, "answer": res_text, "sources": []}

        # `res` may be either a string or a dict with 'result' and 'source_documents'
        answer = None
        source_docs = []

        if isinstance(res, dict):
            answer = res.get("result") or res.get("answer")
            source_documents = res.get("source_documents") or res.get("source_documents") or []
        else:
            answer = str(res)
            source_documents = []

        for doc in source_documents:
            source_docs.append({
                "source": getattr(doc.metadata, "get", lambda *a, **k: doc.metadata.get("source", "N/A"))("source", "N/A") if hasattr(doc, "metadata") else "N/A",
                "content": getattr(doc, "page_content", str(doc))[:200]
            })

        return {"question": question, "answer": answer, "sources": source_docs}