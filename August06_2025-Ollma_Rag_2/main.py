# main.py

from fastapi import FastAPI, Query
from pydantic import BaseModel
from rag_pipeline import RAGPipeline

app = FastAPI()
rag = RAGPipeline()

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "RAG API is running."}

@app.post("/query")
def ask_question(request: QueryRequest):
    return rag.ask(request.question)
