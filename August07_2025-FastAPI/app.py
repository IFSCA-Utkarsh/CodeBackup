# main.py

from fastapi import FastAPI
from pydantic import BaseModel
from backend import RAGPipeline

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
