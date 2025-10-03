import os
import json
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Import your RAG pipeline
try:
    from rag_pipeline import RAGPipeline
except ImportError:
    RAGPipeline = None
    print("[main] Warning: rag_pipeline.py not found. Using dummy mode.")

# ------------------------
# Initialize FastAPI
# ------------------------
app = FastAPI(title="Tech Trends AI Chatbot", version="0.1.0")

# Enable CORS for all origins (tune as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# Simple API Key Auth
# ------------------------
API_KEY = os.getenv("API_KEY")  # set this in your environment

def verify_key(x_api_key: Optional[str]):
    """Raise 401 if API key invalid when API_KEY is set; if not set, auth is disabled."""
    if API_KEY is None:
        # Auth disabled (no env var provided)
        return
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

# ------------------------
# Load RAG pipeline or dummy fallback
# ------------------------
rag = None
try:
    if RAGPipeline:
        rag = RAGPipeline()
        print("[main] RAG pipeline initialized successfully.")
except Exception as e:
    print(f"[main] Warning: failed to initialize RAG pipeline: {e}")

if rag is None:
    class DummyRAG:
        def ask(self, question):
            return {
                "answer": f"(Dummy) No RAG pipeline loaded. Here's a placeholder answer for: {question}",
                "sources": []
            }
    rag = DummyRAG()
    print("[main] Using DummyRAG for responses.")

# ------------------------
# Request model for /api/chat
# ------------------------
class ChatRequest(BaseModel):
    question: str

# ------------------------
# WebSocket chat endpoint
# ------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_key = websocket.headers.get("x-api-key") or websocket.query_params.get("api_key")
    # Enforce key only if API_KEY is set
    if API_KEY is not None and client_key != API_KEY:
        await websocket.close(code=1008)  # policy violation
        return

    await websocket.accept()
    try:
        while True:
            question = await websocket.receive_text()
            try:
                result = rag.ask(question)
            except Exception as e:
                result = {
                    "question": question,
                    "answer": f"Error running pipeline: {e}",
                    "sources": [],
                }
            await websocket.send_json(result)
    except WebSocketDisconnect:
        pass

# ------------------------
# SSE chat endpoint
# ------------------------
@app.post("/api/chat")
async def chat_sse(payload: ChatRequest, x_api_key: Optional[str] = Header(default=None)):
    verify_key(x_api_key)

    question = payload.question

    # Run RAG pipeline
    try:
        result = rag.ask(question)
    except Exception as e:
        result = {
            "answer": f"Error running pipeline: {e}",
            "sources": [],
        }

    answer = result.get("answer", "")
    sources = result.get("sources", [])

    # SSE generator (single JSON event + [DONE])
    def event_stream():
        full_payload = {
            "type": "definition",
            "term": question.strip(),
            "definition": answer.strip(),
            "sources": sources
        }
        yield f"data: {json.dumps(full_payload)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
