import json
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
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

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
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

# Fallback if RAG not loaded
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
async def chat_sse(payload: ChatRequest):
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

    # SSE generator
    def event_stream():
    # Combine into one final structured response for the frontend
        full_payload = {
        "type": "definition",
        "term": question.strip(),
        "definition": answer.strip(),
        "sources": sources
        }

    # Send the definition in one chunk
        yield f"data: {json.dumps(full_payload)}\n\n"

    # Mark the stream as done (optional for your parser)
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
