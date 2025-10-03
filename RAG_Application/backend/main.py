import os
import json
import csv
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from jose import JWTError, jwt
from pydantic import BaseModel

# ------------------------
# Try to import RAG pipeline
# ------------------------
try:
    from rag_pipeline import RAGPipeline, PDF_FOLDER
except ImportError:
    RAGPipeline = None
    PDF_FOLDER = "RAGData"
    print("[main] Warning: rag_pipeline.py not found. Using dummy mode.")

# ------------------------
# JWT Config
# ------------------------
SECRET_KEY = os.getenv("JWT_SECRET", "supersecretjwtkey")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
EMPLOYEE_CSV = os.getenv("EMPLOYEE_CSV", "employees.csv")


def load_employees():
    employees = {}
    if not os.path.exists(EMPLOYEE_CSV):
        print(f"[auth] ⚠️ Employees file not found: {EMPLOYEE_CSV}")
        return employees
    with open(EMPLOYEE_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            employees[row["id"]] = row["password"]
    print(f"[auth] ✅ Loaded {len(employees)} employees")
    return employees


EMPLOYEES = load_employees()


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None or user_id not in EMPLOYEES:
            raise HTTPException(status_code=401, detail="Invalid user")
        return user_id
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


# ------------------------
# FastAPI App
# ------------------------
app = FastAPI(title="IFSCA IntelliChat", version="1.0.0")

# Enable CORS (still useful if you later serve frontend separately)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all for now (tighten in prod)
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# Load RAG pipeline
# ------------------------
rag = None
try:
    if RAGPipeline:
        rag = RAGPipeline()
        print("[main] ✅ RAG pipeline initialized")
except Exception as e:
    print(f"[main] Warning: failed to initialize RAG pipeline: {e}")

if rag is None:
    class DummyRAG:
        def ask(self, user_id, question):
            return {
                "answer": f"(Dummy) No RAG pipeline loaded. Placeholder answer for: {question}",
                "sources": []
            }
    rag = DummyRAG()
    print("[main] Using DummyRAG for responses")


# ------------------------
# Models
# ------------------------
class ChatRequest(BaseModel):
    question: str


# ------------------------
# Login Endpoint
# ------------------------
@app.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user_id = form_data.username
    password = form_data.password

    if user_id not in EMPLOYEES or EMPLOYEES[user_id] != password:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": user_id})
    return {"access_token": token, "token_type": "bearer"}


# ------------------------
# Chat API (SSE)
# ------------------------
@app.post("/api/chat")
async def chat_sse(payload: ChatRequest, authorization: str = Header(default=None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    token = authorization.split(" ")[1]
    user_id = verify_token(token)

    question = payload.question
    try:
        result = rag.ask(user_id, question)   # ✅ pass user_id
    except Exception as e:
        result = {"answer": f"Error running pipeline: {e}", "sources": []}

    answer = result.get("answer", "")
    sources = result.get("sources", [])

    def event_stream():
        full_payload = {
            "type": "definition",
            "term": question.strip(),
            "definition": answer.strip(),
            "sources": sources,
            "user": user_id,
        }
        yield f"data: {json.dumps(full_payload)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ------------------------
# WebSocket Chat
# ------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=1008)
        return

    try:
        user_id = verify_token(token)
    except Exception:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    try:
        while True:
            question = await websocket.receive_text()
            result = rag.ask(user_id, question)   # ✅ pass user_id
            result["user"] = user_id
            await websocket.send_json(result)
    except WebSocketDisconnect:
        pass


# ------------------------
# Serve Frontend React Build
# ------------------------
FRONTEND_DIST = os.path.join(os.path.dirname(__file__), "../frontend/dist")

if os.path.exists(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")

    @app.get("/{full_path:path}")
    async def serve_react_app(full_path: str):
        """Serve React index.html for any unknown path (SPA fallback)."""
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))
