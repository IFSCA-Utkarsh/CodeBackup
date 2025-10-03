from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend import RAGPipeline
from auth import authenticate, is_valid_token

app = FastAPI()
rag = RAGPipeline()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class LoginRequest(BaseModel):
    user_id: str
    password: str

@app.post("/login")
def login(req: LoginRequest):
    token = authenticate(req.user_id, req.password)
    if token:
        return {"token": token}
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.websocket("/ws/{token}")
async def websocket_endpoint(websocket: WebSocket, token: str):
    if not is_valid_token(token):
        await websocket.close(code=1008)
        return
    await websocket.accept()
    try:
        while True:
            question = await websocket.receive_text()
            result = rag.ask(question)
            await websocket.send_json(result)
    except WebSocketDisconnect:
        pass
