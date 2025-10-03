from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import SETTINGS
from auth import router as auth_router
from websocket_handler import router as ws_router

app = FastAPI(title="Tech Trends Chatbot - Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# include authentication routes (POST /login)
app.include_router(auth_router)

# include websocket routes (mounted at /ws)
app.include_router(ws_router)

# simple healthcheck
@app.get("/health")
def health():
    return {"status": "ok"}