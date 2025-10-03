from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from auth import is_valid_token
from rag_pipeline import RAGPipeline

router = APIRouter()

# single RAG instance shared by websocket connections
rag = RAGPipeline()

@router.websocket("/ws/{token}")
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