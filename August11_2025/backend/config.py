from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PERSIST_DIR: str = "chroma_db"
    PDF_FOLDER: str = "RAGData"
    EMBEDDING_MODEL: str = "nomic-embed-text"
    LLM_MODEL: str = "phi:2" # "llama3-chatqa"
    LLM_NUM_CTX: int = 2048

SETTINGS = Settings()