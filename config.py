from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    PROJECT_NAME: str = "Neuro-Symbolic Agentic RAG Engine"
    VERSION: str = "1.0.0"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    PERSIST_DIRECTORY: str = "./chroma_db"
    API_PORT: int = 8000
    API_HOST: str = "0.0.0.0"

settings = Settings()
