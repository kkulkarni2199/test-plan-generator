"""
Configuration management for the Knowledge Graph API
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # API Settings
    app_name: str = "Knowledge Graph API"
    app_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    # External Data Source
    external_api_url: str = "https://api.example.com/standards"
    external_api_key: Optional[str] = None

    # LLM Configuration
    llm_provider: str = "openai"  # "openai" or "gemini"
    openai_api_key: Optional[str] = "not-needed"  # Local model doesn't need key
    openai_api_base: str = "http://localhost:1234/v1"  # Local LLM server
    openai_model: str = "qwen/qwen3-vl-4b"  # Local Qwen model
    openai_temperature: float = 0.2
    openai_max_tokens: int = 4096

    # Google Gemini Configuration
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-3-flash-preview"

    # Database
    database_url: str = "sqlite+aiosqlite:///./knowledge_graph.db"
    graph_storage_path: str = "./graph_data"
    vector_db_path: str = "./chroma_db"

    # Storage Paths
    upload_dir: str = "./uploads"
    output_dir: str = "./output"
    temp_dir: str = "./temp"
    data_dir: str = "./data"

    # Logging
    log_level: str = "INFO"
    log_file: str = "./logs/app.log"

    # Embedding Model
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # Processing
    batch_size: int = 32
    max_workers: int = 4

    # API Rate Limiting
    rate_limit_per_minute: int = 100

    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()

# Create necessary directories
def create_directories():
    """Create required directories if they don't exist"""
    dirs = [
        settings.upload_dir,
        settings.output_dir,
        settings.temp_dir,
        settings.data_dir,
        settings.graph_storage_path,
        settings.vector_db_path,
        os.path.dirname(settings.log_file)
    ]

    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

# Initialize on import
create_directories()
