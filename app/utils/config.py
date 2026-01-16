"""Configuration management for the sentiment analysis tool."""

import os
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # General settings
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Model settings
    default_model: str = Field(
        default="nlptown/bert-base-multilingual-uncased-sentiment",
        env="DEFAULT_MODEL"
    )
    cache_dir: str = Field(default="./models", env="CACHE_DIR")
    max_length: int = Field(default=512, env="MAX_LENGTH")
    
    # Translation settings
    google_translate_api_key: Optional[str] = Field(default=None, env="GOOGLE_TRANSLATE_API_KEY")
    translation_cache_ttl: int = Field(default=3600, env="TRANSLATION_CACHE_TTL")
    
    # Redis settings
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    redis_db: int = Field(default=0, env="REDIS_DB")
    
    # API settings
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=4, env="API_WORKERS")
    
    # Streamlit settings
    streamlit_port: int = Field(default=8501, env="STREAMLIT_PORT")
    streamlit_host: str = Field(default="localhost", env="STREAMLIT_HOST")
    
    # Performance settings
    batch_size: int = Field(default=32, env="BATCH_SIZE")
    max_concurrent_requests: int = Field(default=10, env="MAX_CONCURRENT_REQUESTS")
    request_timeout: int = Field(default=30, env="REQUEST_TIMEOUT")
    
    # Supported languages
    supported_languages: List[str] = [
        "en", "es", "fr", "de", "zh", "pt", "it", "ru", "ja", "ar"
    ]
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
