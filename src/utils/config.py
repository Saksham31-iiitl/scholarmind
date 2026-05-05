"""Centralized configuration using pydantic-settings."""
from __future__ import annotations
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # FREE options:  groq | gemini | huggingface | ollama
    # PAID options:  anthropic | openai
    llm_provider: str = "groq"
    llm_model: str = "llama-3.1-8b-instant"

    # Free API Keys
    groq_api_key: str = ""           # FREE → https://console.groq.com
    google_api_key: str = ""         # FREE → https://aistudio.google.com
    huggingface_api_key: str = ""    # FREE → https://huggingface.co/settings/tokens

    # Paid (leave blank)
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # Neo4j (accepts NEO4J_USER or NEO4J_USERNAME)
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_username: str = ""
    neo4j_password: str = "scholarmind123"

    # Paths
    vector_store_path: Path = Path("./data/vector_store")
    raw_data_path: Path = Path("./data/raw")
    processed_data_path: Path = Path("./data/processed")

    # Retrieval (both downloaded FREE from HuggingFace automatically)
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k_retrieve: int = 20
    top_k_rerank: int = 5

    # Retrieval options (set USE_DENSE=true locally for full hybrid retrieval)
    use_dense: bool = False
    use_reranker: bool = False

    # Agents
    max_critic_iterations: int = 3


settings = Settings()