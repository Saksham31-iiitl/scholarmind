"""Unified LLM factory — free and paid providers."""
from __future__ import annotations
from langchain_core.language_models import BaseChatModel
from src.utils.config import settings
from src.utils.logging import logger


def get_llm(temperature: float = 0.2, max_tokens: int = 2048) -> BaseChatModel:
    provider = settings.llm_provider.lower()
    logger.info(f"LLM provider={provider} model={settings.llm_model}")

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=settings.llm_model,
            api_key=settings.groq_api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=settings.llm_model,
            google_api_key=settings.google_api_key,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    if provider == "huggingface":
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
        endpoint = HuggingFaceEndpoint(
            repo_id=settings.llm_model,
            huggingfacehub_api_token=settings.huggingface_api_key,
            temperature=temperature,
            max_new_tokens=max_tokens,
        )
        return ChatHuggingFace(llm=endpoint)

    if provider == "ollama":
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(model=settings.llm_model, temperature=temperature)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.llm_model,
            api_key=settings.anthropic_api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    raise ValueError(f"Unknown provider: '{provider}'")