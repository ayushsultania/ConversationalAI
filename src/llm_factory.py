import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


def get_llm(
    provider: str,
    model_name: str,
    api_key: str = "",
    base_url: str = "",
    temperature: float = 0.3,
):
    """
    Returns a unified LangChain ChatModel for Gemini, Llama (Groq/Ollama),
    OpenAI, or any OpenAI-compatible API endpoint.
    """
    provider = provider.lower()

    if provider == "google gemini":
        key = api_key or os.getenv("GOOGLE_API_KEY", "")
        if not key:
            raise ValueError("Google Gemini API Key is required.")
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=key,
            temperature=temperature,
        )

    elif provider == "llama (groq cloud)":
        key = api_key or os.getenv("GROQ_API_KEY", "")
        if not key:
            raise ValueError("Groq API Key is required for cloud Llama models.")
        return ChatGroq(
            model=model_name,
            groq_api_key=key,
            temperature=temperature,
        )

    elif provider == "llama (local ollama)":
        return ChatOllama(
            model=model_name,
            base_url=base_url or "http://localhost:11434",
            temperature=temperature,
        )

    elif provider == "openai":
        key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("OpenAI API Key is required.")
        return ChatOpenAI(
            model=model_name,
            api_key=key,
            temperature=temperature,
        )

    elif provider == "custom openai-compatible":
        if not api_key or not base_url:
            raise ValueError("Both Base URL and API Key are required for custom endpoints.")
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
        )

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")