import sys
import types
import pydantic.v1
import langchain
import langchain_core.prompts
import langchain_core.documents
import langchain_classic.memory
import langchain_classic.chains
import langchain_classic.chains.combine_documents
import langchain_classic.indexes
import langchain_text_splitters
import langchain_community.document_loaders

# Compatibility shims for LangChain 0.2.x imports on LangChain 1.x
sys.modules["langchain_core.pydantic_v1"] = pydantic.v1
sys.modules["langchain.prompts"] = langchain_core.prompts
sys.modules["langchain.memory"] = langchain_classic.memory
sys.modules["langchain.chains"] = langchain_classic.chains
sys.modules["langchain.chains.combine_documents"] = langchain_classic.chains.combine_documents
sys.modules["langchain.indexes"] = langchain_classic.indexes
sys.modules["langchain.text_splitter"] = langchain_text_splitters
sys.modules["langchain.document_loaders"] = langchain_community.document_loaders

schema_shim = types.ModuleType("langchain.schema")
schema_shim.__path__ = []  # Marks it as a package so submodules can be imported
schema_shim.Document = langchain_core.documents.Document
sys.modules["langchain.schema"] = schema_shim
sys.modules["langchain.schema.document"] = langchain_core.documents



import os
import streamlit as st
from dotenv import load_dotenv

from src.gemini_chain import GeminiChain, GeminiRAGChain
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory
)
from src.openai_chain import OpenAIChain, OpenAIRAGChain
from src.ollama_chain import OllamaChain, OllamaRAGChain
# Load environment variables from .env file
load_dotenv()
# =====================================================
# MODEL CONFIGURATION
# =====================================================

# =====================================================
# MODEL & ENV CONFIGURATION
# =====================================================

PROVIDER_CONFIG = {
    "Google Gemini": {
        "env_var": "GEMINI_API_KEY",  # or "GOOGLE_API_KEY" depending on your setup
        "base_url": None,
        "models": [
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash"
        ]
    },
    "OpenAI": {
        "env_var": "OPENAI_API_KEY",
        "base_url": None,
        "models": [
            "gpt-4o",
            "gpt-4o-mini",
            "o3-mini"
        ]
    },
    "Ollama (Local)": {
        "env_var": None,
        "base_url": "http://localhost:11434/v1",
        "models": [
            "llama3.2:1b",
            "llama3.2:latest",
            "llama3.2"
        ]
    }
}


# =====================================================
# INITIALIZE SESSION STATE
# =====================================================

def initialize_session():
    default_provider = "Google Gemini"
    defaults = {
        "send_input": False,
        "knowledge_change": False,
        "user_question": "",
        "pdf_chat": False,
        "selected_provider": default_provider,
        "selected_model": PROVIDER_CONFIG[default_provider]["models"][0]
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value


# =====================================================
# CACHE
# =====================================================

def clear_cache():
    st.cache_resource.clear()

def on_provider_change():
    provider = st.session_state.selected_provider
    st.session_state.selected_model = PROVIDER_CONFIG[provider]["models"][0]
    clear_cache()

    if st.session_state.pdf_chat and st.session_state.get("uploaded_file"):
            st.session_state.knowledge_change = True

def clear_chat_history():
    st.session_state["history"] = []
    st.session_state.user_question = ""
    st.session_state.send_input = False
    clear_cache()

# =====================================================
# LOAD CHAIN
# =====================================================

@st.cache_resource
def load_chain(_chat_memory, provider, model_name, pdf_mode=False):
    config_entry = PROVIDER_CONFIG[provider]
    env_var_name = config_entry["env_var"]
    base_url = config_entry["base_url"]
    if env_var_name is None:
        api_key = "ollama"  # Dummy key required by ChatOpenAI client
    else:
        api_key = os.getenv(env_var_name)
        if not api_key:
            raise EnvironmentError(f"Missing `{env_var_name}` in environment variables.")

    if provider == "Ollama (Local)" or env_var_name is None:
        if pdf_mode:
            return OllamaRAGChain(_chat_memory, model_name=model_name)
        return OllamaChain(_chat_memory, model_name=model_name)

    env_var_name = PROVIDER_CONFIG[provider]["env_var"]
    api_key = os.getenv(env_var_name)

    if not api_key:
        raise EnvironmentError(f"Missing `{env_var_name}` in environment variables.")

    if provider == "Google Gemini":
        if pdf_mode:
            return GeminiRAGChain(_chat_memory, model_name=model_name, api_key=api_key)
        return GeminiChain(_chat_memory, model_name=model_name, api_key=api_key)

    elif provider == "OpenAI":
        if pdf_mode:
            return OpenAIRAGChain(_chat_memory, model_name=model_name, api_key=api_key, base_url=base_url)
        return OpenAIChain(_chat_memory, model_name=model_name, api_key=api_key, base_url=base_url)
       

    elif provider == "Anthropic":
        raise NotImplementedError(f"Anthropic chain not yet implemented for {model_name}")
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    # if pdf_mode:
    #     return GeminiRAGChain(_chat_memory)

    # return GeminiChain(_chat_memory)


# =====================================================
# PDF TOGGLE
# =====================================================

def toggle_pdf_chat():

    clear_cache()

    if (
        st.session_state.pdf_chat
        and st.session_state.uploaded_file
    ):
        st.session_state.knowledge_change = True


# =====================================================
# FILE UPLOAD
# =====================================================

def on_file_upload():

    if st.session_state.uploaded_file:

        if not st.session_state.pdf_chat:
            st.session_state.pdf_chat = True

        clear_cache()
        st.session_state.knowledge_change = True

    else:

        clear_cache()
        st.session_state.pdf_chat = False


# =====================================================
# INPUT HANDLING
# =====================================================

def store_user_question():

    question = st.session_state.user_input.strip()

    if question:
        st.session_state.user_question = question
        st.session_state.send_input = True

    st.session_state.user_input = ""


# =====================================================
# MAIN
# =====================================================

def main():

    st.set_page_config(
        page_title="Multi-LLM RAG Chat",
        page_icon="🤖",
        layout="wide"
    )

    initialize_session()

    st.title("🤖 Multi-LLM RAG Chat")

    # -------------------------------------------------
    # SIDEBAR
    # -------------------------------------------------

    with st.sidebar:

        st.header("Settings")
        # 1. API Provider Dropdown
        selected_provider = st.selectbox(
            "Select Provider",
            options=list(PROVIDER_CONFIG.keys()),
            key="selected_provider",
            on_change=on_provider_change
        )

        # 2. Dependent Model Dropdown
        selected_model = st.selectbox(
            "Select Model",
            options=PROVIDER_CONFIG[selected_provider]["models"],
            key="selected_model",
            on_change=clear_cache
        )

        # Optional: Visual indicator confirming .env key is detected
        env_var_name = PROVIDER_CONFIG[selected_provider]["env_var"]
        if env_var_name is None:
            st.caption("🟢 Local endpoint (no API key required)")
        elif os.getenv(env_var_name):
            st.caption(f"✅ `{env_var_name}` loaded")
        else:
            st.caption(f"⚠️ `{env_var_name}` not found in .env")

        st.divider()

        st.toggle(
            "PDF Chat",
            key="pdf_chat",
            on_change=toggle_pdf_chat
        )

        uploaded_pdf = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            accept_multiple_files=True,
            key="uploaded_file",
            on_change=on_file_upload
        )

        st.divider()

        if st.session_state.pdf_chat:
            st.success(f"Mode: RAG + PDF | {selected_provider} ({selected_model})")
        else:
            st.info(f"Mode: General Chat | {selected_provider} ({selected_model})")

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            clear_chat_history()
            st.rerun()

        # if st.session_state.pdf_chat:
        #     st.success("Mode: RAG + PDF")
        # else:
        #     st.info("Mode: General Gemini Chat")

    # -------------------------------------------------
    # CHAT HISTORY
    # -------------------------------------------------

    chat_memory = StreamlitChatMessageHistory(
        key="history"
    )

    # Render history

    for msg in chat_memory.messages:
        with st.chat_message(msg.type):
            st.write(msg.content)

    # -------------------------------------------------
    # LOAD CHAIN
    # -------------------------------------------------

    # llm_chain = load_chain(
    #     chat_memory,
    #     st.session_state.pdf_chat
    # )
    try:
        llm_chain = load_chain(
            _chat_memory=chat_memory,
            provider=selected_provider,
            model_name=selected_model,
            pdf_mode=st.session_state.pdf_chat
        )
    except Exception as e:
        st.error(f"Initialization Error: {str(e)}")
        st.stop()
    # -------------------------------------------------
    # UPDATE VECTOR DB
    # -------------------------------------------------

    if st.session_state.knowledge_change:

        try:

            with st.spinner(
                "Processing PDFs and updating knowledge base..."
            ):

                llm_chain.update_chain(
                    uploaded_pdf
                )

                st.session_state.knowledge_change = False

            st.sidebar.success(
                "Knowledge base updated"
            )

        except Exception as e:

            st.sidebar.error(
                f"PDF processing failed: {str(e)}"
            )

    # -------------------------------------------------
    # CHAT INPUT
    # -------------------------------------------------

    prompt = st.chat_input(
        "Ask anything..."
    )

    if prompt:

        st.session_state.user_question = prompt
        st.session_state.send_input = True

    # -------------------------------------------------
    # GENERATE RESPONSE
    # -------------------------------------------------

    if (
        st.session_state.send_input
        and st.session_state.user_question
    ):

        user_query = st.session_state.user_question

        with st.chat_message("user"):
            st.write(user_query)

        try:

            with st.chat_message("assistant"):

                with st.spinner("Thinking..."):

                    response = llm_chain.run(
                        user_input=user_query
                    )

                    st.write(response)

            st.session_state.user_question = ""
            st.session_state.send_input = False

        except Exception as e:

            with st.chat_message("assistant"):

                st.error(
                    f"Error: {str(e)}"
                )

            st.session_state.send_input = False


# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":
    main()