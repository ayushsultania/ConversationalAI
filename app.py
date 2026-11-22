import streamlit as st

from src.gemini_chain import GeminiChain, GeminiRAGChain
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory
)


# =====================================================
# INITIALIZE SESSION STATE
# =====================================================

def initialize_session():

    defaults = {
        "send_input": False,
        "knowledge_change": False,
        "user_question": "",
        "pdf_chat": False
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value


# =====================================================
# CACHE
# =====================================================

def clear_cache():
    st.cache_resource.clear()


# =====================================================
# LOAD CHAIN
# =====================================================

@st.cache_resource
def load_chain(_chat_memory, pdf_mode=False):

    if pdf_mode:
        return GeminiRAGChain(_chat_memory)

    return GeminiChain(_chat_memory)


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
        page_title="Gemini RAG Chat",
        page_icon="🤖",
        layout="wide"
    )

    initialize_session()

    st.title("🤖 Gemini PDF Chat")

    # -------------------------------------------------
    # SIDEBAR
    # -------------------------------------------------

    with st.sidebar:

        st.header("Settings")

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
            st.success("Mode: RAG + PDF")
        else:
            st.info("Mode: General Gemini Chat")

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

    llm_chain = load_chain(
        chat_memory,
        st.session_state.pdf_chat
    )

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