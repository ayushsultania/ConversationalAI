import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from src.llm_factory import get_llm
from src.cv_analyzer import extract_text_from_cv, generate_cv_questionnaire
from src.gemini_chain import get_conversational_response
from src.pdf_handler import get_pdf_chunks  # Keep your existing PDF splitter
from src.vectorstore import get_vectorstore  # Keep your existing vectorstore loader

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Multi-LLM Conversational AI & CV Studio",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM MODERN CSS ---
st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 1.5rem 2rem;
        border-radius: 14px;
        color: #f8fafc;
        margin-bottom: 1.5rem;
        border: 1px solid #334155;
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .main-header p {
        margin: 0.35rem 0 0 0;
        color: #94a3b8;
        font-size: 0.95rem;
    }
    .model-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: #1d4ed8;
        color: #ffffff;
        margin-top: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "questionnaire" not in st.session_state:
    st.session_state.questionnaire = ""

# --- SIDEBAR: MULTI-LLM CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Model & Agent Settings")

    provider = st.selectbox(
        "Select LLM Provider",
        [
            "Google Gemini",
            "Llama (Groq Cloud)",
            "Llama (Local Ollama)",
            "OpenAI",
            "Custom OpenAI-Compatible",
        ],
    )

    # Provider-specific model presets & key inputs
    api_key = ""
    base_url = ""

    if provider == "Google Gemini":
        model_name = st.selectbox(
            "Gemini Model",
            ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
        )
        api_key = st.text_input("Google API Key", type="password", placeholder="AIza...")

    elif provider == "Llama (Groq Cloud)":
        model_name = st.selectbox(
            "Llama Model (via Groq)",
            [
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "llama3-70b-8192",
            ],
        )
        api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")

    elif provider == "Llama (Local Ollama)":
        model_name = st.text_input("Ollama Model Name", value="llama3.2")
        base_url = st.text_input("Ollama Base URL", value="http://localhost:11434")

    elif provider == "OpenAI":
        model_name = st.selectbox("OpenAI Model", ["gpt-4o", "gpt-4o-mini", "o3-mini"])
        api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")

    else:  # Custom OpenAI-Compatible (OpenRouter, Together, Mistral, DeepSeek, etc.)
        base_url = st.text_input(
            "API Base URL",
            placeholder="https://openrouter.ai/api/v1",
        )
        model_name = st.text_input(
            "Model ID",
            placeholder="meta-llama/llama-3.3-70b-instruct",
        )
        api_key = st.text_input("Custom API Key", type="password")

    temperature = st.slider("Creativity (Temperature)", 0.0, 1.0, 0.3, 0.05)
    enable_web_search = st.toggle("🌐 Enable Web Search Augmentation", value=True)

    st.divider()
    st.subheader("📚 Knowledge Base (PDFs)")
    uploaded_docs = st.file_uploader(
        "Upload PDFs for RAG Chat",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if uploaded_docs and st.button("Index Documents", use_container_width=True):
        with st.spinner("Chunking and indexing PDFs..."):
            chunks = get_pdf_chunks(uploaded_docs)
            vectorstore = get_vectorstore(chunks)
            st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            st.success(f"Indexed {len(uploaded_docs)} document(s)!")

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# --- HEADER BANNER ---
st.markdown(
    f"""
    <div class="main-header">
        <h1>✨ Multi-Agent Conversational AI & CV Studio</h1>
        <p>Chat with your documents, search the web, and generate tailored interview questionnaires from any CV.</p>
        <span class="model-badge">Active Engine: {provider} • {model_name}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- MAIN TABS ---
tab_chat, tab_cv = st.tabs(["💬 Multi-LLM & RAG Chat", "📄 CV Questionnaire Generator"])

# ==========================================
# TAB 1: CONVERSATIONAL AI + SEARCH + RAG
# ==========================================
with tab_chat:
    for msg in st.session_state.chat_history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    user_prompt = st.chat_input("Ask anything about your PDFs, code, or live web topics...")
    if user_prompt:
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner(f"Thinking with {model_name}..."):
                try:
                    llm = get_llm(provider, model_name, api_key, base_url, temperature)
                    answer = get_conversational_response(
                        llm=llm,
                        question=user_prompt,
                        chat_history=st.session_state.chat_history,
                        retriever=st.session_state.retriever,
                        enable_web_search=enable_web_search,
                    )
                    st.markdown(answer)
                    st.session_state.chat_history.append(HumanMessage(content=user_prompt))
                    st.session_state.chat_history.append(AIMessage(content=answer))
                except Exception as e:
                    st.error(f"Error: {e}")

# ==========================================
# TAB 2: CV QUESTIONNAIRE GENERATOR
# ==========================================
with tab_cv:
    col1, col2 = st.columns([1, 1.5], gap="large")

    with col1:
        st.subheader("1. Upload Candidate CV")
        cv_file = st.file_uploader("Upload Resume / CV (PDF)", type=["pdf"], key="cv_uploader")
        target_role = st.text_input(
            "Target Role (Optional)",
            placeholder="e.g., Senior ML Engineer, Full-Stack Developer",
        )
        num_questions = st.slider("Number of Questions", min_value=5, max_value=20, value=10)
        cv_web_search = st.checkbox(
            "Enrich questions with live web search trends",
            value=True,
        )

        generate_btn = st.button(
            "🚀 Generate Tailored Questionnaire",
            type="primary",
            use_container_width=True,
            disabled=cv_file is None,
        )

    with col2:
        st.subheader("2. Generated Interview Questionnaire")
        if generate_btn and cv_file is not None:
            with st.spinner("Analyzing CV, searching role trends, and crafting questions..."):
                try:
                    llm = get_llm(provider, model_name, api_key, base_url, temperature)
                    cv_text = extract_text_from_cv(cv_file)
                    if not cv_text:
                        st.warning("Could not extract readable text from this PDF.")
                    else:
                        result = generate_cv_questionnaire(
                            llm=llm,
                            cv_text=cv_text,
                            target_role=target_role,
                            use_web_search=cv_web_search,
                            num_questions=num_questions,
                        )
                        st.session_state.questionnaire = result
                except Exception as e:
                    st.error(f"Failed to generate questionnaire: {e}")

        if st.session_state.questionnaire:
            st.markdown(st.session_state.questionnaire)
            st.download_button(
                label="📥 Download Questionnaire (Markdown)",
                data=st.session_state.questionnaire,
                file_name="cv_interview_questionnaire.md",
                mime="text/markdown",
            )
        else:
            st.info(
                "Upload a candidate's CV on the left and click **Generate Tailored Questionnaire** "
                "to create role-specific technical, scenario, and behavioral questions."
            )