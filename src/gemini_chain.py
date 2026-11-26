from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from src.cv_analyzer import web_search_context


def get_conversational_response(
    llm,
    question: str,
    chat_history: list,
    retriever=None,
    enable_web_search: bool = False,
) -> str:
    """
    Generates a response using document context (RAG), optional web search,
    and conversation history across any selected LLM provider.
    """
    doc_context = ""
    if retriever is not None:
        docs = retriever.invoke(question)
        doc_context = "\n\n".join(doc.page_content for doc in docs)

    search_context = ""
    if enable_web_search:
        search_context = web_search_context(question)

    system_prompt = (
        "You are a sharp, helpful AI assistant. Use the provided Document Context and/or "
        "Web Search Results to answer the user's question accurately. If neither contains "
        "the answer, rely on your general knowledge and clearly say so.\n\n"
        "--- Document Context ---\n{doc_context}\n\n"
        "--- Web Search Context ---\n{search_context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    return chain.invoke(
        {
            "doc_context": doc_context or "No uploaded documents.",
            "search_context": search_context or "Web search disabled.",
            "chat_history": chat_history,
            "question": question,
        }
    )