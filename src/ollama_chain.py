import os
from dotenv import load_dotenv

load_dotenv()

from langchain_ollama import ChatOllama
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain
)
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)

from src.vectorstore import VectorDB
from src.utils import load_config


class OllamaChain:

    def __init__(self, chat_memory, model_name=None, base_url=None):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful AI assistant."),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        config = load_config()
        # resolved_model = model_name or "llama3.2"
        resolved_model = model_name or "llama3.2:1b"
        resolved_base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        self.llm = ChatOllama(
            model=resolved_model,
            base_url=resolved_base_url,
            temperature=config["chat_model"]["temperature"],
            num_ctx=4096
        )

        self.chat_memory = chat_memory

        chain = prompt | self.llm | StrOutputParser()

        self.conversation_chain = RunnableWithMessageHistory(
            chain,
            lambda session_id: self.chat_memory,
            input_messages_key="input",
            history_messages_key="chat_history"
        )

    def run(self, user_input):
        return self.conversation_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "default"}}
        )


class OllamaRAGChain:

    def __init__(
        self,
        chat_memory,
        model_name=None,
        base_url=None,
        uploaded_file=None
    ):
        self.vector_db = VectorDB(
            db_name="pinecone",
            index_name="any"
        )

        if uploaded_file:
            self.update_knowledge_base(uploaded_file)

        config = load_config()
        # resolved_model = model_name or "llama3.2"
        resolved_model = model_name or "llama3.2:1b"
        resolved_base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        self.llm = ChatOllama(
            model=resolved_model,
            base_url=resolved_base_url,
            temperature=config["chat_model"]["temperature"],
            num_ctx=4096
        )

        self.chat_memory = chat_memory

        contextual_q_system_prompt = """
        Given a chat history and the latest user question which might
        refer to context in the chat history, formulate a standalone
        question.

        Do NOT answer the question.
        Only reformulate if necessary.
        """

        self.contextual_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextual_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        self.history_aware_retriever = create_history_aware_retriever(
            self.llm,
            self.vector_db.as_retriever(),
            self.contextual_q_prompt
        )

        qa_system_prompt = """
        You are an assistant for question-answering tasks.

        Use the retrieved context to answer.

        If the answer is not available in the context,
        say you do not know.

        Context:
        {context}
        """

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        self.question_answer_chain = create_stuff_documents_chain(
            self.llm,
            qa_prompt
        )

        rag_chain = create_retrieval_chain(
            self.history_aware_retriever,
            self.question_answer_chain
        )

        self.conversation_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            lambda session_id: self.chat_memory,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

    def run(self, user_input):
        response = self.conversation_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "default"}}
        )
        return response["answer"]

    def update_chain(self, uploaded_pdf):
        self.update_knowledge_base(uploaded_pdf)

        self.history_aware_retriever = create_history_aware_retriever(
            self.llm,
            self.vector_db.as_retriever(),
            self.contextual_q_prompt
        )

        rag_chain = create_retrieval_chain(
            self.history_aware_retriever,
            self.question_answer_chain
        )

        self.conversation_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            lambda session_id: self.chat_memory,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

    def update_knowledge_base(self, uploaded_pdf):
        self.vector_db.index(uploaded_pdf)