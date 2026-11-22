import os

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain.memory import ConversationBufferWindowMemory

from langchain.chains import (
    LLMChain,
    create_history_aware_retriever,
    create_retrieval_chain
)

from langchain.chains.combine_documents import (
    create_stuff_documents_chain
)

from langchain_core.runnables.history import (
    RunnableWithMessageHistory
)

from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document

from src.vectorstore import VectorDB
from src.utils import load_config


# Set API key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


def format_docs(docs: list[Document]):
    return "\n\n".join(doc.page_content for doc in docs)


class GeminiChain:

    def __init__(self, chat_memory):

        prompt = PromptTemplate(
            template="""
            Previous conversation:
            {chat_history}

            User Question:
            {input}

            Assistant:
            """,
            input_variables=["chat_history", "input"]
        )

        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            chat_memory=chat_memory,
            k=3,
            return_messages=True
        )

        config = load_config()

        self.llm = ChatGoogleGenerativeAI(
            model=config["chat_model"]["model"],
            temperature=config["chat_model"]["temperature"]
            )

        self.llm_chain = LLMChain(
            prompt=prompt,
            llm=self.llm,
            memory=self.memory,
            output_parser=StrOutputParser()
        )

    def run(self, user_input):

        response = self.llm_chain.invoke(
            {"input": user_input}
        )

        return response["text"]


class GeminiRAGChain:

    def __init__(
        self,
        chat_memory,
        uploaded_file=None
    ):

        self.vector_db = VectorDB(
            db_name="pinecone",
            index_name="any"
        )

        if uploaded_file:
            self.update_knowledge_base(uploaded_file)

        config = load_config()

        self.llm = ChatGoogleGenerativeAI(
            model=config["chat_model"]["model"],
            temperature=config["chat_model"]["temperature"]
            )

        self.chat_memory = chat_memory

        contextual_q_system_prompt = """
        Given a chat history and the latest user question which might
        refer to context in the chat history, formulate a standalone
        question.

        Do NOT answer the question.
        Only reformulate if necessary.
        """

        self.contextual_q_prompt = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", contextual_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}")
                ]
            )
        )

        self.history_aware_retriever = (
            create_history_aware_retriever(
                self.llm,
                self.vector_db.as_retriever(),
                self.contextual_q_prompt
            )
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

        self.question_answer_chain = (
            create_stuff_documents_chain(
                self.llm,
                qa_prompt
            )
        )

        rag_chain = create_retrieval_chain(
            self.history_aware_retriever,
            self.question_answer_chain
        )

        self.conversation_rag_chain = (
            RunnableWithMessageHistory(
                rag_chain,
                lambda session_id: self.chat_memory,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )
        )

    def run(self, user_input):

        config = {
            "configurable": {
                "session_id": "default"
            }
        }

        response = self.conversation_rag_chain.invoke(
            {"input": user_input},
            config=config
        )

        return response["answer"]

    def update_chain(self, uploaded_pdf):

        self.update_knowledge_base(uploaded_pdf)

        self.history_aware_retriever = (
            create_history_aware_retriever(
                self.llm,
                self.vector_db.as_retriever(),
                self.contextual_q_prompt
            )
        )

        rag_chain = create_retrieval_chain(
            self.history_aware_retriever,
            self.question_answer_chain
        )

        self.conversation_rag_chain = (
            RunnableWithMessageHistory(
                rag_chain,
                lambda session_id: self.chat_memory,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )
        )

    def update_knowledge_base(self, uploaded_pdf):
        self.vector_db.index(uploaded_pdf)