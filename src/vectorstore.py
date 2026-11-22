import os
import shutil
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec, PodSpec
from langchain_pinecone import PineconeVectorStore
from langchain_chroma import Chroma
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings
)
from langchain.indexes import (
    SQLRecordManager,
    index
)
from src.pdf_handler import (
    extract_pdf,
    load_pdf_directory,
    split_pdf
)
load_dotenv()


# -----------------------------------------------------
# EMBEDDING MODEL
# -----------------------------------------------------

def get_embedding_model():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )


# -----------------------------------------------------
# PINECONE SETUP
# -----------------------------------------------------

def setup_pinecone(
    index_name,
    embedding_model,
    embedding_dim,
    metric="cosine",
    use_serverless=True
):

    pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )

    if use_serverless:
        spec = ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    else:
        spec = PodSpec()

    # Create index only if not existing

    existing_indexes = pc.list_indexes().names()

    if index_name not in existing_indexes:

        pc.create_index(
            name=index_name,
            dimension=embedding_dim,
            metric=metric,
            spec=spec
        )

    return PineconeVectorStore(
        index_name=index_name,
        embedding=embedding_model
    )


# -----------------------------------------------------
# CHROMA SETUP
# -----------------------------------------------------

def setup_chroma(
    collection_name,
    embedding_model,
    persist_directory
):

    os.makedirs(
        persist_directory,
        exist_ok=True
    )

    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=persist_directory
    )


# -----------------------------------------------------
# VECTOR DATABASE
# -----------------------------------------------------

class VectorDB:

    def __init__(
        self,
        db_name,
        index_name,
        cache_dir=None
    ):

        embedding_model = get_embedding_model()

        embedding_dim = 3072

        if cache_dir is None:
            cache_dir = "./.cache/database"

        self.cache_dir = cache_dir

        os.makedirs(
            self.cache_dir,
            exist_ok=True
        )

        # -------------------
        # Pinecone
        # -------------------

        if db_name.lower() == "pinecone":

            self.vectorstore = setup_pinecone(
                index_name=index_name,
                embedding_model=embedding_model,
                embedding_dim=embedding_dim,
                metric="cosine"
            )

        # -------------------
        # Chroma
        # -------------------

        else:

            self.vectorstore = setup_chroma(
                collection_name=index_name,
                embedding_model=embedding_model,
                persist_directory=self.cache_dir
            )

        namespace = f"{db_name}/{index_name}"

        self.record_manager = SQLRecordManager(
            namespace=namespace,
            db_url=f"sqlite:///{self.cache_dir}/record_manager_cache.sql"
        )

        self.record_manager.create_schema()

    # -------------------------------------------------
    # ADD DOCUMENTS
    # -------------------------------------------------

    def index(
        self,
        uploaded_file
    ):

        directory = extract_pdf(
            uploaded_file
        )

        docs = load_pdf_directory(
            directory
        )

        chunks = split_pdf(
            docs
        )

        index(
            docs_source=chunks,
            record_manager=self.record_manager,
            vector_store=self.vectorstore,
            cleanup="full",
            source_id_key="source"
        )

        # cleanup extracted files

        for file in os.listdir(directory):

            file_path = os.path.join(
                directory,
                file
            )

            if os.path.isfile(file_path):
                os.remove(file_path)

    # -------------------------------------------------
    # RETRIEVER
    # -------------------------------------------------

    def as_retriever(
        self,
        search_k=5
    ):

        return self.vectorstore.as_retriever(
            search_kwargs={
                "k": search_k
            }
        )

    # -------------------------------------------------
    # CLEANUP
    # -------------------------------------------------

    def __del__(self):

        try:

            if os.path.exists(
                self.cache_dir
            ):
                shutil.rmtree(
                    self.cache_dir,
                    ignore_errors=True
                )

        except Exception:
            pass