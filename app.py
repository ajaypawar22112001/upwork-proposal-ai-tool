import os
import tempfile
from pathlib import Path

import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

# Directories for storing temporary data and vector embeddings
TMP_DIR = Path(__file__).resolve().parent / "data" / "tmp"
VECTOR_STORE_DIR = Path(__file__).resolve().parent / "data" / "vector_store"

st.set_page_config(page_title="RAG Chatbot")
st.title("Retrieval-Augmented Generation (RAG) Engine")


# Ensure required directories exist
TMP_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)


def load_documents():
    """Load PDF documents from the temporary directory."""
    loader = DirectoryLoader(
        TMP_DIR.as_posix(), glob="**/*.pdf", loader_cls=PyPDFLoader
    )
    return loader.load()


def split_documents(documents):
    """Split documents into smaller chunks for efficient retrieval."""
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_documents(documents)


def create_vector_store(texts):
    """Create a FAISS vector store using Hugging Face embeddings."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectordb = FAISS.from_documents(texts, embedding=embeddings)
    return vectordb.as_retriever(search_kwargs={"k": 7})


def query_llm(retriever, query):
    """Query the local LLM (Ollama) using retrieved documents."""
    llm = Ollama(model="deepseek-r1:latest")  # Use DeepSeek model from Ollama

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )

    result = qa_chain({"question": query, "chat_history": st.session_state.messages})
    st.session_state.messages.append((query, result["answer"]))
    return result["answer"]


def process_documents():
    """Handle document processing (upload, chunking, embedding)."""
    if not st.session_state.source_docs:
        st.warning("Please upload documents before processing.")
        return

    try:
        # Save uploaded documents to temporary directory
        for source_doc in st.session_state.source_docs:
            with tempfile.NamedTemporaryFile(
                delete=False, dir=TMP_DIR.as_posix(), suffix=".pdf"
            ) as tmp_file:
                tmp_file.write(source_doc.read())

        # Load and process documents
        documents = load_documents()
        texts = split_documents(documents)

        # Create vector store
        st.session_state.retriever = create_vector_store(texts)

        # Cleanup temporary files
        for file in TMP_DIR.iterdir():
            file.unlink()

        st.success("Documents processed successfully!")

    except Exception as e:
        st.error(f"An error occurred: {e}")


def boot():
    """Initialize the Streamlit UI components."""
    st.sidebar.header("Configuration")

    # File uploader
    st.session_state.source_docs = st.file_uploader(
        "Upload PDF Documents", type=["pdf"], accept_multiple_files=True
    )

    # Process button
    st.sidebar.button("Process Documents", on_click=process_documents)

    # Initialize conversation history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:
        st.chat_message("human").write(message[0])
        st.chat_message("ai").write(message[1])

    # Handle new user input
    if query := st.chat_input():
        st.chat_message("human").write(query)
        response = query_llm(st.session_state.retriever, query)
        st.chat_message("ai").write(response)


if __name__ == "__main__":
    boot()
