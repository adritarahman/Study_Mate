import streamlit as st
import os
import shutil
import tempfile
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Configuration
DB_DIR = "chroma-db"

# --- Page Config ---
st.set_page_config(page_title="RAG PDF Assistant", page_icon="📄", layout="centered")
st.title("📄 Chat with your PDF")

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None
if "chat_active" not in st.session_state:
    st.session_state.chat_active = True

# --- Helper Functions ---
def clear_database():
    """Deletes the existing Chroma database directory."""
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)

def process_and_store_pdf(uploaded_file):
    """Saves uploaded file temporarily, chunks it, and stores it in Chroma."""
    # PyPDFLoader requires a file path, so we save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    try:
        # Load and Split
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(docs)
        
        # Embed and Store
        embedding_model = MistralAIEmbeddings()
        Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=DB_DIR
        )
    finally:
        # Clean up the temporary file
        os.remove(temp_path)

# --- Sidebar: File Upload ---
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    # Logic to handle new file uploads and overwrite the DB
    if uploaded_file:
        if st.session_state.processed_file != uploaded_file.name:
            with st.spinner("Processing new PDF and updating database..."):
                clear_database()  # Delete old DB
                process_and_store_pdf(uploaded_file) # Create new DB
                
                # Update session state
                st.session_state.processed_file = uploaded_file.name
                st.session_state.messages = [] # Clear chat history for new file
                st.session_state.chat_active = True
                
            st.success(f"Database created for {uploaded_file.name}!")
    else:
        st.info("Please upload a PDF to begin.")

# --- Main Chat Interface ---
if uploaded_file and st.session_state.processed_file:
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if st.session_state.chat_active:
        query = st.chat_input("Ask a question about your document (Type '0' to exit)")
        
        if query:
            # Handle the "0" exit condition
            if query.strip() == "0":
                st.session_state.chat_active = False
                st.warning("Chat ended by user. Please refresh the page or upload a new file to start over.")
                st.rerun()
                
            # Display user message
            with st.chat_message("user"):
                st.markdown(query)
            st.session_state.messages.append({"role": "user", "content": query})
            
            # Generate AI Response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # 1. Initialize DB and Retriever
                    embedding_model = MistralAIEmbeddings()
                    vectorstore = Chroma(
                        persist_directory=DB_DIR,
                        embedding_function=embedding_model
                    )
                    retriever = vectorstore.as_retriever(
                        search_type='mmr',
                        search_kwargs={
                            "k": 4,
                            "fetch_k": 10,
                            "lambda_mult": 0.5
                        }
                    )
                    
                    # 2. Retrieve Context
                    docs = retriever.invoke(query)
                    context = "\n\n".join([doc.page_content for doc in docs])
                    
                    # 3. Setup LLM and Prompt
                    llm = ChatMistralAI(model="mistral-small-2506")
                    prompt = ChatPromptTemplate.from_messages([
                        ('system', """You are a helpful AI assistant.
                        Use ONLY the provided context to answer the question.
                        If the answer is not present in the context, say: "I could not find the answer in the document."
                        """),
                        ('human', "Context:\n{context}\n\nQuestion:\n{question}")
                    ])
                    
                    # 4. Invoke LLM
                    final_prompt = prompt.invoke({
                        "context": context,
                        "question": query
                    })
                    response = llm.invoke(final_prompt)
                    
                    # 5. Display and Save Response
                    st.markdown(response.content)
                    st.session_state.messages.append({"role": "assistant", "content": response.content})
else:
    if not uploaded_file:
        st.write("👈 Upload a PDF file in the sidebar to start asking questions.")