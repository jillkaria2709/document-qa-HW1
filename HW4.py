import streamlit as st
from openai import OpenAI
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
from chromadb.utils import embedding_functions
import PyPDF2
import tiktoken
import os

# Setup OpenAI client using API key from Streamlit secrets
openai_client = OpenAI(api_key=st.secrets["openai_key"])

# Function to build ChromaDB collection and add PDF documents with embeddings
def initialize_vector_database(uploaded_pdfs):
    if 'course_vector_db' not in st.session_state:
        # Initialize ChromaDB with persistent storage
        vector_db_client = chromadb.PersistentClient()
        st.session_state.course_vector_db = vector_db_client.get_or_create_collection(name="CourseCollection")
        
        # Configure OpenAI embedding function
        embedding_function = embedding_functions.OpenAIEmbeddingFunction(api_key=st.secrets["openai_key"], model_name="text-embedding-3-small")
        
        # Loop through uploaded PDF files, extract text, and add to vector database
        for pdf_file in uploaded_pdfs:
            try:
                # Extract text from PDF
                extracted_text = ""
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                for page in pdf_reader.pages:
                    extracted_text += page.extract_text()
                
                # Add the document with its embedding to ChromaDB collection
                st.session_state.course_vector_db.add(
                    documents=[extracted_text],
                    metadatas=[{"filename": pdf_file.name}],
                    ids=[pdf_file.name]
                )
            except Exception as err:
                st.error(f"Failed to process {pdf_file.name}: {err}")
        
        st.success("Vector database has been successfully created!")

# Function to calculate the number of tokens from a string using a specific encoding
def calculate_tokens(text: str, encoding: str) -> int:
    tokenizer = tiktoken.get_encoding(encoding)
    return len(tokenizer.encode(text))

# Function to query the vector database and retrieve relevant context
def retrieve_context(query_text, max_allowed_tokens=6000):
    if 'course_vector_db' in st.session_state:
        query_result = st.session_state.course_vector_db.query(
            query_texts=[query_text],
            n_results=5,
            include=["documents", "metadatas"]
        )
        
        context_data = ""
        for doc, metadata in zip(query_result['documents'][0], query_result['metadatas'][0]):
            new_context = f"From document '{metadata['filename']}':\n{doc}\n\n"
            if calculate_tokens(context_data + new_context, "cl100k_base") <= max_allowed_tokens:
                context_data += new_context
            else:
                break
        
        return context_data
    return ""

# Function to interact with OpenAI API and generate a response
def get_llm_response(chat_messages):
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=chat_messages,
            max_tokens=200
        )
        return response.choices[0].message.content
    except Exception as err:
        return f"Error: {str(err)}"

# Streamlit interface
st.title("Course Information Query Chatbot")

# Upload PDF files
pdf_files_uploaded = st.file_uploader("Upload PDF files for course details", accept_multiple_files=True, type=["pdf"])

# Create ChromaDB vector database if PDFs are uploaded and button is clicked
if st.button("Initialize Vector Database") and pdf_files_uploaded:
    initialize_vector_database(pdf_files_uploaded)

# Initialize chat history
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Display chat history
for msg in st.session_state.conversation:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input for chatbot
if user_input := st.chat_input("Ask your question about the course:"):
    # Display user input
    st.chat_message("user").markdown(user_input)
    st.session_state.conversation.append({"role": "user", "content": user_input})

    # Fetch relevant context from vector database
    context_from_db = retrieve_context(user_input)

    # Prepare messages for the OpenAI model
    system_prompt = "You are an assistant helping with course-related queries using provided context. If the answer isn't in the context, inform the user."
    user_prompt = f"Context: {context_from_db}\n\nQuery: {user_input}"
    
    # Check token usage and truncate context if needed
    total_token_count = calculate_tokens(system_prompt, "cl100k_base") + calculate_tokens(user_prompt, "cl100k_base")
    if total_token_count > 5000:
        remaining_tokens = 5000 - calculate_tokens(system_prompt, "cl100k_base") - calculate_tokens(f"Query: {user_input}", "cl100k_base")
        context_from_db = context_from_db[:remaining_tokens]
        user_prompt = f"Context: {context_from_db}\n\nQuery: {user_input}"

    chat_history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Generate response from OpenAI
    assistant_reply = get_llm_response(chat_history)

    # Display assistant's response
    with st.chat_message("assistant"):
        st.markdown(assistant_reply)
    st.session_state.conversation.append({"role": "assistant", "content": assistant_reply})
