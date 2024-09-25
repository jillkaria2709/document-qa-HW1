import streamlit as st
from openai import OpenAI
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
from chromadb.utils import embedding_functions
import os
import tiktoken
from bs4 import BeautifulSoup

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["openai_key"])

# Function to create ChromaDB collection and embed HTML documents
def create_chromadb_collection(html_files):
    if 'HW4_vectorDB' not in st.session_state:
        # Initialize ChromaDB client with persistent storage
        client = chromadb.PersistentClient(Settings())
        st.session_state.HW4_vectorDB = client.get_or_create_collection(name="HTMLCollection")

        # Set up OpenAI embedding function
        openai_embedder = embedding_functions.OpenAIEmbeddingFunction(api_key=st.secrets["openai_key"], model_name="text-embedding-3-small")

        # Loop through provided HTML files, extract text, and add to the vector database
        for file in html_files:
            try:
                # Read HTML file and extract text
                html_content = file.read().decode("utf-8")
                soup = BeautifulSoup(html_content, "html.parser")
                html_text = soup.get_text()

                # Add document to ChromaDB collection with embeddings
                st.session_state.HW4_vectorDB.add(
                    documents=[html_text],
                    metadatas=[{"filename": file.name}],
                    ids=[file.name]
                )
            except Exception as e:
                st.error(f"Error processing {file.name}: {e}")
        
        st.success("ChromaDB collection has been created successfully!")

# Function to check if the vector DB file exists
def vector_db_exists():
    return 'HW4_vectorDB' in st.session_state

# Function to count tokens
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Function to query the vector database and get relevant context
def get_relevant_context(query, max_tokens=6000):
    if 'HW4_vectorDB' in st.session_state:
        results = st.session_state.HW4_vectorDB.query(
            query_texts=[query],
            n_results=5,
            include=["documents", "metadatas"]
        )
        
        context = ""
        for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
            new_context = f"From document '{metadata['filename']}':\n{doc}\n\n"
            if num_tokens_from_string(context + new_context, "cl100k_base") <= max_tokens:
                context += new_context
            else:
                break
        
        return context
    return ""

# Function to generate response using OpenAI
def generate_response(messages):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=200
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# Function to manage memory conversation buffer (up to 5 messages)
def update_conversation_buffer(role, content):
    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.session_state.messages.append({"role": role, "content": content})
    # Keep only the last 5 messages in memory
    if len(st.session_state.messages) > 5:
        st.session_state.messages.pop(0)

# Streamlit application
st.title("Course Information Chatbot")

# Load HTML files
html_files = st.file_uploader("Upload your HTML files", accept_multiple_files=True, type=["html"])

# Check if vector DB already exists and create ChromaDB collection if not created
if st.button("Create ChromaDB Collection") and html_files:
    if not vector_db_exists():
        create_chromadb_collection(html_files)
    else:
        st.success("Vector DB already exists.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What would you like to know about the course?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Update conversation buffer
    update_conversation_buffer("user", prompt)

    # Get relevant context from the vector database
    context = get_relevant_context(prompt)

    # Prepare messages for the LLM
    system_message = "You are a helpful assistant that answers questions about a course based on the provided context. If the answer is not in the context, say you don't have that information."
    user_message = f"Context: {context}\n\nQuestion: {prompt}"
    
    # Check total tokens and truncate if necessary
    total_tokens = num_tokens_from_string(system_message, "cl100k_base") + num_tokens_from_string(user_message, "cl100k_base")
    if total_tokens > 5000:  # Leave some room for the response
        context_tokens = 5000 - num_tokens_from_string(system_message, "cl100k_base") - num_tokens_from_string(f"Question: {prompt}", "cl100k_base")
        context = context[:context_tokens]
        user_message = f"Context: {context}\n\nQuestion: {prompt}"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    # Generate response
    response = generate_response(messages)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Update conversation buffer with assistant response
    update_conversation_buffer("assistant", response)
