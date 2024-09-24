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

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["openai_key"])

# Function to create ChromaDB collection and embed PDF documents
def create_chromadb_collection(pdf_files):
    if 'Lab4_vectorDB' not in st.session_state:
        # Initialize ChromaDB client with persistent storage
        client = chromadb.PersistentClient(path="./chroma_db")
        st.session_state.Lab4_vectorDB = client.get_or_create_collection(name="Lab4Collection")
        
        # Set up OpenAI embedding function
        openai_embedder = embedding_functions.OpenAIEmbeddingFunction(api_key=st.secrets["openai_key"], model_name="text-embedding-3-small")
        
        # Loop through provided PDF files, convert to text, and add to the vector database
        for file in pdf_files:
            try:
                # Read PDF file and extract text
                pdf_text = ""
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    pdf_text += page.extract_text()
                
                # Add document to ChromaDB collection with embeddings
                st.session_state.Lab4_vectorDB.add(
                    documents=[pdf_text],
                    metadatas=[{"filename": file.name}],
                    ids=[file.name]
                )
            except Exception as e:
                st.error(f"Error processing {file.name}: {e}")
        
        st.success("ChromaDB collection has been created successfully!")

# Function to count tokens
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Function to query the vector database and get relevant context
def get_relevant_context(query, max_tokens=6000):
    if 'Lab4_vectorDB' in st.session_state:
        results = st.session_state.Lab4_vectorDB.query(
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
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit application
st.title("Course Information Chatbot")

# Load PDF files
pdf_files = st.file_uploader("Upload your PDF files", accept_multiple_files=True, type=["pdf"])

# Create ChromaDB collection and embed documents if not already created
if st.button("Create ChromaDB Collection") and pdf_files:
    create_chromadb_collection(pdf_files)

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
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get relevant context from the vector database
    context = get_relevant_context(prompt)

    # Prepare messages for the LLM
    system_message = "You are a helpful assistant that answers questions about a course based on the provided context. If the answer is not in the context, say you don't have that information."
    user_message = f"Context: {context}\n\nQuestion: {prompt}"
    
    # Check total tokens and truncate if necessary
    total_tokens = num_tokens_from_string(system_message, "cl100k_base") + num_tokens_from_string(user_message, "cl100k_base")
    if total_tokens > 7500:  # Leave some room for the response
        context_tokens = 7500 - num_tokens_from_string(system_message, "cl100k_base") - num_tokens_from_string(f"Question: {prompt}", "cl100k_base")
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
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})