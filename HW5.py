import streamlit as st
from openai import OpenAI
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
from chromadb.utils import embedding_functions
import PyPDF2

# Set OpenAI client with the API key
client = OpenAI(api_key=st.secrets["openai_key"])

# Initialize the ChromaDB client with persistent storage
def initialize_chromadb():
    if 'HW4' not in st.session_state:
        client = chromadb.PersistentClient(path="chromadb_storage")  # Ensure persistence
        st.session_state.HW4 = client.get_or_create_collection(name="HW4_collection")

# Function to create ChromaDB collection from PDFs
def create_chromadb_collection(pdf_files):
    initialize_chromadb()  # Initialize if not already

    # Set up OpenAI embedding function
    openai_embedder = embedding_functions.OpenAIEmbeddingFunction(
        api_key=st.secrets["openai_key"], 
        model_name="text-embedding-ada-002"  # Use a supported embedding model
    )
    
    # Manually track added files to avoid duplicates
    if 'added_files' not in st.session_state:
        st.session_state.added_files = []

    # Loop through provided PDF files, convert to text, and add to the vector database
    for file in pdf_files:
        try:
            # Check if the document is already added to avoid duplicates
            if file.name in st.session_state.added_files:
                st.warning(f"{file.name} is already in the collection.")
                continue

            # Read PDF file and extract text
            pdf_text = ""
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                pdf_text += page.extract_text()

            # Add document to ChromaDB collection with embeddings
            st.session_state.HW4.add(
                documents=[pdf_text],
                metadatas=[{"filename": file.name}],
                ids=[file.name]
            )
            
            # Track added file to avoid duplicates
            st.session_state.added_files.append(file.name)
            st.success(f"Added {file.name} to the ChromaDB collection.")
        except Exception as e:
            st.error(f"Error processing {file.name}: {e}")

# Function to return relevant course or club information from ChromaDB
def get_relevant_info(query, collection_name="HW4_collection"):
    if 'HW4' in st.session_state:
        # Perform the query in the ChromaDB collection
        results = st.session_state.HW4.query(
            query_texts=[query],
            n_results=5,
            include=["documents", "metadatas"]
        )
        
        context = ""
        for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
            new_context = f"From document '{metadata['filename']}':\n{doc}\n\n"
            context += new_context
        
        # Save context to session state for continuity
        if 'context' not in st.session_state:
            st.session_state.context = ""
        
        # Append new context to the session state for future queries
        st.session_state.context += context
        
        return context
    return ""

# Function to generate response using OpenAI, with relevant context included
def generate_response_with_context(query):
    try:
        # Load the accumulated context from previous queries
        context = st.session_state.context if 'context' in st.session_state else ""

        system_message = "You are a helpful assistant that answers questions about a course or club based on the provided context. If the answer is not in the context, say you don't have that information."
        user_message = f"Context: {context}\n\nQuestion: {query}"

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=150
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit application
st.title("Understanding your courses or clubs!")

# Load PDF files
pdf_files = st.file_uploader("Upload your PDF files", accept_multiple_files=True, type=["pdf"])

# Create ChromaDB collection and embed documents if not already created
if st.button("Create ChromaDB") and pdf_files:
    create_chromadb_collection(pdf_files)

# Initialize chat history if not already done
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input (here, it's 'query' instead of 'prompt')
if query := st.chat_input("What course or club info do you need?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(query)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})

    # Get relevant information using the query
    relevant_info = get_relevant_info(query)

    # Generate response with context
    response = generate_response_with_context(query)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
