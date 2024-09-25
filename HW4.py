import streamlit as st
from openai import OpenAI
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
from chromadb.utils import embedding_functions
import PyPDF2
import os

client = OpenAI(api_key=st.secrets["openai_key"])

def create_chromadb_collection(pdf_files):
    if 'HW4' not in st.session_state:
        # Initialize ChromaDB client with persistent storage
        client = chromadb.PersistentClient()
        st.session_state.HW4 = client.get_or_create_collection(name="HW4_collection")
        
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
                st.session_state.HW4.add(
                    documents=[pdf_text],
                    metadatas=[{"filename": file.name}],
                    ids=[file.name]
                )
            except Exception as e:
                st.error(f"Error processing {file.name}: {e}")
        
        st.success("ChromaDB has been created!")

# Function to query the vector database and get relevant context
def get_relevant_context(query):
    if 'HW4' in st.session_state:
        results = st.session_state.HW4.query(
            query_texts=[query],
            n_results=5,
            include=["documents", "metadatas"]
        )
        
        context = ""
        for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
            new_context = f"From document '{metadata['filename']}':\n{doc}\n\n"
            context += new_context
        
        return context
    return ""

# Function to generate response using OpenAI
def generate_response(messages):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=150
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit application
st.title("Understanding your courses!")

# Load PDF files
pdf_files = st.file_uploader("Upload your PDF files", accept_multiple_files=True, type=["pdf"])

# Create ChromaDB collection and embed documents if not already created
if st.button("Create ChromaDB") and pdf_files:
    create_chromadb_collection(pdf_files)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What questions do you have?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    context = get_relevant_context(prompt)

    # Prepare messages for the LLM
    system_message = "You are a helpful assistant that answers questions about a course based on the provided context. If the answer is not in the context, say you don't have that information."
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
