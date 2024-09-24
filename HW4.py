import streamlit as st
from openai import OpenAI
import os
from PyPDF2 import PdfReader
import chromadb

# SQLite adjustments for chromadb
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Initialize the ChromaDB Persistent Client
openai_client = chromadb.PersistentClient()

# Initialize session state for OpenAI client
if 'openai_client' not in st.session_state:
    api_key = st.secrets("openai_key")
    st.session_state.openai_client = OpenAI(api_key=api_key)

# Function to extract text from PDFs in the "pdfs" folder
def extract_text_from_pdfs(folder_path="pdfs"):
    pdf_texts = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            pdf_texts[filename] = text
    return pdf_texts

# Function to add PDFs to a collection
def add_to_collection(collection, text, filename):
    openai_client = st.session_state.openai_client
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    embedding = response.data[0].embedding

    collection.add(
        documents=[text],
        ids=[filename],
        embeddings=[embedding]
    )

# Predefined folder for PDF files
pdf_folder = "pdfs"

# Extract PDFs and add to collection
if os.path.exists(pdf_folder):
    st.write(f"Loading PDFs from: {pdf_folder}")
    
    # Extract text from PDFs
    pdf_texts = extract_text_from_pdfs(pdf_folder)
    
    # Create a collection
    collection = openai_client.create_collection("pdf_collection")
    
    # Add each PDF text to the collection
    for filename, text in pdf_texts.items():
        add_to_collection(collection, text, filename)
        st.write(f"Added {filename} to the collection")

# Sidebar for topic selection
topic = st.sidebar.selectbox("Topic", ("Text Mining", "GenAI"))

# Query the collection with the topic
openai_client = st.session_state.openai_client
response = openai_client.embeddings.create(
    input=topic,
    model="text-embedding-3-small"
)

query_embedding = response.data[0].embedding

# Retrieve relevant results from the collection
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

# Display relevant documents
st.write(f"Relevant documents for the topic '{topic}':")
for i in range(len(results['documents'][0])):
    doc = results['documents'][0][i]
    doc_id = results['ids'][0][i]
    st.write(f"The following file/syllabus might be helpful: {doc_id}")
