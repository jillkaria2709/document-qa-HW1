import streamlit as st
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
import os
import PyPDF2

st.title('My LAB4 Question Answering chatbox')

openAImodel = st.sidebar.selectbox("Which model?", ("mini", "regular"))
buffer_size = st.sidebar.slider("Buffer Size", min_value=1, max_value=10, value=2, step=1)

if openAImodel == "mini":
    model_to_use = "gpt-4o-mini"
else:
    model_to_use = "gpt-4o"

if 'client' not in st.session_state:
    api_key = st.secrets["openai_key"]
    st.session_state.client = OpenAI(api_key=api_key)

if 'messages' not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help?"}]

def create_chroma_collection():
    # Initialize ChromaDB client
    chroma_client = chromadb.Client()
    
    # Create OpenAI embedding function
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=st.secrets["openai_key"],
        model_name="text-embedding-3-small"
    )
    
    # Create or get the collection
    collection = chroma_client.create_collection(
        name="Lab4Collection",
        embedding_function=openai_ef
    )
    
    # Process and add PDF files
    pdf_dir = "path_to_your_pdf_files"  # Replace with actual path
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_dir, filename)
            with open(file_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
            
            # Add document to collection
            collection.add(
                documents=[text],
                metadatas=[{"filename": filename}],
                ids=[filename]
            )
    
    return collection

# Create the ChromaDB collection if not already in session state
if 'Lab4_vectorDB' not in st.session_state:
    st.session_state.Lab4_vectorDB = create_chroma_collection()

# Display all messages
for msg in st.session_state.messages:
    chat_msg = st.chat_message(msg["role"])
    chat_msg.write(msg["content"])

# Input prompt
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Maintain the buffer size
    if len(st.session_state.messages) > buffer_size * 2:
        st.session_state.messages = st.session_state.messages[-buffer_size * 2:]

    with st.chat_message("user"):
        st.markdown(prompt)

    # Query the vector database
    results = st.session_state.Lab4_vectorDB.query(
        query_texts=[prompt],
        n_results=3
    )

    # Add relevant document information to the context
    context = "Relevant documents:\n"
    for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
        context += f"- {metadata['filename']}: {doc[:100]}...\n"

    # Prepare messages for OpenAI, including the context
    messages = st.session_state.messages + [
        {"role": "system", "content": f"Use this context to inform your response: {context}"}
    ]

    client = st.session_state.client
    stream = client.chat.completions.create(
        model=model_to_use,
        messages=messages,
        stream=True
    )

    with st.chat_message("assistant"):
        response = st.write_stream(stream)

    # Ensure response is less than 150 words
    response = ' '.join(response.split()[:150])
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Automatically ask for more information
    more_info_question = "Want more info? (Yes/No)"
    st.session_state.messages.append({"role": "assistant", "content": more_info_question})

# Handle the user's response for more information
if prompt and prompt.lower() in ["yes", "no"]:
    if prompt.lower() == "yes":
        st.session_state.messages.append({"role": "assistant", "content": "Continuing..."})
    elif prompt.lower() == "no":
        st.session_state.messages.append({"role": "assistant", "content": "What else?"})