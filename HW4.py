import streamlit as st
import chromadb
import os
import openai
import fitz  

# Initialize OpenAI API key
openai.api_key = st.secrets["openai_key"]

# Function to generate OpenAI embeddings
def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.embeddings.create(input=text, model=model)
    return response.data[0].embedding

# Function to create ChromaDB collection
def create_lab4_collection():
    if "Lab4_vectorDB" not in st.session_state:
        client = chromadb.Client()
        collection = client.create_collection(name="Lab4Collection")

        pdf_folder = "/mount/src/document-qa-1/pdf"
        if not os.path.exists(pdf_folder):
            st.error(f"PDF folder '{pdf_folder}' does not exist.")
            return
        
        pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
        if not pdf_files or len(pdf_files) != 7:
            st.error("Please ensure there are exactly 7 PDF files in the directory.")
            return

        documents = []
        metadatas = []
        ids = [file for file in pdf_files]

        for file in pdf_files:
            file_path = os.path.join(pdf_folder, file)
            with fitz.open(file_path) as doc:
                text = ""
                for page in doc:
                    text += page.get_text()

            if not text:
                st.warning(f"No text extracted from '{file}'")
                continue
            
            try:
                embedding = get_embedding(text)
                documents.append(embedding)
                metadatas.append({"filename": file, "content": text})  
            except Exception as e:
                st.error(f"Error generating embedding for {file}: {str(e)}")
                continue
        
        try:
            collection.add(ids=ids, embeddings=documents, metadatas=metadatas)
        except Exception as e:
            st.error(f"Error adding documents to ChromaDB: {str(e)}")
            return
        
        st.session_state.Lab4_vectorDB = collection
        st.success("ChromaDB collection 'Lab4Collection' created and stored in session state.")

# Function to get LLM response with context indication
def get_llm_response(query, context):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Here is some relevant information from the course syllabi:\n\n{context}\n\nNow, answer this question: {query}"}
    ]
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",  # Use the specified model
            messages=messages
        )
        return response.choices[0].message.content  # Accessing the response correctly
    except Exception as e:
        st.error(f"Error getting response from the LLM: {str(e)}")
        return "Sorry, I couldn't generate a response."

# Function to search the vector database
def search_lab4_collection(query_text):
    if "Lab4_vectorDB" in st.session_state:
        collection = st.session_state.Lab4_vectorDB
        
        # Generate embedding for query text
        try:
            query_embedding = get_embedding(query_text)
        except Exception as e:
            st.error(f"Error generating embedding for the query: {str(e)}")
            return None, None
        
        # Query the ChromaDB using query_embeddings
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=3
            )
            print(f"Raw results: {results}")  # Log the raw results for debugging
        except Exception as e:
            st.error(f"Error querying ChromaDB: {str(e)}")
            return None, None
        
        # Accessing metadatas properly
        metadatas = results['metadatas'][0]  # Access the first list in 'metadatas'
        
        # Gather context from the results
        context = "\n".join([result.get('content', '') for result in metadatas if isinstance(result, dict)])
        
        # If context is available, generate LLM response
        if context:
            llm_response = get_llm_response(query_text, context)
            return metadatas, llm_response  # Return both metadatas and the response
        else:
            st.warning("No relevant context found for the query.")
            return None, None
    else:
        st.warning("Lab4 vector database not found. Please create it first.")
        return None, None

# Streamlit interface
st.title("Lab4 Vector Database")

# Create ChromaDB collection if not already in session state
if st.button("Create Lab4 Collection"):
    create_lab4_collection()

# Chat interface
query = st.text_input("Enter your question:")
if st.button("Ask"):
    if query:
        metadatas, llm_response = search_lab4_collection(query)
        if metadatas:
            # Display the top results if any
            st.write(f"Search results for '{query}':")
            for i, result in enumerate(metadatas):
                filename = result.get('filename', 'Unknown file')
                st.write(f"{i + 1}. {filename}")
        else:
            st.warning("No relevant documents found.")

        # Display the chatbot response
        if llm_response:
            st.write("Chatbot response:")
            st.write(llm_response)
    else:
        st.warning("Please enter a question.")
