import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI, OpenAIError
import fitz  # PyMuPDF for reading PDF files

# Show title and description
st.title("📄 Question the Document or URL")
st.write(
    "Upload a document or enter a URL below, and ask a question – GPT or other LLMs will answer! "
    "API keys for different LLMs are fetched from secrets."
)

# Function to read PDF files using PyMuPDF
def read_pdf(file):
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

# Function to read content from a URL
def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None

# Function to call OpenAI API
def call_openai_api(api_key, document, question):
    try:
        client = OpenAI(api_key=api_key)
        messages = [
            {
                "role": "user",
                "content": f"Here's a document: {document} \n\n---\n\n {question}",
            }
        ]
        # Generate an answer using the OpenAI API.
        response = client.chat_completions.create(
            model="gpt-4o-mini",  # or "gpt-4" depending on your use case
            messages=messages,
            stream=True,
        )
        return response  # Return the stream to be processed
    except OpenAIError as e:
        return f"OpenAI API error: {e}"

# Function to call TogetherAI API
def call_togetherai_api(api_key, document, question):
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "input": f"{document} \n\n---\n\n {question}",
            "model": "together-large",  # Adjust this model name as per TogetherAI's available models
        }
        response = requests.post("https://api.togetherai.com/v1/completions", json=data, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            return result.get("completion", "No completion found in TogetherAI response.")
        else:
            return f"TogetherAI API error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error calling TogetherAI API: {e}"

# Function to call Gemini API
def call_gemini_api(api_key, document, question):
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "input": f"{document} \n\n---\n\n {question}",
            "model": "gemini-xl",  # Adjust model as per Gemini's available models
        }
        response = requests.post("https://api.gemini.com/v1/completions", json=data, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            return result.get("completion", "No completion found in Gemini response.")
        else:
            return f"Gemini API error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error calling Gemini API: {e}"

# Sidebar options for LLM selection
st.sidebar.header("LLM Settings")
llm_option = st.sidebar.selectbox("Choose LLM", ["OpenAI", "TogetherAI", "Gemini"])
use_advanced_model = st.sidebar.checkbox("Use advanced model")

# Main content for document or URL input
st.write("You can either upload a document or enter a URL.")
url = st.text_input("Enter a URL:")

uploaded_file = st.file_uploader("Or upload a document (.pdf or .txt)", type=("pdf", "txt"))

# Read document content from the URL or uploaded file
if url:
    document = read_url_content(url)
elif uploaded_file:
    file_extension = uploaded_file.name.split('.')[-1]
    
    if file_extension == 'txt':
        document = uploaded_file.read().decode()
    elif file_extension == 'pdf':
        document = read_pdf(uploaded_file)
    else:
        st.error("Unsupported file type.")
        document = None
else:
    document = None  # Reset if both file and URL are empty

# Ask the user for a question via `st.text_area`.
question = st.text_area(
    "Now ask a question about the document or URL!",
    placeholder="Can you give me a short summary?",
    disabled=not document,
)

# Handle API calls based on the chosen LLM
if document and question:
    # Retrieve the appropriate API key from secrets based on the LLM selected
    api_key = st.secrets.get(llm_option.lower(), {}).get("api_key", None)

    if api_key:
        if llm_option == "OpenAI":
            response = call_openai_api(api_key, document, question)
            st.write_stream(response)  # Stream the OpenAI response
        elif llm_option == "TogetherAI":
            response = call_togetherai_api(api_key, document, question)
            st.write(response)
        elif llm_option == "Gemini":
            response = call_gemini_api(api_key, document, question)
            st.write(response)
    else:
        st.warning(f"API key for {llm_option} is missing. Please check your configuration.")
else:
    if not question:
        st.warning("Please enter a question.")
    if not document:
        st.warning("Please upload a document or provide a URL to analyze.")
