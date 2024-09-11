import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI, OpenAIError
import fitz  # PyMuPDF for reading PDF files

# Show title and description
st.title("📄 Question the Document or URL")
st.write(
    "Upload a document or enter a URL below, and ask a question – GPT or other LLMs will answer! "
    "To use this app, API keys for different LLMs are fetched from secrets."
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

# Sidebar options for LLM and language selection
st.sidebar.header("LLM and Language Settings")
llm_option = st.sidebar.selectbox("Choose LLM", ["OpenAI", "Claude", "Cohere", "Gemini", "Mistral"])
language = st.sidebar.selectbox("Select Output Language", ["English", "French", "Spanish"])
use_advanced_model = st.sidebar.checkbox("Use advanced model")

# Retrieve API keys from Streamlit secrets
llm_api_keys = {
    "OpenAI": st.secrets["openai"]["api_key"],
    "TogetherAI": st.secrets["togetherai"]["together_api_key"],
    "Gemini": st.secrets["gemini"]["gemini_api_key"],
}

# Main content for document or URL input
st.write("You can either upload a document or enter a URL.")
url = st.text_input("Enter a URL:")

uploaded_file = st.file_uploader("Or upload a document (.pdf or .txt)", type=("pdf", "txt"))

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

# Check if the document and question are provided
if document and question:
    api_key = llm_api_keys.get(llm_option)  # Get the correct API key for the selected LLM

    if api_key:
        try:
            # LLM-specific API integration
            if llm_option == "OpenAI":
                client = OpenAI(api_key=api_key)
                model = "gpt-4o-mini" if use_advanced_model else "gpt-3.5-turbo"

                messages = [
                    {"role": "user", "content": f"Here's a document: {document} \n\n---\n\n {question}"}
                ]

                # Generate an answer using OpenAI
                response = client.chat.completions.create(model=model, messages=messages)
                st.write(response.choices[0].message["content"])

            elif llm_option == "TogetherAi":
                # Integrate Claude API logic here using the retrieved api_key
                pass

            elif llm_option == "Gemini":
                # Integrate Gemini API logic here using the retrieved api_key
                pass

        except OpenAIError as e:
            st.error(f"OpenAI API error: {e}")
    else:
        st.warning(f"API key for {llm_option} is missing. Please check your secrets.toml configuration.")
else:
    if not question:
        st.warning("Please enter a question.")
