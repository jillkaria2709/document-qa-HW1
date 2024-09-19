import streamlit as st
import openai
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse
from groq import Groq

# Title and description
st.title("📄 My Homework 3 Question Answering Chatbox")

# Sidebar options
st.sidebar.header("Options")

# Option to input two URLs
url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")

# Option to pick the LLM vendor (OpenAI, Gemini, Groq, OpenRouter)
llm_vendor = st.sidebar.selectbox("Select LLM Vendor", ["OpenAI", "Gemini", "Groq", "OpenRouter"])

# Option to pick the type of conversation memory
memory_type = st.sidebar.selectbox("Select Conversation Memory Type", ["Buffer of 5 questions", "Conversation Summary", "Buffer of 5,000 tokens"])

# Static memory mode
memory_mode = "URL Memory"  # Change based on actual logic or condition

# Display memory mode in the sidebar
st.sidebar.write(f"Memory Mode: {memory_mode}")

# Determine the buffer size based on the selected memory type
if memory_type == "Buffer of 5 questions":
    buffer_size = 5
elif memory_type == "Buffer of 5,000 tokens":
    buffer_size = 5000  
else:
    buffer_size = 10  

# Model selection based on LLM vendor
if llm_vendor == "OpenAI":
    model_to_use = st.sidebar.selectbox("Select OpenAI Model", ["gpt-4", "gpt-3.5-turbo"])
elif llm_vendor == "Gemini":
    model_to_use = "gemini-1.5-flash"  # Always use gemini-1.5-flash
elif llm_vendor == "Groq":
    model_to_use = "llama3-8b-8192"  # Always use llama3-8b-8192
elif llm_vendor == "OpenRouter":
    model_to_use = "mattshumer/reflection-70b:free"  # Example model for OpenRouter

# Buffer size slider
buffer_size = st.sidebar.slider("Buffer Size", min_value=1, max_value=10, value=2, step=1)

# Set up LLM clients based on the vendor
if llm_vendor == "OpenAI":
    openai.api_key = st.secrets["openai_key"]
elif llm_vendor == "Gemini":
    genai.configure(api_key=st.secrets["gemini_api_key"])
elif llm_vendor == "Groq":
    groq_client = Groq(api_key=st.secrets["grok_api_key"])
elif llm_vendor == "OpenRouter":
    openrouter_api_key = st.secrets["openrouter_api_key"]

# A dictionary to store parsed URL content in session state
if "parsed_urls" not in st.session_state:
    st.session_state["parsed_urls"] = {}

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you"}]

# Display chat messages
for msg in st.session_state.messages:
    chat_msg = st.chat_message(msg["role"])
    chat_msg.write(msg["content"])

# URL validation function
def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

# Extract text from URL
def extract_text_from_url(url):
    try:
        if url in st.session_state["parsed_urls"]:
            return st.session_state["parsed_urls"][url]  # Return cached content if already parsed
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()
        # Clean up the text
        text = re.sub(r'\s+', ' ', text).strip()
        st.session_state["parsed_urls"][url] = text  # Cache the parsed content
        
        # Add message to the sidebar
        st.sidebar.success(f"URL {url} studied and parsed.")
        
        return text
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching URL {url}: {e}")
        return ""

# Parse URLs and cache the content in memory
if url1 and is_valid_url(url1):
    extract_text_from_url(url1)
if url2 and is_valid_url(url2):
    extract_text_from_url(url2)

# Function to check if the user's question is related to the URL
def is_question_related_to_url(prompt):
    keywords = ["content", "details", "info from", "link", "URL"]
    return any(keyword in prompt.lower() for keyword in keywords)

# Function to call OpenRouter API
def call_openrouter_api(api_key, document, instruction):
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": model_to_use,  # Use the selected model
            "messages": [
                {"role": "user", "content": f"Here's a document: {document} \n\n---\n\n {instruction}"},
            ],
        }
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=data, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            return result.get("choices", [{}])[0].get("message", {}).get("content", "No completion found.")
        else:
            return f"OpenRouter API error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error calling OpenRouter API: {e}"

# Handling user input
if prompt := st.chat_input("Ask your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Limit the buffer to the specified size
    if len(st.session_state.messages) > buffer_size * 2:
        st.session_state.messages = st.session_state.messages[-buffer_size * 2:]

    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response based on memory mode
    if memory_mode == "URL Memory" and is_question_related_to_url(prompt):
        url_texts = list(st.session_state["parsed_urls"].values())
        combined_text = "\n".join(url_texts)  # Combine all URL content

        if combined_text:
            response_from_url = ""
            # Generate response using URL content if available
            try:
                if llm_vendor == "OpenAI":
                    response = openai.ChatCompletion.create(
                        model=model_to_use,
                        messages=[
                            {"role": "system", "content": 'You answer questions based on URL content.'},
                            {"role": "user", "content": f"{combined_text}\n\n{prompt}"}
                        ],
                        temperature=0  # Adjust temperature if needed
                    )
                    response_from_url = response.choices[0].message['content']
                    
                elif llm_vendor == "Gemini":
                    model = genai.GenerativeModel(model_to_use)
                    response = model.generate_content(f"{combined_text}\n\n{prompt}")
                    response_from_url = response.text
                    
                elif llm_vendor == "Groq":
                    chat_completion = groq_client.chat.completions.create(
                        messages=[
                            {"role": "user", "content": f"{combined_text}\n\n{prompt}"}
                        ],
                        model=model_to_use
                    )
                    response_from_url = chat_completion.choices[0].message['content']
                    
                elif llm_vendor == "OpenRouter":
                    response_from_url = call_openrouter_api(openrouter_api_key, combined_text, prompt)
                
                # Display and save URL-based response
                with st.chat_message("assistant"):
                    st.write(response_from_url)
                st.session_state.messages.append({"role": "assistant", "content": response_from_url})

            except Exception as e:
                st.error(f"Error generating response: {e}")
                
    elif memory_mode == "Self Memory":
        # Prepare messages from memory for the general conversation
        combined_messages = st.session_state.messages + [{"role": "system", "content": "Use conversation history for context."}]
        
        try:
            if llm_vendor == "OpenAI":
                response = openai.ChatCompletion.create(
                    model=model_to_use,
                    messages=[
                        {"role": "system", "content": 'You answer questions based on conversation history.'},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0  # Adjust temperature if needed
                )
                response_from_memory = response.choices[0].message['content']
                
            elif llm_vendor == "Gemini":
                model = genai.GenerativeModel(model_to_use)
                response = model.generate_content(f"{prompt} - context: {combined_messages}")
                response_from_memory = response.text
                
            elif llm_vendor == "Groq":
                chat_completion = groq_client.chat.completions.create(
                    messages=[
                        {"role": "user", "content": f"{prompt}"}
                    ],
                    model=model_to_use
                )
                response_from_memory = chat_completion.choices[0].message['content']
                
            elif llm_vendor == "OpenRouter":
                response_from_memory = call_openrouter_api(openrouter_api_key, "", prompt)
            
            # Display and save self-memory-based response
            with st.chat_message("assistant"):
                st.write(response_from_memory)
            st.session_state.messages.append({"role": "assistant", "content": response_from_memory})

        except Exception as e:
            st.error(f"Error generating response: {e}")

    elif memory_mode == "Combined Memory":
        # Combine URL and self memory for response generation
        url_texts = list(st.session_state["parsed_urls"].values())
        combined_url_text = "\n".join(url_texts)
        
        try:
            if llm_vendor == "OpenAI":
                response = openai.ChatCompletion.create(
                    model=model_to_use,
                    messages=[
                        {"role": "system", "content": 'Use both URL content and conversation history for responses.'},
                        {"role": "user", "content": f"{combined_url_text}\n\n{prompt}"}
                    ],
                    temperature=0  # Adjust temperature if needed
                )
                response_from_combined = response.choices[0].message['content']
                
            elif llm_vendor == "Gemini":
                model = genai.GenerativeModel(model_to_use)
                response = model.generate_content(f"{combined_url_text}\n\n{prompt}")
                response_from_combined = response.text
                
            elif llm_vendor == "Groq":
                chat_completion = groq_client.chat.completions.create(
                    messages=[
                        {"role": "user", "content": f"{combined_url_text}\n\n{prompt}"}
                    ],
                    model=model_to_use
                )
                response_from_combined = chat_completion.choices[0].message['content']
                
            elif llm_vendor == "OpenRouter":
                response_from_combined = call_openrouter_api(openrouter_api_key, combined_url_text, prompt)
            
            # Display and save combined-memory-based response
            with st.chat_message("assistant"):
                st.write(response_from_combined)
            st.session_state.messages.append({"role": "assistant", "content": response_from_combined})

        except Exception as e:
            st.error(f"Error generating response: {e}")
