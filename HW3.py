import streamlit as st
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from openai import OpenAI
import json

st.title('My HW3 Question Answering Chatbox')

# Sidebar for model selection and input URLs
model_provider = st.sidebar.selectbox("Which model?", ("OpenAI", "Gemini", "OpenRouter"))
buffer_size = st.sidebar.slider("Buffer Size", min_value=1, max_value=10, value=2, step=1)

url_1 = st.sidebar.text_input("Enter the first URL:")
url_2 = st.sidebar.text_input("Enter the second URL:")

# Fetch and parse URLs
def fetch_and_parse(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        return soup.get_text()  # Store full text
    except Exception as e:
        return f"Error fetching URL: {e}"

if st.sidebar.button("Fetch URLs"):
    if url_1:
        st.session_state['url_1_content'] = fetch_and_parse(url_1)
    if url_2:
        st.session_state['url_2_content'] = fetch_and_parse(url_2)

# Display the content of the first URL
if 'url_1_content' in st.session_state and st.sidebar.button("Print First URL Studied"):
    st.write("**First URL Content:**")
    st.write(st.session_state['url_1_content'][:500])  # Limiting output to 500 characters

# Initialize LLM clients
if 'client' not in st.session_state:
    api_keys = {
        "OpenAI": st.secrets["openai_key"],
        "Gemini": st.secrets["gemini_api_key"],
        "OpenRouter": st.secrets["openrouter_api_key"]
    }
    st.session_state.clients = {
        "OpenAI": OpenAI(api_key=api_keys["OpenAI"]),
        "Gemini": genai.Client(api_key=api_keys["Gemini"]),
        "OpenRouter": None  # Set up OpenRouter client later
    }

if 'messages' not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help?"}]

# Display all messages
for msg in st.session_state.messages:
    chat_msg = st.chat_message(msg["role"])
    chat_msg.write(msg["content"])

# Handle user input
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Maintain the buffer size
    if len(st.session_state.messages) > buffer_size * 2:
        st.session_state.messages = st.session_state.messages[-buffer_size * 2:]

    with st.chat_message("user"):
        st.markdown(prompt)

    response_text = ""
    if model_provider == "OpenAI":
        client = st.session_state.clients["OpenAI"]
        stream = client.chat.completions.create(
            model="gpt-4" if st.sidebar.selectbox("OpenAI Model", ("mini", "regular")) == "regular" else "gpt-4-mini",
            messages=st.session_state.messages,
            stream=True
        )

        for chunk in stream:
            if 'choices' in chunk and len(chunk['choices']) > 0:
                message = chunk['choices'][0].get('message', {})
                response_text += message.get('content', '')
                st.chat_message("assistant").write(response_text)

    elif model_provider == "Gemini":
        try:
            client = st.session_state.clients["Gemini"]
            response = client.chat.completions.create(
                model="gemini-model",
                messages=st.session_state.messages,
                stream=True
            )

            for chunk in response:
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    message = chunk['choices'][0].get('message', {})
                    response_text += message.get('content', '')
                    st.chat_message("assistant").write(response_text)

        except Exception as e:
            st.write(f"Error with Gemini API: {e}")

    elif model_provider == "OpenRouter":
        def call_openrouter_streaming_api(api_key, document, instruction):
            try:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                data = {
                    "model": "openrouter-model",
                    "messages": [
                        {"role": "user", "content": f"Here's a document: {document} \n\n---\n\n {instruction}"},
                    ],
                }
                response = requests.post("https://openrouter.ai/api/v1/chat/completions/stream", json=data, headers=headers, stream=True)

                response_text = ""
                for line in response.iter_lines():
                    if line:
                        chunk = line.decode('utf-8')
                        result = json.loads(chunk)
                        if 'choices' in result and len(result['choices']) > 0:
                            message = result['choices'][0].get('message', {})
                            response_text += message.get('content', '')
                            st.chat_message("assistant").write(response_text)

                return response_text
            except Exception as e:
                return f"Error calling OpenRouter API: {e}"

        openrouter_response = call_openrouter_streaming_api(st.secrets["openrouter_api_key"], st.session_state['url_1_content'][:1000], prompt)
        st.session_state.messages.append({"role": "assistant", "content": openrouter_response[:150]})
        st.write(openrouter_response)

    st.session_state.messages.append({"role": "assistant", "content": response_text[:150]})

    # Automatically ask for more information
    more_info_question = "Want more info? (Yes/No)"
    st.session_state.messages.append({"role": "assistant", "content": more_info_question})

# Handle the user's response for more information
if prompt and prompt.lower() in ["yes", "no"]:
    if prompt.lower() == "yes":
        st.session_state.messages.append({"role": "assistant", "content": "Continuing..."})
    elif prompt.lower() == "no":
        st.session_state.messages.append({"role": "assistant", "content": "What else?"})
