# Function to check if the user's question is related to the URL
def is_question_related_to_url(prompt):
    keywords = ["content", "details", "info from", "link", "URL"]
    return any(keyword in prompt.lower() for keyword in keywords)

# Handling user input
if prompt := st.chat_input("Ask your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Limit the buffer to the specified size
    if len(st.session_state.messages) > buffer_size * 2:
        st.session_state.messages = st.session_state.messages[-buffer_size * 2:]

    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Determine if the question is related to the URL content
    url_related = is_question_related_to_url(prompt)
    
    # Combine conversation history and URL texts only if the prompt is related to the URL
    if url_related:
        url_texts = list(st.session_state["parsed_urls"].values())
    else:
        url_texts = []  # No URL content for general questions

    combined_messages = st.session_state.messages + [{"role": "system", "content": "\n".join(url_texts)}]

    # Initialize the response variable
    reply = ""

    # OpenAI Response Handling
    if llm_vendor == "OpenAI":
        client = openai.OpenAI(api_key=st.secrets["openai_key"])  # Initialize OpenAI client with secret key
        messages = [
            {"role": "system", "content": 'You answer questions about web services.'},
            {"role": "user", "content": prompt}  # Pass the user's prompt as the message content
        ]
        # Call the OpenAI API to get the response
        response = client.chat.completions.create(
            model=model_to_use,
            messages=messages,
            temperature=0  # Adjust temperature if needed
        )
        reply = response.choices[0].message.content  # Get the response content

    # Gemini Response Handling
    elif llm_vendor == "Gemini":
        model = genai.GenerativeModel(model_to_use)
        response = model.generate_content("\n".join([msg["content"] for msg in combined_messages]))
        reply = response.text
    
    # Groq Response Handling
    elif llm_vendor == "Groq":
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_to_use
        )
        reply = chat_completion.choices[0].message.content  # Get the response content

    # Display the model's reply
    with st.chat_message("assistant"):
        st.write(reply)

    # Append the assistant's reply to the session state for conversation history
    st.session_state.messages.append({"role": "assistant", "content": reply})

    # Show a message if the answer was based on the URL or model's general knowledge
    if url_related:
        st.sidebar.write("The answer was derived from the studied URLs.")
    else:
        st.sidebar.write("The answer was generated from the model's general knowledge.")
    
    # Limit messages to buffer size after completing the flow
    if len(st.session_state.messages) > buffer_size * 2:
        st.session_state.messages = st.session_state.messages[-buffer_size*2:]
