import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set the page title
st.title("News Reporting Bot")

# File uploader for CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Check if a file has been uploaded
if uploaded_file:
    # Read the uploaded CSV file
    news_df = pd.read_csv(uploaded_file)
    
    # Display the first few rows of the CSV file
    st.write("CSV file loaded successfully!")
    st.write(news_df.head())  # Show the first few rows of the data

    # Input prompt for user to query the news
    prompt = st.text_input("Enter your prompt (e.g., 'find the most interesting news' or 'find news about topic X'):")

    # Define function to rank news by 'interestingness'
    def rank_news(news_df, query, top_n=5):
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(news_df['Document'])  # Use 'Document' for the article content
        query_vector = vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        ranked_news_indices = similarities.argsort()[::-1][:top_n]
        return news_df.iloc[ranked_news_indices]

    # Define function to find news about a specific topic
    def find_news_by_topic(news_df, topic):
        filtered_news = news_df[news_df['Document'].str.contains(topic, case=False, na=False)]  # Search in 'Document'
        return filtered_news

    # Handle user input when "Submit" button is clicked
    if st.button("Submit"):
        if prompt:
            if "interesting" in prompt.lower():
                # Handle "most interesting news" query
                st.write(f"Finding the most interesting news for query: '{prompt}'")
                top_news = rank_news(news_df, prompt)
                for idx, row in top_news.iterrows():
                    st.write(f"**Title:** {row['Document']}")  # Use 'Document' for the title/content
                    st.write(f"**URL:** [Link]({row['URL']})")
                    st.write("---")
            else:
                # Handle news about a specific topic
                topic_news = find_news_by_topic(news_df, prompt)
                if not topic_news.empty:
                    st.write(f"News articles about '{prompt}':")
                    for idx, row in topic_news.iterrows():
                        st.write(f"**Title:** {row['Document']}")  # Use 'Document' for the title/content
                        st.write(f"**URL:** [Link]({row['URL']})")
                        st.write("---")
                else:
                    st.write(f"No articles found about '{prompt}'")
        else:
            st.error("Please enter a prompt.")
