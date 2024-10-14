import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load CSV file
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    else:
        return None

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

# Streamlit app
def main():
    st.title("News Reporting Bot")
    st.write("This bot helps you find and rank news articles.")

    # File uploader for CSV
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        # Load the news data
        news_df = load_data(uploaded_file)
        st.write("CSV file loaded successfully!")
        st.write(news_df.head())  # Display the first few rows of the data to check

        # Input prompt
        prompt = st.text_input("Enter your prompt (e.g., 'find the most interesting news' or 'find news about topic X'):")

        # Handle user input
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

if __name__ == "__main__":
    main()
