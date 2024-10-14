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

    # Sidebar options
    st.sidebar.title("Options")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        # Load and display data
        news_df = load_data(uploaded_file)
        st.write("CSV file loaded successfully!")
        st.write(news_df.head())  # Display the first few rows of the data to check
        
        # Rest of your code to select options
        option = st.sidebar.selectbox(
            "Choose an option:",
            ("Find most interesting news", "Find news about a topic")
        )

        if option == "Find most interesting news":
            query = st.text_input("Enter a query for 'interesting' news (e.g., 'legal', 'technology'):")

            if st.button("Find Most Interesting News"):
                if query:
                    top_news = rank_news(news_df, query)
                    st.write(f"Top news articles for '{query}':")
                    for idx, row in top_news.iterrows():
                        st.write(f"**Title:** {row['Document']}")  # Use 'Document' for the title/content
                        st.write(f"**URL:** [Link]({row['URL']})")
                        st.write("---")
                else:
                    st.error("Please enter a query.")

        elif option == "Find news about a topic":
            topic = st.text_input("Enter a topic (e.g., 'legal', 'economy'):")

            if st.button("Find News on Topic"):
                if topic:
                    topic_news = find_news_by_topic(news_df, topic)
                    st.write(f"News articles about '{topic}':")
                    for idx, row in topic_news.iterrows():
                        st.write(f"**Title:** {row['Document']}")  # Use 'Document' for the title/content
                        st.write(f"**URL:** [Link]({row['URL']})")
                        st.write("---")
                else:
                    st.error("Please enter a topic.")

if __name__ == "__main__":
    main()
