import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load CSV file
@st.cache
def load_data():
    return pd.read_csv(r"/workspaces/document-qa-HW1/Example_news_info_for_testing.csv")  # Make sure the CSV file is uploaded to Streamlit Cloud

# Define function to rank news by 'interestingness'
def rank_news(news_df, query, top_n=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(news_df['story'])
    
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    ranked_news_indices = similarities.argsort()[::-1][:top_n]
    return news_df.iloc[ranked_news_indices]

# Define function to find news about a specific topic
def find_news_by_topic(news_df, topic):
    filtered_news = news_df[news_df['story'].str.contains(topic, case=False, na=False)]
    return filtered_news

# Streamlit app
def main():
    st.title("News Reporting Bot")
    st.write("This bot helps you find and rank news articles.")

    # Load the news data
    news_df = load_data()

    # Sidebar options
    st.sidebar.title("Options")
    option = st.sidebar.selectbox(
        "Choose an option:",
        ("Find most interesting news", "Find news about a topic")
    )

    # Find most interesting news
    if option == "Find most interesting news":
        query = st.text_input("Enter a query for 'interesting' news (e.g., 'legal', 'technology'):")

        if st.button("Find Most Interesting News"):
            if query:
                top_news = rank_news(news_df, query)
                st.write(f"Top news articles for '{query}':")
                for idx, row in top_news.iterrows():
                    st.write(f"**Title:** {row['title']}")
                    st.write(f"**Story:** {row['story']}")
                    st.write("---")
            else:
                st.error("Please enter a query.")

    # Find news about a topic
    elif option == "Find news about a topic":
        topic = st.text_input("Enter a topic (e.g., 'legal', 'economy'):")

        if st.button("Find News on Topic"):
            if topic:
                topic_news = find_news_by_topic(news_df, topic)
                st.write(f"News articles about '{topic}':")
                for idx, row in topic_news.iterrows():
                    st.write(f"**Title:** {row['title']}")
                    st.write(f"**Story:** {row['story']}")
                    st.write("---")
            else:
                st.error("Please enter a topic.")

if __name__ == "__main__":
    main()
