import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set display options for pandas
pd.set_option('display.max_colwidth', None)

# Load the dataset (adjust the path as necessary)
@st.cache_data
def load_data():
    data = pd.read_csv("/Users/ridhampuri/Desktop/streamlit/paper_model/arxiv_data.csv")
    data = data[["titles", "summaries"]]
    return data

data = load_data()

# Text embedding with TF-IDF vectorizer
tfidf_vec = TfidfVectorizer()
tfidf_vec.fit(data["titles"])
data_tfidf = tfidf_vec.transform(data["titles"])

# Function to find the most similar paper
def find_most_similar_paper(query, tfidf_vectorizer, data_tfidf, data):
    query_vec = tfidf_vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vec, data_tfidf).flatten()
    most_similar_idx = cosine_similarities.argmax()
    return data.iloc[most_similar_idx], cosine_similarities[most_similar_idx]

# Streamlit app layout
st.title("ML Paper Recommender")

# User input for query
query = st.text_input("Enter your query", "I want a paper about deep learning and fake news")

if query:
    most_similar_paper, similarity_score = find_most_similar_paper(query, tfidf_vec, data_tfidf, data)
    
    st.write(f"## Most Similar Paper for: '{query}'")
    st.write(f"### Title: {most_similar_paper['titles']}")
    st.write(f"**Summary:** {most_similar_paper['summaries']}")
    st.write(f"**Similarity Score:** {similarity_score:.4f}")

# Run the app with: streamlit run app.py
