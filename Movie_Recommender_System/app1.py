# Import all necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re

# Download NLTK data (optional, can be done once)
# nltk.download('punkt')
# nltk.download('stopwords')

# Full Recommendation System function
# Place all your data loading and preprocessing logic inside this function
def recommend_movies(movie_name):
    # Load and preprocess data
    df = pd.read_csv(r"E:\NLP\movies.csv")
    selected_features = ["genres", "keywords", 'overview', "title"]
    df = df[selected_features]
    df = df.dropna().reset_index(drop=True)
    df["combined"] = df["genres"] + " " + df["keywords"] + " " + df["overview"] + " " + df["title"]
    data = df[["title", "combined"]]

    # Preprocess text
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    def preprocess_text(text):
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
        return " ".join(tokens)

    data["clean_text"] = data["combined"].apply(preprocess_text)

    # Vectorize and calculate similarity
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    tfidf_matrix = tfidf_vectorizer.fit_transform(data["clean_text"])
    cosine_sim = cosine_similarity(tfidf_matrix)

    titles = data['title'].str.lower().tolist()
    movie_name = movie_name.lower()

    if movie_name in titles:
        index_of_movie = data[data.title.str.lower() == movie_name].index[0]
    else:
        close_match = difflib.get_close_matches(movie_name, titles)
        if not close_match:
            return "Movie not found. Please check the spelling or try another movie."
        index_of_movie = data[data.title.str.lower() == close_match[0]].index[0]

    similarity_score = list(enumerate(cosine_sim[index_of_movie]))
    sort_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    top_movies = sort_similar_movies[1:6]

    recommended_list = []
    for movie in top_movies:
        index = movie[0]
        title_movie = data[data.index == index]['title'].values[0]
        recommended_list.append(title_movie)

    return recommended_list

# Streamlit App
st.title('🎬 Movie Recommendation System')
st.write("Enter the name of your favorite movie and get recommendations!")

movie_name = st.text_input('Enter a movie name:')

if movie_name:
    recommendations = recommend_movies(movie_name)

    if isinstance(recommendations, str):
        st.error(recommendations)
    else:
        st.subheader(f"Movies suggested for you based on '{movie_name}':")
        for i, movie in enumerate(recommendations, 1):
            st.write(f"{i}. {movie}")