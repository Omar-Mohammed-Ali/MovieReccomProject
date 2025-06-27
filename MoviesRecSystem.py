import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import streamlit as st

# Load dataset
movies = pd.read_csv("movies.csv")
movies['genres'] = movies['genres'].fillna('')

# Combine features
movies['combined_features'] = movies['title'] + ' ' + movies['genres']

# Vectorize combined features
vectorizer = TfidfVectorizer(stop_words='english')
feature_matrix = vectorizer.fit_transform(movies['combined_features'])

# Build Nearest Neighbors model
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(feature_matrix)

# Recommendation function
def content_based_recommendation(title):
    index = movies[movies['title'] == title].index[0]

    movie_vector = feature_matrix[index]
    distances, indices = model.kneighbors(movie_vector, n_neighbors=6)

    recommended_indices = indices.flatten()[1:]
    recommended_titles = movies.iloc[recommended_indices]['title'].tolist()

    return recommended_titles

# Streamlit UI
st.set_page_config(page_title="Movie Recommendation System 🎬", layout="centered")

st.title("🎥 Movie Recommender")
st.markdown("_Find your next favorite movie based on genres you love!_")
st.markdown("💡 **Start typing a movie name below and select it from the dropdown.**")

# Dropdown input
movie_input = st.selectbox("🎞️ Select a Movie", movies['title'].tolist())

# Recommendation button
if st.button("🔍 Recommend"):
    recommendations = content_based_recommendation(movie_input)
    
    st.markdown(f"### Showing recommendations for: **{movie_input}**")
    for i, rec in enumerate(recommendations, 1):
        st.write(f"🎬 {i}. {rec}")
