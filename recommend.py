
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the real movie dataset
movies_data = pd.read_csv("tmdb_5000_movies.csv")
credits_data = pd.read_csv("tmdb_5000_credits.csv")

# Keep useful information
credits_data = credits_data[["movie_id", "cast", "crew"]]

movies_data = movies_data.merge(
    credits_data,
    left_on="id",
    right_on="movie_id"
)

movies = movies_data[
    ["title", "genres", "overview", "cast", "crew"]
].copy()

# Convert JSON-like text into readable names
def extract_names(text):
    try:
        items = ast.literal_eval(text)
        return " ".join(
            item["name"].replace(" ", "")
            for item in items
            if "name" in item
        )
    except (ValueError, TypeError):
        return ""

def extract_cast(text):
    try:
        items = ast.literal_eval(text)
        return " ".join(
            item["name"].replace(" ", "")
            for item in items[:3]
        )
    except (ValueError, TypeError):
        return ""

def extract_director(text):
    try:
        items = ast.literal_eval(text)
        for item in items:
            if item.get("job") == "Director":
                return item["name"].replace(" ", "")
        return ""
    except (ValueError, TypeError):
        return ""

movies["genres"] = movies["genres"].apply(extract_names)
movies["cast"] = movies["cast"].apply(extract_cast)
movies["director"] = movies["crew"].apply(extract_director)

movies["overview"] = movies["overview"].fillna("")

# Combine movie features
movies["features"] = (
    movies["genres"] + " " +
    movies["overview"] + " " +
    movies["cast"] + " " +
    movies["director"]
)

movies = movies.drop_duplicates(subset="title")
movies = movies.reset_index(drop=True)

# Convert features into numerical vectors
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000
)

feature_matrix = vectorizer.fit_transform(movies["features"])

# Calculate similarity between movies
similarity_matrix = cosine_similarity(feature_matrix)


def recommend_movies(movie_title, number=5):
    matches = movies.index[
        movies["title"].str.lower() == movie_title.strip().lower()
    ].tolist()

    if not matches:
        return f"Movie '{movie_title}' was not found."

    movie_index = matches[0]

    scores = list(enumerate(similarity_matrix[movie_index]))
    scores.sort(key=lambda item: item[1], reverse=True)

    recommendations = [
        movies.iloc[index]["title"]
        for index, score in scores
        if index != movie_index
    ]

    return recommendations[:number]


if __name__ == "__main__":
    print("CineMatch Movie Recommendation System")
    print(f"Movies loaded: {len(movies)}")
    print("\nEnter a movie title to find similar movies.")

    selected_movie = input("Movie title: ")
    results = recommend_movies(selected_movie)

    print("\nRecommended movies:")
    if isinstance(results, str):
        print(results)
    else:
        for i, title in enumerate(results, start=1):
            print(f"{i}. {title}")
