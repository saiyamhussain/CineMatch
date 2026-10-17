# 🎬 CineMatch — Movie Recommendation System

CineMatch is a Python-based movie recommendation web application that helps users discover movies similar to their favourites. It uses **Content-Based Filtering**, **TF-IDF Vectorization**, and **Cosine Similarity** to recommend movies based on their textual features and genres.

The application is built with Streamlit and integrates with The Movie Database (TMDB) API to retrieve movie posters and additional information.

## ✨ Features

- 🎥 **Content-Based Recommendations:** Discover movies similar to a selected title.
- 🧠 **TF-IDF Vectorization:** Convert movie metadata into numerical representations.
- 🔍 **Cosine Similarity:** Calculate similarity between movies.
- 🎭 **Genre-Based Matching:** Include genre similarity in the recommendation score.
- 🖼️ **Movie Posters:** Retrieve posters and additional movie details through the TMDB API.
- 🎨 **Interactive UI:** Explore movies through a Streamlit web interface.
- ⚡ **Caching:** Reuse computed similarity matrices and cached API results to reduce unnecessary work.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application interface |
| Pandas | Data loading and preprocessing |
| Scikit-learn | TF-IDF and cosine similarity |
| Requests | TMDB API integration |
| TMDB API | Movie posters and additional information |

## ⚙️ Recommendation Algorithm

CineMatch follows a content-based recommendation approach.

1. **Data Loading:** Load movie metadata and credits from CSV files.
2. **Data Preprocessing:** Extract genres, keywords, cast, director and overview information.
3. **Feature Engineering:** Combine relevant movie metadata into textual features.
4. **TF-IDF Vectorization:** Convert the text and genres into numerical matrices.
5. **Similarity Calculation:** Use cosine similarity to compare the selected movie with other movies.
6. **Weighted Scoring:** Combine text similarity and genre similarity.
7. **Ranking:** Exclude the selected movie and return the highest-ranked recommendations.

The current scoring formula is:

`Final Score = (0.7 × Text Similarity) + (0.3 × Genre Similarity)`

Text similarity contributes 70% of the score, while genre similarity contributes 30%.

## 📂 Project Structure

```text
CineMatch/
├── app.py
├── recommend.py
├── test_app.py
├── requirements.txt
├── secrets.toml.example
├── tmdb_5000_movies.csv
├── tmdb_5000_credits.csv
├── .gitignore
└── README.md
```

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/saiyamhussain/CineMatch.git
cd CineMatch
```

### 2. Create a virtual environment

**Windows:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the TMDB API token

Create an account at [The Movie Database](https://www.themoviedb.org/) and obtain an API access token if you want to use the poster and movie-details integration.

Create a file named `secrets.toml` inside a `.streamlit` directory in the project root:

```toml
TMDB_BEARER_TOKEN = "your_tmdb_bearer_token"
```

Replace the placeholder with your actual token.

**Security:** Never commit `.streamlit/secrets.toml` or publish your API token. The repository includes `secrets.toml.example` as a configuration reference.

### 5. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal, usually `http://localhost:8501`.

## 🧪 Testing

The repository includes `test_app.py` for project tests.

Run the tests with:

```bash
python -m pytest test_app.py
```

Install `pytest` separately if it is not already included in your environment.

## 🔮 Future Improvements

- Improve recommendation quality using franchise and movie-collection metadata.
- Add recommendation evaluation metrics and systematic testing.
- Improve API error handling and fallback poster behaviour.
- Add filters for genres, ratings and release years.
- Deploy the application so users can access it online.

## 👨‍💻 Author

**Saiyam Hussain**

GitHub: [@saiyamhussain](https://github.com/saiyamhussain)

## 🙏 Acknowledgements

- [The Movie Database (TMDB)](https://www.themoviedb.org/) for movie metadata and poster information.
- [Streamlit](https://streamlit.io/) for the web application framework.
- [Scikit-learn](https://scikit-learn.org/) for machine-learning utilities.

Movie data and images provided through TMDB are subject to TMDB's applicable terms and attribution requirements.
