from pathlib import Path
import time
import ast
import os

import pandas as pd
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# CineMatch: simple, complete Streamlit movie recommender.
ROOT = Path(__file__).resolve().parent
MOVIES_CSV = ROOT / "tmdb_5000_movies.csv"
CREDITS_CSV = ROOT / "tmdb_5000_credits.csv"
TMDB_SEARCH = "https://api.themoviedb.org/3/search/movie"
TMDB_POSTERS = "https://image.tmdb.org/t/p/w500"


st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(ellipse at top left, #351116 0%, transparent 48%),
        linear-gradient(145deg, #080809, #171013 60%, #09090c);
    color: #f5f5f5;
}

.block-container {
    max-width: 1180px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #ffffff !important;
}

.stButton > button {
    background: linear-gradient(110deg, #ed2438, #a81022);
    color: #ffffff !important;
    border: 1px solid #ff5660;
    border-radius: 10px;
    font-weight: 700;
    min-height: 44px;
}

.stButton > button:hover {
    background: #ff3045;
    color: #ffffff !important;
    border-color: #ff8a91;
    box-shadow: 0 4px 20px #ed243840;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(145deg, #1c191d, #111115);
    border: 1px solid #38272c;
    border-radius: 16px;
}

[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: #bd3544;
}

[data-testid="stImage"] img {
    border-radius: 12px;
}

[data-baseweb="select"] > div {
    background: #171317;
    border-color: #513038;
    border-radius: 10px;
}

hr {
    border-color: #38272c;
}

</style>
""", unsafe_allow_html=True)



def parse_cell(cell):
    if not isinstance(cell, str) or not cell.strip():
        return []
    try:
        value = ast.literal_eval(cell)
    except (ValueError, SyntaxError, TypeError):
        return []
    return value if isinstance(value, list) else []


def extract_names(cell):
    return " ".join(
        item.get("name", "")
        for item in parse_cell(cell)
        if isinstance(item, dict) and item.get("name")
    )



@st.cache_data(show_spinner=False)
def read_movies():
    if not MOVIES_CSV.is_file():
        raise FileNotFoundError(f"Missing file: {MOVIES_CSV.name}")
    if not CREDITS_CSV.is_file():
        raise FileNotFoundError(f"Missing file: {CREDITS_CSV.name}")

    movies = pd.read_csv(MOVIES_CSV)
    credits = pd.read_csv(CREDITS_CSV)

    if "id" not in movies.columns or "movie_id" not in credits.columns:
        raise ValueError("The CSV files do not have the expected id/movie_id columns.")

    # Keep the movie title from the movies dataset to prevent duplicate title columns.
    credits = credits.drop(columns=["title"], errors="ignore")
    data = movies.merge(
        credits,
        left_on="id",
        right_on="movie_id",
        how="left",
    )

    for col in ("genres", "keywords", "overview", "title"):
        if col not in data.columns:
            data[col] = ""
        data[col] = data[col].fillna("").astype(str)

    data["genres_text"] = data["genres"].map(extract_names)
    data["keywords_text"] = data["keywords"].map(extract_names)

    if "cast" in data.columns:
        data["cast_text"] = data["cast"].fillna("").map(
            lambda cell: " ".join(extract_names(cell).split()[:10])
        )
    else:
        data["cast_text"] = ""

    if "crew" in data.columns:
        def directors(cell):
            return " ".join(
                person.get("name", "")
                for person in parse_cell(cell)
                if isinstance(person, dict) and person.get("job") == "Director"
            )

        data["director_text"] = data["crew"].fillna("").map(directors)
    else:
        data["director_text"] = ""

    data["features"] = (
        data["genres_text"] + " "
        + data["keywords_text"] + " "
        + data["overview"] + " "
        + data["cast_text"] + " "
        + data["director_text"]
    ).str.lower()

    data = data[data["title"].str.strip().ne("")].reset_index(drop=True)

    if data.empty:
        raise ValueError("No movie titles were found in the datasets.")

    return data



def tmdb_token():
    try:
        value = st.secrets.get("TMDB_BEARER_TOKEN", "")
    except Exception:
        value = ""
    return str(value or os.getenv("TMDB_BEARER_TOKEN", "")).strip()




@st.cache_data(ttl=86400, show_spinner=False)
def poster_for(title, year, token):
    if not token:
        return None

    headers = {
        "Authorization": f"Bearer {token}",
        "accept": "application/json",
    }

    searches = []

    params = {"query": title, "include_adult": "false"}
    if year and str(year).isdigit():
        params["year"] = str(year)

    searches.append(params)

    if "year" in params:
        searches.append({
            "query": title,
            "include_adult": "false",
        })

    for params in searches:
        results = []

        # Retry temporary connection failures up to 3 times.
        for attempt in range(3):
            try:
                response = requests.get(
                    TMDB_SEARCH,
                    headers=headers,
                    params=params,
                    timeout=(5, 15),
                )
                response.raise_for_status()
                results = response.json().get("results", [])
                break

            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt == 2:
                    print(
                        f"[TMDB ERROR] {title!r}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                else:
                    time.sleep(1.5 * (attempt + 1))

            except (requests.RequestException, ValueError, TypeError) as exc:
                print(
                    f"[TMDB ERROR] {title!r}: "
                    f"{type(exc).__name__}: {exc}"
                )
                break

        target = title.strip().casefold()

        exact = [
            item for item in results
            if target in {
                str(item.get("title") or "").strip().casefold(),
                str(item.get("original_title") or "").strip().casefold(),
            }
        ]

        candidates = exact + [
            item for item in results if item not in exact
        ]

        for item in candidates:
            poster_path = item.get("poster_path")

            if poster_path:
                return {
                    "url": TMDB_POSTERS + poster_path,
                    "overview": item.get("overview") or "",
                    "rating": item.get("vote_average"),
                }

    return None



@st.cache_resource
def build_similarity_matrices(features, genres):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=40000,
        ngram_range=(1, 2)
    )
    text_matrix = vectorizer.fit_transform(features)

    genre_vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b"
    )
    genre_matrix = genre_vectorizer.fit_transform(genres)

    return text_matrix, genre_matrix


def recommendations(data, chosen_title, limit):
    features = tuple(data["features"].fillna("").astype(str))
    genres = tuple(data["genres_text"].fillna("").astype(str))

    matrix, genre_matrix = build_similarity_matrices(
        features, genres
    )

    positions = data.index[
        data["title"].eq(chosen_title)
    ].tolist()

    if not positions:
        return data.iloc[0:0].copy()

    source = positions[0]

    text_scores = cosine_similarity(
        matrix[source], matrix
    ).ravel()

    genre_scores = cosine_similarity(
        genre_matrix[source], genre_matrix
    ).ravel()

    scores = (0.3 * genre_scores) + (0.7 * text_scores)

    # Exclude the selected movie itself.
    scores[source] = -1

    ranked = scores.argsort()[::-1]
    ranked = ranked[:limit]

    return data.iloc[ranked].copy()



def show_movie(movie, token):
    title = str(movie.get("title", "Untitled"))
    date = str(movie.get("release_date", "") or "")
    year = date[:4] if len(date) >= 4 else ""
    info = poster_for(title, year, token) if token else None

    with st.container(border=True):
        poster_col, text_col = st.columns([1, 2], gap="medium")

        with poster_col:
            if info and info.get("url"):
                st.image(info["url"], use_container_width=True)
            else:
                st.markdown(
                    """
                    <div style="
                        height:230px;
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        background:#21181c;
                        border:1px solid #49262c;
                        border-radius:10px;
                        font-size:42px;
                    ">🎬</div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption("Poster not available")

        with text_col:
            st.markdown(
                f"""
                <div style="
                    color:#ffffff;
                    font-size:21px;
                    font-weight:800;
                    line-height:1.3;
                    margin:4px 0 8px;
                ">{title}</div>
                """,
                unsafe_allow_html=True,
            )

            if year:
                st.caption(f"📅 {year}")
                genres = str(movie.get("genres_text", "") or "").strip()

            if genres:
                st.caption(f"🎭 {genres.title()}")

            rating = (info or {}).get("rating")
            if rating is not None:
                st.markdown(
                    f"""
                    <div style="
                        display:inline-block;
                        background:#32171d;
                        color:#ff6873;
                        border:1px solid #63303a;
                        border-radius:7px;
                        padding:4px 9px;
                        font-size:12px;
                        font-weight:700;
                        margin:4px 0 10px;
                    ">★ TMDB {float(rating):.1f}/10</div>
                    """,
                    unsafe_allow_html=True,
                )

            summary = (
                (info or {}).get("overview")
                or str(movie.get("overview", "") or "")
            )

            if summary:
                st.write(
                    summary[:320] + ("…" if len(summary) > 320 else "")
                )


def app():

    st.markdown("""
<div style="padding: 42px 32px 36px; border: 1px solid #49262c; border-radius: 18px; margin-bottom: 24px; background: radial-gradient(ellipse at 80% 20%, #65202b 0%, transparent 48%), linear-gradient(120deg, #160b0e 0%, #241015 55%, #09090c 100%); box-shadow: 0 12px 40px #00000055;">
    <div style="color:#ff5964; font-size:11px; font-weight:800; letter-spacing:3px;">
        YOUR PERSONAL MOVIE DISCOVERY ENGINE
    </div>
    <div style="color:#ffffff; font-size:56px; line-height:1.1; font-weight:900; margin-top:8px;">
        CineMatch<span style="color:#ed343e;">.</span>
    </div>
    <div style="color:#c2bdc0; font-size:16px; margin-top:10px;">
        Love a movie. Discover your next obsession.
    </div>
</div>
""", unsafe_allow_html=True)


    st.write("Choose a movie to discover films with similar stories, genres, and people.")

    # Keep the first render independent of the recommender and TMDB requests.
    try:
        movies = read_movies()
    except Exception as exc:
        st.error("Could not load the movie data.")
        st.code(str(exc))
        st.info("Place both TMDB CSV files beside app.py and restart Streamlit.")
        st.stop()

     
    st.markdown(
    f"""
    <div style="
        background:#1b1418;
        border:1px solid #63303a;
        border-left:4px solid #ed343e;
        border-radius:9px;
        padding:12px 16px;
        color:#f5dfe2;
        font-size:14px;
        margin:12px 0 20px;
    ">
        <span style="color:#ff5964;">●</span>
        &nbsp; Movie database ready
        <span style="color:#ff8a91;font-weight:700;">
            · {len(movies):,} titles
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

    token = tmdb_token()
    if not token:
        st.warning("TMDB token not found; recommendations still work, but posters may be unavailable.")

    title_options = sorted(movies["title"].unique().tolist())
    chosen = st.selectbox(
    "🎬 Choose a movie",
    title_options,
    help="Select a movie you enjoyed to discover similar films."
)
    limit = st.slider(
    "How many movies should we recommend?",
    min_value=2,
    max_value=10,
    value=6,
    help="Choose between 2 and 10 similar movies."
)

    if st.button("✨ Discover My Next Movie", type="primary", use_container_width=True):
        try:
            with st.spinner("Calculating similar movies..."):
                result = recommendations(movies, chosen, limit)
        except Exception as exc:
            st.error("The recommendation step failed.")
            st.code(str(exc))
            st.stop()


        st.markdown(
    """
    <div style="margin:28px 0 18px;">
        <div style="
            color:#ff5964;
            font-size:11px;
            font-weight:800;
            letter-spacing:3px;
        ">YOUR NEXT WATCH</div>
        <div style="
            color:#ffffff;
            font-size:30px;
            font-weight:900;
            margin-top:5px;
        ">Movies picked for you<span style="color:#ed343e;">.</span></div>
        <div style="color:#aaa1a5;font-size:14px;margin-top:5px;">
            Because great movies lead to even better discoveries.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
        if result.empty:
            st.info("No similar movies were found.")
        else:
            for i in range(0, len(result), 2):
                cols = st.columns(2)
                for offset, col in enumerate(cols):
                    idx = i + offset
                    if idx < len(result):
                        with col:
                            show_movie(result.iloc[idx].to_dict(), token)


app()