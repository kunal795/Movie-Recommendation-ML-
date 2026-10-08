import html
import random
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from recommender import POSTER_BASE, SORT_OPTIONS, MovieEngine

BASE = Path(__file__).parent
DATA_PATH = BASE / "movies_clean.csv.gz"
JOBLIB_FILES = ("tfidf_vectorizer.joblib", "tfidf_matrix.joblib", "movies_df.joblib")
REQUIRED_COLS = {"title", "year", "genres", "overview", "tagline", "vote_average",
                 "vote_count", "popularity", "poster_path"}
QUICK_TITLES = ["Toy Story", "The Dark Knight", "Inception", "Titanic", "Frozen"]
EXAMPLE_QUERIES = [
    "A lonely robot falls in love in space",
    "Detectives hunt a serial killer in a rainy city",
    "Two best friends go on a hilarious road trip",
    "A young wizard attends a magical school",
    "A crew plans one last big heist",
]

st.set_page_config(page_title="CineMatch - Movie Recommender", page_icon="🎬", layout="wide")

# ------------------------------------------------------------------ styling
# (no blank lines inside this block, otherwise Markdown breaks the HTML)
st.markdown(
    """
<style>
.hero{padding:6px 0 2px}
.hero h1{font-size:2.6rem;font-weight:800;margin:0;line-height:1.15;
  background:linear-gradient(90deg,#f43f5e,#f59e0b);-webkit-background-clip:text;background-clip:text;color:transparent}
.hero p{opacity:.75;margin:4px 0 0;font-size:1.05rem}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:16px;margin:8px 0 18px}
.card{border-radius:12px;overflow:hidden;background:rgba(127,127,127,.10);border:1px solid rgba(127,127,127,.22);
  transition:transform .15s ease,box-shadow .15s ease}
.card:hover{transform:translateY(-4px);box-shadow:0 10px 24px rgba(0,0,0,.28)}
.poster{position:relative;aspect-ratio:2/3;background-size:cover;background-position:center;display:flex;align-items:center;justify-content:center}
.noposter{font-size:3rem;font-weight:800;color:rgba(255,255,255,.55)}
.badge{position:absolute;font-size:.72rem;font-weight:700;padding:2px 8px;border-radius:999px;background:rgba(0,0,0,.78);color:#fff}
.badge.r{top:8px;left:8px;color:#fbbf24}
.badge.s{bottom:8px;left:8px;background:#f43f5e}
.ov{position:absolute;inset:0;background:rgba(0,0,0,.86);color:#f1f5f9;padding:12px;font-size:.78rem;line-height:1.4;
  opacity:0;transition:opacity .2s ease;overflow:hidden}
.card:hover .ov{opacity:1}
.info{padding:8px 10px 10px}
.info .t{font-weight:700;font-size:.9rem;line-height:1.25;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.info .m{opacity:.7;font-size:.78rem;margin-bottom:2px}
.chip{display:inline-block;font-size:.68rem;padding:1px 8px;margin:3px 4px 0 0;border-radius:999px;
  background:rgba(244,63,94,.14);border:1px solid rgba(244,63,94,.38)}
.detail{display:flex;gap:20px;padding:16px;border-radius:14px;background:rgba(127,127,127,.10);
  border:1px solid rgba(127,127,127,.22);margin:12px 0 22px}
.dposter{flex:0 0 130px;aspect-ratio:2/3;border-radius:10px;background-size:cover;background-position:center}
.dtext h3{margin:0 0 6px;font-size:1.45rem}
.dtext h3 span{opacity:.6;font-weight:400;font-size:1.1rem}
.rating{font-weight:700;color:#f59e0b;margin:6px 0}
.tagline{font-style:italic;opacity:.8;margin:4px 0 8px}
.plot{line-height:1.55;opacity:.92}
@media (max-width:640px){.detail{flex-direction:column}.dposter{width:120px;flex:none}.hero h1{font-size:2rem}}
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ data
@st.cache_resource(show_spinner="Loading the movie model...")
def load_engine():
    # Way 1: saved model files (joblib), like the credit risk app
    if all((BASE / f).exists() for f in JOBLIB_FILES):
        vec = joblib.load(BASE / JOBLIB_FILES[0])
        mat = joblib.load(BASE / JOBLIB_FILES[1])
        df = joblib.load(BASE / JOBLIB_FILES[2])
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError("OLD_MOVIES_DF:" + ", ".join(sorted(missing)))
        return MovieEngine(df, vectorizer=vec, matrix=mat)
    # Way 2: build the TF-IDF index from the cleaned CSV (no pickle files needed)
    if DATA_PATH.exists():
        return MovieEngine(pd.read_csv(DATA_PATH))
    return None


try:
    engine = load_engine()
except ValueError as err:
    if str(err).startswith("OLD_MOVIES_DF:"):
        st.error(
            "**`movies_df.joblib` is the old version** - it is missing these columns: "
            f"`{str(err).split(':', 1)[1].strip()}`.\n\n"
            "**Fix:** in your notebook, run the cell from `save_model_cell.py` (it adds poster, year and "
            "vote count and re-saves the 3 files), copy the new `.joblib` files into this folder, "
            "then restart the app."
        )
        st.stop()
    raise
if engine is None:
    st.error(
        "Model files not found. Put the 3 `.joblib` files saved from your notebook "
        "(`tfidf_vectorizer`, `tfidf_matrix`, `movies_df`) in the same folder as `app.py`."
    )
    st.stop()


# ------------------------------------------------------------------ html helpers
def poster_style(path: str, size="cover") -> str:
    fallback = "linear-gradient(135deg,#475569,#1e293b)"
    if path:
        return html.escape(f"background-image:url('{POSTER_BASE}{path}'),{fallback};", quote=True)
    return html.escape(f"background-image:{fallback};", quote=True)


def chips(genres: str, limit=3) -> str:
    return "".join(
        f'<span class="chip">{html.escape(g)}</span>' for g in genres.split("|")[:limit] if g
    )


def short(text: str, limit: int) -> str:
    text = text.strip()
    return html.escape(text if len(text) <= limit else text[:limit].rstrip() + "...")


def card(m: dict) -> str:
    title = html.escape(m["title"])
    rating = f'<span class="badge r">★ {m["vote_average"]:.1f}</span>' if m["vote_count"] > 0 else ""
    sim = m.get("similarity")
    sim_badge = f'<span class="badge s">{sim * 100:.0f}% similar</span>' if pd.notna(sim) else ""
    letter = "" if m["poster_path"] else f'<div class="noposter">{html.escape(m["title"][:1])}</div>'
    overview = short(m["overview"], 230) or "No description available."
    year = m["year"] if m["year"] else ""
    return (
        f'<div class="card"><div class="poster" style="{poster_style(m["poster_path"])}">'
        f"{letter}{rating}{sim_badge}<div class=\"ov\">{overview}</div></div>"
        f'<div class="info"><div class="t" title="{title}">{title}</div>'
        f'<div class="m">{year}</div><div>{chips(m["genres"])}</div></div></div>'
    )


def grid(df: pd.DataFrame) -> str:
    return '<div class="grid">' + "".join(card(m) for m in df.to_dict("records")) + "</div>"


def detail(m: dict) -> str:
    tagline = m["tagline"].strip()
    tag_html = f'<div class="tagline">“{html.escape(tagline)}”</div>' if tagline else ""
    rating = (
        f'<div class="rating">★ {m["vote_average"]:.1f} <span style="opacity:.6;font-weight:400">'
        f'({int(m["vote_count"]):,} votes)</span></div>'
        if m["vote_count"] > 0
        else ""
    )
    year = f' <span>({m["year"]})</span>' if m["year"] else ""
    return (
        f'<div class="detail"><div class="dposter" style="{poster_style(m["poster_path"])}"></div>'
        f'<div class="dtext"><h3>{html.escape(m["title"])}{year}</h3>{chips(m["genres"], 5)}'
        f'{rating}{tag_html}<div class="plot">{short(m["overview"], 600)}</div></div></div>'
    )


def show_results(res: pd.DataFrame, heading: str, key: str):
    st.subheader(heading)
    if res.empty:
        st.info("No movies match these filters. Try lowering the minimum rating / votes or removing a genre.")
        return
    st.markdown(grid(res), unsafe_allow_html=True)
    left, right = st.columns([4, 1])
    with left:
        if res["similarity"].notna().any():
            st.caption("Hover over a poster to read the plot. 'Similar' is the TF-IDF cosine similarity of the movie text.")
        else:
            st.caption("Ranked by IMDB-style weighted rating (rating balanced with number of votes).")
    with right:
        export = res[["title", "year", "genres", "vote_average", "vote_count", "similarity"]]
        st.download_button("⬇ Download CSV", export.to_csv(index=False).encode("utf-8"),
                           file_name="recommendations.csv", mime="text/csv", key=key,
                           use_container_width=True)


# ------------------------------------------------------------------ callbacks
def set_pick(label):
    st.session_state["movie_pick"] = label


def surprise():
    st.session_state["movie_pick"] = random.choice(engine.popular_labels)


def set_desc(text):
    st.session_state["desc_text"] = text


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("🎛️ Filters")
    st.caption("These apply to every tab.")
    n = st.slider("Number of recommendations", 6, 24, 12, step=2)
    genres = st.multiselect("Genres (match any)", engine.all_genres)
    min_rating = st.slider("Minimum rating", 0.0, 9.0, 0.0, step=0.5)
    min_votes = st.slider("Minimum number of votes", 0, 1000, 20, step=10,
                          help="Hides obscure movies with only a handful of ratings.")
    sort_by = st.radio("Sort results by", SORT_OPTIONS,
                       help="The best 100 matches are found first, then sorted this way.")
    st.divider()
    st.caption(f"📚 {engine.n_movies:,} movies indexed")

# ------------------------------------------------------------------ header
st.markdown(
    '<div class="hero"><h1>🎬 CineMatch</h1>'
    "<p>Tell us a movie you love, or describe what you feel like watching, and we'll find your next favourite.</p></div>",
    unsafe_allow_html=True,
)

tab_similar, tab_describe, tab_top, tab_about = st.tabs(
    ["🎯 Movies like...", "✍️ Describe a movie", "🏆 Top picks", "ℹ️ How it works"]
)

# ---- Tab 1: similar movies
with tab_similar:
    c1, c2 = st.columns([5, 1])
    with c1:
        st.selectbox("Search for a movie", engine.labels, index=None,
                     placeholder="Start typing a movie name, e.g. Jumanji...",
                     key="movie_pick", label_visibility="collapsed")
    with c2:
        st.button("🎲 Surprise me", on_click=surprise, use_container_width=True)

    quick = [lab for lab in (engine.label_for_title(t) for t in QUICK_TITLES) if lab]
    if quick:
        st.caption("Or try one of these:")
        for col, lab in zip(st.columns(len(quick)), quick):
            col.button(lab.split(" (")[0], key=f"quick_{lab}", on_click=set_pick,
                       args=(lab,), use_container_width=True)

    picked = st.session_state.get("movie_pick")
    if picked and picked in engine.label_to_idx:
        idx = engine.label_to_idx[picked]
        movie = engine.movie(idx)
        st.markdown(detail(movie), unsafe_allow_html=True)
        res = engine.similar(idx, n, genres, min_rating, min_votes, sort_by)
        show_results(res, f"Because you liked {movie['title']}", key="dl_similar")
    else:
        st.info("👆 Pick a movie above to get recommendations.")

# ---- Tab 2: free-text description
with tab_describe:
    st.write("Describe the plot, mood or characters you want. The more detail, the better the match.")
    st.text_area("Your description", key="desc_text", height=110, label_visibility="collapsed",
                 placeholder="e.g. A detective investigates a mysterious murder on a train...")
    st.caption("Try an example:")
    for col, text in zip(st.columns(len(EXAMPLE_QUERIES)), EXAMPLE_QUERIES):
        col.button(text, key=f"ex_{text}", on_click=set_desc, args=(text,), use_container_width=True)
    st.button("🔍 Find movies", type="primary")

    query = st.session_state.get("desc_text", "").strip()
    if query:
        res = engine.from_text(query, n, genres, min_rating, min_votes, sort_by)
        if res.empty and not engine.from_text(query, 1).shape[0]:
            st.warning("None of those words are in the movie index. Try different or more descriptive words.")
        else:
            show_results(res, "Movies that match your description", key="dl_describe")

# ---- Tab 3: top picks
with tab_top:
    label = ", ".join(genres) if genres else "all genres"
    show_results(engine.top_picks(n, genres, min_rating, min_votes),
                 f"Top rated movies ({label})", key="dl_top")

# ---- Tab 4: about
with tab_about:
    st.markdown(
        """
### How the recommender works
This is a **content-based** recommender: it looks at what a movie is *about*, not at what other users watched.

1. **Build the text** for each movie from its overview, genres and tagline.
2. **Clean it**: lowercase, remove punctuation and stopwords, lemmatize.
3. **TF-IDF**: turn each text into a vector. Rare, meaningful words (like "jungle" or "heist") get more weight than common ones.
4. **Cosine similarity**: compare the chosen movie's vector with all others. The closest vectors are the recommendations.
5. **Filters**: genre, rating and vote filters from the sidebar are applied on top.

### Features
- **Movies like...** similar movies to one you pick (with plot and details).
- **Describe a movie** type what you want in plain English and the same TF-IDF model finds matches.
- **Top picks** best-rated movies using an IMDB-style weighted rating.
- **Download** any list of results as a CSV.

### Good to know
- Similarity scores of 10-40% are normal for text; they are not probabilities.
- Movies with a very short or missing description give weaker recommendations.
- Content-based models do not know personal taste; they only compare movie text.
"""
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Movies indexed", f"{engine.n_movies:,}")
    m2.metric("TF-IDF features", f"{engine.n_features:,}")
    m3.metric("Genres", len(engine.all_genres))

st.divider()
st.caption("Data: The Movies Dataset (TMDB metadata, via Kaggle). Posters are loaded from TMDB. "
           "This product is not endorsed or certified by TMDB.")
