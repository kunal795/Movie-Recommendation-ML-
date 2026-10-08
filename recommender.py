"""Content-based movie recommender (TF-IDF + cosine similarity), no UI code."""
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

POSTER_BASE = "https://image.tmdb.org/t/p/w342"
SORT_OPTIONS = ["Best match", "Highest rated", "Most popular"]


class MovieEngine:
    def __init__(self, df: pd.DataFrame, vectorizer=None, matrix=None, search_min_votes: int = 10):
        """Pass a saved `vectorizer` + `matrix` (from joblib) to skip training,
        or leave them out and the TF-IDF index is built from df['tags']."""
        df = df.reset_index(drop=True).copy()
        for c in ["title", "genres", "overview", "tagline", "poster_path", "tags"]:
            if c in df.columns:
                df[c] = df[c].fillna("").astype(str)
        for c in ["vote_average", "vote_count", "popularity"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)

        # IMDB-style weighted rating, so a 10/10 from 2 votes does not win
        voted = df[df["vote_count"] > 0]
        c_mean = voted["vote_average"].mean()
        m_votes = voted["vote_count"].quantile(0.80)
        v, r = df["vote_count"], df["vote_average"]
        df["weighted"] = v / (v + m_votes) * r + m_votes / (v + m_votes) * c_mean
        self.df = df

        if matrix is not None and vectorizer is not None:
            if matrix.shape[0] != len(df):
                raise ValueError(
                    f"Saved matrix has {matrix.shape[0]} rows but movie data has {len(df)} rows."
                )
            self.vectorizer = vectorizer
            self.matrix = matrix.tocsr()
        else:
            # same settings as the notebook
            self.vectorizer = TfidfVectorizer(
                max_features=50000, ngram_range=(1, 2), stop_words="english", dtype=np.float32
            )
            self.matrix = self.vectorizer.fit_transform(df["tags"]).tocsr()

        self.genre_sets = [set(g.split("|")) if g else set() for g in df["genres"]]
        self.all_genres = sorted(set().union(*self.genre_sets))

        # searchable labels: "Title (Year)", only for movies with enough votes
        pool = df[df["vote_count"] >= search_min_votes].sort_values("popularity", ascending=False)
        self.label_to_idx, self.idx_to_label = {}, {}
        for idx, t, y in zip(pool.index, pool["title"], pool["year"]):
            label = f"{t} ({y})" if y else t
            while label in self.label_to_idx:
                label += " *"
            self.label_to_idx[label] = idx
            self.idx_to_label[idx] = label
        self.labels = list(self.label_to_idx)
        self.popular_labels = [
            self.idx_to_label[i] for i in pool[pool["vote_count"] >= 500].index
        ]

    # ------------------------------------------------------------ helpers
    @property
    def n_movies(self):
        return len(self.df)

    @property
    def n_features(self):
        return self.matrix.shape[1]

    def movie(self, idx: int) -> dict:
        return self.df.iloc[idx].to_dict()

    def label_for_title(self, title: str):
        """Most-voted movie with this exact title, as a search label."""
        hit = self.df[(self.df["title"] == title) & (self.df.index.isin(self.idx_to_label))]
        if hit.empty:
            return None
        return self.idx_to_label[hit["vote_count"].idxmax()]

    def _mask(self, genres, min_rating, min_votes):
        mask = (self.df["vote_average"].values >= min_rating) & (
            self.df["vote_count"].values >= min_votes
        )
        if genres:
            wanted = set(genres)
            mask &= np.array([bool(wanted & s) for s in self.genre_sets])
        return mask

    def _rank(self, sims, mask, n, sort_by, pool=100):
        sims = sims.copy()
        sims[~mask] = -1.0
        order = np.argsort(-sims)[:pool]
        order = order[sims[order] > 0]
        out = self.df.iloc[order].copy()
        out["similarity"] = sims[order]
        if sort_by == "Highest rated":
            out = out.sort_values("weighted", ascending=False)
        elif sort_by == "Most popular":
            out = out.sort_values("popularity", ascending=False)
        return out.head(n)

    # ------------------------------------------------------------ public API
    def similar(self, idx, n=12, genres=None, min_rating=0.0, min_votes=0, sort_by="Best match"):
        """Movies most similar to movie `idx` (TF-IDF cosine similarity)."""
        sims = (self.matrix @ self.matrix[idx].T).toarray().ravel()
        sims[idx] = -1.0  # never recommend the movie itself
        return self._rank(sims, self._mask(genres, min_rating, min_votes), n, sort_by)

    def from_text(self, query, n=12, genres=None, min_rating=0.0, min_votes=0, sort_by="Best match"):
        """Movies that match a free-text description."""
        cleaned = re.sub(r"[^a-zA-Z\s]", "", str(query).lower())
        q = self.vectorizer.transform([cleaned])
        if q.nnz == 0:
            return self.df.iloc[0:0].assign(similarity=[])
        sims = (self.matrix @ q.T).toarray().ravel()
        return self._rank(sims, self._mask(genres, min_rating, min_votes), n, sort_by)

    def top_picks(self, n=12, genres=None, min_rating=0.0, min_votes=0):
        """Best-rated movies by weighted rating."""
        mask = self._mask(genres, min_rating, min_votes)
        out = self.df[mask].sort_values("weighted", ascending=False).head(n).copy()
        out["similarity"] = np.nan
        return out
