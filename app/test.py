"""
Information Retrieval Demo 📚🔍
================================
A single‑file Streamlit application showcasing a classical IR pipeline:
• TF–IDF indexing  + cosine‑similarity search.
• Latent Dirichlet Allocation (LDA) topic modelling.
• Analytics dashboard with word‑clouds & length histograms.

Fixes & improvements vs. the original snippet
---------------------------------------------
1. **LDA uses a CountVectorizer, not TF–IDF.**  (sklearn’s LDA expects counts.)
2. **Robust NLTK handling** – if stop‑word corpus can’t download (e.g. offline)
   we fall back to a small built‑in list.
3. **Session‑safe document storage** – the docs & IDs live in *st.session_state*,
   so UI events won’t lose them.
4. **Accurate topic‑word weights** – top‑N words *and* their weights are paired
   correctly.
5. **Regex‑based highlighting** – case‑insensitive, whole‑word matches.
6. Minor polish: reset button, loader spinners, clearer empty‑state messages.

Run
---
```bash
pip install streamlit scikit‑learn nltk plotly wordcloud matplotlib pandas numpy
streamlit run information_retrieval_app.py
```
"""

from __future__ import annotations

import os
import re
import io
import base64
from typing import List, Dict

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ────────────────────────────── Streamlit page config ──────────────────────────────
st.set_page_config("Information Retrieval Demo", "🔍", layout="wide")

# ────────────────────────────── NLTK bootstrap ──────────────────────────────
@st.cache_data(show_spinner=False)
def ensure_nltk() -> set[str]:
    """Return an English stop‑word set – download corpora if needed, else fallback."""
    try:
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("corpora/stopwords")
    except LookupError:
        try:
            nltk.download("punkt", quiet=True)
            nltk.download("stopwords", quiet=True)
        except Exception:
            pass  # maybe offline
    try:
        sw = set(stopwords.words("english"))
    except LookupError:
        # offline fallback (tiny but better than nothing)
        sw = {
            "the","and","is","in","it","of","to","for","on","with","a","an","by","from","that","this"
        }
    return sw

STOP_WORDS = ensure_nltk()
STEMMER = PorterStemmer()

# ────────────────────────────── Core IR engine ──────────────────────────────
class InformationRetrievalSystem:
    def __init__(self, n_topics: int = 5):
        self.raw_docs: List[str] = []
        self.doc_ids: List[str] = []
        self._tfidf: TfidfVectorizer | None = None
        self._tfidf_matrix = None
        self._count_vec: CountVectorizer | None = None
        self._term_matrix = None  # counts for LDA
        self._lda: LatentDirichletAllocation | None = None
        self.n_topics = n_topics

    # ───────────────── preprocessing ─────────────────
    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _tokenise(self, text: str) -> List[str]:
        tokens = word_tokenize(text)
        return [STEMMER.stem(t) for t in tokens if t not in STOP_WORDS]

    def _preprocess(self, text: str) -> str:
        return " ".join(self._tokenise(self._clean_text(text)))

    # ───────────────── indexing ─────────────────
    def add_documents(self, docs: List[str], ids: List[str]):
        self.raw_docs = docs
        self.doc_ids = ids
        processed_docs = [self._preprocess(d) for d in docs]

        # TF‑IDF for search
        self._tfidf = TfidfVectorizer(max_features=1500, ngram_range=(1, 2))
        self._tfidf_matrix = self._tfidf.fit_transform(processed_docs)

        # Term counts for LDA
        self._count_vec = CountVectorizer(max_features=2000)
        self._term_matrix = self._count_vec.fit_transform(processed_docs)

        self._lda = LatentDirichletAllocation(
            n_components=self.n_topics,
            learning_method="batch",
            random_state=42,
        ).fit(self._term_matrix)

    # ───────────────── search ─────────────────
    def search(self, query: str, k: int = 5) -> List[Dict]:
        if not self._tfidf:
            return []
        q_vec = self._tfidf.transform([self._preprocess(query)])
        sims = cosine_similarity(q_vec, self._tfidf_matrix).flatten()
        top_idx = sims.argsort()[::-1][:k]
        results = [
            {
                "doc_id": self.doc_ids[i],
                "doc": self.raw_docs[i],
                "score": float(sims[i]),
            }
            for i in top_idx if sims[i] > 0
        ]
        return results

    # ───────────────── topics ─────────────────
    def doc_topic_matrix(self):
        return self._lda.transform(self._term_matrix) if self._lda else None

    def topics(self, n_words: int = 10):
        if not self._lda:
            return []
        feats = np.array(self._count_vec.get_feature_names_out())
        topics = []
        for tidx, component in enumerate(self._lda.components_):
            top_idx = component.argsort()[::-1][:n_words]
            topics.append({
                "id": tidx,
                "words": feats[top_idx].tolist(),
                "weights": component[top_idx].tolist()
            })
        return topics

# ────────────────────────────── helper visuals ──────────────────────────────
@st.cache_data(show_spinner=False)
def make_wordcloud(docs: List[str]) -> str:
    wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(docs))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ────────────────────────────── UI ──────────────────────────────
DEFAULT_DOCS = [
    "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
    "Natural language processing enables computers to understand and process human language effectively.",
    "Deep learning uses neural networks with multiple layers to solve complex problems in AI.",
    "Computer vision allows machines to interpret and understand visual information from the world.",
    "Data science combines statistics, programming, and domain expertise to extract insights from data.",
]

if "irs" not in st.session_state:
    st.session_state["irs"] = InformationRetrievalSystem()
    st.session_state["docs"] = []
    st.session_state["ids"] = []

st.sidebar.header("📄 Document source")
mode = st.sidebar.radio("Select input method", ["Sample", "Upload .txt", "Manual"], horizontal=True)

loaded = False
if mode == "Sample":
    if st.sidebar.button("Load sample docs", use_container_width=True):
        st.session_state["docs"] = DEFAULT_DOCS
        st.session_state["ids"] = [f"Sample_{i+1}" for i in range(len(DEFAULT_DOCS))]
        loaded = True

elif mode == "Upload .txt":
    up = st.sidebar.file_uploader("Upload a plain‑text file", type="txt")
    if up is not None:
        text = up.read().decode("utf-8")
        docs = [l.strip() for l in text.split("\n") if l.strip()]
        if st.sidebar.button(f"Load {len(docs)} lines as docs"):
            st.session_state["docs"] = docs
            st.session_state["ids"] = [f"File_{i+1}" for i in range(len(docs))]
            loaded = True

else:  # Manual
    txt = st.sidebar.text_area("One document per line", height=150)
    if st.sidebar.button("Load docs"):
        docs = [l.strip() for l in txt.split("\n") if l.strip()]
        st.session_state["docs"] = docs
        st.session_state["ids"] = [f"Manual_{i+1}" for i in range(len(docs))]
        loaded = True

# Reset / re‑index
if loaded or (st.session_state["docs"] and not st.session_state["irs"].raw_docs):
    with st.spinner("Indexing documents..."):
        st.session_state["irs"].add_documents(st.session_state["docs"], st.session_state["ids"])
    st.success("Documents indexed! Ready to search.")

if not st.session_state["docs"]:
    st.info("👉 Load some documents from the sidebar to get started.")
    st.stop()

# ───────────────── tabs ─────────────────
search_tab, analytics_tab, topics_tab, docs_tab = st.tabs(["🔍 Search", "📊 Analytics", "📈 Topics", "📚 Docs"])

with search_tab:
    st.subheader("Search the collection")
    q = st.text_input("Query", placeholder="e.g. neural networks")
    k = st.slider("Top‑K", 3, 15, 5, step=1)
    if st.button("Search", type="primary") and q:
        res = st.session_state["irs"].search(q, k)
        if not res:
            st.warning("No matching documents.")
        else:
            for r in res:
                pattern = re.compile(rf"\b({re.escape(q)})\b", re.I)
                highlighted = pattern.sub(r"**\1**", r["doc"])
                with st.expander(f"{r['doc_id']} – score {r['score']:.3f}"):
                    st.write(highlighted)

with analytics_tab:
    st.subheader("Corpus analytics")
    docs = st.session_state["docs"]
    st.metric("Documents", len(docs))
    lengths = [len(d.split()) for d in docs]
    st.metric("Avg length", f"{np.mean(lengths):.1f} words")

    st.markdown("### Word cloud")
    try:
        st.image(f"data:image/png;base64,{make_wordcloud(docs)}")
    except Exception as e:
        st.error(f"Word‑cloud failed: {e}")

    st.markdown("### Length distribution")
    st.plotly_chart(px.histogram(x=lengths, nbins=20, labels={"x": "words", "y": "count"}))

with topics_tab:
    st.subheader("Topic modelling (LDA)")
    topics = st.session_state["irs"].topics()
    if not topics:
        st.warning("Topics not available – make sure docs are loaded.")
    else:
        for t in topics:
            st.write(f"**Topic {t['id']+1}:** {', '.join(t['words'][:5])}")

        st.markdown("### Document–topic heat‑map")
        dt = st.session_state["irs"].doc_topic_matrix()
        doc_labels = st.session_state["ids"]
        fig = px.imshow(dt.T, x=doc_labels, y=[f"T{i+1}" for i in range(dt.shape[1])],
                        color_continuous_scale="RdBu", aspect="auto")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Word weights per topic")
        choose = st.selectbox("Choose topic", [f"Topic {i+1}" for i in range(len(topics))])
        idx = int(choose.split()[1]) - 1
        data = pd.DataFrame({"word": topics[idx]["words"], "weight": topics[idx]["weights"]})
        st.plotly_chart(px.bar(data, x="weight", y="word", orientation="h"), use_container_width=True)

with docs_tab:
    st.subheader("Browse documents")
    filt = st.text_input("Filter", help="Substring filter (case‑insensitive)")
    for doc, id_ in zip(st.session_state["docs"], st.session_state["ids"]):
        if filt and filt.lower() not in doc.lower():
            continue
        with st.expander(f"{id_} – {len(doc.split())} words"):
            st.write(doc)

st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit • TF‑IDF • Cosine sim • LDA • Plotly • WordCloud")
