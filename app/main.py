import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64

# --- Page Configuration ---
st.set_page_config(
    page_title="Advanced Information Retrieval System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- NLTK Data Download ---
@st.cache_data
def download_nltk_data():
    """Downloads necessary NLTK data if not already present."""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)

download_nltk_data()

# --- Core IR System Class ---
class InformationRetrievalSystem:
    def __init__(self, max_features=1000, n_topics=5, ngram_range=(1, 2)):
        self.documents = []
        self.document_ids = []
        self.tfidf_vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        self.lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.tfidf_matrix = None
        self.doc_topics = None

    def preprocess_text(self, text):
        """Preprocesses text: lowercase, remove non-alphanumeric chars, tokenize, remove stopwords, and stem."""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)

    def add_documents(self, documents, doc_ids=None):
        """Adds and processes documents for the IR system."""
        self.documents = documents
        self.document_ids = doc_ids or [f"Doc_{i+1}" for i in range(len(documents))]
        
        preprocessed_docs = [self.preprocess_text(doc) for doc in self.documents]
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(preprocessed_docs)
        self.lda_model.fit(self.tfidf_matrix)
        self.doc_topics = self.lda_model.transform(self.tfidf_matrix)

    def search(self, query, top_k=5):
        """Searches for documents similar to the query using cosine similarity."""
        if self.tfidf_matrix is None:
            return []
        
        preprocessed_query = self.preprocess_text(query)
        query_vector = self.tfidf_vectorizer.transform([preprocessed_query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        return [
            {
                'document_id': self.document_ids[idx],
                'document': self.documents[idx],
                'similarity': similarities[idx],
                'rank': i + 1
            }
            for i, idx in enumerate(top_indices) if similarities[idx] > 0
        ]

    def get_top_words_per_topic(self, n_words=10):
        """Retrieves the top N words for each topic from the LDA model."""
        if not hasattr(self.lda_model, 'components_'):
            return []
            
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_words_indices = topic.argsort()[-n_words:][::-1]
            topics.append({
                'topic_id': topic_idx,
                'words': [feature_names[i] for i in top_words_indices],
                'weights': topic[top_words_indices]
            })
        return topics

# --- UI Helper Functions ---
def create_wordcloud(text_data):
    """Generates and returns a base64 encoded word cloud image."""
    combined_text = ' '.join(text_data)
    if not combined_text.strip():
        return None
        
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def highlight_text(text, query):
    """Highlights query terms in a given text, ignoring case."""
    highlighted_text = text
    for term in set(query.lower().split()):
        # Use a regex for case-insensitive whole-word matching
        pattern = re.compile(r'\b(' + re.escape(term) + r')\b', re.IGNORECASE)
        highlighted_text = pattern.sub(r'**\1**', highlighted_text)
    return highlighted_text

# --- Main Application ---
def main():
    st.title("🚀 Advanced Information Retrieval System")
    st.markdown("An interactive system for searching, analyzing, and exploring document collections.")

    # --- Sidebar for Configuration ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        doc_input_method = st.selectbox(
            "Select Document Source:",
            ["Sample Documents", "Upload Text File", "Custom Input"]
        )

        documents, doc_ids = load_documents(doc_input_method)

        if documents:
            st.success(f"{len(documents)} documents loaded successfully.")
            
            st.header("🔧 System Parameters")
            max_features = st.slider("Max TF-IDF Features:", 500, 2000, 1000, 100)
            n_topics = st.slider("Number of Topics (LDA):", 2, 10, 5, 1)
            
            if st.button("🔄 Re-process Documents", use_container_width=True):
                initialize_system(documents, doc_ids, max_features, n_topics)
        else:
            st.info("Please load documents to proceed.")

    # --- Main Content ---
    if 'ir_system' in st.session_state and st.session_state.ir_system.documents:
        display_main_interface()
    else:
        st.info("Load documents and click 'Re-process' in the sidebar to begin.")

def load_documents(method):
    """Loads documents based on the selected method in the sidebar."""
    if method == "Sample Documents":
        docs = [
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing helps computers understand human language.",
            "Deep learning uses neural networks for complex problem-solving.",
            "Computer vision allows machines to interpret visual information.",
            "Data science combines statistics and programming to extract insights."
        ]
        return docs, [f"Sample_{i+1}" for i in range(len(docs))]

    elif method == "Upload Text File":
        uploaded_file = st.sidebar.file_uploader("Upload a .txt file", type=['txt'])
        if uploaded_file:
            content = uploaded_file.read().decode('utf-8')
            docs = [line.strip() for line in content.split('\n') if line.strip()]
            return docs, [f"File_{i+1}" for i in range(len(docs))]
            
    elif method == "Custom Input":
        custom_docs = st.sidebar.text_area("Enter documents (one per line):", height=200)
        if custom_docs:
            docs = [line.strip() for line in custom_docs.split('\n') if line.strip()]
            return docs, [f"Custom_{i+1}" for i in range(len(docs))]
            
    return [], []

def initialize_system(documents, doc_ids, max_features, n_topics):
    """Initializes or re-initializes the IR system with given parameters."""
    with st.spinner("Processing documents... This may take a moment."):
        st.session_state.ir_system = InformationRetrievalSystem(
            max_features=max_features, n_topics=n_topics
        )
        st.session_state.ir_system.add_documents(documents, doc_ids)
    st.toast("✅ System processed successfully!", icon="🎉")


def display_main_interface():
    """Renders the main tabs for search, analytics, and document exploration."""
    tab_list = ["🔍 Search", "📊 Analytics", "📈 Topic Modeling", "📚 Document Explorer"]
    search_tab, analytics_tab, topics_tab, docs_tab = st.tabs(tab_list)

    with search_tab:
        render_search_tab()
    with analytics_tab:
        render_analytics_tab()
    with topics_tab:
        render_topic_modeling_tab()
    with docs_tab:
        render_document_explorer_tab()

def render_search_tab():
    """Renders the UI for the search functionality."""
    st.header("Find Relevant Documents")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input("Enter your search query:", placeholder="e.g., machine learning applications")
    with col2:
        top_k = st.selectbox("Results to show:", [3, 5, 10, 20], index=1)

    if query:
        results = st.session_state.ir_system.search(query, top_k)
        if results:
            st.subheader(f"Top {len(results)} Results for '{query}'")
            for res in results:
                with st.expander(f"**#{res['rank']} - {res['document_id']}** (Similarity: {res['similarity']:.3f})"):
                    st.markdown(highlight_text(res['document'], query))
        else:
            st.warning("No relevant documents found for your query. Try different keywords.")

def render_analytics_tab():
    """Renders the document analytics and visualizations."""
    st.header("Corpus Analytics")
    ir_system = st.session_state.ir_system
    
    # Key Metrics
    col1, col2, col3 = st.columns(3)
    doc_lengths = [len(doc.split()) for doc in ir_system.documents]
    col1.metric("Total Documents", len(ir_system.documents))
    col2.metric("Total Words", sum(doc_lengths))
    col3.metric("Avg. Doc Length", f"{np.mean(doc_lengths):.1f} words")
    
    # Visualizations
    st.subheader("Visual Insights")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Word Cloud**")
        wordcloud_img = create_wordcloud(ir_system.documents)
        if wordcloud_img:
            st.image(f"data:image/png;base64,{wordcloud_img}", use_column_width=True)
        else:
            st.info("Not enough text to generate a word cloud.")
            
    with col2:
        st.markdown("**Document Length Distribution**")
        fig = px.histogram(x=doc_lengths, nbins=20, title="Document Lengths")
        fig.update_layout(showlegend=False, yaxis_title="Frequency", xaxis_title="Word Count")
        st.plotly_chart(fig, use_container_width=True)

def render_topic_modeling_tab():
    """Renders the topic modeling results and visualizations."""
    st.header("Topic Modeling Insights (LDA)")
    ir_system = st.session_state.ir_system
    topics = ir_system.get_top_words_per_topic()
    
    if not topics:
        st.warning("Topic modeling data is not available.")
        return

    st.subheader("Discovered Topics")
    for topic in topics:
        st.info(f"**Topic {topic['topic_id'] + 1}:** {', '.join(topic['words'][:7])}")

    st.subheader("Interactive Topic Exploration")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_topic_idx = st.selectbox(
            "Select Topic to Analyze:", 
            options=range(len(topics)),
            format_func=lambda x: f"Topic {x + 1}"
        )
        selected_topic = topics[selected_topic_idx]
        
        st.markdown("**Top Words & Weights**")
        df = pd.DataFrame(selected_topic['words'], columns=['Word'])
        st.dataframe(df, use_container_width=True)

    with col2:
        fig = px.bar(
            x=selected_topic['weights'], 
            y=selected_topic['words'], 
            orientation='h', 
            title=f"Word Weights for Topic {selected_topic_idx + 1}"
        )
        fig.update_yaxes(categoryorder='total ascending')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Document-Topic Distribution")
    fig = px.imshow(
        ir_system.doc_topics.T,
        labels=dict(x="Document ID", y="Topic", color="Probability"),
        x=ir_system.document_ids,
        y=[f"Topic {i+1}" for i in range(len(topics))],
        aspect="auto"
    )
    st.plotly_chart(fig, use_container_width=True)


def render_document_explorer_tab():
    """Renders the UI for exploring and filtering all documents."""
    st.header("Explore the Document Collection")
    ir_system = st.session_state.ir_system
    
    filter_text = st.text_input("Filter documents by keyword:")
    
    filtered_docs = [
        (doc_id, doc) for doc_id, doc in zip(ir_system.document_ids, ir_system.documents)
        if not filter_text or filter_text.lower() in doc.lower()
    ]
    
    if filtered_docs:
        st.info(f"Showing {len(filtered_docs)} of {len(ir_system.documents)} documents.")
        for doc_id, doc in filtered_docs:
            with st.expander(f"**{doc_id}** - *{doc[:80]}...*"):
                st.write(doc)
                st.caption(f"Word count: {len(doc.split())}")
    else:
        st.warning("No documents match your filter criteria.")

if __name__ == "__main__":
    main()