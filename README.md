# 🔍 Information Retrieval System — Streamlit App

This project is a full-featured **Information Retrieval (IR)** demo web app built using **Python** and **Streamlit**. It showcases classical NLP and IR techniques such as **TF–IDF vectorization**, **cosine similarity** search, and **LDA-based topic modeling**, all presented through an interactive interface with real-time visualizations.

---

## 🚀 Features

- ✅ **TF–IDF + Cosine Similarity** for ranking document relevance
- 🧠 **LDA (Latent Dirichlet Allocation)** for unsupervised topic modeling
- 📊 **Interactive visualizations** using Plotly and WordCloud
- 🗂️ Multiple document input modes: sample docs, uploaded `.txt` file, or manual entry
- ⚙️ Built-in preprocessing: stopword removal, stemming, case-folding
- 🧼 Robust handling of empty input, offline NLTK setup, and session-safe document management

---

## 📁 Project Structure

Information-Retrieval-System/
├── app/
│ ├── main.py # Core logic and Streamlit backend
│ ├── test.py # Streamlit UI entry point
│ ├── test.txt # Sample document lines
├── tools/
│ ├── bing_tool.py # (Optional) Search engine tool
│ ├── google_tool.py
│ ├── yahoo_tool.py
├── agent.py # (Optional) Custom agent logic
├── summarizer.py # (Optional) Summarization logic
├── README.md
├── requirements.txt


---

## 🛠️ Installation

### 1. Clone this repository

```bash
git clone https://github.com/tzprograms/Information-Retrieval-System.git
cd Information-Retrieval-System

Install
pip install streamlit scikit-learn nltk wordcloud plotly matplotlib pandas numpy
streamlit run app/test.py
