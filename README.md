name: information-retrieval-system
description: >
  A Streamlit-based Information Retrieval demo app featuring TF-IDF search, cosine similarity,
  LDA topic modeling, and interactive analytics using NLTK, Plotly, and WordCloud.
  
version: 1.0.0
author: Tejas Soham
license: MIT
source: https://github.com/tzprograms/Information-Retrieval-System
keywords:
  - streamlit
  - information retrieval
  - tf-idf
  - lda
  - topic modeling
  - nlp
  - cosine similarity

dependencies:
  - python >=3.7
  - streamlit
  - scikit-learn
  - nltk
  - wordcloud
  - matplotlib
  - pandas
  - numpy
  - plotly

run:
  command: streamlit run app/test.py
  working_dir: ./

entrypoint: app/test.py

scripts:
  install:
    - pip install -r requirements.txt

data:
  - app/test.txt

include:
  - README.md
  - requirements.txt
  - app/**
  - tools/**
  - summarizer.py
  - agent.py


