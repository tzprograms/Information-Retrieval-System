import streamlit as st
from agent import build_agent
from summarizer import summarize_text

GOOGLE_API_KEY = "AIzaSyDWMR9N7wS-qrBwOtVOq0omuGNLu3fn-mo"
CSE_ID = "e2ad0f10232e74371"


st.title("🔍 GenAI Web Search Agent Summarizer")

query = st.text_input("Enter your search query:")

if st.button("Run Agent & Summarize"):
    agent = build_agent(GOOGLE_API_KEY, CSE_ID)
    with st.spinner("Using agents to retrieve results..."):
        result = agent.chat(query)
    with st.spinner("Summarizing..."):
        summary = summarize_text(result.text)
    st.subheader("🔎 Summary:")
    st.write(summary)
