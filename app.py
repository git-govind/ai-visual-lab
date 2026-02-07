import streamlit as st
from modules import (
    welcome, intro, data_basics, regression, classification,
    neural_networks, cnn_visuals,
    genai_tokenization, genai_attention, genai_generation,
    capstone
)

st.set_page_config(
    page_title="AI Visual Canvas",
    layout="wide"
)

if "completed" not in st.session_state:
    st.session_state.completed = set()

st.sidebar.title("🧠 AI Visual Canvas")
st.sidebar.caption("Learn AI by Seeing It Think")

progress = len(st.session_state.completed) / 10
st.sidebar.progress(progress)

page = st.sidebar.radio(
    "Learning Path",
    [
        "Welcome",
        "Intro to AI",
        "Data Basics",
        "Regression",
        "Classification",
        "Neural Networks",
        "CNN Visuals",
        "GenAI: Tokenization",
        "GenAI: Attention",
        "GenAI: Generation",
        "Capstone"
    ]
)

pages = {
    "Welcome": welcome,
    "Intro to AI": intro,
    "Data Basics": data_basics,
    "Regression": regression,
    "Classification": classification,
    "Neural Networks": neural_networks,
    "CNN Visuals": cnn_visuals,
    "GenAI: Tokenization": genai_tokenization,
    "GenAI: Attention": genai_attention,
    "GenAI: Generation": genai_generation,
    "Capstone": capstone
}

pages[page].run()

st.markdown("---")
st.caption("© 2026 AI Visual Canvas | Built by Govind Tiwari")
