import streamlit as st
from modules import (
    welcome,
    intro,
    data_basics,
    regression,
    classification,
    neural_networks,
    cnn_visuals,
    genai_tokenization,
    genai_attention,
    genai_generation,
    capstone
)

# Configure page
st.set_page_config(
    page_title="AI Visual Lab",
    page_icon="🤖",
    layout="wide"
)

# Sidebar navigation
st.sidebar.title("Navigation")
modules = {
    "Welcome": welcome,
    "Introduction": intro,
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

selected_module = st.sidebar.radio("Select Module", list(modules.keys()))

# Run selected module
if selected_module in modules:
    modules[selected_module].run()
