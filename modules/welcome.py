import streamlit as st


def run():
    st.title("🤖 Welcome to AI Visual Lab")
    
    st.markdown("""
    ## Interactive AI Learning Platform
    
    Welcome to the AI Visual Lab! This interactive platform helps you understand
    artificial intelligence concepts through visualizations and hands-on examples.
    
    ### What You'll Learn:
    - **Data Basics**: Understanding data structures and preprocessing
    - **Regression**: Linear and polynomial regression models
    - **Classification**: Binary and multi-class classification
    - **Neural Networks**: Deep learning fundamentals
    - **CNN Visuals**: Convolutional Neural Networks visualization
    - **GenAI Tokenization**: How language models process text
    - **GenAI Attention**: Attention mechanisms in transformers
    - **GenAI Generation**: Text generation techniques
    - **Capstone**: Apply your knowledge to real-world projects
    
    ### Getting Started
    Select a module from the sidebar to begin your AI journey!
    """)
    
    st.info("💡 Tip: Each module includes interactive visualizations and explanations.")
