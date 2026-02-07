import streamlit as st

def run():
    st.title("🔤 GenAI Lab: Tokenization")

    text = st.text_input(
        "Enter text",
        "AI models do not understand words"
    )

    tokens = text.split()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.write("### Tokens")
        st.write(tokens)

    with col2:
        st.metric("Token Count", len(tokens))
        st.caption("LLMs operate on tokens, not words")

    st.info(
        "Why this matters: token count affects cost, speed, and context limits."
    )

    if st.button("✅ Mark Complete"):
        st.session_state.completed.add("genai_tokenization")
