import streamlit as st
import re


def run():
    st.title("🔤 GenAI: Tokenization")
    
    st.markdown("""
    ## How Language Models Process Text
    
    Tokenization is the first step in processing text for language models.
    """)
    
    # Interactive tokenization
    st.subheader("Try Tokenization")
    
    user_text = st.text_area("Enter text to tokenize:", 
                              "Hello, world! This is an example of tokenization.")
    
    tokenization_method = st.selectbox(
        "Tokenization Method",
        ["Word-based", "Character-based", "Subword (simplified)"]
    )
    
    if tokenization_method == "Word-based":
        tokens = user_text.split()
    elif tokenization_method == "Character-based":
        tokens = list(user_text)
    else:  # Subword
        # Simplified subword tokenization
        tokens = re.findall(r'\w+|[^\w\s]', user_text)
        tokens = [t for word in tokens for t in (word if len(word) <= 4 else [word[:len(word)//2], word[len(word)//2:]])]
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Number of Tokens", len(tokens))
        st.metric("Number of Characters", len(user_text))
    
    with col2:
        st.metric("Compression Ratio", f"{len(user_text) / max(len(tokens), 1):.2f}")
    
    st.subheader("Tokens")
    st.write(tokens)
    
    # Visualize tokens
    st.subheader("Token Visualization")
    colors = ['#FFB6C1', '#87CEEB', '#98FB98', '#DDA0DD', '#F0E68C', '#FFA07A']
    html = ""
    for i, token in enumerate(tokens):
        color = colors[i % len(colors)]
        html += f'<span style="background-color: {color}; padding: 2px 5px; margin: 2px; border-radius: 3px; display: inline-block;">{token}</span>'
    
    st.markdown(html, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("Why Tokenization Matters")
    st.write("""
    - **Efficiency**: Reduces vocabulary size
    - **Flexibility**: Handles unknown words
    - **Balance**: Trade-off between granularity and vocabulary size
    """)
    
    st.info("💡 Modern LLMs like GPT use Byte-Pair Encoding (BPE) for tokenization!")
