import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter
import pandas as pd


def run():
    st.title("🔤 GenAI: Tokenization")
    
    st.markdown("""
    ## How Language Models Process Text
    
    Before understanding words, AI must break text into tokens. Let's explore how!
    """)
    
    # Text input
    st.subheader("✏️ Enter Your Text")
    
    default_text = """The quick brown fox jumps over the lazy dog. 
Artificial intelligence is transforming technology!
Tokenization is fundamental to NLP. 🚀"""
    
    text = st.text_area(
        "Text to tokenize:",
        default_text,
        height=120
    )
    
    # Tokenization methods
    st.subheader("🔧 Tokenization Methods")
    
    tokenization_method = st.selectbox(
        "Choose tokenization method:",
        ["Word-based", "Character-based", "Subword (BPE-like)", "Whitespace", "Sentence"]
    )
    
    # Perform tokenization based on method
    if tokenization_method == "Word-based":
        # Remove punctuation and split
        tokens = re.findall(r'\b\w+\b', text.lower())
        description = "Splits text into words, removing punctuation. Simple but large vocabulary."
        
    elif tokenization_method == "Character-based":
        # Split into characters
        tokens = list(text)
        description = "Splits text into individual characters. Small vocabulary but long sequences."
        
    elif tokenization_method == "Subword (BPE-like)":
        # Simulate BPE-like tokenization
        words = text.lower().split()
        tokens = []
        for word in words:
            if len(word) > 5:
                # Split long words into subwords
                mid = len(word) // 2
                tokens.extend([word[:mid] + "##", word[mid:]])
            else:
                tokens.append(word)
        description = "Splits into subwords. Balance between word and character level. Used by GPT, BERT."
        
    elif tokenization_method == "Whitespace":
        # Simple whitespace split
        tokens = text.split()
        description = "Splits on whitespace only. Keeps punctuation attached to words."
        
    else:  # Sentence
        # Split into sentences
        tokens = re.split(r'[.!?]+', text)
        tokens = [t.strip() for t in tokens if t.strip()]
        description = "Splits text into sentences. Useful for document-level processing."
    
    # Display results
    st.info(description)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📊 Token Count", len(tokens))
    with col2:
        vocab_size = len(set(tokens))
        st.metric("📚 Unique Tokens", vocab_size)
    with col3:
        avg_length = np.mean([len(t) for t in tokens])
        st.metric("📏 Avg Token Length", f"{avg_length:.1f}")
    
    # Visualize tokens
    st.markdown("---")
    st.subheader("🎨 Token Visualization")
    
    # Color-code tokens
    colors = plt.cm.Set3(np.linspace(0, 1, min(len(set(tokens)), 12)))
    token_to_color = {token: colors[i % len(colors)] for i, token in enumerate(set(tokens))}
    
    # Create HTML with colored tokens
    html = '<div style="line-height: 2.5; font-size: 16px;">'
    for token in tokens[:100]:  # Limit display
        color = token_to_color.get(token, colors[0])
        rgb = ','.join([str(int(c*255)) for c in color[:3]])
        html += f'<span style="background-color: rgba({rgb}, 0.6); padding: 3px 6px; margin: 2px; border-radius: 4px; border: 1px solid #333;">{token}</span> '
    
    if len(tokens) > 100:
        html += f'<span style="color: #666;">... and {len(tokens)-100} more tokens</span>'
    html += '</div>'
    
    st.markdown(html, unsafe_allow_html=True)
    
    # Token frequency analysis
    st.markdown("---")
    st.subheader("📈 Token Frequency Analysis")
    
    token_counts = Counter(tokens)
    most_common = token_counts.most_common(15)
    
    if most_common:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Plot frequency distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            
            tokens_list, counts_list = zip(*most_common)
            bars = ax.barh(range(len(tokens_list)), counts_list, color='steelblue', edgecolor='black')
            
            # Add value labels
            for i, (bar, count) in enumerate(zip(bars, counts_list)):
                ax.text(count + 0.1, i, str(count), va='center', fontweight='bold')
            
            ax.set_yticks(range(len(tokens_list)))
            ax.set_yticklabels(tokens_list)
            ax.set_xlabel('Frequency', fontsize=11, fontweight='bold')
            ax.set_title('Top 15 Most Frequent Tokens', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            st.pyplot(fig)
        
        with col2:
            st.write("**Frequency Table:**")
            freq_df = pd.DataFrame({
                "Token": [t for t, _ in most_common],
                "Count": [c for _, c in most_common]
            })
            st.dataframe(freq_df, use_container_width=True)
            
            # Stats
            total_tokens = len(tokens)
            unique_tokens = len(set(tokens))
            st.metric("Vocabulary Diversity", f"{unique_tokens/total_tokens:.2%}")
    
    # Tokenization comparison
    st.markdown("---")
    st.subheader("⚔️ Tokenization Method Comparison")
    
    # Apply all methods to same text
    sample_text = "Tokenization is preprocessing text for AI models!"
    
    methods_results = {
        "Word": re.findall(r'\b\w+\b', sample_text.lower()),
        "Character": list(sample_text),
        "Subword": re.findall(r'\w+|[^\w\s]', sample_text),
        "Whitespace": sample_text.split()
    }
    
    comparison_data = {
        "Method": list(methods_results.keys()),
        "Tokens": [len(tokens) for tokens in methods_results.values()],
        "Unique": [len(set(tokens)) for tokens in methods_results.values()],
        "Example": [str(tokens[:5]) for tokens in methods_results.values()]
    }
    
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
    
    # Why tokenization matters
    st.markdown("---")
    st.subheader("⚡ Why Tokenization Matters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **💰 Cost**
        
        APIs charge per token:
        - GPT-4: ~$0.03/1K tokens
        - Token count affects billing
        - Efficient tokenization saves money
        
        Example:
        - "AI" = 1 token
        - "Artificial Intelligence" = 2 tokens
        """)
    
    with col2:
        st.markdown("""
        **⏱️ Speed**
        
        Processing time scales with tokens:
        - More tokens = slower
        - Token limit: 4K, 8K, 32K, 128K
        - Exceeding limit = truncation/error
        
        Example:
        - 1000 tokens ≈ 750 words
        - 4K tokens ≈ 3000 words
        """)
    
    with col3:
        st.markdown("""
        **🧠 Context**
        
        Models have token limits:
        - GPT-3.5: 4K tokens
        - GPT-4: 8K-128K tokens
        - Claude: 100K-200K tokens
        
        Context includes:
        - Input prompt
        - Output response
        - System instructions
        """)
    
    st.success("""
    **🎯 Key Takeaways:**
    - Tokenization converts text to numbers that models can process
    - Different methods have different trade-offs
    - Subword tokenization balances vocabulary size and flexibility
    - Token count affects cost, speed, and context limits
    - Modern LLMs use sophisticated subword algorithms (BPE, WordPiece)
    """)
    
    if st.button("✅ Mark Module Complete"):
        if 'completed' not in st.session_state:
            st.session_state.completed = set()
        st.session_state.completed.add("genai_tokenization")
        st.balloons()
        st.success("Module completed! Continue to GenAI: Attention ➡️")
