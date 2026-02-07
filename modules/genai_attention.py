import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def run():
    st.title("👁️ GenAI: Attention Mechanism")
    
    st.markdown("""
    ## Understanding Attention in Transformers
    
    The attention mechanism allows models to focus on relevant parts of the input.
    """)
    
    # Interactive attention visualization
    st.subheader("Self-Attention Visualization")
    
    sentence = st.text_input("Enter a sentence:", "The cat sat on the mat")
    tokens = sentence.split()
    
    if len(tokens) > 1:
        # Create attention matrix (simplified)
        n = len(tokens)
        attention_matrix = np.random.rand(n, n)
        
        # Apply softmax to each row
        attention_matrix = np.exp(attention_matrix) / np.exp(attention_matrix).sum(axis=1, keepdims=True)
        
        # Visualize attention
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(attention_matrix, 
                    xticklabels=tokens,
                    yticklabels=tokens,
                    annot=True,
                    fmt='.2f',
                    cmap='YlOrRd',
                    ax=ax,
                    cbar_kws={'label': 'Attention Weight'})
        ax.set_xlabel('Key (From)')
        ax.set_ylabel('Query (To)')
        ax.set_title('Self-Attention Matrix')
        
        st.pyplot(fig)
        
        # Explain attention
        st.markdown("---")
        st.subheader("How Attention Works")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Query (Q)**")
            st.write("What am I looking for?")
        
        with col2:
            st.write("**Key (K)**")
            st.write("What do I contain?")
        
        with col3:
            st.write("**Value (V)**")
            st.write("What do I actually represent?")
        
        st.markdown("""
        ### Attention Formula:
        """)
        st.latex(r'\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V')
        
        st.markdown("---")
        st.subheader("Multi-Head Attention")
        st.write("""
        - Multiple attention mechanisms run in parallel
        - Each "head" learns different relationships
        - Outputs are concatenated and linearly transformed
        """)
        
        # Visualize multi-head
        num_heads = st.slider("Number of Attention Heads", 1, 8, 4)
        
        fig, axes = plt.subplots(1, min(num_heads, 4), figsize=(15, 4))
        if num_heads == 1:
            axes = [axes]
        
        for i, ax in enumerate(axes[:num_heads]):
            head_attention = np.random.rand(n, n)
            head_attention = np.exp(head_attention) / np.exp(head_attention).sum(axis=1, keepdims=True)
            
            sns.heatmap(head_attention,
                       xticklabels=tokens,
                       yticklabels=tokens,
                       cmap='viridis',
                       ax=ax,
                       cbar=False)
            ax.set_title(f'Head {i+1}')
        
        st.pyplot(fig)
        
        st.info("💡 Attention allows models to capture long-range dependencies in sequences!")
    else:
        st.warning("Please enter a sentence with at least 2 words.")
