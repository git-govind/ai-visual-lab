import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


def run():
    st.title("✨ GenAI: Text Generation")
    
    st.markdown("""
    ## How Language Models Generate Text
    
    Language models predict the next token based on previous context.
    """)
    
    # Interactive text generation simulation
    st.subheader("Generation Strategy")
    
    strategy = st.selectbox(
        "Sampling Strategy",
        ["Greedy", "Temperature Sampling", "Top-k Sampling", "Top-p (Nucleus) Sampling"]
    )
    
    # Simulate probability distribution
    vocab = ["the", "cat", "dog", "sat", "ran", "on", "quickly", "slowly", "mat", "park"]
    logits = np.random.randn(len(vocab))
    
    if strategy == "Greedy":
        probs = np.exp(logits) / np.sum(np.exp(logits))
        st.write("**Greedy Decoding**: Always pick the most probable token")
        
    elif strategy == "Temperature Sampling":
        temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
        probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
        st.write(f"**Temperature = {temperature}**")
        st.write("- Lower temperature → more deterministic (focused)")
        st.write("- Higher temperature → more random (creative)")
        
    elif strategy == "Top-k Sampling":
        k = st.slider("k", 1, len(vocab), 5)
        top_k_indices = np.argsort(logits)[-k:]
        probs = np.zeros(len(vocab))
        top_k_probs = np.exp(logits[top_k_indices]) / np.sum(np.exp(logits[top_k_indices]))
        probs[top_k_indices] = top_k_probs
        st.write(f"**Top-k = {k}**: Sample from top {k} most probable tokens")
        
    else:  # Top-p
        p = st.slider("p", 0.1, 1.0, 0.9, 0.05)
        sorted_indices = np.argsort(logits)[::-1]
        sorted_probs = np.exp(logits[sorted_indices]) / np.sum(np.exp(logits[sorted_indices]))
        cumsum = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cumsum, p) + 1
        probs = np.zeros(len(vocab))
        probs[sorted_indices[:cutoff]] = sorted_probs[:cutoff] / sorted_probs[:cutoff].sum()
        st.write(f"**Top-p = {p}**: Sample from smallest set with cumulative probability ≥ {p}")
    
    # Visualize probabilities
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(vocab, probs, color='steelblue', edgecolor='black')
    
    # Highlight selected token
    selected_idx = np.argmax(probs) if strategy == "Greedy" else np.random.choice(len(vocab), p=probs/probs.sum())
    bars[selected_idx].set_color('orange')
    
    ax.set_xlabel('Vocabulary')
    ax.set_ylabel('Probability')
    ax.set_title(f'Next Token Probability Distribution ({strategy})')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    st.success(f"**Selected Token**: {vocab[selected_idx]}")
    
    st.markdown("---")
    
    st.subheader("Generation Process")
    st.write("""
    1. **Input**: Prompt/context tokens
    2. **Encode**: Convert to embeddings
    3. **Process**: Pass through transformer layers
    4. **Predict**: Generate probability distribution for next token
    5. **Sample**: Select next token based on strategy
    6. **Repeat**: Add token to context and continue
    """)
    
    # Autoregressive generation visualization
    st.subheader("Autoregressive Generation")
    
    prompt = "The cat"
    generated = prompt.split()
    
    num_steps = st.slider("Generation Steps", 1, 10, 5)
    
    for i in range(num_steps):
        next_word = np.random.choice(vocab)
        generated.append(next_word)
    
    # Visualize step by step
    html = f'<div style="font-size: 18px; line-height: 2;">'
    html += f'<span style="background-color: #90EE90; padding: 5px; border-radius: 3px;">{prompt}</span> '
    
    for i, word in enumerate(generated[len(prompt.split()):]):
        color = f'hsl({i * 30 % 360}, 70%, 80%)'
        html += f'<span style="background-color: {color}; padding: 5px; margin: 2px; border-radius: 3px;">{word}</span> '
    
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)
    
    st.info("💡 Modern LLMs generate text one token at a time, conditioned on all previous tokens!")
