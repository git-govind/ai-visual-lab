import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


def run():
    st.title("📚 Introduction to AI")
    
    st.markdown("""
    ## What is Artificial Intelligence?
    
    Artificial Intelligence (AI) is the simulation of human intelligence in machines
    that are programmed to think and learn like humans.
    
    ### Key Concepts:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Machine Learning")
        st.write("""
        - Supervised Learning
        - Unsupervised Learning
        - Reinforcement Learning
        """)
    
    with col2:
        st.subheader("Deep Learning")
        st.write("""
        - Neural Networks
        - CNN (Computer Vision)
        - Transformers (NLP)
        """)
    
    st.markdown("---")
    st.subheader("AI Pipeline")
    
    # Simple visualization of AI pipeline
    fig, ax = plt.subplots(figsize=(10, 2))
    stages = ["Data\nCollection", "Preprocessing", "Model\nTraining", "Evaluation", "Deployment"]
    x_pos = np.arange(len(stages))
    
    ax.barh(x_pos, [1]*len(stages), color='skyblue')
    ax.set_yticks(x_pos)
    ax.set_yticklabels(stages)
    ax.set_xlim(0, 1)
    ax.axis('off')
    
    for i, stage in enumerate(stages):
        ax.text(0.5, i, stage, ha='center', va='center', fontsize=12, fontweight='bold')
    
    st.pyplot(fig)
