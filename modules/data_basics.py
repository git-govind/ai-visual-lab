import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def run():
    st.title("📊 Data Basics")
    
    st.markdown("""
    ## Understanding Data
    
    Data is the foundation of all AI models. Let's explore basic data concepts.
    """)
    
    # Generate sample data
    st.subheader("Sample Dataset")
    np.random.seed(42)
    data = pd.DataFrame({
        'feature_1': np.random.randn(100),
        'feature_2': np.random.randn(100),
        'feature_3': np.random.randint(0, 10, 100),
        'target': np.random.choice(['A', 'B', 'C'], 100)
    })
    
    st.dataframe(data.head(10))
    
    # Statistical summary
    st.subheader("Statistical Summary")
    st.write(data.describe())
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribution")
        fig, ax = plt.subplots()
        ax.hist(data['feature_1'], bins=20, edgecolor='black')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Correlation")
        fig, ax = plt.subplots()
        sns.heatmap(data[['feature_1', 'feature_2', 'feature_3']].corr(), 
                    annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    
    st.markdown("---")
    st.info("💡 Understanding your data is crucial before building any AI model!")
