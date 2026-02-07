import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns


def run():
    st.title("🎯 Classification")
    
    st.markdown("""
    ## Predicting Categories
    
    Classification models predict discrete class labels for input data.
    """)
    
    # Interactive parameters
    st.sidebar.subheader("Classification Parameters")
    n_samples = st.sidebar.slider("Number of Samples", 100, 500, 200)
    n_features = st.sidebar.slider("Number of Features", 2, 10, 2)
    model_type = st.sidebar.selectbox("Model Type", ["Logistic Regression", "Decision Tree"])
    
    # Generate data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    if model_type == "Logistic Regression":
        model = LogisticRegression()
    else:
        model = DecisionTreeClassifier()
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Distribution")
        if n_features >= 2:
            fig, ax = plt.subplots()
            scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.6)
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            plt.colorbar(scatter, ax=ax)
            st.pyplot(fig)
    
    with col2:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    st.metric("Accuracy", f"{accuracy:.2%}")
    
    st.markdown("---")
    st.info("💡 Try different models and parameters to see how they affect performance!")
