import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def run():
    st.title("🎓 Capstone Project")
    
    st.markdown("""
    ## Apply Your Knowledge
    
    Congratulations on completing the AI Visual Lab modules! Now it's time to apply
    what you've learned to a real-world project.
    """)
    
    st.markdown("---")
    
    st.subheader("Project Ideas")
    
    projects = {
        "Image Classifier": {
            "description": "Build a CNN to classify images into categories",
            "skills": ["Data preprocessing", "CNN architecture", "Model training"],
            "difficulty": "Intermediate"
        },
        "Text Generator": {
            "description": "Create a model that generates text in a specific style",
            "skills": ["Tokenization", "Transformers", "Text generation"],
            "difficulty": "Advanced"
        },
        "Sentiment Analyzer": {
            "description": "Classify text sentiment as positive, negative, or neutral",
            "skills": ["NLP", "Classification", "Feature extraction"],
            "difficulty": "Beginner"
        },
        "Recommendation System": {
            "description": "Build a system to recommend items based on user preferences",
            "skills": ["Collaborative filtering", "Matrix factorization", "Neural networks"],
            "difficulty": "Intermediate"
        },
        "Object Detection": {
            "description": "Detect and localize objects in images",
            "skills": ["CNN", "Bounding boxes", "Transfer learning"],
            "difficulty": "Advanced"
        }
    }
    
    selected_project = st.selectbox("Choose a project:", list(projects.keys()))
    
    if selected_project:
        project = projects[selected_project]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Description:**")
            st.write(project["description"])
            
        with col2:
            st.write("**Difficulty:**")
            st.write(project["difficulty"])
        
        st.write("**Skills Required:**")
        for skill in project["skills"]:
            st.write(f"- {skill}")
    
    st.markdown("---")
    
    st.subheader("Project Checklist")
    
    steps = [
        "Define the problem and objectives",
        "Collect and explore the data",
        "Preprocess and prepare the data",
        "Choose an appropriate model architecture",
        "Train and validate the model",
        "Evaluate model performance",
        "Fine-tune and optimize",
        "Deploy and monitor"
    ]
    
    for i, step in enumerate(steps, 1):
        st.checkbox(f"{i}. {step}")
    
    st.markdown("---")
    
    st.subheader("Resources")
    st.write("""
    - **Datasets**: Kaggle, UCI ML Repository, HuggingFace Datasets
    - **Documentation**: TensorFlow, PyTorch, Scikit-learn
    - **Communities**: Stack Overflow, GitHub, Reddit r/MachineLearning
    - **Courses**: Coursera, fast.ai, DeepLearning.AI
    """)
    
    st.success("🎉 Good luck with your project! Remember, the best way to learn is by doing!")
