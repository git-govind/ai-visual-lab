import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import time


def run():
    st.title("📚 Introduction to AI")
    
    # Animated intro
    st.markdown("""
    ## What is Artificial Intelligence?
    
    Artificial Intelligence (AI) enables machines to **learn**, **reason**, and **adapt**
    like humans. It's transforming every industry from healthcare to entertainment!
    """)
    
    # Interactive AI evolution timeline
    st.subheader("🕰️ Evolution of AI")
    
    timeline_year = st.slider("Explore AI History", 1950, 2026, 2020, step=10)
    
    ai_milestones = {
        1950: "Alan Turing proposes the Turing Test",
        1960: "First neural networks & ELIZA chatbot",
        1970: "Expert systems emerge",
        1980: "Backpropagation algorithm discovered",
        1990: "Machine learning boom begins",
        2000: "Big data & computational power surge",
        2010: "Deep learning revolution (ImageNet)",
        2020: "Transformers & GPT models dominate",
        2026: "Multi-modal AI & AGI research"
    }
    
    closest_year = min(ai_milestones.keys(), key=lambda x: abs(x - timeline_year))
    st.info(f"**{closest_year}**: {ai_milestones[closest_year]}")
    
    st.markdown("---")
    
    # Interactive AI categories
    st.subheader("🎯 AI Categories")
    
    tab1, tab2, tab3 = st.tabs(["🤖 Machine Learning", "🧠 Deep Learning", "🔮 Types of Learning"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📊 Supervised Learning
            Learn from **labeled data**
            
            **Examples:**
            - 📧 Email spam detection
            - 🏠 House price prediction
            - 🖼️ Image classification
            - 🗣️ Speech recognition
            """)
            
            # Simple supervised learning visualization
            fig, ax = plt.subplots(figsize=(6, 4))
            np.random.seed(42)
            X_class1 = np.random.randn(30, 2) + [2, 2]
            X_class2 = np.random.randn(30, 2) + [-2, -2]
            
            ax.scatter(X_class1[:, 0], X_class1[:, 1], c='blue', label='Class A', s=100, alpha=0.6, edgecolors='black')
            ax.scatter(X_class2[:, 0], X_class2[:, 1], c='red', label='Class B', s=100, alpha=0.6, edgecolors='black')
            
            # Decision boundary
            x_line = np.linspace(-5, 5, 100)
            ax.plot(x_line, x_line, 'g--', linewidth=2, label='Decision Boundary')
            
            ax.set_xlabel('Feature 1', fontsize=10)
            ax.set_ylabel('Feature 2', fontsize=10)
            ax.set_title('Supervised Classification', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
        
        with col2:
            st.markdown("""
            ### 🔍 Unsupervised Learning
            Discover **patterns** in unlabeled data
            
            **Examples:**
            - 👥 Customer segmentation
            - 🧬 Gene clustering
            - 🎵 Music recommendation
            - 🌐 Anomaly detection
            """)
            
            # Clustering visualization
            fig, ax = plt.subplots(figsize=(6, 4))
            np.random.seed(42)
            
            centers = [[2, 2], [-2, -2], [2, -2]]
            colors = ['red', 'blue', 'green']
            
            for i, (center, color) in enumerate(zip(centers, colors)):
                cluster = np.random.randn(25, 2) * 0.6 + center
                ax.scatter(cluster[:, 0], cluster[:, 1], c=color, 
                          label=f'Cluster {i+1}', s=100, alpha=0.6, edgecolors='black')
                ax.scatter(*center, c=color, s=300, marker='*', 
                          edgecolors='black', linewidth=2)
            
            ax.set_xlabel('Feature 1', fontsize=10)
            ax.set_ylabel('Feature 2', fontsize=10)
            ax.set_title('Unsupervised Clustering', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
        
        st.markdown("""
        ### 🎮 Reinforcement Learning
        Learn through **trial and error** with rewards
        
        **Examples:**
        - 🎯 Game playing (Chess, Go, Dota)
        - 🚗 Autonomous driving
        - 🤖 Robotics control
        - 💰 Trading strategies
        """)
    
    with tab2:
        st.markdown("""
        ### 🧠 Deep Learning
        
        Deep learning uses **neural networks** with multiple layers to learn hierarchical representations.
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            #### 🖼️ Computer Vision
            **CNN** (Convolutional Neural Networks)
            
            Applications:
            - Face recognition
            - Object detection
            - Medical imaging
            - Self-driving cars
            """)
        
        with col2:
            st.markdown("""
            #### 🗣️ Natural Language
            **Transformers** & **RNNs**
            
            Applications:
            - ChatGPT
            - Translation
            - Sentiment analysis
            - Text generation
            """)
        
        with col3:
            st.markdown("""
            #### 🎨 Generative AI
            **GANs** & **Diffusion Models**
            
            Applications:
            - Image generation
            - Video synthesis
            - Music creation
            - 3D modeling
            """)
        
        # Neural network depth visualization
        st.subheader("Why 'Deep' Learning?")
        
        depth = st.slider("Network Depth (Number of Layers)", 1, 10, 5)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        layer_width = 10 / (depth + 1)
        
        for i in range(depth + 2):
            x = i * layer_width
            neurons = 5 if i in [0, depth + 1] else 8
            y_positions = np.linspace(2, 8, neurons)
            
            # Draw neurons
            for y in y_positions:
                color = 'lightgreen' if i == 0 else 'lightcoral' if i == depth + 1 else 'lightblue'
                circle = plt.Circle((x, y), 0.3, color=color, ec='black', linewidth=2)
                ax.add_patch(circle)
            
            # Draw connections
            if i < depth + 1:
                next_y_positions = np.linspace(2, 8, 5 if i == depth else 8)
                for y1 in y_positions:
                    for y2 in next_y_positions:
                        ax.plot([x + 0.3, x + layer_width - 0.3], [y1, y2], 
                               'gray', alpha=0.2, linewidth=0.5)
            
            # Labels
            if i == 0:
                ax.text(x, 1, 'Input', ha='center', fontweight='bold', fontsize=11)
            elif i == depth + 1:
                ax.text(x, 1, 'Output', ha='center', fontweight='bold', fontsize=11)
            else:
                ax.text(x, 1, f'Layer {i}', ha='center', fontweight='bold', fontsize=10)
        
        ax.set_xlim(-1, 11)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title(f'{depth}-Layer Deep Neural Network', fontsize=14, fontweight='bold', pad=20)
        
        st.pyplot(fig)
        
        st.info(f"💡 **Depth = {depth} layers**: Each layer learns more abstract features!")
    
    with tab3:
        st.markdown("### 🔄 Learning Paradigms Comparison")
        
        comparison_data = {
            'Paradigm': ['Supervised', 'Unsupervised', 'Reinforcement'],
            'Data Type': ['Labeled', 'Unlabeled', 'Sequential'],
            'Goal': ['Predict', 'Discover', 'Optimize'],
            'Feedback': ['Correct answers', 'None', 'Rewards/penalties'],
            'Example': ['Classification', 'Clustering', 'Game playing']
        }
        
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True)
        
        # Interactive example selector
        st.subheader("🎪 Try It Yourself!")
        
        learning_type = st.selectbox(
            "Choose a learning type to visualize:",
            ["Supervised: Classification", "Unsupervised: Clustering", "Reinforcement: Grid World"]
        )
        
        if learning_type == "Supervised: Classification":
            noise = st.slider("Data noise level", 0.1, 2.0, 0.5)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            np.random.seed(42)
            
            n_points = 100
            X1 = np.random.randn(n_points, 2) * noise + [2, 2]
            X2 = np.random.randn(n_points, 2) * noise + [-2, -2]
            
            ax.scatter(X1[:, 0], X1[:, 1], c='blue', label='Class 0', alpha=0.6, s=50)
            ax.scatter(X2[:, 0], X2[:, 1], c='red', label='Class 1', alpha=0.6, s=50)
            
            ax.set_title('Binary Classification Problem', fontweight='bold', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
        elif learning_type == "Unsupervised: Clustering":
            n_clusters = st.slider("Number of clusters", 2, 6, 3)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            np.random.seed(42)
            
            colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))
            
            for i in range(n_clusters):
                angle = 2 * np.pi * i / n_clusters
                center = [3 * np.cos(angle), 3 * np.sin(angle)]
                cluster = np.random.randn(30, 2) * 0.5 + center
                ax.scatter(cluster[:, 0], cluster[:, 1], c=[colors[i]], 
                          label=f'Cluster {i+1}', alpha=0.6, s=50)
            
            ax.set_title('Data Clustering', fontweight='bold', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            
            st.pyplot(fig)
            
        else:  # Reinforcement Learning
            st.write("**Grid World Environment**: Agent learns to reach the goal ⭐")
            
            grid_size = 5
            fig, ax = plt.subplots(figsize=(6, 6))
            
            # Draw grid
            for i in range(grid_size + 1):
                ax.plot([0, grid_size], [i, i], 'k-', linewidth=1)
                ax.plot([i, i], [0, grid_size], 'k-', linewidth=1)
            
            # Agent, goal, and obstacles
            agent_pos = (0, 0)
            goal_pos = (4, 4)
            obstacles = [(1, 1), (2, 2), (3, 1)]
            
            # Draw elements
            ax.text(agent_pos[0] + 0.5, agent_pos[1] + 0.5, '🤖', 
                   ha='center', va='center', fontsize=30)
            ax.text(goal_pos[0] + 0.5, goal_pos[1] + 0.5, '⭐', 
                   ha='center', va='center', fontsize=30)
            
            for obs in obstacles:
                rect = plt.Rectangle(obs, 1, 1, facecolor='gray', alpha=0.5)
                ax.add_patch(rect)
                ax.text(obs[0] + 0.5, obs[1] + 0.5, '🚫', 
                       ha='center', va='center', fontsize=20)
            
            ax.set_xlim(0, grid_size)
            ax.set_ylim(0, grid_size)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title('Reinforcement Learning Environment', fontweight='bold', fontsize=12)
            
            st.pyplot(fig)
            
            st.write("- **Green reward**: +10 for reaching goal")
            st.write("- **Red penalty**: -5 for hitting obstacles")
            st.write("- **Small penalty**: -1 for each step (encourages efficiency)")
    
    st.markdown("---")
    
    # AI Pipeline with detailed animation
    st.subheader("🔄 The AI Development Pipeline")
    
    pipeline_step = st.select_slider(
        "Explore each step:",
        options=["Data Collection", "Preprocessing", "Model Training", "Evaluation", "Deployment"]
    )
    
    step_details = {
        "Data Collection": {
            "icon": "📊",
            "desc": "Gather relevant data from various sources",
            "tasks": ["Identify data sources", "Extract data", "Store data", "Handle missing values"],
            "tools": ["Web scraping", "APIs", "Databases", "Sensors"]
        },
        "Preprocessing": {
            "icon": "🧹",
            "desc": "Clean and prepare data for training",
            "tasks": ["Clean data", "Handle outliers", "Feature engineering", "Normalization"],
            "tools": ["Pandas", "NumPy", "Scikit-learn", "Feature scaling"]
        },
        "Model Training": {
            "icon": "🎓",
            "desc": "Train the AI model on prepared data",
            "tasks": ["Select algorithm", "Train model", "Tune hyperparameters", "Validate"],
            "tools": ["TensorFlow", "PyTorch", "Scikit-learn", "XGBoost"]
        },
        "Evaluation": {
            "icon": "📈",
            "desc": "Assess model performance and accuracy",
            "tasks": ["Test on new data", "Calculate metrics", "Compare models", "Error analysis"],
            "tools": ["Accuracy", "Precision/Recall", "F1-score", "Confusion matrix"]
        },
        "Deployment": {
            "icon": "🚀",
            "desc": "Deploy model to production environment",
            "tasks": ["Model optimization", "API creation", "Monitor performance", "Update model"],
            "tools": ["Docker", "Flask/FastAPI", "Cloud platforms", "MLOps"]
        }
    }
    
    step_info = step_details[pipeline_step]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"## {step_info['icon']}")
        st.markdown(f"### {pipeline_step}")
        st.write(step_info['desc'])
    
    with col2:
        st.markdown("**Key Tasks:**")
        for task in step_info['tasks']:
            st.write(f"✓ {task}")
        
        st.markdown("**Common Tools:**")
        tools_text = " • ".join(step_info['tools'])
        st.info(tools_text)
    
    # Visual pipeline flow
    fig, ax = plt.subplots(figsize=(14, 3))
    
    stages = list(step_details.keys())
    n_stages = len(stages)
    
    for i, stage in enumerate(stages):
        x = i * 2.5
        
        # Highlight current stage
        color = 'lightgreen' if stage == pipeline_step else 'lightblue'
        alpha = 1.0 if stage == pipeline_step else 0.5
        
        # Draw box
        rect = FancyBboxPatch((x, 0.5), 2, 1, boxstyle="round,pad=0.1", 
                              facecolor=color, edgecolor='black', 
                              linewidth=2, alpha=alpha)
        ax.add_patch(rect)
        
        # Add icon and text
        ax.text(x + 1, 1, step_details[stage]['icon'], 
               ha='center', va='center', fontsize=20)
        ax.text(x + 1, 0.3, stage.replace(' ', '\n'), 
               ha='center', va='top', fontsize=8, fontweight='bold')
        
        # Draw arrow
        if i < n_stages - 1:
            arrow = FancyArrowPatch((x + 2, 1), (x + 2.5, 1),
                                   arrowstyle='->', mutation_scale=20,
                                   linewidth=2, color='gray')
            ax.add_patch(arrow)
    
    ax.set_xlim(-0.5, n_stages * 2.5)
    ax.set_ylim(0, 2)
    ax.axis('off')
    
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Success metrics
    st.subheader("📊 Why AI Matters")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Healthcare", "↑ 40%", "Diagnosis accuracy", help="AI improves medical diagnosis")
    with col2:
        st.metric("Business", "↑ 60%", "Productivity gains", help="Automation benefits")
    with col3:
        st.metric("Research", "10x", "Faster discoveries", help="Accelerated scientific research")
    with col4:
        st.metric("Industries", "100%", "Will use AI by 2030", help="Universal adoption predicted")
    
    st.success("""
    🎯 **Ready to dive deeper?** Each module in this course provides hands-on, 
    interactive learning experiences that will transform you into an AI expert!
    """)
    
    # Mark as complete button
    if st.button("✅ Mark Module Complete"):
        if 'completed' not in st.session_state:
            st.session_state.completed = set()
        st.session_state.completed.add("intro")
        st.balloons()
        st.success("Module completed! Continue to Data Basics ➡️")
