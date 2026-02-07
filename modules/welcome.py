import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time


def run():
    st.title("🤖 Welcome to AI Visual Lab")
    
    # Animated welcome message
    welcome_text = "Your Interactive Journey into Artificial Intelligence"
    st.markdown(f"## {welcome_text}")
    
    # Progress indicator
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📚 Total Modules", "11", help="Complete learning modules")
    with col2:
        completed = len(st.session_state.get('completed', set()))
        st.metric("✅ Your Progress", f"{completed}/11", help="Modules completed")
    with col3:
        progress_pct = (completed / 11) * 100
        st.metric("🎯 Completion", f"{progress_pct:.0f}%", help="Overall progress")
    
    st.markdown("---")
    
    # Interactive learning path visualization
    st.subheader("🎓 Your Learning Path")
    
    modules = [
        ("📊 Data Basics", "Understand data structures"),
        ("📈 Regression", "Predict continuous values"),
        ("🎯 Classification", "Categorize data points"),
        ("🧠 Neural Networks", "Deep learning fundamentals"),
        ("🖼️ CNN Visuals", "Computer vision mastery"),
        ("🔤 Tokenization", "Text preprocessing"),
        ("👁️ Attention", "Transformer mechanisms"),
        ("✨ Generation", "Create with AI"),
        ("🎓 Capstone", "Build complete systems")
    ]
    
    # Create visual learning path
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create path points
    n_modules = len(modules)
    x = np.linspace(0, 10, n_modules)
    y = np.sin(x * 0.5) * 2 + 5
    
    # Draw path
    ax.plot(x, y, 'b--', linewidth=2, alpha=0.3, label='Learning Path')
    
    # Draw module points
    colors = ['green' if i < completed else 'lightgray' for i in range(n_modules)]
    ax.scatter(x, y, s=500, c=colors, edgecolors='black', linewidth=2, zorder=3)
    
    # Add labels
    for i, (name, _) in enumerate(modules):
        ax.text(x[i], y[i], str(i+1), ha='center', va='center', 
                fontweight='bold', fontsize=12, color='white', zorder=4)
    
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Your AI Learning Journey', fontsize=16, fontweight='bold', pad=20)
    
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Expandable module details
    st.subheader("📖 Course Modules")
    
    module_details = {
        "📊 Data Basics": {
            "desc": "Master data manipulation, visualization, and preprocessing techniques",
            "topics": ["Data structures", "Statistical analysis", "Data visualization", "Feature engineering"]
        },
        "📈 Regression": {
            "desc": "Learn to predict continuous values with linear and polynomial models",
            "topics": ["Linear regression", "Polynomial regression", "Model evaluation", "Feature scaling"]
        },
        "🎯 Classification": {
            "desc": "Predict categories using various classification algorithms",
            "topics": ["Logistic regression", "Decision trees", "Model metrics", "Confusion matrices"]
        },
        "🧠 Neural Networks": {
            "desc": "Understand deep learning architecture and activation functions",
            "topics": ["Network architecture", "Activation functions", "Backpropagation", "Loss functions"]
        },
        "🖼️ CNN Visuals": {
            "desc": "Explore convolutional neural networks for computer vision",
            "topics": ["Convolution operations", "Pooling layers", "Feature maps", "Image classification"]
        },
        "🔤 Tokenization": {
            "desc": "Learn how language models process and encode text",
            "topics": ["Text tokenization", "Subword encoding", "Vocabulary building", "Token embeddings"]
        },
        "👁️ Attention": {
            "desc": "Master the attention mechanism powering modern transformers",
            "topics": ["Self-attention", "Multi-head attention", "Query-Key-Value", "Attention weights"]
        },
        "✨ Generation": {
            "desc": "Generate text using various sampling strategies",
            "topics": ["Autoregressive generation", "Temperature sampling", "Top-k sampling", "Nucleus sampling"]
        },
        "🎓 Capstone": {
            "desc": "Build and deploy a complete AI system from scratch",
            "topics": ["End-to-end pipeline", "Model training", "Evaluation", "Deployment"]
        }
    }
    
    for emoji_name, details in module_details.items():
        with st.expander(f"{emoji_name}"):
            st.write(f"**{details['desc']}**")
            st.write("**Key Topics:**")
            for topic in details['topics']:
                st.write(f"  • {topic}")
    
    st.markdown("---")
    
    # Interactive tips
    st.subheader("💡 Learning Tips")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **🎯 Interactive Learning**
        - Adjust parameters in real-time
        - See immediate visual feedback
        - Experiment freely
        """)
        
        st.info("""
        **📊 Visualizations**
        - Every concept is visualized
        - Interactive charts and graphs
        - Step-by-step animations
        """)
    
    with col2:
        st.warning("""
        **🔬 Hands-On Practice**
        - Try different datasets
        - Modify algorithms
        - Compare results
        """)
        
        st.error("""
        **🏆 Challenge Yourself**
        - Complete all modules
        - Build the capstone project
        - Share your creations
        """)
    
    st.markdown("---")
    
    # Call to action
    st.subheader("🚀 Ready to Start?")
    st.write("Select **Intro to AI** from the sidebar to begin your journey!")
    
    # Fun fact
    facts = [
        "🧠 The human brain has ~86 billion neurons, while GPT-4 has ~1.7 trillion parameters!",
        "🤖 The first neural network was created in 1958 by Frank Rosenblatt (the Perceptron).",
        "📈 AI can now predict protein folding, a problem that puzzled scientists for 50 years!",
        "🎨 AI-generated art sold for $432,500 at Christie's auction in 2018.",
        "🗣️ Modern language models can understand context from thousands of words away!"
    ]
    
    st.info(f"**Did you know?** {np.random.choice(facts)}")
