import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time


def run():
    st.title("🤖 Welcome to AI Visual Canvas")
    
    # Animated welcome message
    welcome_text = "Your Interactive Journey into Artificial Intelligence"
    st.markdown(f"## {welcome_text}")
    
    # Progress indicator
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📚 Total Modules", "10", help="Complete learning modules")
    with col2:
        completed = len(st.session_state.get('completed', set()))
        st.metric("✅ Your Progress", f"{completed}/10", help="Modules completed")
    with col3:
        progress_pct = (completed / 10) * 100
        st.metric("🎯 Completion", f"{progress_pct:.0f}%", help="Overall progress")
    
    st.markdown("---")
    
    # Interactive learning path visualization
    st.subheader("🎓 Your Learning Path")
    
    modules = [
        ("Intro", "intro", "📚"),
        ("Data Basics", "data_basics", "📊"),
        ("Regression", "regression", "📈"),
        ("Classification", "classification", "🎯"),
        ("Neural Networks", "neural_networks", "🧠"),
        ("CNN Visuals", "cnn_visuals", "🖼️"),
        ("Tokenization", "genai_tokenization", "🔤"),
        ("Attention", "genai_attention", "👁️"),
        ("Generation", "genai_generation", "✨"),
        ("Capstone", "capstone", "🎓")
    ]
    
    # Create visual learning path with enhanced design
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='#f8f9fa')
    ax.set_facecolor('#ffffff')
    
    # Create path points in a more dynamic curved pattern
    n_modules = len(modules)
    x = np.linspace(0, 12, n_modules)
    # Create a smooth wave pattern that rises
    y = 3 + 2.5 * np.sin(x * 0.6) + x * 0.3
    
    # Draw gradient path connecting the modules
    for i in range(n_modules - 1):
        # Create smooth curve between points
        x_segment = np.linspace(x[i], x[i+1], 50)
        y_segment = np.interp(x_segment, x, y)
        
        # Color gradient from blue to purple
        colors_gradient = plt.cm.viridis(np.linspace(0.2, 0.9, n_modules))
        ax.plot(x_segment, y_segment, color=colors_gradient[i], 
                linewidth=4, alpha=0.6, solid_capstyle='round')
    
    # Draw module points with beautiful styling
    for i in range(n_modules):
        # Determine color based on completion
        if i < completed:
            color = '#10b981'  # Green for completed
            edge_color = '#059669'
            size = 800
        else:
            color = '#e5e7eb'  # Light gray for incomplete
            edge_color = '#9ca3af'
            size = 700
        
        # Draw outer glow effect
        ax.scatter(x[i], y[i], s=size + 200, c=color, alpha=0.2, 
                  edgecolors='none', zorder=2)
        
        # Draw main circle
        ax.scatter(x[i], y[i], s=size, c=color, 
                  edgecolors=edge_color, linewidth=3, zorder=3)
        
        # Add number inside circle
        ax.text(x[i], y[i], str(i+1), ha='center', va='center', 
                fontweight='bold', fontsize=14, color='white', zorder=4)
        
        # Get module info
        name, module_name, emoji = modules[i]
        
        # Add module display name below circle
        ax.text(x[i], y[i] - 1.2, name, ha='center', va='top', 
                fontweight='bold', fontsize=10, color='#374151', zorder=4)
        
        # Add module technical name below display name
        ax.text(x[i], y[i] - 1.7, f'({module_name})', ha='center', va='top', 
                fontsize=7, color='#6b7280', style='italic', zorder=4)
    
    # Add progress arrow if there are completed modules
    if completed > 0 and completed < n_modules:
        arrow_x = x[completed - 1] + (x[completed] - x[completed - 1]) * 0.5
        arrow_y = np.interp(arrow_x, x, y) + 2
        ax.annotate('YOU ARE HERE', xy=(x[completed], y[completed]), 
                   xytext=(arrow_x, arrow_y),
                   fontsize=10, fontweight='bold', color='#dc2626',
                   ha='center',
                   arrowprops=dict(arrowstyle='->', color='#dc2626', lw=2))
    
    ax.set_xlim(-1, 13)
    ax.set_ylim(-1, 12)
    ax.axis('off')
    
    # Add decorative title with gradient effect
    title_text = 'Your AI Learning Journey'
    ax.text(6, 11, title_text, fontsize=20, fontweight='bold', 
            ha='center', color='#1f2937')
    
    # Add progress text
    progress_text = f'{completed}/{n_modules} Modules Completed'
    ax.text(6, 10.2, progress_text, fontsize=12, ha='center', 
            color='#6b7280', style='italic')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Expandable module details
    st.subheader("📖 Course Modules")
    
    module_details = {
        "1. Intro (intro) 📚": {
            "desc": "Get started with AI fundamentals and core concepts",
            "topics": ["What is AI?", "Machine learning basics", "AI applications", "Learning roadmap"]
        },
        "2. Data Basics (data_basics) 📊": {
            "desc": "Master data manipulation, visualization, and preprocessing techniques",
            "topics": ["Data structures", "Statistical analysis", "Data visualization", "Feature engineering"]
        },
        "3. Regression (regression) 📈": {
            "desc": "Learn to predict continuous values with linear and polynomial models",
            "topics": ["Linear regression", "Polynomial regression", "Model evaluation", "Feature scaling"]
        },
        "4. Classification (classification) 🎯": {
            "desc": "Predict categories using various classification algorithms",
            "topics": ["Logistic regression", "Decision trees", "Model metrics", "Confusion matrices"]
        },
        "5. Neural Networks (neural_networks) 🧠": {
            "desc": "Understand deep learning architecture and activation functions",
            "topics": ["Network architecture", "Activation functions", "Backpropagation", "Loss functions"]
        },
        "6. CNN Visuals (cnn_visuals) 🖼️": {
            "desc": "Explore convolutional neural networks for computer vision",
            "topics": ["Convolution operations", "Pooling layers", "Feature maps", "Image classification"]
        },
        "7. Tokenization (genai_tokenization) 🔤": {
            "desc": "Learn how language models process and encode text",
            "topics": ["Text tokenization", "Subword encoding", "Vocabulary building", "Token embeddings"]
        },
        "8. Attention (genai_attention) 👁️": {
            "desc": "Master the attention mechanism powering modern transformers",
            "topics": ["Self-attention", "Multi-head attention", "Query-Key-Value", "Attention weights"]
        },
        "9. Generation (genai_generation) ✨": {
            "desc": "Generate text using various sampling strategies",
            "topics": ["Autoregressive generation", "Temperature sampling", "Top-k sampling", "Nucleus sampling"]
        },
        "10. Capstone (capstone) 🎓": {
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
