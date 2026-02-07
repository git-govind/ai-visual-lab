import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


def run():
    st.title("🧠 Neural Networks")
    
    st.markdown("""
    ## Deep Learning Fundamentals
    
    Neural networks are computing systems inspired by biological neural networks.
    """)
    
    # Interactive parameters
    st.sidebar.subheader("Network Architecture")
    n_hidden = st.sidebar.slider("Hidden Layers", 1, 5, 2)
    neurons_per_layer = st.sidebar.slider("Neurons per Layer", 2, 10, 4)
    
    # Visualize neural network architecture
    st.subheader("Network Architecture")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    layers = [3] + [neurons_per_layer] * n_hidden + [2]
    layer_positions = np.linspace(0, 10, len(layers))
    
    max_neurons = max(layers)
    
    # Draw neurons
    for i, (layer_size, x_pos) in enumerate(zip(layers, layer_positions)):
        y_positions = np.linspace(0, max_neurons, layer_size + 2)[1:-1]
        for y_pos in y_positions:
            circle = plt.Circle((x_pos, y_pos), 0.3, color='skyblue', ec='black', linewidth=2)
            ax.add_patch(circle)
            
        # Draw connections to next layer
        if i < len(layers) - 1:
            next_y_positions = np.linspace(0, max_neurons, layers[i+1] + 2)[1:-1]
            for y1 in y_positions:
                for y2 in next_y_positions:
                    ax.plot([x_pos + 0.3, layer_positions[i+1] - 0.3], 
                           [y1, y2], 'gray', alpha=0.3, linewidth=0.5)
    
    # Labels
    ax.text(layer_positions[0], -1, 'Input\nLayer', ha='center', fontsize=10, fontweight='bold')
    for i in range(1, len(layers) - 1):
        ax.text(layer_positions[i], -1, f'Hidden\nLayer {i}', ha='center', fontsize=10, fontweight='bold')
    ax.text(layer_positions[-1], -1, 'Output\nLayer', ha='center', fontsize=10, fontweight='bold')
    
    ax.set_xlim(-1, 11)
    ax.set_ylim(-2, max_neurons + 1)
    ax.axis('off')
    
    st.pyplot(fig)
    
    # Activation functions
    st.subheader("Activation Functions")
    
    x = np.linspace(-5, 5, 100)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig, ax = plt.subplots()
        ax.plot(x, 1 / (1 + np.exp(-x)))
        ax.set_title('Sigmoid')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots()
        ax.plot(x, np.tanh(x))
        ax.set_title('Tanh')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col3:
        fig, ax = plt.subplots()
        ax.plot(x, np.maximum(0, x))
        ax.set_title('ReLU')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    st.markdown("---")
    st.info("💡 Neural networks learn by adjusting weights through backpropagation!")
