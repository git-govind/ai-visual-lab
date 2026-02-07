import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D


def run():
    st.title("🧠 Neural Networks")
    
    st.markdown("""
    ## Deep Learning Fundamentals
    
    Neural networks are the backbone of modern AI. Let's build and understand them!
    """)
    
    # Interactive architecture builder
    st.subheader("🏛️ Build Your Neural Network")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        input_neurons = st.slider("Input Layer Neurons", 2, 10, 3)
    with col2:
        n_hidden = st.slider("Number of Hidden Layers", 1, 5, 2)
    with col3:
        output_neurons = st.slider("Output Layer Neurons", 1, 10, 2)
    
    neurons_per_layer = st.slider("Neurons per Hidden Layer", 2, 12, 4)
    
    # Visualize neural network architecture
    st.subheader("📊 Network Architecture Visualization")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    layers = [input_neurons] + [neurons_per_layer] * n_hidden + [output_neurons]
    n_layers = len(layers)
    layer_positions = np.linspace(0, 12, n_layers)
    
    max_neurons = max(layers)
    
    # Store neuron positions for connections
    neuron_positions = []
    
    # Draw neurons and connections
    for i, (layer_size, x_pos) in enumerate(zip(layers, layer_positions)):
        y_positions = np.linspace(1, max_neurons + 1, layer_size + 2)[1:-1]
        layer_neurons = []
        
        for j, y_pos in enumerate(y_positions):
            # Determine color based on layer type
            if i == 0:
                color = 'lightgreen'
                label = 'Input'
            elif i == n_layers - 1:
                color = 'lightcoral'
                label = 'Output'
            else:
                color = 'lightblue'
                label = 'Hidden'
            
            # Draw neuron
            circle = Circle((x_pos, y_pos), 0.25, color=color, ec='black', linewidth=2, zorder=4)
            ax.add_patch(circle)
            layer_neurons.append((x_pos, y_pos))
            
            # Add activation value (simulated)
            if i > 0:  # Not input layer
                activation = np.random.rand()
                intensity = int(activation * 255)
                ax.text(x_pos, y_pos, f"{activation:.2f}", 
                       ha='center', va='center', fontsize=7, 
                       fontweight='bold', zorder=5)
        
        neuron_positions.append(layer_neurons)
        
        # Draw connections to next layer
        if i < n_layers - 1:
            next_y_positions = np.linspace(1, max_neurons + 1, layers[i+1] + 2)[1:-1]
            for y1 in y_positions:
                for y2 in next_y_positions:
                    # Random weight for visualization
                    weight = np.random.randn()
                    alpha = min(abs(weight) / 2, 0.8)
                    color = 'red' if weight < 0 else 'blue'
                    linewidth = min(abs(weight) + 0.3, 2)
                    
                    ax.plot([x_pos + 0.25, layer_positions[i+1] - 0.25], 
                           [y1, y2], color=color, alpha=alpha, 
                           linewidth=linewidth, zorder=1)
    
    # Layer labels
    ax.text(layer_positions[0], 0.3, 'Input\nLayer', 
           ha='center', fontsize=11, fontweight='bold')
    for i in range(1, n_layers - 1):
        ax.text(layer_positions[i], 0.3, f'Hidden\nLayer {i}', 
               ha='center', fontsize=11, fontweight='bold')
    ax.text(layer_positions[-1], 0.3, 'Output\nLayer', 
           ha='center', fontsize=11, fontweight='bold')
    
    # Network stats
    total_params = sum([layers[i] * layers[i+1] for i in range(n_layers-1)])
    ax.text(6, max_neurons + 2, f'Total Parameters: {total_params:,}', 
           ha='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax.set_xlim(-1, 13)
    ax.set_ylim(0, max_neurons + 3)
    ax.axis('off')
    ax.set_title('Neural Network Architecture', fontsize=15, fontweight='bold', pad=20)
    
    st.pyplot(fig)
    
    # Network statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Layers", n_layers)
    with col2:
        st.metric("Total Neurons", sum(layers))
    with col3:
        st.metric("Connections", total_params)
    with col4:
        depth_category = "Shallow" if n_layers <= 3 else "Deep" if n_layers <= 5 else "Very Deep"
        st.metric("Depth", depth_category)
    
    st.markdown("---")
    
    # Activation functions
    st.subheader("⚡ Activation Functions")
    
    activation_choice = st.selectbox(
        "Select activation function to explore:",
        ["Sigmoid", "Tanh", "ReLU", "Leaky ReLU", "ELU", "Softmax"]
    )
    
    x = np.linspace(-5, 5, 100)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if activation_choice == "Sigmoid":
            y = 1 / (1 + np.exp(-x))
            derivative = y * (1 - y)
            formula = r"$\sigma(x) = \frac{1}{1 + e^{-x}}$"
            description = "Squashes values to (0, 1). Used in binary classification."
            
        elif activation_choice == "Tanh":
            y = np.tanh(x)
            derivative = 1 - y**2
            formula = r"$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$"
            description = "Squashes values to (-1, 1). Zero-centered, better than sigmoid."
            
        elif activation_choice == "ReLU":
            y = np.maximum(0, x)
            derivative = (x > 0).astype(float)
            formula = r"$\text{ReLU}(x) = \max(0, x)$"
            description = "Most popular! Simple, fast, and works well in practice."
            
        elif activation_choice == "Leaky ReLU":
            alpha = 0.01
            y = np.where(x > 0, x, alpha * x)
            derivative = np.where(x > 0, 1, alpha)
            formula = r"$\text{LeakyReLU}(x) = \max(0.01x, x)$"
            description = "Fixes 'dying ReLU' problem with small negative slope."
            
        elif activation_choice == "ELU":
            alpha = 1.0
            y = np.where(x > 0, x, alpha * (np.exp(x) - 1))
            derivative = np.where(x > 0, 1, alpha * np.exp(x))
            formula = r"$\text{ELU}(x) = \begin{cases} x & x > 0 \\ \alpha(e^x - 1) & x \leq 0 \end{cases}$"
            description = "Smooth version of ReLU with negative values."
            
        else:  # Softmax (showing for single input)
            y = np.exp(x) / np.sum(np.exp(x))
            derivative = y * (1 - y)
            formula = r"$\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$"
            description = "Converts logits to probabilities. Used in multi-class output."
        
        # Plot function
        ax.plot(x, y, 'b-', linewidth=3, label=f'{activation_choice} Function')
        ax.plot(x, derivative, 'r--', linewidth=2, label='Derivative', alpha=0.7)
        
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        
        ax.set_xlabel('Input (x)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Output', fontsize=12, fontweight='bold')
        ax.set_title(f'{activation_choice} Activation Function', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
    with col2:
        st.markdown("**Formula:**")
        st.latex(formula)
        
        st.markdown("**Description:**")
        st.info(description)
        
        st.markdown("**Properties:**")
        if activation_choice == "Sigmoid":
            st.write("• Range: (0, 1)")
            st.write("• Vanishing gradient issue")
            st.write("• Not zero-centered")
        elif activation_choice == "Tanh":
            st.write("• Range: (-1, 1)")
            st.write("• Zero-centered")
            st.write("• Still has vanishing gradient")
        elif activation_choice == "ReLU":
            st.write("• Range: [0, ∞)")
            st.write("• Computationally efficient")
            st.write("• Can 'die' during training")
        elif activation_choice == "Leaky ReLU":
            st.write("• Range: (-∞, ∞)")
            st.write("• Prevents dying neurons")
            st.write("• Slight computation overhead")
        elif activation_choice == "ELU":
            st.write("• Range: (-α, ∞)")
            st.write("• Smooth everywhere")
            st.write("• More expensive to compute")
        else:  # Softmax
            st.write("• Range: (0, 1)")
            st.write("• Outputs sum to 1")
            st.write("• Used in output layer")
    
    # Comparison of activations
    st.subheader("🔍 Compare All Activation Functions")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    
    activations = {
        'Sigmoid': lambda x: 1 / (1 + np.exp(-x)),
        'Tanh': lambda x: np.tanh(x),
        'ReLU': lambda x: np.maximum(0, x),
        'Leaky ReLU': lambda x: np.where(x > 0, x, 0.01 * x),
        'ELU': lambda x: np.where(x > 0, x, 1.0 * (np.exp(x) - 1)),
        'Softplus': lambda x: np.log(1 + np.exp(x))
    }
    
    x = np.linspace(-5, 5, 200)
    
    for idx, (name, func) in enumerate(activations.items()):
        y = func(x)
        axes[idx].plot(x, y, linewidth=2.5, color=f'C{idx}')
        axes[idx].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        axes[idx].axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        axes[idx].set_title(name, fontsize=11, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xlabel('x', fontsize=9)
        axes[idx].set_ylabel('f(x)', fontsize=9)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Forward propagation simulation
    st.subheader("➡️ Forward Propagation Demo")
    
    st.write("Watch how information flows through the network:")
    
    # Simple 3-layer network demo
    input_val = st.slider("Input Value", -5.0, 5.0, 2.0, 0.1)
    
    # Simulate forward pass
    w1, b1 = 0.5, 0.3  # Layer 1 weights
    w2, b2 = 0.8, -0.2  # Layer 2 weights
    
    # Layer 1
    z1 = w1 * input_val + b1
    if activation_choice == "ReLU":
        a1 = max(0, z1)
    else:  # Sigmoid
        a1 = 1 / (1 + np.exp(-z1))
    
    # Layer 2
    z2 = w2 * a1 + b2
    if activation_choice == "ReLU":
        output = max(0, z2)
    else:
        output = 1 / (1 + np.exp(-z2))
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Input", f"{input_val:.3f}")
        st.write("")
        
    with col2:
        st.metric("Hidden Layer", f"{a1:.3f}")
        st.caption(f"z = {w1}*x + {b1} = {z1:.3f}")
        st.caption(f"a = {activation_choice}(z) = {a1:.3f}")
        
    with col3:
        st.metric("Output", f"{output:.3f}")
        st.caption(f"z = {w2}*a + {b2} = {z2:.3f}")
        st.caption(f"out = {activation_choice}(z) = {output:.3f}")
    
    st.markdown("---")
    
    # Loss functions
    st.subheader("🎯 Loss Functions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Mean Squared Error (MSE)**
        """)
        st.latex(r"L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2")
        st.write("• Used for regression tasks")
        st.write("• Penalizes large errors more")
        st.write("• Always non-negative")
        
        st.markdown("""
        **Binary Cross-Entropy**
        """)
        st.latex(r"L = -\frac{1}{n}\sum_{i=1}^{n}[y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]")
        st.write("• Used for binary classification")
        st.write("• Measures probability difference")
        st.write("• Works with sigmoid output")
    
    with col2:
        st.markdown("""
        **Categorical Cross-Entropy**
        """)
        st.latex(r"L = -\sum_{i=1}^{n}\sum_{c=1}^{C}y_{i,c}\log(\hat{y}_{i,c})")
        st.write("• Used for multi-class classification")
        st.write("• Works with softmax output")
        st.write("• C is number of classes")
        
        st.markdown("""
        **Mean Absolute Error (MAE)**
        """)
        st.latex(r"L = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|")
        st.write("• Less sensitive to outliers")
        st.write("• Linear penalty")
        st.write("• More robust than MSE")
    
    # Backpropagation explanation
    st.markdown("---")
    st.subheader("⬅️ Backpropagation: How Networks Learn")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        Backpropagation is the algorithm that allows neural networks to learn. Here's how it works:
        
        **Steps:**
        1. **Forward Pass**: Input flows through network to produce output
        2. **Calculate Loss**: Compare output with true label
        3. **Backward Pass**: Compute gradients of loss w.r.t. each weight
        4. **Update Weights**: Adjust weights to reduce loss
        
        **Chain Rule:**
        """)
        st.latex(r"\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}")
        
        st.markdown("""
        **Gradient Descent Update:**
        """)
        st.latex(r"w_{new} = w_{old} - \eta \cdot \frac{\partial L}{\partial w}")
        
        st.write("Where η (eta) is the learning rate")
    
    with col2:
        learning_rate = st.slider("Learning Rate (η)", 0.001, 1.0, 0.1, 0.01)
        
        # Simulate gradient descent
        x_gd = np.linspace(-2, 2, 100)
        loss = x_gd**2  # Simple quadratic loss
        
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(x_gd, loss, 'b-', linewidth=2, label='Loss Function')
        
        # Show gradient descent steps
        current_w = 1.8
        for step in range(5):
            gradient = 2 * current_w
            ax.plot(current_w, current_w**2, 'ro', markersize=8)
            current_w = current_w - learning_rate * gradient
            if abs(current_w) < 0.01:
                break
        
        ax.set_xlabel('Weight Value', fontsize=10, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=10, fontweight='bold')
        ax.set_title(f'Gradient Descent (lr={learning_rate})', fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
    st.success("""
    💡 **Key Insights:**
    - Deeper networks can learn more complex patterns
    - Activation functions introduce non-linearity (essential for learning)
    - Backpropagation efficiently computes gradients
    - Learning rate controls how fast the network learns
    - Different architectures suit different problems
    """)
    
    if st.button("✅ Mark Module Complete"):
        if 'completed' not in st.session_state:
            st.session_state.completed = set()
        st.session_state.completed.add("neural_networks")
        st.balloons()
        st.success("Module completed! Continue to CNN Visuals ➡️")
