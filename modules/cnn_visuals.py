import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns


def run():
    st.title("🖼️ Convolutional Neural Networks")
    
    st.markdown("""
    ## Computer Vision with CNNs
    
    CNNs revolutionized computer vision by automatically learning visual features!
    """)
    
    # Interactive parameters
    st.sidebar.subheader("⚙️ CNN Configuration")
    
    image_type = st.sidebar.selectbox(
        "Image Pattern:",
        ["Random", "Vertical Edges", "Horizontal Edges", "Diagonal Pattern", "Checkerboard"]
    )
    
    kernel_type = st.sidebar.selectbox(
        "Filter/Kernel Type:",
        ["Average (Blur)", "Vertical Edge", "Horizontal Edge", "Sharpen", "Custom"]
    )
    
    if kernel_type == "Custom":
        kernel_size = st.sidebar.slider("Kernel Size", 3, 7, 3, step=2)
    else:
        kernel_size = 3
    
    stride = st.sidebar.slider("Stride", 1, 3, 1)
    padding = st.sidebar.slider("Padding", 0, 2, 0)
    
    # Create sample image based on pattern
    image_size = 16
    np.random.seed(42)
    
    if image_type == "Random":
        image = np.random.rand(image_size, image_size)
    elif image_type == "Vertical Edges":
        image = np.zeros((image_size, image_size))
        image[:, :image_size//2] = 1.0
        image += np.random.rand(image_size, image_size) * 0.1
    elif image_type == "Horizontal Edges":
        image = np.zeros((image_size, image_size))
        image[:image_size//2, :] = 1.0
        image += np.random.rand(image_size, image_size) * 0.1
    elif image_type == "Diagonal Pattern":
        image = np.eye(image_size) + np.flip(np.eye(image_size), axis=1)
        image = np.clip(image, 0, 1)
        image += np.random.rand(image_size, image_size) * 0.1
    else:  # Checkerboard
        image = np.zeros((image_size, image_size))
        for i in range(image_size):
            for j in range(image_size):
                if (i // 2 + j // 2) % 2 == 0:
                    image[i, j] = 1.0
        image += np.random.rand(image_size, image_size) * 0.1
    
    # Create kernel based on type
    if kernel_type == "Average (Blur)":
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    elif kernel_type == "Vertical Edge":
        kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    elif kernel_type == "Horizontal Edge":
        kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    elif kernel_type == "Sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    else:  # Custom
        kernel = np.random.randn(kernel_size, kernel_size)
    
    # Add padding if specified
    if padding > 0:
        image_padded = np.pad(image, padding, mode='constant', constant_values=0)
    else:
        image_padded = image
    
    # Perform convolution
    output_size_h = (image_padded.shape[0] - kernel_size) // stride + 1
    output_size_w = (image_padded.shape[1] - kernel_size) // stride + 1
    output = np.zeros((output_size_h, output_size_w))
    
    for i in range(output_size_h):
        for j in range(output_size_w):
            h_start = i * stride
            w_start = j * stride
            patch = image_padded[h_start:h_start+kernel_size, w_start:w_start+kernel_size]
            output[i, j] = np.sum(patch * kernel)
    
    # Apply ReLU activation
    output_relu = np.maximum(0, output)
    
    # Main visualization
    st.subheader("🔍 Convolution Operation Visualization")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.write("**Input Image**")
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(image, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'{image.shape[0]}x{image.shape[1]}', fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        st.pyplot(fig)
        
    with col2:
        st.write("**Kernel/Filter**")
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(kernel, cmap='RdBu_r', vmin=-3, vmax=3)
        ax.set_title(f'{kernel.shape[0]}x{kernel.shape[1]}', fontweight='bold')
        
        # Add grid and values
        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[1]):
                ax.text(j, i, f'{kernel[i, j]:.2f}', 
                       ha='center', va='center', fontsize=10, fontweight='bold')
        
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        st.pyplot(fig)
    
    with col3:
        st.write("**Feature Map**")
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(output, cmap='viridis')
        ax.set_title(f'{output.shape[0]}x{output.shape[1]}', fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        st.pyplot(fig)
    
    with col4:
        st.write("**After ReLU**")
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(output_relu, cmap='viridis')
        ax.set_title(f'{output_relu.shape[0]}x{output_relu.shape[1]}', fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        st.pyplot(fig)
    
    st.info(f"""
    ⚙️ **Computation Details:**
    - Input: {image.shape[0]}x{image.shape[1]} | Kernel: {kernel.shape[0]}x{kernel.shape[1]} | Stride: {stride} | Padding: {padding}
    - Output size: ({image_padded.shape[0]} - {kernel_size}) / {stride} + 1 = {output_size_h}x{output_size_w}
    - Parameters: {kernel.size + 1:,} (weights + bias)
    """)
    
    st.markdown("---")
    
    # Pooling operations
    st.subheader("🏊 Pooling Layers")
    
    pool_type = st.selectbox("Pooling Operation:", ["Max Pooling", "Average Pooling"])
    pool_size = st.slider("Pool Size", 2, 4, 2)
    
    # Perform pooling
    pooled_h = output_relu.shape[0] // pool_size
    pooled_w = output_relu.shape[1] // pool_size
    pooled = np.zeros((pooled_h, pooled_w))
    
    for i in range(pooled_h):
        for j in range(pooled_w):
            patch = output_relu[i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size]
            if pool_type == "Max Pooling":
                pooled[i, j] = np.max(patch)
            else:
                pooled[i, j] = np.mean(patch)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Before Pooling**")
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(output_relu, cmap='viridis')
        
        # Draw pooling regions
        for i in range(pooled_h):
            for j in range(pooled_w):
                rect = Rectangle((j*pool_size-0.5, i*pool_size-0.5), 
                               pool_size, pool_size,
                               fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)
        
        ax.set_title(f'Size: {output_relu.shape[0]}x{output_relu.shape[1]}', fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        st.pyplot(fig)
    
    with col2:
        st.write("**After Pooling**")
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(pooled, cmap='viridis')
        ax.set_title(f'Size: {pooled.shape[0]}x{pooled.shape[1]}', fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        st.pyplot(fig)
    
    st.success("""
    💡 **Key Takeaways:**
    - CNNs use convolution to detect patterns
    - Pooling reduces dimensions and adds translation invariance
    - Parameters are shared across the image (efficient!)
    - Deep layers learn increasingly complex features
    - Architecture design is crucial for performance
    """)
    
    if st.button("✅ Mark Module Complete"):
        if 'completed' not in st.session_state:
            st.session_state.completed = set()
        st.session_state.completed.add("cnn_visuals")
        st.balloons()
        st.success("Module completed! Continue to GenAI: Tokenization ➡️")
