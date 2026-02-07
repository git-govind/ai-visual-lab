import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


def run():
    st.title("🖼️ Convolutional Neural Networks")
    
    st.markdown("""
    ## Computer Vision with CNNs
    
    CNNs are specialized neural networks designed for processing grid-like data such as images.
    """)
    
    # Interactive parameters
    st.sidebar.subheader("CNN Parameters")
    kernel_size = st.sidebar.slider("Kernel Size", 3, 7, 3, step=2)
    
    st.subheader("Convolution Operation")
    
    # Create sample image
    image = np.random.rand(8, 8)
    
    # Create kernel
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    
    # Perform convolution (simplified)
    output_size = image.shape[0] - kernel_size + 1
    output = np.zeros((output_size, output_size))
    
    for i in range(output_size):
        for j in range(output_size):
            output[i, j] = np.sum(image[i:i+kernel_size, j:j+kernel_size] * kernel)
    
    # Visualize
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Input Image**")
        fig, ax = plt.subplots()
        im = ax.imshow(image, cmap='gray')
        ax.set_title(f'Size: {image.shape[0]}x{image.shape[1]}')
        plt.colorbar(im, ax=ax)
        st.pyplot(fig)
    
    with col2:
        st.write("**Kernel/Filter**")
        fig, ax = plt.subplots()
        im = ax.imshow(kernel, cmap='viridis')
        ax.set_title(f'Size: {kernel_size}x{kernel_size}')
        plt.colorbar(im, ax=ax)
        st.pyplot(fig)
    
    with col3:
        st.write("**Output Feature Map**")
        fig, ax = plt.subplots()
        im = ax.imshow(output, cmap='gray')
        ax.set_title(f'Size: {output_size}x{output_size}')
        plt.colorbar(im, ax=ax)
        st.pyplot(fig)
    
    st.markdown("---")
    
    st.subheader("CNN Architecture Layers")
    st.write("""
    - **Convolutional Layer**: Applies filters to detect features
    - **Pooling Layer**: Reduces spatial dimensions
    - **Activation Layer**: Introduces non-linearity (ReLU)
    - **Fully Connected Layer**: Classification at the end
    """)
    
    st.info("💡 CNNs automatically learn hierarchical features from images!")
