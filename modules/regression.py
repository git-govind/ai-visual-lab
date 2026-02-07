import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def run():
    st.title("📈 Regression Analysis")
    
    st.markdown("""
    ## Predicting Continuous Values
    
    Regression models predict continuous output values based on input features.
    """)
    
    # Interactive parameters
    st.sidebar.subheader("Regression Parameters")
    noise_level = st.sidebar.slider("Noise Level", 0.0, 2.0, 0.5, 0.1)
    degree = st.sidebar.slider("Polynomial Degree", 1, 5, 1)
    
    # Generate data
    np.random.seed(42)
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = 2 * X.ravel() + 3 + np.random.randn(100) * noise_level
    
    # Fit model
    if degree == 1:
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
    else:
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, alpha=0.5, label='Data points')
    ax.plot(X, y_pred, 'r-', linewidth=2, label=f'Degree {degree} fit')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title('Regression Model')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Model metrics
    mse = np.mean((y - y_pred) ** 2)
    st.metric("Mean Squared Error", f"{mse:.4f}")
    
    st.markdown("---")
    st.info("💡 Adjust the parameters in the sidebar to see how they affect the model!")
