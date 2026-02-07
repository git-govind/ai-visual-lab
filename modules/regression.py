import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import time


def run():
    st.title("📈 Regression Analysis")
    
    st.markdown("""
    ## Predicting Continuous Values
    
    Regression models predict continuous output values. Let's explore different techniques!
    """)
    
    # Interactive parameters
    st.sidebar.subheader("🎛️ Model Configuration")
    
    data_source = st.sidebar.radio("Data Pattern:", 
                                   ["Linear", "Quadratic", "Sine Wave", "Complex"])
    noise_level = st.sidebar.slider("Noise Level", 0.0, 3.0, 0.8, 0.1)
    n_samples = st.sidebar.slider("Number of Samples", 50, 300, 100, 10)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Algorithm Selection")
    algorithm = st.sidebar.selectbox("Algorithm:", 
                                     ["Linear Regression", "Polynomial Regression", 
                                      "Ridge Regression", "Lasso Regression"])
    
    if "Polynomial" in algorithm:
        degree = st.sidebar.slider("Polynomial Degree", 1, 8, 2)
    else:
        degree = 1
    
    if algorithm in ["Ridge Regression", "Lasso Regression"]:
        alpha = st.sidebar.slider("Regularization (α)", 0.01, 10.0, 1.0, 0.1)
    
    # Generate data based on pattern
    np.random.seed(42)
    X = np.linspace(0, 10, n_samples).reshape(-1, 1)
    
    if data_source == "Linear":
        y_true = 2 * X.ravel() + 3
    elif data_source == "Quadratic":
        y_true = 0.5 * X.ravel()**2 - 3 * X.ravel() + 10
    elif data_source == "Sine Wave":
        y_true = 5 * np.sin(X.ravel()) + X.ravel()
    else:  # Complex
        y_true = 0.3 * X.ravel()**2 + 2 * np.sin(2 * X.ravel()) + X.ravel()
    
    y = y_true + np.random.randn(n_samples) * noise_level
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit model
    if algorithm == "Linear Regression":
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_pred = model.predict(X)
        
    elif algorithm == "Polynomial Regression":
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        X_train_poly = poly.transform(X_train)
        X_test_poly = poly.transform(X_test)
        
        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        y_pred_train = model.predict(X_train_poly)
        y_pred_test = model.predict(X_test_poly)
        y_pred = model.predict(X_poly)
        
    elif algorithm == "Ridge Regression":
        if degree > 1:
            poly = PolynomialFeatures(degree=degree)
            X_poly = poly.fit_transform(X)
            X_train_poly = poly.transform(X_train)
            X_test_poly = poly.transform(X_test)
        else:
            X_poly, X_train_poly, X_test_poly = X, X_train, X_test
        
        model = Ridge(alpha=alpha)
        model.fit(X_train_poly, y_train)
        y_pred_train = model.predict(X_train_poly)
        y_pred_test = model.predict(X_test_poly)
        y_pred = model.predict(X_poly)
        
    else:  # Lasso
        if degree > 1:
            poly = PolynomialFeatures(degree=degree)
            X_poly = poly.fit_transform(X)
            X_train_poly = poly.transform(X_train)
            X_test_poly = poly.transform(X_test)
        else:
            X_poly, X_train_poly, X_test_poly = X, X_train, X_test
        
        model = Lasso(alpha=alpha)
        model.fit(X_train_poly, y_train)
        y_pred_train = model.predict(X_train_poly)
        y_pred_test = model.predict(X_test_poly)
        y_pred = model.predict(X_poly)
    
    # Main visualization
    st.subheader("📊 Model Visualization")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot training and test data
    ax.scatter(X_train, y_train, alpha=0.6, s=60, c='blue', 
              edgecolors='black', linewidth=0.5, label='Training Data', zorder=3)
    ax.scatter(X_test, y_test, alpha=0.6, s=60, c='red', 
              edgecolors='black', linewidth=0.5, label='Test Data', zorder=3)
    
    # Plot true function (if low noise)
    if noise_level < 1.0:
        ax.plot(X, y_true, 'g--', linewidth=2, alpha=0.5, label='True Function', zorder=1)
    
    # Plot prediction
    ax.plot(X, y_pred, 'orange', linewidth=3, label=f'{algorithm}', zorder=2)
    
    ax.set_xlabel('Input Feature (X)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Target Value (y)', fontsize=12, fontweight='bold')
    ax.set_title(f'{algorithm} - {data_source} Pattern', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Model performance metrics
    st.markdown("---")
    st.subheader("📊 Model Performance")
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Training Set**")
        st.metric("MSE (Mean Squared Error)", f"{train_mse:.4f}")
        st.metric("RMSE (Root MSE)", f"{np.sqrt(train_mse):.4f}")
        st.metric("MAE (Mean Absolute Error)", f"{train_mae:.4f}")
        st.metric("R² Score", f"{train_r2:.4f}")
    
    with col2:
        st.write("**Test Set**")
        st.metric("MSE (Mean Squared Error)", f"{test_mse:.4f}")
        st.metric("RMSE (Root MSE)", f"{np.sqrt(test_mse):.4f}")
        st.metric("MAE (Mean Absolute Error)", f"{test_mae:.4f}")
        st.metric("R² Score", f"{test_r2:.4f}")
    
    # Overfitting/Underfitting indicator
    if abs(train_r2 - test_r2) > 0.2:
        if train_r2 > test_r2:
            st.warning("⚠️ **Possible Overfitting**: Model performs much better on training data!")
        else:
            st.info("ℹ️ **Unusual**: Test performance exceeds training (might be due to small data)")
    elif test_r2 < 0.5:
        st.warning("⚠️ **Possible Underfitting**: Model might be too simple for the data!")
    else:
        st.success("✅ **Good Fit**: Model generalizes well to test data!")
    
    # Residuals analysis
    st.markdown("---")
    st.subheader("🔍 Residuals Analysis")
    
    residuals_train = y_train - y_pred_train
    residuals_test = y_test - y_pred_test
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Residuals plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(y_pred_train, residuals_train, alpha=0.6, s=50, c='blue', label='Training')
        ax.scatter(y_pred_test, residuals_test, alpha=0.6, s=50, c='red', label='Test')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=2)
        ax.set_xlabel('Predicted Values', fontsize=11, fontweight='bold')
        ax.set_ylabel('Residuals', fontsize=11, fontweight='bold')
        ax.set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        # Residuals distribution
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(residuals_train, bins=20, alpha=0.6, color='blue', edgecolor='black', label='Training')
        ax.hist(residuals_test, bins=15, alpha=0.6, color='red', edgecolor='black', label='Test')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
        ax.set_xlabel('Residual Value', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title('Residuals Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # Predicted vs Actual
    st.subheader("🎯 Predicted vs Actual")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_train, y_pred_train, alpha=0.6, s=60, c='blue', 
              edgecolors='black', linewidth=0.5, label='Training')
    ax.scatter(y_test, y_pred_test, alpha=0.6, s=60, c='red', 
              edgecolors='black', linewidth=0.5, label='Test')
    
    # Perfect prediction line
    all_y = np.concatenate([y_train, y_test])
    min_y, max_y = all_y.min(), all_y.max()
    ax.plot([min_y, max_y], [min_y, max_y], 'g--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Values', fontsize=11, fontweight='bold')
    ax.set_ylabel('Predicted Values', fontsize=11, fontweight='bold')
    ax.set_title('Prediction Accuracy', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    st.pyplot(fig)
    
    # Educational content
    st.markdown("---")
    st.subheader("📚 Understanding the Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Metrics Explained:**
        
        - **MSE**: Average squared difference between predicted and actual values
          (Lower is better, sensitive to outliers)
        
        - **RMSE**: Square root of MSE, in original units
          (More interpretable than MSE)
        
        - **MAE**: Average absolute difference
          (Less sensitive to outliers than MSE)
        
        - **R² Score**: Proportion of variance explained (0 to 1)
          (1.0 = perfect, 0.0 = as good as predicting mean)
        """)
    
    with col2:
        st.warning("""
        **Common Issues:**
        
        - **Underfitting**: Model too simple
          → Increase polynomial degree
        
        - **Overfitting**: Model too complex
          → Decrease degree or use regularization
        
        - **High Noise**: Poor performance despite good model
          → Get more/better data
        
        - **Regularization**: Ridge/Lasso prevent overfitting
          → Adjust α parameter
        """)
    
    st.success("""
    💡 **Try This:**
    - Change the data pattern and see which algorithm works best
    - Increase polynomial degree until you see overfitting
    - Use Ridge/Lasso to prevent overfitting in high-degree polynomials
    - Observe how noise affects model performance
    """)
    
    if st.button("✅ Mark Module Complete"):
        if 'completed' not in st.session_state:
            st.session_state.completed = set()
        st.session_state.completed.add("regression")
        st.balloons()
        st.success("Module completed! Continue to Classification ➡️")
