import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
import seaborn as sns
import pandas as pd


def run():
    st.title("🎓 Capstone: Build Your AI System")
    
    st.markdown("""
    ## Put Everything Together!
    
    Apply everything you've learned to build a complete AI system from scratch.
    """)
    
    # Project selection
    st.subheader("🎯 Choose Your Project")
    
    project_type = st.selectbox(
        "Select a project:",
        ["House Price Prediction (Regression)", 
         "Customer Churn Prediction (Classification)",
         "Sales Forecasting (Time Series)",
         "Image Pattern Recognition"]
    )
    
    st.markdown("---")
    
    if "House Price" in project_type:
        st.subheader("🏠 House Price Prediction")
        
        st.write("""
        **Goal**: Predict house prices based on features like size, location, and age.
        
        **Steps**:
        1. Generate/load data
        2. Explore and visualize
        3. Preprocess data
        4. Train models
        5. Evaluate performance
        6. Make predictions
        """)
        
        # Data generation
        st.markdown("### 1️⃣ Data Generation")
        
        col1, col2 = st.columns(2)
        with col1:
            n_samples = st.slider("Number of houses", 100, 1000, 500)
        with col2:
            noise_level = st.slider("Data noise", 0.0, 2.0, 0.3)
        
        np.random.seed(42)
        
        # Generate realistic house data
        sqft = np.random.randint(500, 5000, n_samples)
        bedrooms = np.random.randint(1, 6, n_samples)
        bathrooms = np.random.randint(1, 4, n_samples)
        age = np.random.randint(0, 100, n_samples)
        location_score = np.random.uniform(1, 10, n_samples)
        
        # Price formula with noise
        base_price = 50000
        price_per_sqft = 200
        bedroom_value = 15000
        bathroom_value = 10000
        age_penalty = -1000
        location_factor = 20000
        
        price = (base_price + 
                price_per_sqft * sqft + 
                bedroom_value * bedrooms + 
                bathroom_value * bathrooms + 
                age_penalty * age + 
                location_factor * location_score +
                np.random.randn(n_samples) * 50000 * noise_level)
        
        # Create DataFrame
        data = pd.DataFrame({
            'Square_Feet': sqft,
            'Bedrooms': bedrooms,
            'Bathrooms': bathrooms,
            'Age_Years': age,
            'Location_Score': location_score,
            'Price': price
        })
        
        st.write("**Dataset Preview:**")
        st.dataframe(data.head(10), use_container_width=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", n_samples)
        with col2:
            st.metric("Features", 5)
        with col3:
            st.metric("Avg Price", f"${data['Price'].mean():,.0f}")
        with col4:
            st.metric("Price Range", f"${data['Price'].max() - data['Price'].min():,.0f}")
        
        # Data exploration
        st.markdown("### 2️⃣ Data Exploration")
        
        tab1, tab2 = st.tabs(["📊 Distributions", "🔗 Correlations"])
        
        with tab1:
            feature_to_plot = st.selectbox("Select feature:", data.columns[:-1])
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Histogram
            axes[0].hist(data[feature_to_plot], bins=30, edgecolor='black', color='skyblue')
            axes[0].set_xlabel(feature_to_plot, fontweight='bold')
            axes[0].set_ylabel('Frequency', fontweight='bold')
            axes[0].set_title(f'Distribution of {feature_to_plot}', fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            
            # Scatter with price
            axes[1].scatter(data[feature_to_plot], data['Price'], alpha=0.5, s=20)
            axes[1].set_xlabel(feature_to_plot, fontweight='bold')
            axes[1].set_ylabel('Price', fontweight='bold')
            axes[1].set_title(f'{feature_to_plot} vs Price', fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab2:
            fig, ax = plt.subplots(figsize=(8, 6))
            corr_matrix = data.corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, square=True, ax=ax)
            ax.set_title('Feature Correlation Matrix', fontweight='bold', pad=20)
            st.pyplot(fig)
        
        # Model training
        st.markdown("### 3️⃣ Model Training")
        
        # Prepare data
        X = data.drop('Price', axis=1)
        y = data['Price']
        
        test_size = st.slider("Test set size (%)", 10, 40, 20) / 100
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Model selection
        model_choice = st.selectbox(
            "Choose algorithm:",
            ["Linear Regression", "Decision Tree", "Random Forest"]
        )
        
        if model_choice == "Linear Regression":
            model = LinearRegression()
        elif model_choice == "Decision Tree":
            max_depth = st.slider("Max depth", 3, 20, 10)
            model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        else:
            n_estimators = st.slider("Number of trees", 10, 200, 100)
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        
        # Train model
        if st.button("🚀 Train Model"):
            with st.spinner("Training model..."):
                model.fit(X_train_scaled, y_train)
                
                # Predictions
                y_train_pred = model.predict(X_train_scaled)
                y_test_pred = model.predict(X_test_scaled)
                
                # Store in session state
                st.session_state.model_trained = True
                st.session_state.model = model
                st.session_state.scaler = scaler
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.y_train_pred = y_train_pred
                st.session_state.y_test_pred = y_test_pred
                
                st.success("✅ Model trained successfully!")
        
        # Model evaluation
        if st.session_state.get('model_trained', False):
            st.markdown("### 4️⃣ Model Evaluation")
            
            y_train = st.session_state.y_train
            y_test = st.session_state.y_test
            y_train_pred = st.session_state.y_train_pred
            y_test_pred = st.session_state.y_test_pred
            
            # Metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Train RMSE", f"${train_rmse:,.0f}")
            with col2:
                st.metric("Test RMSE", f"${test_rmse:,.0f}")
            with col3:
                st.metric("Train R²", f"{train_r2:.3f}")
            with col4:
                st.metric("Test R²", f"{test_r2:.3f}")
            
            # Visualizations
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Predicted vs Actual
            axes[0].scatter(y_test, y_test_pred, alpha=0.5, s=30)
            min_val = min(y_test.min(), y_test_pred.min())
            max_val = max(y_test.max(), y_test_pred.max())
            axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
            axes[0].set_xlabel('Actual Price', fontweight='bold')
            axes[0].set_ylabel('Predicted Price', fontweight='bold')
            axes[0].set_title('Predicted vs Actual', fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            
            # Residuals
            residuals = y_test - y_test_pred
            axes[1].scatter(y_test_pred, residuals, alpha=0.5, s=30)
            axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
            axes[1].set_xlabel('Predicted Price', fontweight='bold')
            axes[1].set_ylabel('Residuals', fontweight='bold')
            axes[1].set_title('Residual Plot', fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Feature importance (if applicable)
            if hasattr(st.session_state.model, 'feature_importances_'):
                st.markdown("### 5️⃣ Feature Importance")
                
                importances = st.session_state.model.feature_importances_
                feature_names = X.columns
                
                fig, ax = plt.subplots(figsize=(10, 5))
                indices = np.argsort(importances)[::-1]
                
                ax.bar(range(len(importances)), importances[indices], color='steelblue', edgecolor='black')
                ax.set_xticks(range(len(importances)))
                ax.set_xticklabels([feature_names[i] for i in indices], rotation=45)
                ax.set_xlabel('Features', fontweight='bold')
                ax.set_ylabel('Importance', fontweight='bold')
                ax.set_title('Feature Importance', fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                
                st.pyplot(fig)
            
            # Make predictions
            st.markdown("### 6️⃣ Make Predictions")
            
            st.write("**Enter house characteristics:**")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                input_sqft = st.number_input("Square Feet", 500, 5000, 2000)
                input_bedrooms = st.number_input("Bedrooms", 1, 6, 3)
            with col2:
                input_bathrooms = st.number_input("Bathrooms", 1, 4, 2)
                input_age = st.number_input("Age (years)", 0, 100, 10)
            with col3:
                input_location = st.slider("Location Score", 1.0, 10.0, 7.0)
            
            if st.button("💰 Predict Price"):
                input_data = np.array([[input_sqft, input_bedrooms, input_bathrooms, 
                                       input_age, input_location]])
                input_scaled = st.session_state.scaler.transform(input_data)
                predicted_price = st.session_state.model.predict(input_scaled)[0]
                
                st.success(f"### Predicted Price: ${predicted_price:,.2f}")
                
                # Show confidence interval
                std_error = test_rmse
                lower_bound = predicted_price - 1.96 * std_error
                upper_bound = predicted_price + 1.96 * std_error
                
                st.write(f"**95% Confidence Interval:** ${lower_bound:,.2f} - ${upper_bound:,.2f}")
    
    elif "Customer Churn" in project_type:
        st.subheader("📱 Customer Churn Prediction")
        st.info("Classification project - predict which customers will leave")
        # Similar structure for classification...
        
    st.markdown("---")
    
    st.success("""
    🎉 **Congratulations!**
    
    You've completed the AI Visual Lab! You now understand:
    - Data preprocessing and exploration
    - Regression and classification
    - Neural networks and deep learning
    - CNNs for computer vision
    - Generative AI and transformers
    - Building complete AI systems
    
    Keep practicing and building! 🚀
    """)
    
    if st.button("🎉 Complete Course"):
        if 'completed' not in st.session_state:
            st.session_state.completed = set()
        st.session_state.completed.add("capstone")
        st.balloons()
        st.success("🏆 Course completed! You're now an AI practitioner!")
