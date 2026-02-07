import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from matplotlib.colors import ListedColormap


def run():
    st.title("🎯 Classification")
    
    st.markdown("""
    ## Predicting Categories
    
    Classification models assign data points to discrete categories. Let's explore!
    """)
    
    # Interactive parameters
    st.sidebar.subheader("🎲 Dataset Configuration")
    
    dataset_type = st.sidebar.selectbox(
        "Dataset Pattern:",
        ["Linear Separable", "Moons", "Circles", "Random"]
    )
    
    n_samples = st.sidebar.slider("Number of Samples", 100, 800, 300, 50)
    noise_level = st.sidebar.slider("Noise Level", 0.0, 0.5, 0.1, 0.05)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 Model Selection")
    
    model_type = st.sidebar.selectbox(
        "Algorithm:",
        ["Logistic Regression", "Decision Tree", "Random Forest", 
         "Support Vector Machine", "K-Nearest Neighbors"]
    )
    
    # Model-specific parameters
    if model_type == "Decision Tree":
        max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
    elif model_type == "Random Forest":
        n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100, 10)
        max_depth = st.sidebar.slider("Max Depth", 1, 20, 10)
    elif model_type == "Support Vector Machine":
        kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly"])
        C = st.sidebar.slider("C (Regularization)", 0.1, 10.0, 1.0, 0.1)
    elif model_type == "K-Nearest Neighbors":
        n_neighbors = st.sidebar.slider("Number of Neighbors (k)", 1, 20, 5)
    
    # Generate data based on pattern
    np.random.seed(42)
    
    if dataset_type == "Linear Separable":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            n_clusters_per_class=1,
            class_sep=2.0,
            random_state=42
        )
        if noise_level > 0:
            X += np.random.randn(*X.shape) * noise_level
            
    elif dataset_type == "Moons":
        X, y = make_moons(n_samples=n_samples, noise=noise_level, random_state=42)
        
    elif dataset_type == "Circles":
        X, y = make_circles(n_samples=n_samples, noise=noise_level, factor=0.5, random_state=42)
        
    else:  # Random
        X, y = make_classification(
            n_samples=n_samples,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            n_clusters_per_class=2,
            random_state=42
        )
        X += np.random.randn(*X.shape) * noise_level
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_scaled = scaler.transform(X)
    
    # Train model
    if model_type == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "Decision Tree":
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    elif model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    elif model_type == "Support Vector Machine":
        model = SVC(kernel=kernel, C=C, random_state=42)
    else:  # K-Nearest Neighbors
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    # Fit model
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Main visualization - Decision Boundary
    st.subheader("🗺️ Decision Boundary Visualization")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create mesh grid
    h = 0.02  # step size in mesh
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict on mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ['#FF0000', '#0000FF']
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
    
    # Plot training points
    train_scatter = ax.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], 
                               c=y_train, cmap=ListedColormap(cmap_bold),
                               s=80, alpha=0.6, edgecolors='black', linewidth=1,
                               marker='o', label='Training Data')
    
    # Plot test points
    test_scatter = ax.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], 
                              c=y_test, cmap=ListedColormap(cmap_bold),
                              s=100, alpha=0.9, edgecolors='yellow', linewidth=2,
                              marker='s', label='Test Data')
    
    ax.set_xlabel('Feature 1 (scaled)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature 2 (scaled)', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_type} - Decision Boundary', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Performance metrics
    st.markdown("---")
    st.subheader("📊 Model Performance Metrics")
    
    # Calculate all metrics
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    precision = precision_score(y_test, y_pred_test, average='weighted')
    recall = recall_score(y_test, y_pred_test, average='weighted')
    f1 = f1_score(y_test, y_pred_test, average='weighted')
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Test Accuracy", f"{test_accuracy:.2%}", 
                 delta=f"{(test_accuracy - train_accuracy):.2%}" if train_accuracy < test_accuracy else None)
    with col2:
        st.metric("Precision", f"{precision:.2%}")
    with col3:
        st.metric("Recall", f"{recall:.2%}")
    with col4:
        st.metric("F1-Score", f"{f1:.2%}")
    
    # Training vs Test comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Training Set**")
        st.metric("Accuracy", f"{train_accuracy:.2%}")
        st.metric("Samples", len(y_train))
        
    with col2:
        st.write("**Test Set**")
        st.metric("Accuracy", f"{test_accuracy:.2%}")
        st.metric("Samples", len(y_test))
    
    # Overfitting indicator
    if train_accuracy - test_accuracy > 0.1:
        st.warning("⚠️ **Possible Overfitting**: Training accuracy significantly higher than test accuracy!")
    elif test_accuracy > 0.9:
        st.success("✅ **Excellent Performance**: Model generalizes very well!")
    elif test_accuracy > 0.7:
        st.info("ℹ️ **Good Performance**: Model performs reasonably well.")
    else:
        st.error("❌ **Poor Performance**: Consider different model or parameters.")
    
    # Confusion Matrix
    st.markdown("---")
    st.subheader("🔢 Confusion Matrix")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Test Set Confusion Matrix**")
        cm = confusion_matrix(y_test, y_pred_test)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
        
        st.pyplot(fig)
        
        # Calculate metrics from confusion matrix
        tn, fp, fn, tp = cm.ravel()
        
        st.write("**Matrix Breakdown:**")
        st.write(f"- True Positives (TP): {tp}")
        st.write(f"- True Negatives (TN): {tn}")
        st.write(f"- False Positives (FP): {fp}")
        st.write(f"- False Negatives (FN): {fn}")
    
    with col2:
        st.write("**Classification Report**")
        report = classification_report(y_test, y_pred_test, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
        
        # Visualize precision, recall, f1
        fig, ax = plt.subplots(figsize=(6, 5))
        
        metrics_data = report_df.loc[['0', '1'], ['precision', 'recall', 'f1-score']]
        metrics_data.plot(kind='bar', ax=ax, rot=0, width=0.8)
        
        ax.set_xlabel('Class', fontsize=11, fontweight='bold')
        ax.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax.set_title('Per-Class Metrics', fontsize=12, fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.legend(['Precision', 'Recall', 'F1-Score'], loc='lower right')
        ax.grid(True, alpha=0.3, axis='y')
        
        st.pyplot(fig)
    
    # Prediction probabilities (if available)
    if hasattr(model, 'predict_proba'):
        st.markdown("---")
        st.subheader("📈 Prediction Confidence")
        
        y_proba = model.predict_proba(X_test_scaled)
        
        # Show confidence distribution
        fig, ax = plt.subplots(figsize=(10, 5))
        
        confidence = np.max(y_proba, axis=1)
        correct = y_pred_test == y_test
        
        ax.hist(confidence[correct], bins=20, alpha=0.7, color='green', 
               edgecolor='black', label='Correct Predictions')
        ax.hist(confidence[~correct], bins=20, alpha=0.7, color='red', 
               edgecolor='black', label='Incorrect Predictions')
        
        ax.set_xlabel('Prediction Confidence', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title('Model Confidence Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        avg_confidence = confidence.mean()
        st.metric("Average Confidence", f"{avg_confidence:.2%}")
    
    # Educational content
    st.markdown("---")
    st.subheader("📚 Understanding Classification Metrics")
    
    tab1, tab2, tab3 = st.tabs(["📊 Metrics Guide", "🎯 Use Cases", "💡 Tips"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Accuracy**
            - Overall correctness
            - (TP + TN) / Total
            - Good for balanced datasets
            
            **Precision**
            - "Of all positive predictions, how many were correct?"
            - TP / (TP + FP)
            - Important when false positives are costly
            
            **Recall (Sensitivity)**
            - "Of all actual positives, how many did we find?"
            - TP / (TP + FN)
            - Important when false negatives are costly
            """)
        
        with col2:
            st.markdown("""
            **F1-Score**
            - Harmonic mean of precision and recall
            - 2 × (Precision × Recall) / (Precision + Recall)
            - Good balance for imbalanced datasets
            
            **Confusion Matrix**
            - Shows all prediction outcomes
            - Reveals where model makes mistakes
            - Essential for understanding model behavior
            
            **ROC-AUC** (not shown)
            - Trade-off between true/false positive rates
            - Single score summarizing performance
            """)
    
    with tab2:
        st.markdown("""
        ### 🏥 Medical Diagnosis
        - **High Recall** needed: Don't miss sick patients (minimize false negatives)
        - Example: Cancer screening
        
        ### 📧 Spam Detection
        - **High Precision** needed: Don't mark important emails as spam (minimize false positives)
        - Example: Email filters
        
        ### 🔒 Fraud Detection
        - **Balance** needed: Catch fraudsters but don't block legitimate transactions
        - Example: Credit card fraud
        
        ### 🎯 Customer Churn
        - **High Recall** preferred: Identify all at-risk customers
        - Example: Subscription services
        """)
    
    with tab3:
        st.success("""
        **🎓 Best Practices:**
        
        1. **Always split your data** into train/test sets
        2. **Check for class imbalance** - affects metric interpretation
        3. **Use multiple metrics** - accuracy alone can be misleading
        4. **Visualize decision boundaries** - understand model behavior
        5. **Cross-validate** for more robust performance estimates
        6. **Try different algorithms** - no single best choice
        7. **Tune hyperparameters** - can significantly improve performance
        8. **Watch for overfitting** - train vs test performance gap
        """)
    
    st.info("""
    💡 **Experiment:**
    - Try different dataset patterns (Moons, Circles)
    - Compare algorithms on the same data
    - Observe how noise affects performance
    - Tune hyperparameters for better results
    """)
    
    if st.button("✅ Mark Module Complete"):
        if 'completed' not in st.session_state:
            st.session_state.completed = set()
        st.session_state.completed.add("classification")
        st.balloons()
        st.success("Module completed! Continue to Neural Networks ➡️")


import pandas as pd
