import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def run():
    st.title("📊 Data Basics")
    
    st.markdown("""
    ## Understanding Data: The Foundation of AI
    
    Data is the fuel that powers AI models. Learn to explore, visualize, and understand data!
    """)
    
    # Interactive dataset generator
    st.subheader("🎲 Generate Your Dataset")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        n_samples = st.slider("Number of samples", 50, 500, 200)
    with col2:
        n_features = st.slider("Number of features", 2, 5, 3)
    with col3:
        noise_level = st.slider("Noise level", 0.0, 2.0, 0.5)
    
    # Generate sample data
    np.random.seed(42)
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    data_dict = {name: np.random.randn(n_samples) * (1 + noise_level) 
                 for name in feature_names}
    data_dict['target'] = np.random.choice(['A', 'B', 'C'], n_samples)
    data_dict['score'] = (np.random.randn(n_samples) * 10 + 50).clip(0, 100)
    
    data = pd.DataFrame(data_dict)
    
    # Display options
    st.subheader("📋 Dataset Preview")
    
    view_option = st.radio("View:", ["First 10 rows", "Last 10 rows", "Random sample", "All data"], horizontal=True)
    
    if view_option == "First 10 rows":
        st.dataframe(data.head(10), use_container_width=True)
    elif view_option == "Last 10 rows":
        st.dataframe(data.tail(10), use_container_width=True)
    elif view_option == "Random sample":
        st.dataframe(data.sample(10), use_container_width=True)
    else:
        st.dataframe(data, use_container_width=True)
    
    # Statistical summary
    st.markdown("---")
    st.subheader("📈 Statistical Analysis")
    
    tab1, tab2, tab3 = st.tabs(["📊 Summary Statistics", "📉 Distributions", "🔗 Correlations"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Descriptive Statistics**")
            st.dataframe(data.describe(), use_container_width=True)
            
        with col2:
            st.write("**Data Types & Missing Values**")
            info_df = pd.DataFrame({
                'Column': data.columns,
                'Type': data.dtypes.astype(str),
                'Non-Null': data.count(),
                'Null': data.isnull().sum(),
                'Unique': data.nunique()
            })
            st.dataframe(info_df, use_container_width=True)
        
        # Key metrics
        st.write("**Quick Insights**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", f"{len(data):,}")
        with col2:
            st.metric("Total Columns", len(data.columns))
        with col3:
            st.metric("Memory Usage", f"{data.memory_usage(deep=True).sum() / 1024:.1f} KB")
        with col4:
            st.metric("Duplicates", data.duplicated().sum())
    
    with tab2:
        st.write("**Feature Distributions**")
        
        selected_feature = st.selectbox("Select feature to visualize:", feature_names)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig, ax = plt.subplots(figsize=(8, 5))
            n, bins, patches = ax.hist(data[selected_feature], bins=30, 
                                       edgecolor='black', alpha=0.7, color='steelblue')
            
            # Add mean and median lines
            mean_val = data[selected_feature].mean()
            median_val = data[selected_feature].median()
            
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
            
            ax.set_xlabel(selected_feature, fontsize=11, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax.set_title(f'Distribution of {selected_feature}', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
        
        with col2:
            # Box plot
            fig, ax = plt.subplots(figsize=(8, 5))
            box = ax.boxplot([data[selected_feature]], labels=[selected_feature],
                            patch_artist=True, widths=0.5)
            
            for patch in box['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.7)
            
            ax.set_ylabel('Value', fontsize=11, fontweight='bold')
            ax.set_title(f'Box Plot of {selected_feature}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add statistics
            q1 = data[selected_feature].quantile(0.25)
            q3 = data[selected_feature].quantile(0.75)
            iqr = q3 - q1
            
            ax.text(1.3, q1, f'Q1: {q1:.2f}', va='center', fontsize=9)
            ax.text(1.3, median_val, f'Median: {median_val:.2f}', va='center', fontsize=9, fontweight='bold')
            ax.text(1.3, q3, f'Q3: {q3:.2f}', va='center', fontsize=9)
            
            st.pyplot(fig)
        
        # Statistical tests
        st.write("**Statistical Properties**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean", f"{data[selected_feature].mean():.3f}")
        with col2:
            st.metric("Std Dev", f"{data[selected_feature].std():.3f}")
        with col3:
            skewness = stats.skew(data[selected_feature])
            st.metric("Skewness", f"{skewness:.3f}")
        with col4:
            kurtosis = stats.kurtosis(data[selected_feature])
            st.metric("Kurtosis", f"{kurtosis:.3f}")
    
    with tab3:
        st.write("**Correlation Analysis**")
        
        # Correlation matrix
        numeric_data = data[feature_names + ['score']]
        corr_matrix = numeric_data.corr()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                       center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                       ax=ax)
            ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
            st.pyplot(fig)
        
        with col2:
            st.write("**Interpretation Guide**")
            st.markdown("""
            **Correlation values:**
            - `+1.0`: Perfect positive
            - `+0.7 to +1.0`: Strong positive
            - `+0.3 to +0.7`: Moderate positive
            - `-0.3 to +0.3`: Weak/No correlation
            - `-0.7 to -0.3`: Moderate negative
            - `-1.0 to -0.7`: Strong negative
            - `-1.0`: Perfect negative
            """)
            
            # Find strongest correlations
            st.write("**Strongest correlations:**")
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))
            
            corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            
            for feat1, feat2, corr in corr_pairs[:3]:
                st.write(f"• `{feat1}` ↔ `{feat2}`: **{corr:.3f}**")
    
    # Scatter plot matrix
    st.markdown("---")
    st.subheader("🔍 Pairwise Relationships")
    
    if len(feature_names) >= 2:
        feat_x = st.selectbox("X-axis feature:", feature_names, index=0)
        feat_y = st.selectbox("Y-axis feature:", feature_names, index=1 if len(feature_names) > 1 else 0)
        color_by = st.selectbox("Color by:", ['None', 'target'], index=0)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if color_by == 'None':
            ax.scatter(data[feat_x], data[feat_y], alpha=0.6, s=50, 
                      c='steelblue', edgecolors='black', linewidth=0.5)
        else:
            categories = data[color_by].unique()
            colors = plt.cm.Set1(np.linspace(0, 1, len(categories)))
            
            for i, category in enumerate(categories):
                mask = data[color_by] == category
                ax.scatter(data.loc[mask, feat_x], data.loc[mask, feat_y],
                          alpha=0.6, s=50, c=[colors[i]], label=category,
                          edgecolors='black', linewidth=0.5)
            ax.legend(title=color_by)
        
        ax.set_xlabel(feat_x, fontsize=11, fontweight='bold')
        ax.set_ylabel(feat_y, fontsize=11, fontweight='bold')
        ax.set_title(f'{feat_x} vs {feat_y}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Data preprocessing demo
    st.subheader("🧹 Data Preprocessing")
    
    preprocess_option = st.multiselect(
        "Apply preprocessing:",
        ["Standardization (Z-score)", "Normalization (Min-Max)", "Log Transform", "Remove Outliers"]
    )
    
    if preprocess_option:
        processed_data = data.copy()
        
        for option in preprocess_option:
            if option == "Standardization (Z-score)":
                for col in feature_names:
                    processed_data[col] = (data[col] - data[col].mean()) / data[col].std()
                st.info("✓ Standardized: Mean=0, Std=1")
                
            elif option == "Normalization (Min-Max)":
                for col in feature_names:
                    processed_data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
                st.info("✓ Normalized: Range=[0, 1]")
                
            elif option == "Log Transform":
                for col in feature_names:
                    processed_data[col] = np.log1p(data[col] - data[col].min() + 1)
                st.info("✓ Log transformed: Reduces skewness")
                
            elif option == "Remove Outliers":
                for col in feature_names:
                    q1 = data[col].quantile(0.25)
                    q3 = data[col].quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    processed_data = processed_data[(processed_data[col] >= lower) & (processed_data[col] <= upper)]
                st.info(f"✓ Outliers removed: {len(data) - len(processed_data)} rows filtered")
        
        # Show before/after comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Before Preprocessing**")
            st.dataframe(data[feature_names].head(), use_container_width=True)
        
        with col2:
            st.write("**After Preprocessing**")
            st.dataframe(processed_data[feature_names].head(), use_container_width=True)
    
    st.markdown("---")
    st.success("""
    💡 **Key Takeaways:**
    - Always explore your data before modeling
    - Check for missing values and outliers
    - Understand feature distributions and correlations
    - Preprocess data appropriately for your model
    """)
    
    if st.button("✅ Mark Module Complete"):
        if 'completed' not in st.session_state:
            st.session_state.completed = set()
        st.session_state.completed.add("data_basics")
        st.balloons()
        st.success("Module completed! Continue to Regression ➡️")
