"""
Streamlit Web Application for Wine Quality Classification
ML Assignment 2 - BITS Pilani
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)
import seaborn as sns
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Wine Quality Classification",
    page_icon="🍷",
    layout="wide"
)

# Title
st.title("🍷 Wine Quality Classification System")
st.markdown("**ML Assignment 2 - Multiple Classification Models**")
st.markdown("---")

# Sidebar
st.sidebar.header("📊 Model Selection")
st.sidebar.markdown("Choose a classification model to evaluate:")

# Model selection
model_options = {
    'Logistic Regression': 'model/logistic_regression.pkl',
    'Decision Tree': 'model/decision_tree.pkl',
    'K-Nearest Neighbor': 'model/k_nearest_neighbor.pkl',
    'Naive Bayes': 'model/naive_bayes.pkl',
    'Random Forest': 'model/random_forest.pkl',
    'XGBoost': 'model/xgboost.pkl'
}

selected_model = st.sidebar.selectbox(
    "Select Model:",
    list(model_options.keys())
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About Dataset")
st.sidebar.info("""
**Wine Quality Dataset (UCI)**
- Features: 11 physicochemical properties
- Target: Quality classification (Good/Bad)
- Good Wine: Quality ≥ 6
- Bad Wine: Quality < 6
""")

# Load scaler
@st.cache_resource
def load_scaler():
    try:
        with open('model/scaler.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return None

# Load model
@st.cache_resource
def load_model(model_path):
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Main content
tab1, tab2, tab3 = st.tabs(["📤 Upload Data", "📈 Model Evaluation", "ℹ️ Info"])

with tab1:
    st.header("Upload Test Dataset")
    st.markdown("Upload a CSV file with wine quality features for classification.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Load data
            data = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! Shape: {data.shape}")
            
            # Display data
            st.subheader("Dataset Preview")
            st.dataframe(data.head(10), use_container_width=True)
            
            # Check for target column
            if 'quality_class' in data.columns:
                X = data.drop(['quality_class'], axis=1)
                y = data['quality_class']
                has_labels = True
            elif 'quality' in data.columns:
                X = data.drop(['quality'], axis=1)
                y = (data['quality'] >= 6).astype(int)
                has_labels = True
            else:
                X = data
                has_labels = False
                st.warning("⚠️ No target column found. Predictions only (no evaluation metrics).")
            
            # Store in session state
            st.session_state['X'] = X
            st.session_state['y'] = y if has_labels else None
            st.session_state['has_labels'] = has_labels
            
        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        st.info("👆 Please upload a CSV file to begin evaluation.")

with tab2:
    st.header(f"Model Evaluation: {selected_model}")
    
    if 'X' in st.session_state:
        X = st.session_state['X']
        y = st.session_state['y']
        has_labels = st.session_state['has_labels']
        
        # Load model and scaler
        model = load_model(model_options[selected_model])
        scaler = load_scaler()
        
        if model is not None:
            # Prepare features
            if selected_model in ['Logistic Regression', 'K-Nearest Neighbor'] and scaler is not None:
                X_prepared = scaler.transform(X)
            else:
                X_prepared = X
            
            # Make predictions
            try:
                y_pred = model.predict(X_prepared)
                y_pred_proba = model.predict_proba(X_prepared)[:, 1]
                
                # Display predictions
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Prediction Distribution")
                    pred_counts = pd.Series(y_pred).value_counts()
                    fig, ax = plt.subplots(figsize=(6, 4))
                    pred_counts.plot(kind='bar', ax=ax, color=['#ff6b6b', '#4ecdc4'])
                    ax.set_xlabel('Class (0=Bad, 1=Good)')
                    ax.set_ylabel('Count')
                    ax.set_title('Predicted Class Distribution')
                    plt.xticks(rotation=0)
                    st.pyplot(fig)
                
                with col2:
                    st.subheader("Prediction Stats")
                    st.metric("Total Predictions", len(y_pred))
                    st.metric("Bad Wine (0)", int(sum(y_pred == 0)))
                    st.metric("Good Wine (1)", int(sum(y_pred == 1)))
                
                # Show evaluation metrics if labels available
                if has_labels and y is not None:
                    st.markdown("---")
                    st.subheader("📊 Evaluation Metrics")
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y, y_pred)
                    auc = roc_auc_score(y, y_pred_proba)
                    precision = precision_score(y, y_pred)
                    recall = recall_score(y, y_pred)
                    f1 = f1_score(y, y_pred)
                    mcc = matthews_corrcoef(y, y_pred)
                    
                    # Display metrics in columns
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Accuracy", f"{accuracy:.4f}")
                    col1.metric("Precision", f"{precision:.4f}")
                    col2.metric("AUC Score", f"{auc:.4f}")
                    col2.metric("Recall", f"{recall:.4f}")
                    col3.metric("F1 Score", f"{f1:.4f}")
                    col3.metric("MCC", f"{mcc:.4f}")
                    
                    st.markdown("---")
                    
                    # Confusion Matrix
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Confusion Matrix")
                        cm = confusion_matrix(y, y_pred)
                        fig, ax = plt.subplots(figsize=(6, 5))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                                   xticklabels=['Bad (0)', 'Good (1)'],
                                   yticklabels=['Bad (0)', 'Good (1)'])
                        ax.set_ylabel('Actual')
                        ax.set_xlabel('Predicted')
                        ax.set_title('Confusion Matrix')
                        st.pyplot(fig)
                    
                    with col2:
                        st.subheader("Classification Report")
                        report = classification_report(y, y_pred, 
                                                      target_names=['Bad Wine', 'Good Wine'],
                                                      output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)
                    
                else:
                    st.info("Upload data with labels (quality_class or quality column) to see evaluation metrics.")
                
            except Exception as e:
                st.error(f"Error making predictions: {e}")
        else:
            st.error("Failed to load model. Please check if model files exist.")
    else:
        st.warning("⚠️ Please upload a dataset in the 'Upload Data' tab first.")

with tab3:
    st.header("ℹ️ Application Information")
    
    st.markdown("""
    ### Wine Quality Classification System
    
    This application demonstrates 6 different machine learning classification models trained on the UCI Wine Quality dataset.
    
    #### 🎯 Models Implemented:
    1. **Logistic Regression** - Linear probabilistic classifier
    2. **Decision Tree** - Tree-based rule learning
    3. **K-Nearest Neighbor** - Instance-based learning
    4. **Naive Bayes** - Probabilistic classifier based on Bayes' theorem
    5. **Random Forest** - Ensemble of decision trees
    6. **XGBoost** - Gradient boosting ensemble method
    
    #### 📊 Evaluation Metrics:
    - **Accuracy**: Overall correctness of predictions
    - **AUC**: Area Under ROC Curve
    - **Precision**: Accuracy of positive predictions
    - **Recall**: Coverage of actual positives
    - **F1 Score**: Harmonic mean of precision and recall
    - **MCC**: Matthews Correlation Coefficient
    
    #### 🗂️ Dataset Features:
    The dataset contains 11 physicochemical properties:
    - Fixed acidity
    - Volatile acidity
    - Citric acid
    - Residual sugar
    - Chlorides
    - Free sulfur dioxide
    - Total sulfur dioxide
    - Density
    - pH
    - Sulphates
    - Alcohol
    
    #### 🎓 Assignment Details:
    - **Course**: Machine Learning
    - **Institution**: BITS Pilani
    - **Assignment**: ML Assignment 2
    """)
    
    st.markdown("---")
    st.markdown("**Note**: For best results, ensure your CSV has the same features as the training data.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>ML Assignment 2 - BITS Pilani | Wine Quality Classification</p>
</div>
""", unsafe_allow_html=True)
