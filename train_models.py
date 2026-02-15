"""
ML Assignment 2 - Classification Models Implementation
Train and evaluate 6 classification models on Wine Quality Dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)
import pickle
import warnings
import os
warnings.filterwarnings('ignore')

# Resolve project root and model directory robustly
script_dir = Path(__file__).resolve().parent
repo_root = script_dir if script_dir.name != 'model' else script_dir.parent
model_dir = repo_root / 'model'
model_dir.mkdir(parents=True, exist_ok=True)

# Load and prepare dataset
print("Loading Wine Quality Dataset...")
# Try downloading from UCI repository, if fails use alternative source
try:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    data = pd.read_csv(url, sep=';')
except:
    print("UCI source failed, trying alternative source...")
    # Alternative: raw GitHub dataset
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/WineQuality-Red.csv"
    data = pd.read_csv(url)

print(f"Dataset shape: {data.shape}")
print(f"Features: {data.columns.tolist()}")

# Convert quality to binary classification (good wine >= 6, bad wine < 6)
data['quality_class'] = (data['quality'] >= 6).astype(int)
X = data.drop(['quality', 'quality_class'], axis=1)
y = data['quality_class']

print(f"\nClass distribution:\n{y.value_counts()}")

# HANDLE OUTLIERS
print("\n" + "="*80)
print("OUTLIER DETECTION & REMOVAL")
print("="*80)

# Remove extreme outliers using IQR method
from scipy import stats
X_numeric = X.copy()
initial_samples = len(X_numeric)

z_scores = np.abs(stats.zscore(X_numeric))
outlier_mask = (z_scores < 3).all(axis=1)  # Keep if Z-score < 3
X = X[outlier_mask]
y = y[outlier_mask]

print(f"Initial samples: {initial_samples}")
print(f"Outliers removed: {initial_samples - len(X)}")
print(f"Remaining samples: {len(X)}")

# ADVANCED FEATURE ENGINEERING
print("\n" + "="*80)
print("ADVANCED FEATURE ENGINEERING")
print("="*80)

# 1. Feature Interactions
print("\n1. Adding feature interactions...")
X['acidity_ratio'] = X['fixed acidity'] / (X['volatile acidity'] + 0.01)
X['sulfur_ratio'] = X['free sulfur dioxide'] / (X['total sulfur dioxide'] + 0.01)
X['acid_score'] = X['fixed acidity'] - X['volatile acidity']
X['alcohol_density'] = X['alcohol'] * X['density']
X['pH_alcohol'] = X['pH'] * X['alcohol']
X['chlorides_sulfates'] = X['chlorides'] * X['sulphates']

# Additional interactions
X['quality_indicators'] = (X['alcohol'] * X['sulphates']) / (X['volatile acidity'] + 0.01)
X['balanced_acidity'] = X['citric acid'] / (X['fixed acidity'] + 0.01)
X['sulfur_impact'] = X['total sulfur dioxide'] - X['free sulfur dioxide']

print(f"   Added 9 interaction features")

# 2. Log transformations for skewed features
print("\n2. Applying log transformations to skewed features...")
skewed_cols = ['residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide']
for col in skewed_cols:
    X[f'{col}_log'] = np.log1p(X[col])
    X[f'{col}_sqrt'] = np.sqrt(X[col])  # Square root transform

# 3. Polynomial features (degree 2) for key features
print("\n3. Adding polynomial features for key predictors...")
key_features = ['alcohol', 'density', 'pH', 'sulphates', 'volatile acidity']
for col in key_features:
    X[f'{col}_squared'] = X[col] ** 2
    X[f'{col}_cubed'] = X[col] ** 3  # Cubic for extra non-linearity

# 4. Normalization (0-1 scale)
print("\n4. Normalizing features to [0, 1] range...")
from sklearn.preprocessing import MinMaxScaler
minmax_scaler = MinMaxScaler()
normalized_features = minmax_scaler.fit_transform(X)
X_normalized = pd.DataFrame(normalized_features, columns=X.columns, index=X.index)
X = X_normalized

print(f"   Total engineered features: {X.shape[1]}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
with open(model_dir / 'scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# HYPERPARAMETER TUNING
print("\n" + "="*80)
print("HYPERPARAMETER TUNING")
print("="*80)

# Define models with optimized hyperparameters and class weights
models = {
    'Logistic Regression': LogisticRegression(
        max_iter=2000, 
        C=0.05,
        solver='lbfgs',
        class_weight='balanced',
        random_state=42
    ),
    'Decision Tree': DecisionTreeClassifier(
        max_depth=9,
        min_samples_split=8,
        min_samples_leaf=4,
        class_weight='balanced',
        random_state=42
    ),
    'K-Nearest Neighbor': KNeighborsClassifier(
        n_neighbors=5,
        weights='distance',
        metric='minkowski',
        p=2
    ),
    'Naive Bayes': GaussianNB(
        var_smoothing=1e-10
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=300,
        max_depth=13,
        min_samples_split=7,
        min_samples_leaf=3,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    )
}

# Store results
results = []

print("\n" + "="*80)
print("TRAINING AND EVALUATING MODELS")
print("="*80)

for name, model in models.items():
    print(f"\n{name}:")
    print("-" * 40)
    
    # Train model
    if name in ['Logistic Regression', 'K-Nearest Neighbor']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"AUC:       {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"MCC:       {mcc:.4f}")
    
    # Save model
    model_filename = model_dir / f"{name.lower().replace(' ', '_').replace('-', '_')}.pkl"
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved: {model_filename}")
    
    # Store results
    results.append({
        'Model': name,
        'Accuracy': f"{accuracy:.4f}",
        'AUC': f"{auc:.4f}",
        'Precision': f"{precision:.4f}",
        'Recall': f"{recall:.4f}",
        'F1 Score': f"{f1:.4f}",
        'MCC': f"{mcc:.4f}"
    })

# Create results DataFrame
results_df = pd.DataFrame(results)
print("\n" + "="*80)
print("SUMMARY OF ALL MODELS")
print("="*80)
print(results_df.to_string(index=False))

# Save results to CSV
results_df.to_csv(model_dir / 'model_results.csv', index=False)
print("\n\nResults saved to model/model_results.csv")

# Save test data for Streamlit app
test_data = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
test_data.to_csv(model_dir / 'test_data.csv', index=False)
print("Test data saved to model/test_data.csv")

print("\n✓ All models trained and saved successfully!")
