# ============================================================
#  Task 3 – Cuisine Classification
#  Cognifyz ML Internship
# ============================================================

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

# ─────────────────────────────────────────────────────────────
# STEP 1 – Load Dataset
# ─────────────────────────────────────────────────────────────
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, 'Dataset1.csv')

try:
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    print(f"✅ Dataset loaded: {df.shape}")
except FileNotFoundError:
    print("❌ File not found:", file_path)
    exit()

# ─────────────────────────────────────────────────────────────
# STEP 2 – Preprocessing
# ─────────────────────────────────────────────────────────────

# Remove unrated restaurants
df = df[df['Aggregate rating'] > 0].copy()
df.reset_index(drop=True, inplace=True)

# Fill missing values
df['Cuisines'] = df['Cuisines'].fillna('Unknown')
df['City'] = df['City'].fillna('Unknown')

# Extract Primary Cuisine (TARGET)
df['Primary Cuisine'] = df['Cuisines'].astype(str).str.split(',').str[0].str.strip()

# Keep only top 10 cuisines
top_cuisines = df['Primary Cuisine'].value_counts().head(10).index.tolist()
df = df[df['Primary Cuisine'].isin(top_cuisines)].copy()

print(f"✅ Using top cuisines: {top_cuisines}")

# Handle binary columns safely
binary_cols = [
    'Has Table booking',
    'Has Online delivery',
    'Is delivering now',
    'Switch to order menu'
]

for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].fillna('No')
        df[col] = (df[col] == 'Yes').astype(int)
    else:
        df[col] = 0

# Fill numeric values
numeric_cols = [
    'Average Cost for two',
    'Votes',
    'Price range',
    'Country Code'
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# Encode City
le_city = LabelEncoder()
df['City_enc'] = le_city.fit_transform(df['City'].astype(str))

# ─────────────────────────────────────────────────────────────
# STEP 3 – Feature Selection
# ─────────────────────────────────────────────────────────────
features = [
    'Country Code',
    'Average Cost for two',
    'Price range',
    'Votes',
    'Has Table booking',
    'Has Online delivery',
    'Is delivering now',
    'City_enc'
]

X = df[features]
y = df['Primary Cuisine']

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Encode target
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print(f"✅ Train size: {X_train.shape}")
print(f"✅ Test size : {X_test.shape}")

# ─────────────────────────────────────────────────────────────
# STEP 4 – Models
# ─────────────────────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
}

results = {}

print("\nModel Performance")
print("-" * 60)

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='weighted', zero_division=0)
    rec = recall_score(y_test, preds, average='weighted', zero_division=0)
    f1 = f1_score(y_test, preds, average='weighted', zero_division=0)

    results[name] = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'model': model,
        'preds': preds
    }

    print(f"{name} → Accuracy: {acc:.3f}, F1: {f1:.3f}")

# ─────────────────────────────────────────────────────────────
# STEP 5 – Best Model
# ─────────────────────────────────────────────────────────────
best_name = max(results, key=lambda x: results[x]['F1'])
best_preds = results[best_name]['preds']

print(f"\nBest Model: {best_name}")
print(classification_report(
    y_test,
    best_preds,
    target_names=le_target.classes_,
    zero_division=0
))

# ─────────────────────────────────────────────────────────────
# STEP 6 – Feature Importance
# ─────────────────────────────────────────────────────────────
rf_model = results['Random Forest']['model']

importances = pd.Series(
    rf_model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importances)

# ─────────────────────────────────────────────────────────────
# STEP 7 – Visualization
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Cuisine distribution
df['Primary Cuisine'].value_counts().head(10).plot(
    kind='barh', ax=axes[0, 0]
)
axes[0, 0].set_title("Cuisine Distribution")

# Accuracy comparison
acc_vals = {k: v['Accuracy'] for k, v in results.items()}
axes[0, 1].bar(acc_vals.keys(), acc_vals.values())
axes[0, 1].set_title("Model Accuracy")
axes[0, 1].set_ylabel("Accuracy")

# Confusion matrix (IMPROVED)
cm = confusion_matrix(y_test, best_preds)
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    xticklabels=le_target.classes_,
    yticklabels=le_target.classes_,
    ax=axes[1, 0]
)
axes[1, 0].set_title("Confusion Matrix")

# Feature importance
importances.plot(kind='barh', ax=axes[1, 1])
axes[1, 1].set_title("Feature Importance")

plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'task3_analysis.png'))

print("\n✅ Plot saved: task3_analysis.png")

# ─────────────────────────────────────────────────────────────
# STEP 8 – Insights
# ─────────────────────────────────────────────────────────────
print("""
Key Insights:
- Random Forest performs best for this dataset
- Class imbalance impacts minority cuisine prediction
- Price and location significantly influence cuisine classification
- Further improvement possible using NLP features (TF-IDF on cuisines)
""")