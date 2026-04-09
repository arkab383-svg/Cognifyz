# ============================================================
#  Task 1 – Predict Restaurant Ratings
#  Cognifyz ML Internship
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # For saving plots without GUI
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ─────────────────────────────────────────────────────────────
# STEP 1 – Load Dataset
# ─────────────────────────────────────────────────────────────
file_path = "/Users/arkabose/Cognifyz/Task 1/Dataset .csv"

try:
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    print(f"✅ Dataset loaded successfully!")
except FileNotFoundError:
    print("❌ ERROR: Dataset file not found.")
    print("Check your file path:", file_path)
    exit()

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

print(f"Dataset shape: {df.shape}")

# Check required columns
required_cols = ['Aggregate rating', 'Votes', 'Price range']
for col in required_cols:
    if col not in df.columns:
        print(f"❌ ERROR: Missing column -> {col}")
        print("Available columns:", df.columns.tolist())
        exit()

print(df[required_cols].describe())

# ─────────────────────────────────────────────────────────────
# STEP 2 – Preprocessing
# ─────────────────────────────────────────────────────────────

# Remove unrated restaurants
df = df[df['Aggregate rating'] > 0].copy()
print(f"\nAfter removing unrated: {df.shape}")

# Fill missing values safely
df['Cuisines'] = df['Cuisines'].fillna('Unknown')
df['City'] = df['City'].fillna('Unknown')

# Some datasets may have missing numeric values
numeric_fill_cols = ['Average Cost for two', 'Votes', 'Price range', 'Country Code']
for col in numeric_fill_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# Encode binary Yes/No columns safely
binary_cols = ['Has Table booking', 'Has Online delivery',
               'Is delivering now', 'Switch to order menu']

for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].fillna('No')
        df[col] = (df[col] == 'Yes').astype(int)
    else:
        print(f"⚠️ Column '{col}' not found. Filling with 0.")
        df[col] = 0

# Label Encode City
le_city = LabelEncoder()
df['City_enc'] = le_city.fit_transform(df['City'].astype(str))

# Extract primary cuisine
df['Primary Cuisine'] = df['Cuisines'].astype(str).str.split(',').str[0].str.strip()

# Label Encode Primary Cuisine
le_cuis = LabelEncoder()
df['Cuisine_enc'] = le_cuis.fit_transform(df['Primary Cuisine'].astype(str))

# ─────────────────────────────────────────────────────────────
# STEP 3 – Feature Selection & Train/Test Split
# ─────────────────────────────────────────────────────────────
features = [
    'Country Code',
    'Average Cost for two',
    'Price range',
    'Has Table booking',
    'Has Online delivery',
    'Is delivering now',
    'Votes',
    'City_enc',
    'Cuisine_enc'
]

# Check all features exist
for col in features:
    if col not in df.columns:
        print(f"❌ ERROR: Missing feature column -> {col}")
        exit()

X = df[features]
y = df['Aggregate rating']

# Final NaN check
if X.isnull().sum().sum() > 0:
    print("❌ ERROR: There are still missing values in X")
    print(X.isnull().sum())
    exit()

if y.isnull().sum() > 0:
    print("❌ ERROR: Missing values found in target y")
    exit()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTrain size : {X_train.shape}")
print(f"Test size  : {X_test.shape}")

# ─────────────────────────────────────────────────────────────
# STEP 4 – Train Models
# ─────────────────────────────────────────────────────────────
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(max_depth=8, random_state=42),
    'Random Forest': RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
}

results = {}

print("\n── Model Performance ──────────────────────────────────────")
print(f"{'Model':<22} | {'RMSE':>8} {'MAE':>8} {'R²':>8}")
print("-" * 55)

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    results[name] = {
        'MSE': round(mse, 4),
        'RMSE': round(rmse, 4),
        'MAE': round(mae, 4),
        'R²': round(r2, 4),
        'model': model,
        'preds': preds
    }

    print(f"{name:<22} | {rmse:>8.4f} {mae:>8.4f} {r2:>8.4f}")

# ─────────────────────────────────────────────────────────────
# STEP 5 – Feature Importance (Random Forest)
# ─────────────────────────────────────────────────────────────
rf_model = results['Random Forest']['model']

importances = pd.Series(
    rf_model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\n── Feature Importances (Random Forest) ───────────────────")
print(importances.to_string())

# ─────────────────────────────────────────────────────────────
# STEP 6 – Visualisations
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Restaurant Rating Prediction – Model Analysis',
             fontsize=16, fontweight='bold')

# 6a. Rating distribution
axes[0, 0].hist(df['Aggregate rating'], bins=25,
                color='steelblue', edgecolor='white')
axes[0, 0].set_title('Rating Distribution')
axes[0, 0].set_xlabel('Aggregate Rating')
axes[0, 0].set_ylabel('Count')

# 6b. Feature importance
importances.plot(kind='barh', ax=axes[0, 1], color='coral')
axes[0, 1].set_title('Feature Importance (Random Forest)')
axes[0, 1].invert_yaxis()

# 6c. R² comparison
r2s = {k: v['R²'] for k, v in results.items()}
colors_bar = ['royalblue', 'darkorange', 'forestgreen']
bars = axes[0, 2].bar(list(r2s.keys()), list(r2s.values()), color=colors_bar)
axes[0, 2].set_title('R² Score Comparison')
axes[0, 2].set_ylabel('R²')

for bar in bars:
    axes[0, 2].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f'{bar.get_height():.3f}',
        ha='center',
        fontsize=10
    )

# 6d-f. Actual vs Predicted
scatter_colors = ['royalblue', 'darkorange', 'forestgreen']

for i, (name, res) in enumerate(results.items()):
    ax = axes[1, i]
    ax.scatter(y_test, res['preds'], alpha=0.3, s=10, color=scatter_colors[i])
    ax.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        'r--',
        lw=1.5,
        label='Perfect fit'
    )
    ax.set_title(f'{name}\nR²={res["R²"]}  RMSE={res["RMSE"]}')
    ax.set_xlabel('Actual Rating')
    ax.set_ylabel('Predicted Rating')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('model_analysis.png', dpi=150, bbox_inches='tight')
print("\n✅ Plot saved as model_analysis.png")

# ─────────────────────────────────────────────────────────────
# STEP 7 – Interpretation / Key Findings
# ─────────────────────────────────────────────────────────────
best_model = max(results, key=lambda x: results[x]['R²'])

print(f"""
── Key Findings ───────────────────────────────────────────────
• Best model  : {best_model}
• Votes is often one of the strongest predictors of rating.
• Tree-based models usually perform better than Linear Regression
  because restaurant ratings often have non-linear patterns.
• Features like city, cuisine, and cost can influence ratings,
  but their importance depends on the dataset.
───────────────────────────────────────────────────────────────
""")