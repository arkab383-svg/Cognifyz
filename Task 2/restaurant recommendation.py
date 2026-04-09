# ============================================================
#  Task 2 – Restaurant Recommendation System
#  Cognifyz ML Internship
#  Approach: Content-Based Filtering using Cosine Similarity
# ============================================================

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────
# STEP 1 – Load Dataset
# ─────────────────────────────────────────────────────────────
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, 'Dataset.csv')   # ✅ FIXED

try:
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    print(f"✅ Dataset loaded successfully: {df.shape}")
except FileNotFoundError:
    print("❌ ERROR: File not found:", file_path)
    exit()

# ─────────────────────────────────────────────────────────────
# STEP 2 – Preprocessing
# ─────────────────────────────────────────────────────────────

# Keep only rated restaurants
df = df[df['Aggregate rating'] > 0].copy()
df.reset_index(drop=True, inplace=True)

# Fill missing values
df['Cuisines'] = df['Cuisines'].fillna('Unknown')
df['City'] = df['City'].fillna('Unknown')
df['Votes'] = df['Votes'].fillna(0)

# Extract primary cuisine
df['Primary Cuisine'] = df['Cuisines'].astype(str).str.split(',').str[0].str.strip()

# Encode binary columns safely
binary_cols = ['Has Table booking', 'Has Online delivery']
for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].fillna('No')
        df[col] = (df[col] == 'Yes').astype(int)
    else:
        print(f"⚠️ Missing column: {col}. Filling with 0.")
        df[col] = 0

# Fill numeric missing values if any
numeric_cols = ['Price range', 'Aggregate rating', 'Votes']
for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# Label encode City and Cuisine
le_city = LabelEncoder()
le_cuis = LabelEncoder()

df['City_enc'] = le_city.fit_transform(df['City'].astype(str))
df['Cuisine_enc'] = le_cuis.fit_transform(df['Primary Cuisine'].astype(str))

# Features used for similarity
feature_cols = [
    'Cuisine_enc',
    'Price range',
    'Aggregate rating',
    'Has Table booking',
    'Has Online delivery',
    'City_enc'
]

# Normalize features to 0–1 scale
scaler = MinMaxScaler()
feature_matrix = scaler.fit_transform(df[feature_cols])

print(f"✅ Feature matrix shape: {feature_matrix.shape}")

# ─────────────────────────────────────────────────────────────
# STEP 3 – Build Cosine Similarity Matrix
# ─────────────────────────────────────────────────────────────
print("\n⏳ Computing similarity matrix...")
similarity_matrix = cosine_similarity(feature_matrix)
print(f"✅ Similarity matrix shape: {similarity_matrix.shape}")

# ─────────────────────────────────────────────────────────────
# STEP 4 – Preference-Based Recommendation Function
# ─────────────────────────────────────────────────────────────
def recommend_restaurants(user_preferences, top_n=5):
    """
    Recommend restaurants based on user preferences.
    """
    filtered = df.copy()

    # Filter by cuisine
    if 'cuisine' in user_preferences:
        cuisine_pref = user_preferences['cuisine'].lower()
        filtered = filtered[
            filtered['Primary Cuisine'].str.lower().str.contains(cuisine_pref, na=False)
        ]

    # Filter by minimum rating
    if 'min_rating' in user_preferences:
        filtered = filtered[filtered['Aggregate rating'] >= user_preferences['min_rating']]

    # Filter by price range
    if 'price_range' in user_preferences:
        filtered = filtered[filtered['Price range'] == user_preferences['price_range']]

    # Filter by online delivery
    if 'online_delivery' in user_preferences:
        val = 1 if user_preferences['online_delivery'] else 0
        filtered = filtered[filtered['Has Online delivery'] == val]

    # Filter by table booking
    if 'table_booking' in user_preferences:
        val = 1 if user_preferences['table_booking'] else 0
        filtered = filtered[filtered['Has Table booking'] == val]

    if filtered.empty:
        print("⚠️ No exact matches found. Relaxing some filters...")
        filtered = df.copy()

        if 'min_rating' in user_preferences:
            filtered = filtered[filtered['Aggregate rating'] >= user_preferences['min_rating']]

    # Weighted recommendation score
    max_votes = df['Votes'].max() if df['Votes'].max() != 0 else 1
    filtered = filtered.copy()
    filtered['Score'] = (
        filtered['Aggregate rating'] * 0.7 +
        (filtered['Votes'] / max_votes) * 0.3
    )

    top = filtered.nlargest(top_n, 'Score')

    result = top[[
        'Restaurant Name', 'City', 'Primary Cuisine',
        'Price range', 'Aggregate rating', 'Votes',
        'Has Online delivery', 'Has Table booking', 'Score'
    ]].reset_index(drop=True)

    result.index += 1
    return result

# ─────────────────────────────────────────────────────────────
# STEP 5 – Similarity-Based Recommendation Function
# ─────────────────────────────────────────────────────────────
def find_similar_restaurants(restaurant_name, top_n=5):
    """
    Find restaurants similar to a given restaurant using cosine similarity.
    """
    matches = df[df['Restaurant Name'].str.lower() == restaurant_name.lower()]

    if matches.empty:
        matches = df[df['Restaurant Name'].str.lower().str.contains(
            restaurant_name.lower(), na=False)]

    if matches.empty:
        print(f"❌ Restaurant '{restaurant_name}' not found in dataset.")
        return None

    idx = matches.index[0]

    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Remove itself
    sim_scores = [(i, s) for i, s in sim_scores if i != idx][:top_n]

    top_indices = [i for i, _ in sim_scores]
    scores = [round(s, 4) for _, s in sim_scores]

    result = df.iloc[top_indices][[
        'Restaurant Name', 'City', 'Primary Cuisine',
        'Price range', 'Aggregate rating', 'Votes'
    ]].copy()

    result['Similarity Score'] = scores
    result = result.reset_index(drop=True)
    result.index += 1

    return result

# ─────────────────────────────────────────────────────────────
# STEP 6 – Test the Recommendation System
# ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("TEST 1 – Italian food, rating ≥ 4.0, moderate price")
print("="*65)

prefs1 = {
    'cuisine': 'Italian',
    'min_rating': 4.0,
    'price_range': 2
}
result1 = recommend_restaurants(prefs1, top_n=5)
print(result1.to_string())

print("\n" + "="*65)
print("TEST 2 – Indian food with online delivery, rating ≥ 3.5")
print("="*65)

prefs2 = {
    'cuisine': 'Indian',
    'min_rating': 3.5,
    'online_delivery': True
}
result2 = recommend_restaurants(prefs2, top_n=5)
print(result2.to_string())

print("\n" + "="*65)
print("TEST 3 – Cheap eats (price range 1), rating ≥ 3.8")
print("="*65)

prefs3 = {
    'price_range': 1,
    'min_rating': 3.8
}
result3 = recommend_restaurants(prefs3, top_n=5)
print(result3.to_string())

print("\n" + "="*65)
print("TEST 4 – Similar restaurants to a known restaurant")
print("="*65)

sample_name = df['Restaurant Name'].iloc[0]
print(f"Finding restaurants similar to: '{sample_name}'")
result4 = find_similar_restaurants(sample_name, top_n=5)

if result4 is not None:
    print(result4.to_string())

# ─────────────────────────────────────────────────────────────
# STEP 7 – Visualisations
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Restaurant Recommendation System – Analysis',
             fontsize=16, fontweight='bold')

# 7a. Top 10 cuisines
top_cuisines = df['Primary Cuisine'].value_counts().head(10)
axes[0, 0].barh(top_cuisines.index[::-1], top_cuisines.values[::-1], color='steelblue')
axes[0, 0].set_title('Top 10 Most Common Cuisines')
axes[0, 0].set_xlabel('Number of Restaurants')

# 7b. Rating distribution by price range
price_labels = {1: 'Cheap', 2: 'Moderate', 3: 'Expensive', 4: 'Luxury'}
for pr, label in price_labels.items():
    subset = df[df['Price range'] == pr]['Aggregate rating']
    axes[0, 1].hist(subset, bins=15, alpha=0.6, label=label)

axes[0, 1].set_title('Rating Distribution by Price Range')
axes[0, 1].set_xlabel('Aggregate Rating')
axes[0, 1].set_ylabel('Count')
axes[0, 1].legend()

# 7c. Delivery vs average rating
delivery_rating = df.groupby('Has Online delivery')['Aggregate rating'].mean()
bars = axes[1, 0].bar(
    ['No Delivery', 'Has Delivery'],
    delivery_rating.values,
    color=['salmon', 'mediumseagreen']
)
axes[1, 0].set_title('Avg Rating: Delivery vs No Delivery')
axes[1, 0].set_ylabel('Average Rating')

for bar in bars:
    axes[1, 0].text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.01,
        f'{bar.get_height():.3f}',
        ha='center'
    )

# 7d. Top 10 cities
top_cities = df['City'].value_counts().head(10)
axes[1, 1].barh(top_cities.index[::-1], top_cities.values[::-1], color='mediumpurple')
axes[1, 1].set_title('Top 10 Cities by Restaurant Count')
axes[1, 1].set_xlabel('Number of Restaurants')

plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'task2_analysis.png'), dpi=150, bbox_inches='tight')
print("\n✅ Plot saved as task2_analysis.png")

# ─────────────────────────────────────────────────────────────
# STEP 8 – Key Findings
# ─────────────────────────────────────────────────────────────
print("""
── Key Findings ───────────────────────────────────────────────
• This system uses content-based filtering to recommend
  restaurants based on cuisine, price, rating, city,
  online delivery, and table booking.
• Preference-based recommendations filter and rank
  restaurants according to user needs.
• Similarity-based recommendations use cosine similarity
  to find restaurants with similar feature profiles.
• Weighted scoring using rating and votes improves the
  quality of recommendations.
• This system can be extended with user behavior and
  collaborative filtering for better personalization.
───────────────────────────────────────────────────────────────
""")