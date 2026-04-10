# ============================================================
#  Task 4 – Location-Based Analysis
# ============================================================

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import folium

# ─────────────────────────────────────────────────────────────
# STEP 1 – Load Dataset
# ─────────────────────────────────────────────────────────────
base_dir  = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, 'Dataset2.csv')

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

# Fill missing values
for col in ['Cuisines', 'City', 'Locality']:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown')

# Extract primary cuisine
df['Primary Cuisine'] = df['Cuisines'].astype(str).str.split(',').str[0].str.strip()

# Remove invalid coordinates
df = df.dropna(subset=['Latitude', 'Longitude'])
df = df[(df['Latitude'] != 0) & (df['Longitude'] != 0)].copy()

# Encode Yes/No
for col in ['Has Online delivery', 'Has Table booking']:
    if col in df.columns:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

# ───────── OUTLIER REMOVAL (ACCURACY BOOST) ─────────
df = df[(df['Aggregate rating'] >= 1) & (df['Aggregate rating'] <= 5)]

cost_limit = df['Average Cost for two'].quantile(0.99)
df = df[df['Average Cost for two'] <= cost_limit]

df.reset_index(drop=True, inplace=True)

print(f"✅ Cleaned dataset: {df.shape}")

# ─────────────────────────────────────────────────────────────
# STEP 3 – Feature Engineering (Accuracy Boost)
# ─────────────────────────────────────────────────────────────

# Weighted rating (more reliable than avg rating)
df['Weighted Rating'] = df['Aggregate rating'] * np.log1p(df['Votes'])

# Normalize cost per city
df['Normalized Cost'] = df.groupby('City')['Average Cost for two']\
    .transform(lambda x: (x - x.mean()) / x.std())

# ─────────────────────────────────────────────────────────────
# STEP 4 – City-Level Analysis
# ─────────────────────────────────────────────────────────────

city_stats = df.groupby('City').agg(
    Restaurant_Count=('Restaurant Name', 'count'),
    Avg_Rating=('Aggregate rating', 'mean'),
    Weighted_Rating=('Weighted Rating', 'mean'),
    Avg_Cost=('Average Cost for two', 'mean'),
    Avg_Votes=('Votes', 'mean'),
    Online_Delivery=('Has Online delivery', 'mean')
).reset_index()

city_stats['Online_Delivery'] = (city_stats['Online_Delivery'] * 100).round(1)
city_stats = city_stats.sort_values('Restaurant_Count', ascending=False)

print("\n🏙️ Top Cities:\n", city_stats.head(10))

# ─────────────────────────────────────────────────────────────
# STEP 5 – Advanced Insights
# ─────────────────────────────────────────────────────────────

# Top rated (weighted)
top_weighted = city_stats[city_stats['Restaurant_Count'] >= 10]\
    .nlargest(5, 'Weighted_Rating')

# Expensive cities
expensive = city_stats[city_stats['Restaurant_Count'] >= 10]\
    .nlargest(5, 'Avg_Cost')

print("\n⭐ Top Cities (Weighted Rating):\n", top_weighted[['City','Weighted_Rating']])
print("\n💰 Most Expensive Cities:\n", expensive[['City','Avg_Cost']])

# Top cuisine per city
top_cuisine = df.groupby(['City', 'Primary Cuisine'])\
    .size().reset_index(name='Count')

top_cuisine = top_cuisine.sort_values(['City','Count'], ascending=[True, False])\
    .drop_duplicates('City')

print("\n🍽️ Top Cuisine per City:\n", top_cuisine.head(10))

# ─────────────────────────────────────────────────────────────
# STEP 6 – Correlation Analysis
# ─────────────────────────────────────────────────────────────

corr = df[['Aggregate rating', 'Average Cost for two', 'Votes']].corr()
print("\n📊 Correlation Matrix:\n", corr)

# ─────────────────────────────────────────────────────────────
# STEP 7 – Visualizations
# ─────────────────────────────────────────────────────────────

# Top Cities Plot
plt.figure(figsize=(10,5))
city_stats.head(10).plot(x='City', y='Restaurant_Count', kind='bar')
plt.title("Top Cities by Restaurants")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'top_cities.png'))

# Density Heatmap (IMPORTANT)
plt.figure(figsize=(8,6))
sns.kdeplot(x=df['Longitude'], y=df['Latitude'], fill=True)
plt.title("Restaurant Density Heatmap")
plt.savefig(os.path.join(base_dir, 'density_map.png'))

# Rating vs Cost
plt.figure(figsize=(8,6))
plt.scatter(df['Average Cost for two'], df['Aggregate rating'], alpha=0.3)
plt.title("Rating vs Cost")
plt.xlabel("Cost")
plt.ylabel("Rating")
plt.savefig(os.path.join(base_dir, 'rating_vs_cost.png'))

print("✅ Visualizations saved")

# ─────────────────────────────────────────────────────────────
# STEP 8 – Interactive Map (BONUS ⭐)
# ─────────────────────────────────────────────────────────────

map_center = [df['Latitude'].mean(), df['Longitude'].mean()]
m = folium.Map(location=map_center, zoom_start=5)

sample_df = df.sample(min(1000, len(df)))

for _, row in sample_df.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=3,
        popup=f"{row['Primary Cuisine']} | Rating: {row['Aggregate rating']}",
        fill=True
    ).add_to(m)

m.save(os.path.join(base_dir, "restaurant_map.html"))

print("🌍 Interactive map saved")

# ─────────────────────────────────────────────────────────────
# STEP 9 – Final Insights
# ─────────────────────────────────────────────────────────────

print("\n📊 FINAL INSIGHTS")

print(f"""
• Total Cities: {df['City'].nunique()}
• Total Localities: {df['Locality'].nunique()}

• Most Restaurants: {city_stats.iloc[0]['City']}
• Best City (Weighted): {top_weighted.iloc[0]['City']}

• Strong clustering observed in urban areas (density heatmap)

• Weak correlation between cost and rating → expensive ≠ better

• Weighted ratings provide more reliable ranking

• Dominant cuisine: {df['Primary Cuisine'].value_counts().idxmax()}
""")

print("\n✅ TASK 4 COMPLETED (PRO LEVEL)")