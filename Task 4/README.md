# 📍 Task 4 – Location-Based Restaurant Analysis

---

## 📌 Project Overview

This project focuses on geographical analysis of restaurants using location-based features such as latitude, longitude, city, and locality.
The goal is to uncover patterns in restaurant distribution, pricing, ratings, and cuisine trends across different regions.

---

## 🎯 Objectives

- Analyze restaurant distribution across cities and localities
- Identify high-density restaurant regions
- Compare ratings, cost, and popularity across locations
- Discover dominant cuisines in different cities
- Visualize geographic data using maps and plots

---

## 🛠️ Technologies Used

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas & NumPy | Data processing |
| Matplotlib & Seaborn | Visualization |
| Folium | Interactive maps |

---

## 📂 Dataset

- **File:** `Dataset .csv`
- Contains information such as:
  - Restaurant Name
  - City & Locality
  - Latitude & Longitude
  - Cuisines
  - Ratings & Votes
  - Average Cost for Two
  - Online Delivery Availability

---

## ⚙️ Key Steps Performed

### 🔹 1. Data Preprocessing

- Handled missing values
- Extracted Primary Cuisine
- Removed invalid coordinates
- Encoded categorical features
- Removed outliers for better accuracy

### 🔹 2. Feature Engineering

- Created Weighted Rating (based on votes)
- Normalized cost across cities
- Prepared data for fair comparison

### 🔹 3. Location-Based Analysis

- City-wise restaurant count
- Locality-wise distribution
- Average rating, cost, and votes per city
- Online delivery trends

### 🔹 4. Advanced Insights

- Top cities based on weighted ratings
- Most expensive cities
- Cuisine dominance per city
- Correlation between cost, rating, and votes

### 🔹 5. Visualizations

- 📊 Bar charts (Top cities, ratings, cost)
- 📍 Scatter plot (Latitude vs Longitude colored by rating)
- 🔥 Density heatmap (restaurant clusters)
- 📉 Rating vs Cost analysis

### 🔹 6. Interactive Map

- Created using Folium
- Displays restaurant locations
- Includes cuisine and rating info on markers

---

## 📊 Key Insights

- Restaurants are highly concentrated in urban regions
- Cost does not strongly correlate with rating
- Some cities have high ratings but fewer restaurants
- Online delivery availability varies significantly across cities
- Weighted ratings provide more reliable rankings than raw ratings
- Certain cuisines dominate specific cities reflecting local food culture

---

## 📁 Output Files

| File | Description |
|---|---|
| `task4_analysis.png` | All charts combined |
| `restaurant_map.html` | Interactive Folium map |

---

## 🚀 How to Run

```bash
pip install pandas numpy matplotlib seaborn folium
python task4_location_analysis.py
```

---

## 📁 Project Structure

```
Task 4/
├── task4_location_analysis.py
├── Dataset .csv
├── task4_analysis.png
├── restaurant_map.html
└── README.md
```

---

## 💡 Future Improvements

- Build a Streamlit dashboard for interactive analysis
- Apply clustering (K-Means) for location segmentation
- Integrate real-time map APIs
- Perform predictive modeling on restaurant success by region

---

## 🧑‍💻 Author

**Arka Bose**
B.Tech CSE | Dr. B.C. Roy Engineering College, Durgapur
Cognifyz Machine Learning Internship

---

## ⭐ Project Status

- ✔ Completed
- ✔ Ready for submission
- ✔ Resume-ready
