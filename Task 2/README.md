# Task 2 - Restaurant Recommendation System

## 📌 Objective
The objective of this task is to build a **Restaurant Recommendation System** that suggests restaurants based on user preferences and restaurant similarity.

This task was completed as part of the **Cognifyz Machine Learning Internship**.

---

## 📂 Dataset
The same restaurant dataset used in Task 1 was used for this recommendation system.

The dataset contains information such as:

- Restaurant Name
- City
- Cuisines
- Aggregate Rating
- Votes
- Price Range
- Table Booking Availability
- Online Delivery Availability

---

## 🛠️ Technologies Used
- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Scikit-learn**

---

## 🧠 Recommendation Approach
This project uses **Content-Based Filtering** to recommend restaurants.

Instead of relying on user history, the system compares restaurants based on their features and suggests restaurants that are either:

1. **Best matching for user preferences**
2. **Most similar to a given restaurant**

### Features Used for Recommendation
The following features were used to build the recommendation system:

- `Primary Cuisine`
- `Price range`
- `Aggregate rating`
- `Has Table booking`
- `Has Online delivery`
- `City`

### Similarity Technique Used
The project uses **Cosine Similarity** on a normalized feature matrix to measure how similar restaurants are to each other.

---

## ⚙️ Workflow

### 1. Data Loading
The dataset is loaded using **Pandas** and column names are cleaned.

### 2. Data Preprocessing
The following preprocessing steps were performed:

- Removed restaurants with **0 rating**
- Filled missing values in:
  - `Cuisines`
  - `City`
  - `Votes`
- Extracted **Primary Cuisine** from the `Cuisines` column
- Converted binary categorical columns into numerical format:
  - `Has Table booking`
  - `Has Online delivery`
- Applied **Label Encoding** on:
  - `City`
  - `Primary Cuisine`
- Normalized selected features using **MinMaxScaler**

### 3. Similarity Matrix Construction
A **Cosine Similarity Matrix** was created using the processed feature matrix to measure similarity between restaurants.

---

## 🍽️ Recommendation Modes

### Mode 1 - Preference-Based Recommendation
This mode recommends restaurants based on user-defined preferences such as:

- Preferred cuisine
- Minimum rating
- Price range
- Online delivery availability
- Table booking availability

Restaurants are ranked using a **weighted score**:

```text id="g2wzkm"
Score = (Aggregate Rating × 0.7) + (Normalized Votes × 0.3)
