# Task 1 - Restaurant Rating Prediction

## 📌 Objective
The objective of this task is to build a **Machine Learning Regression Model** that predicts the **aggregate rating of restaurants** using restaurant-related features from the dataset.

This task was completed as part of the **Cognifyz Machine Learning Internship**.

---

## 📂 Dataset
The dataset used in this task contains restaurant information such as:

- Country Code
- City
- Cuisines
- Average Cost for Two
- Price Range
- Table Booking Availability
- Online Delivery Availability
- Votes
- Aggregate Rating

The target variable for prediction is:

- **Aggregate rating**

---

## 🛠️ Technologies Used
- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Scikit-learn**

---

## ⚙️ Workflow

### 1. Data Loading
The dataset is loaded using **Pandas** and basic information about the dataset is displayed.

### 2. Data Preprocessing
The following preprocessing steps were performed:

- Removed restaurants with **0 rating** (unrated restaurants)
- Filled missing values in:
  - `Cuisines`
  - `City`
- Converted binary categorical columns into numerical format:
  - `Has Table booking`
  - `Has Online delivery`
  - `Is delivering now`
  - `Switch to order menu`
- Extracted the **Primary Cuisine** from the `Cuisines` column
- Applied **Label Encoding** on:
  - `City`
  - `Primary Cuisine`

### 3. Feature Selection
The following features were used for prediction:

- `Country Code`
- `Average Cost for two`
- `Price range`
- `Has Table booking`
- `Has Online delivery`
- `Is delivering now`
- `Votes`
- `City_enc`
- `Cuisine_enc`

### 4. Model Training
Three regression models were trained and compared:

- **Linear Regression**
- **Decision Tree Regressor**
- **Random Forest Regressor**

### 5. Model Evaluation
The models were evaluated using:

- **MSE (Mean Squared Error)**
- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**
- **R² Score**

### 6. Feature Importance Analysis
Feature importance was extracted using the **Random Forest Regressor** to understand which features contributed most to restaurant rating prediction.

### 7. Visualization
The project generates a visualization file named:

- `model_analysis.png`

This includes:
- Rating distribution
- Feature importance plot
- R² comparison of models
- Actual vs Predicted scatter plots

---

## 📊 Key Findings
- **Random Forest Regressor** performed the best among the tested models.
- **Votes** was one of the most important predictors of restaurant ratings.
- **Price range**, **city**, and **cuisine type** also influenced ratings.
- Tree-based models performed better than Linear Regression, indicating **non-linear patterns** in the data.

---

## 📁 Project Structure

```bash
Task 1/
├── main.py
├── Dataset.csv
├── model_analysis.png
└── README.md
