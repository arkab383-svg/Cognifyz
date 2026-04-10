# 🍽️ Task 3 – Cuisine Classification

---

## 📌 Objective

The objective of this task is to build a machine learning model that can classify the type of cuisine a restaurant serves based on features such as location, pricing, and customer engagement.

---

## 📂 Dataset

The dataset contains information about restaurants including:

- City
- Country Code
- Average Cost for Two
- Price Range
- Votes
- Table Booking Availability
- Online Delivery Availability
- Cuisines

---

## ⚙️ Steps Performed

### 1. Data Preprocessing

- Removed restaurants with zero ratings
- Handled missing values
- Extracted primary cuisine from multiple cuisines
- Selected top 10 cuisines to reduce class imbalance
- Converted categorical features into numerical format

### 2. Feature Engineering

- Encoded city using Label Encoding
- Converted binary features (Yes/No → 1/0)
- Scaled numerical features using MinMaxScaler

### 3. Model Building

Trained and compared multiple models:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

### 4. Model Evaluation

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score (weighted for class imbalance)

---

## 📊 Results

- **Best Model:** Random Forest Classifier
- Random Forest achieved the highest performance across evaluation metrics
- Class imbalance affects prediction for less frequent cuisines

---

## 📈 Visualizations

The following visualizations were generated:

- Cuisine distribution
- Model accuracy comparison
- Confusion matrix
- Feature importance

**Output file:** `task3_analysis.png`

---

## 🔍 Key Insights

- Location and pricing significantly influence cuisine prediction
- Popular cuisines are easier to classify than rare ones
- Random Forest performs best due to its ensemble nature

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| Python | Core language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | ML models & evaluation |
| Matplotlib | Plotting |
| Seaborn | Advanced visualizations |

---

## ▶️ How to Run

```bash
cd "Task 3"
python3 task3_cuisine_classification.py
```

---

## 📁 Project Structure

```
Task 3/
├── task3_cuisine_classification.py
├── Dataset .csv
├── task3_analysis.png
└── README.md
```

---

## 📌 Conclusion

This project demonstrates a complete machine learning pipeline, including data preprocessing, feature engineering, model training, evaluation, and visualization. It highlights how structured data can be used to solve real-world classification problems.

---

## 🚀 Future Improvements

- Use NLP techniques (TF-IDF) on cuisine text
- Try advanced models like XGBoost
- Deploy as a web app using Flask or Streamlit
