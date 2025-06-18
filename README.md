# 🎬 Movie Buddy: Smart Movie Recommender

This project demonstrates how to build a basic movie recommendation system using real-world MovieLens data. It includes examples of three types of machine learning:

- ✅ Supervised Learning
- 🎯 Unsupervised Learning
- 🧠 Semi-Supervised Learning

---

## 📁 Folder Structure

MovieBuddy/
│
├── data/
│ ├── movies.csv ← Movie metadata
│ └── ratings.csv ← User ratings
│
├── supervised_learning.py ← Predict movie ratings using labeled data
├── unsupervised_learning.py ← Group users with similar taste
├── semi_supervised.py ← Predict ratings with limited labeled data
└── README.md
---

## 🔹 1. `supervised_learning.py` – Predict Movie Ratings

**Type:** Supervised Learning  
**Goal:** Predict how much a user will rate a movie using known ratings.

### 🧠 Inputs:
- Genre of movie
- User ID
- Movie ID

### 🎯 Output:
- Predicted rating (from 0 to 5)

### 🔧 How it works:
- Uses `RandomForestRegressor`
- Trains on known ratings
- Evaluates performance using accuracy score

---

## 🔹 2. `unsupervised_learning.py` – Group Users

**Type:** Unsupervised Learning  
**Goal:** Group users based on the movies they watch and rate.

### 🧠 Input:
- User-movie rating matrix (rows = users, columns = movies)

### 🎯 Output:
- User clusters like:
  - Action Lovers
  - Comedy Fans
  - Drama Addicts

### 🔧 How it works:
- Uses `KMeans` clustering
- Groups users without using any labels
- Analyzes top genres liked by each cluster

---

## 🔹 3. `semi_supervised.py` – Guess Ratings from Few Labels

**Type:** Semi-Supervised Learning  
**Goal:** Predict movie ratings with only a small portion of labeled data.

### 🧠 Input:
- Only 10% of rating data is labeled
- 90% is treated as unknown

### 🎯 Output:
- Model learns to predict the rest using `SelfTrainingClassifier`

### 🔧 How it works:
- Combines labeled and unlabeled data
- Uses `RandomForestClassifier` inside `SelfTrainingClassifier`
- Trains step-by-step by self-labeling

---

## 📥 Dataset Used

Download from: [https://grouplens.org/datasets/movielens/](https://grouplens.org/datasets/movielens/)  
Use: **ml-latest-small.zip**

Place `movies.csv` and `ratings.csv` inside the `data/` folder.

---

## 🚀 Requirements

Install dependencies using pip:

```bash
pip install pandas scikit-learn matplotlib numpy seaborn
```

---

## 🙌 Credits

Created by **Chirag Poshiya** as a learning project to explore various types of Machine Learning with real datasets.

> Feel free to fork, improve, and share. This is part of my journey to become an AI/ML expert.

📫 Connect with me on [LinkedIn](https://www.linkedin.com/in/chiragposhiya)
