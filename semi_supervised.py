import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load data
movies = pd.read_csv('data/movies.csv')
ratings = pd.read_csv('data/ratings.csv')

# Merge to get genres
data = pd.merge(ratings, movies, on='movieId')

# Use only the first genre
data['main_genre'] = data['genres'].apply(lambda x: x.split('|')[0])

# Encode genre, userId, movieId
le_genre = LabelEncoder()
le_user = LabelEncoder()
le_movie = LabelEncoder()

data['genre_encoded'] = le_genre.fit_transform(data['main_genre'])
data['user_encoded'] = le_user.fit_transform(data['userId'])
data['movie_encoded'] = le_movie.fit_transform(data['movieId'])

# Convert rating into classes (round to nearest integer, 0–5)
data['rating_class'] = data['rating'].round().astype(int)

# Features (input)
X = data[['genre_encoded', 'user_encoded', 'movie_encoded']]
# Labels (output)
y = data['rating_class']

# Split into a small labeled set and large unlabeled set
X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(
    X, y, train_size=0.1, stratify=y, random_state=42
)

# Mark unlabeled data with -1
y_unlabeled[:] = -1

# Combine labeled + unlabeled
X_combined = pd.concat([X_labeled, X_unlabeled])
y_combined = pd.concat([y_labeled, y_unlabeled])

# Use Self-Training with Random Forest
base_model = RandomForestClassifier()
semi_supervised_model = SelfTrainingClassifier(base_model)
semi_supervised_model.fit(X_combined, y_combined)

# Evaluate using original labeled test set
X_test, _, y_test, _ = train_test_split(X, y, test_size=0.2, random_state=42)
score = semi_supervised_model.score(X_test, y_test)

print(f"🔍 Semi-Supervised Model Accuracy: {score:.2f}")
