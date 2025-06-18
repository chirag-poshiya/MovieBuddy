import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load data
movies = pd.read_csv('data/movies.csv')
ratings = pd.read_csv('data/ratings.csv')

# Merge
data = pd.merge(ratings, movies, on='movieId')
data['genre_main'] = data['genres'].apply(lambda x: x.split('|')[0])


# Encode
le = LabelEncoder()
data['genre_encoded'] = le.fit_transform(data['genre_main'])
data['user_encoded'] = le.fit_transform(data['userId'])
data['movie_encoded'] = le.fit_transform(data['movieId'])

X = data[['genre_encoded', 'user_encoded', 'movie_encoded']]
y = data['rating']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Result
score = model.score(X_test, y_test)
print(f"Model Accuracy: {score:.2f}")
