import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load datasets
ratings = pd.read_csv('data/ratings.csv')
movies = pd.read_csv('data/movies.csv')

# Step 1: Create user-movie matrix
user_movie = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Step 2: Apply KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
user_movie['cluster'] = kmeans.fit_predict(user_movie)

# Step 3: Merge cluster info back with ratings
user_cluster_df = user_movie['cluster'].reset_index()
ratings_with_cluster = pd.merge(ratings, user_cluster_df, on='userId')
ratings_with_cluster = pd.merge(ratings_with_cluster, movies[['movieId', 'genres']], on='movieId')

# Step 4: Extract main genre (first one)
ratings_with_cluster['main_genre'] = ratings_with_cluster['genres'].apply(lambda x: x.split('|')[0])

# Step 5: Find most liked genre in each cluster
for cluster_id in sorted(ratings_with_cluster['cluster'].unique()):
    print(f"\n🔹 Cluster {cluster_id} users like:")
    cluster_data = ratings_with_cluster[ratings_with_cluster['cluster'] == cluster_id]
    top_genres = cluster_data.groupby('main_genre')['rating'].mean().sort_values(ascending=False).head(5)
    print(top_genres)
