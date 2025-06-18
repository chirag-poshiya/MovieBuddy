import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load ratings data
ratings = pd.read_csv('data/ratings.csv')

# Create a user-movie rating table (rows = users, columns = movies)
user_movie = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Apply KMeans clustering to group users
kmeans = KMeans(n_clusters=4, random_state=42)
user_movie['cluster'] = kmeans.fit_predict(user_movie)

# See how many users are in each group
print(user_movie['cluster'].value_counts())

# Plot the cluster distribution
plt.hist(user_movie['cluster'], bins=range(5), edgecolor='black')
plt.title("User Clusters")
plt.xlabel("Cluster ID")
plt.ylabel("Number of Users")
plt.show()
