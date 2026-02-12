import pandas as pd
import matplotlib.pyplot as plt

# Replace 'ratings.csv' with the actual path to your file
ratings_df = pd.read_csv('ratings.csv')

# 1. Show the first few rows to confirm the structure
print("First few rows:")
print(ratings_df.head())

# 2. Basic info: number of rows and columns, data types, etc.
print("\nInfo about the dataset:")
print(ratings_df.info())

# 3. Number of unique users and movies
n_users = ratings_df['userId'].nunique()
n_movies = ratings_df['movieId'].nunique()

print(f"\nNumber of unique users: {n_users}")
print(f"Number of unique movies: {n_movies}")

# 4. Total number of ratings
n_ratings = len(ratings_df)
print(f"Total number of ratings: {n_ratings}")

# 5. Average rating
avg_rating = ratings_df['rating'].mean()
print(f"Average rating: {avg_rating:.2f}")

# 6. Rating distribution
print("\nRating distribution (counts):")
print(ratings_df['rating'].value_counts().sort_index())

# Optional: visualize the rating distribution
ratings_df['rating'].value_counts().sort_index().plot(kind='bar')
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()

# 7. Calculate sparsity (or density) of the user-item matrix
#    Sparsity = 1 - (number of observed ratings) / (total possible ratings)
#    Density = (number of observed ratings) / (total possible ratings)
total_possible_ratings = n_users * n_movies
density = n_ratings / total_possible_ratings
sparsity = 1 - density

print(f"\nMatrix density: {density*100:.2f}%")
print(f"Matrix sparsity: {sparsity*100:.2f}%")

# 8. Check basic statistics on timestamps (if you need to understand temporal aspects)
if 'timestamp' in ratings_df.columns:
    print("\nTimestamp range:")
    print(f"Min timestamp: {ratings_df['timestamp'].min()}")
    print(f"Max timestamp: {ratings_df['timestamp'].max()}")
